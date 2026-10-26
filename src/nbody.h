/*
 * nbody.h - shared declarations for the hybrid MPI + OpenMP + CUDA
 * Barnes-Hut N-body simulator with an adaptive OpenMP loop-schedule controller.
 *
 * Units: G = 1.  Softened gravity  a_i = sum_j m_j (x_j - x_i) / (r^2 + eps^2)^(3/2).
 */
#ifndef NBODY_H
#define NBODY_H

#include <stddef.h>
#include <stdint.h>

#define MAXLEVEL  21      /* Morton bits per axis                                  */
#define STACK_MAX 256     /* tree-walk stack (max ~7*21+1 entries are ever used)   */
#define PAR_MIN   32768   /* below this N the sort and tree build run on one thread: measured
                             no gain there (libgomp barriers cost ~50 us on Windows)    */

/* High-resolution wall clock in seconds (QueryPerformanceCounter / CLOCK_MONOTONIC).
 * MinGW's omp_get_wtime() only has millisecond resolution, too coarse for per-thread
 * loop timing. */
double wtime(void);

/* ------------------------------------------------------------------ bodies */
typedef struct {
    int n;
    double *x, *y, *z, *vx, *vy, *vz, *m;   /* replicated on every MPI rank           */
    double *ax, *ay, *az;                   /* valid for the rank's own block only    */
    double *cost;                           /* work units per body (own block only,
                                               all bodies with --partition cost)      */
    int sort_cost;                          /* morton_sort also permutes cost[]       */
    int *id;                                /* original body index                    */
    uint64_t *key;                          /* Morton keys, sorted after morton_sort  */
    double bmin[3], bside;                  /* bounding cube of the current step      */
    /* scratch for sorting / permuting */
    uint64_t *key2;
    uint32_t *idx, *idx2;
    double *tmp[8];                         /* gather targets for x..m, cost (swapped in) */
    int *tmpi;
} bodies_t;

int  bodies_alloc(bodies_t *b, int n);
void bodies_free(bodies_t *b);

/* initial conditions: uniform | plummer | collide | collapse | virial | disk, plus optional
 * solid-body rotation (spin, see init.c); eps is the softening (disk circular velocities).
 * Returns 0 on success. */
int  ic_generate(bodies_t *b, const char *kind, int n, uint64_t seed, double spin, double eps);

/* Morton (Z-order) sort of all bodies; permutes x,y,z,vx,vy,vz,m,id (and cost if sort_cost).
 * Runs on the current OpenMP thread count; the result does not depend on it. */
void morton_sort(bodies_t *b);

/* Split [0,n) into nranks Morton-contiguous blocks of (nearly) equal summed cost.
 * Deterministic: every rank computes the same boundaries from the same cost[]. */
void partition_by_cost(const double *cost, int n, int nranks, int *cnts, int *disp);

/* -------------------------------------------------------------------- tree */
typedef struct {
    double cx, cy, cz;      /* centre of mass                       */
    double m;               /* mass                                 */
    double s2;              /* (cell edge length)^2                 */
    int child0, nchild;     /* children are nodes[child0 .. +nchild); nchild==0 -> leaf */
    int lo, hi;             /* body index range [lo,hi) (Morton-contiguous)             */
} node_t;

typedef struct {
    node_t *nodes;
    int cap, count, leafcap;
} tree_t;

int  tree_init(tree_t *t, int n, int leafcap);
void tree_build(tree_t *t, const bodies_t *b);    /* OpenMP tasks; result independent of threads */
void tree_free(tree_t *t);

/* ------------------------------------------------------ schedule controller */
typedef enum { SCHED_STATIC, SCHED_DYNAMIC, SCHED_GUIDED, SCHED_ADAPTIVE } sched_kind_t;
typedef enum { POL_STATIC, POL_DYNAMIC, POL_GUIDED } policy_t;
typedef enum { BAL_DYNAMIC, BAL_GUIDED, BAL_AUTO } balanced_t;

#define CTL_MAXCAND 4

typedef struct {
    sched_kind_t kind;
    policy_t pol;           /* policy currently installed via omp_set_schedule */
    int cur_chunk;          /* chunk size currently installed                  */
    balanced_t balanced;    /* adaptive: what to use when imbalanced           */
    int chunk;              /* chunk size for dynamic / guided                 */
    int k;                  /* controller check interval (steps)               */
    double low, high;       /* hysteresis thresholds on the imbalance          */
    int switches;           /* number of static <-> balanced changes           */
    double last_lambda;     /* lambda used at the most recent check            */
    /* --balanced auto: candidate balanced schedules, each tried for one window */
    int ncand, trial;       /* trial = candidate under test, -1 = not trialling */
    policy_t cand_pol[CTL_MAXCAND];
    int cand_chunk[CTL_MAXCAND];
    double cand_rate[CTL_MAXCAND];          /* measured work units per second  */
    int chosen;             /* candidate picked by the last trial, -1 = none   */
    int trials;             /* number of completed trial rounds                */
    /* per-thread accumulators over the current k-step window */
    int maxthr, nthr, nobs;
    double *acc_t;          /* measured loop time                              */
    double *acc_w;          /* work units actually executed                    */
    double *acc_b;          /* work units of the thread's static block         */
    double acc_wall;        /* sum over steps of the slowest thread's time     */
} controller_t;

int         ctl_init(controller_t *c, sched_kind_t kind, int chunk, int k, double low, double high, int maxthr,
                     balanced_t balanced);
void        ctl_free(controller_t *c);
void        ctl_apply(const controller_t *c);
void        ctl_observe(controller_t *c, const double *thr_time, const double *thr_work,
                        const double *cost, int lo, int hi, int nthr);
int         ctl_update(controller_t *c, int step);
const char *ctl_policy_name(policy_t p);
const char *ctl_balanced_name(balanced_t b);
double      imbalance_of(const double *v, int n);   /* max/mean - 1 */
void        static_blocks(const double *cost, int lo, int hi, int nthreads, double *out);
double      static_equiv_imbalance(const double *cost, int lo, int hi, int nthreads);
double      predicted_static_imbalance(const double *block, const double *t, const double *w, int n);

/* Pin OpenMP threads (global index rank*threads+t) to logical CPUs.  cpus = NULL/"" picks
 * one logical CPU per physical core, fastest cores first.  desc receives the mapping. */
int pin_threads(int rank, int threads, const char *cpus, char *desc, size_t desc_len);

/* --------------------------------------------------------------- simulation */
typedef enum { FORCE_BH, FORCE_DIRECT, FORCE_GPU_DIRECT, FORCE_BH_GPU } force_mode_t;

typedef struct {
    /* configuration */
    int n, rank, nranks, lo, hi;            /* own block is [lo,hi)                  */
    int max_own;                            /* largest own block possible (buffers)  */
    double eps, eps2, theta, theta2, dt;
    int leafcap;
    force_mode_t mode;
    int threads;

    bodies_t b;
    tree_t tree;

    /* per-step outputs of forces_compute() */
    double *thr_time;                       /* per-thread loop time (size threads)   */
    double *thr_work;                       /* per-thread executed work units        */
    unsigned char *owner;                   /* thread that computed each body (own block) */
    int record_owner;                       /* fill owner[] in this force pass (snapshots) */
    int nthr;                               /* threads actually used                 */
    double t_force;                         /* wall time of the CPU force phase      */
    double t_gpu;                           /* wall time spent in GPU phases         */

    /* GPU near-field lists for FORCE_BH_GPU */
    int near_cap;
    int *near_rng, *near_cnt;
    long long near_overflow;
    long long stat_visits, stat_inter;
} sim_t;

int  sim_init_forces(sim_t *s, int gpu_double);   /* allocate force scratch (+GPU)     */
void sim_free_forces(sim_t *s);
int  forces_compute(sim_t *s);                     /* fills ax,ay,az,cost for own block; 0 = ok */
void forces_reference_direct(const sim_t *s, double *rx, double *ry, double *rz); /* own block, O(N^2/P) */
void energy_parts(const sim_t *s, double *ke, double *pe_local, double *px, double *py, double *pz);

#endif
