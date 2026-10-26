/*
 * main.c - driver for the hybrid MPI + OpenMP + CUDA Barnes-Hut N-body simulator.
 *
 * Per timestep (velocity-Verlet / leapfrog KDK), on every MPI rank:
 *   1. half-kick + drift the rank's own block of bodies
 *   2. MPI_Allgatherv positions and velocities  -> full replicated state
 *   3. Morton-sort all bodies (identical on every rank) -> block = spatial domain
 *   4. build the Barnes-Hut octree (replicated)
 *   5. compute forces for the own block   (OpenMP schedule(runtime) [+ CUDA])
 *   6. second half-kick of the own block
 *   7. adaptive controller: measure per-thread loop times, maybe switch schedule
 *
 * With --partition cost the rank blocks are not equal body counts: after the sort, the
 * Morton order is cut into nranks pieces of equal summed work (each body's cost from the
 * previous step, all-gathered after the force phase), i.e. Singh et al.'s cost zones.
 */
#include "nbody.h"
#include "gpu.h"

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char ic[32];
    int n, steps;
    double dt, eps, theta, spin;
    int leafcap;
    char force[32], sched[32];
    int threads, chunk, k;
    double low, high;
    int diag, print, verify, energy, gpu_float, near_cap, quiet, pin;
    char cpus[256];
    char balanced[16];
    char partition[16];
    char snap[512];
    int snap_every;
    unsigned long long seed;
    char trace[512], csv[512], label[128];
} cfg_t;

static int g_rank = 0;

static void usage(void)
{
    if (g_rank) return;
    printf(
"Usage: mpiexec -n P nbody.exe [options]\n"
"  --ic uniform|plummer|collide|collapse|virial|disk  initial conditions (plummer)\n"
"  --spin S                                add solid-body rotation about z (0; 1 ~ circular at r_rms)\n"
"  --n N                                   number of bodies          (4096)\n"
"  --steps S                               timesteps                 (100)\n"
"  --dt DT                                 timestep                  (0.01)\n"
"  --eps EPS                               softening length (>0)     (0.05)\n"
"  --theta T                               Barnes-Hut opening angle  (0.5)\n"
"  --leaf L                                max bodies per tree leaf  (8)\n"
"  --force bh|direct|gpu-direct|bh-gpu     force back end            (bh)\n"
"  --sched static|dynamic|guided|adaptive  OpenMP loop schedule      (static)\n"
"  --threads T                             OpenMP threads per rank   (1)\n"
"  --chunk C                               dynamic/guided chunk      (8)\n"
"  --k K                                   controller interval       (5)\n"
"  --low L --high H                        controller thresholds     (0.05, 0.15)\n"
"  --balanced dynamic|guided|auto          adaptive: schedule used when imbalanced (dynamic);\n"
"                                          auto tries dynamic(C), dynamic(8C), guided(C), keeps the fastest\n"
"  --partition equal|cost                  MPI rank blocks: equal body counts or equal work (equal)\n"
"  --diag D                                energy check every D steps (0 = start/end only)\n"
"  --no-energy                             skip all energy/momentum checks\n"
"  --verify                                compare initial forces with direct summation\n"
"  --gpu-float                             single-precision GPU kernels (default double)\n"
"  --near-cap C                            GPU near-field ranges per body (128)\n"
"  --seed S                                random seed               (12345)\n"
"  --print P                               progress line every P steps (0 = off)\n"
"  --trace FILE                            per-step imbalance/policy trace CSV (rank 0)\n"
"  --snap FILE --snap-every K              write positions + owning thread every K steps (5),\n"
"                                          for scripts/animate.py (not timed)\n"
"  --csv FILE                              append one summary row to FILE\n"
"  --label TEXT                            free-form label for the CSV row\n"
"  --quiet                                 only print the RESULT line\n"
"  --pin                                   pin threads, one per physical core, fastest first\n"
"  --cpus LIST                             pin threads to this CPU list, e.g. 0,2,4-7\n");
}

static void die(const char *msg)
{
    fprintf(stderr, "[rank %d] error: %s\n", g_rank, msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    exit(1);
}

static int parse(int argc, char **argv, cfg_t *c)
{
    memset(c, 0, sizeof *c);
    strcpy(c->ic, "plummer"); strcpy(c->force, "bh"); strcpy(c->sched, "static");
    c->n = 4096; c->steps = 100; c->dt = 0.01; c->eps = 0.05; c->theta = 0.5; c->leafcap = 8;
    c->threads = 1; c->chunk = 8; c->k = 5; c->low = 0.05; c->high = 0.15;
    c->diag = 0; c->print = 10; c->energy = 1; c->near_cap = 128; c->seed = 12345ULL;
    strcpy(c->label, "run");
    strcpy(c->balanced, "dynamic");
    strcpy(c->partition, "equal");
    c->snap_every = 5;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
#define NEEDV() do { if (!v) { if (!g_rank) fprintf(stderr, "missing value for %s\n", a); return -1; } i++; } while (0)
        if      (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(); return 1; }
        else if (!strcmp(a, "--verify"))    c->verify = 1;
        else if (!strcmp(a, "--no-energy")) c->energy = 0;
        else if (!strcmp(a, "--gpu-float")) c->gpu_float = 1;
        else if (!strcmp(a, "--quiet"))     c->quiet = 1;
        else if (!strcmp(a, "--pin"))       c->pin = 1;
        else if (!strcmp(a, "--balanced")){ NEEDV(); snprintf(c->balanced, sizeof c->balanced, "%s", v); }
        else if (!strcmp(a, "--partition")){ NEEDV(); snprintf(c->partition, sizeof c->partition, "%s", v); }
        else if (!strcmp(a, "--snap"))    { NEEDV(); snprintf(c->snap, sizeof c->snap, "%s", v); }
        else if (!strcmp(a, "--snap-every")) { NEEDV(); c->snap_every = atoi(v); }
        else if (!strcmp(a, "--cpus"))    { NEEDV(); snprintf(c->cpus, sizeof c->cpus, "%s", v); c->pin = 1; }
        else if (!strcmp(a, "--ic"))      { NEEDV(); snprintf(c->ic, sizeof c->ic, "%s", v); }
        else if (!strcmp(a, "--force"))   { NEEDV(); snprintf(c->force, sizeof c->force, "%s", v); }
        else if (!strcmp(a, "--sched"))   { NEEDV(); snprintf(c->sched, sizeof c->sched, "%s", v); }
        else if (!strcmp(a, "--trace"))   { NEEDV(); snprintf(c->trace, sizeof c->trace, "%s", v); }
        else if (!strcmp(a, "--csv"))     { NEEDV(); snprintf(c->csv, sizeof c->csv, "%s", v); }
        else if (!strcmp(a, "--label"))   { NEEDV(); snprintf(c->label, sizeof c->label, "%s", v); }
        else if (!strcmp(a, "--n"))       { NEEDV(); c->n = atoi(v); }
        else if (!strcmp(a, "--steps"))   { NEEDV(); c->steps = atoi(v); }
        else if (!strcmp(a, "--dt"))      { NEEDV(); c->dt = atof(v); }
        else if (!strcmp(a, "--eps"))     { NEEDV(); c->eps = atof(v); }
        else if (!strcmp(a, "--theta"))   { NEEDV(); c->theta = atof(v); }
        else if (!strcmp(a, "--spin"))    { NEEDV(); c->spin = atof(v); }
        else if (!strcmp(a, "--leaf"))    { NEEDV(); c->leafcap = atoi(v); }
        else if (!strcmp(a, "--threads")) { NEEDV(); c->threads = atoi(v); }
        else if (!strcmp(a, "--chunk"))   { NEEDV(); c->chunk = atoi(v); }
        else if (!strcmp(a, "--k"))       { NEEDV(); c->k = atoi(v); }
        else if (!strcmp(a, "--low"))     { NEEDV(); c->low = atof(v); }
        else if (!strcmp(a, "--high"))    { NEEDV(); c->high = atof(v); }
        else if (!strcmp(a, "--diag"))    { NEEDV(); c->diag = atoi(v); }
        else if (!strcmp(a, "--print"))   { NEEDV(); c->print = atoi(v); }
        else if (!strcmp(a, "--near-cap")){ NEEDV(); c->near_cap = atoi(v); }
        else if (!strcmp(a, "--seed"))    { NEEDV(); c->seed = strtoull(v, NULL, 10); }
        else { if (!g_rank) fprintf(stderr, "unknown option %s (try --help)\n", a); return -1; }
#undef NEEDV
    }
    /* keep the label CSV/RESULT-safe */
    for (char *p = c->label; *p; p++) if (*p == ',' || *p == ' ' || *p == '=') *p = '_';
    return 0;
}

static void sync_state(sim_t *s, const int *cnts, const int *disp)
{
    if (s->nranks == 1) return;
    double *arr[6] = { s->b.x, s->b.y, s->b.z, s->b.vx, s->b.vy, s->b.vz };
    for (int k = 0; k < 6; k++)
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, arr[k], cnts, disp, MPI_DOUBLE, MPI_COMM_WORLD);
}

static void gather_cost(sim_t *s, const int *cnts, const int *disp)
{
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, s->b.cost, cnts, disp, MPI_DOUBLE, MPI_COMM_WORLD);
}

/* total energy and |P| (collective) */
static double g_ke = 0, g_pe = 0;      /* parts of the most recent total_energy() call */

static void total_energy(const sim_t *s, double *E, double *P)
{
    double loc[5], glob[5];
    energy_parts(s, &loc[0], &loc[1], &loc[2], &loc[3], &loc[4]);
    MPI_Allreduce(loc, glob, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    g_ke = glob[0]; g_pe = glob[1];
    *E = glob[0] + glob[1];
    *P = sqrt(glob[2] * glob[2] + glob[3] * glob[3] + glob[4] * glob[4]);
}

/* instantaneous predicted static-schedule imbalance of the last force loop */
static double inst_pred(const sim_t *s)
{
    double blk[256];
    if (s->nthr <= 1 || s->nthr > 256) return 0.0;
    static_blocks(s->b.cost, s->lo, s->hi, s->nthr, blk);
    return predicted_static_imbalance(blk, s->thr_time, s->thr_work, s->nthr);
}

/* ---------------------------------------------------------- snapshots
 * File: "NBSNAP01", int32 n, int32 workers, int32 0; then per frame:
 *   int32 step, float32 time, int32 policy, float32 lambda (rank 0),
 *   float32 worker_time[workers], float32 x[n], y[n], z[n], uint8 worker[n]
 * worker = rank * threads + OpenMP thread that computed the body's force. */
typedef struct { FILE *f; unsigned char *own; double *wt; float *buf; } snap_t;

static void snap_frame(snap_t *sn, const sim_t *s, int step, double t, int pol, double lam,
                       const int *cnts, const int *disp)
{
    const int n = s->n, W = s->nranks * s->threads;
    for (int i = s->lo; i < s->hi; i++) sn->own[i] = (unsigned char)(s->rank * s->threads + s->owner[i]);
    if (s->nranks > 1)
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sn->own, cnts, disp, MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
    MPI_Gather(s->thr_time, s->threads, MPI_DOUBLE, sn->wt, s->threads, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (s->rank != 0 || !sn->f) return;
    const int32_t hdr[2] = { step, pol };
    const float tf = (float)t, lf = (float)lam;
    fwrite(&hdr[0], 4, 1, sn->f); fwrite(&tf, 4, 1, sn->f); fwrite(&hdr[1], 4, 1, sn->f); fwrite(&lf, 4, 1, sn->f);
    for (int w = 0; w < W; w++) sn->buf[w] = (float)sn->wt[w];
    fwrite(sn->buf, 4, (size_t)W, sn->f);
    const double *arr[3] = { s->b.x, s->b.y, s->b.z };
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < n; i++) sn->buf[i] = (float)arr[k][i];
        fwrite(sn->buf, 4, (size_t)n, sn->f);
    }
    fwrite(sn->own, 1, (size_t)n, sn->f);
}

static unsigned long long checksum(const bodies_t *b)
{
    int *pos = (int *)malloc(sizeof(int) * (size_t)b->n);
    if (!pos) return 0;
    for (int i = 0; i < b->n; i++) pos[b->id[i]] = i;
    unsigned long long h = 1469598103934665603ULL;            /* FNV-1a 64 */
    for (int id = 0; id < b->n; id++) {
        const int i = pos[id];
        const double v[6] = { b->x[i], b->y[i], b->z[i], b->vx[i], b->vy[i], b->vz[i] };
        const unsigned char *p = (const unsigned char *)v;
        for (size_t k = 0; k < sizeof v; k++) { h ^= p[k]; h *= 1099511628211ULL; }
    }
    free(pos);
    return h;
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    g_rank = rank;
    (void)wtime();                    /* initialise the clock before any parallel region */

    cfg_t cfg;
    int pr = parse(argc, argv, &cfg);
    if (pr != 0) { MPI_Finalize(); return pr > 0 ? 0 : 2; }

    /* ---------------------------------------------------------- validation */
    force_mode_t mode;
    if      (!strcmp(cfg.force, "bh"))         mode = FORCE_BH;
    else if (!strcmp(cfg.force, "direct"))     mode = FORCE_DIRECT;
    else if (!strcmp(cfg.force, "gpu-direct")) mode = FORCE_GPU_DIRECT;
    else if (!strcmp(cfg.force, "bh-gpu"))     mode = FORCE_BH_GPU;
    else die("--force must be bh, direct, gpu-direct or bh-gpu");

    sched_kind_t sk;
    if      (!strcmp(cfg.sched, "static"))   sk = SCHED_STATIC;
    else if (!strcmp(cfg.sched, "dynamic"))  sk = SCHED_DYNAMIC;
    else if (!strcmp(cfg.sched, "guided"))   sk = SCHED_GUIDED;
    else if (!strcmp(cfg.sched, "adaptive")) sk = SCHED_ADAPTIVE;
    else die("--sched must be static, dynamic, guided or adaptive");

    if (cfg.n < 2 || cfg.n < nranks) die("--n must be >= 2 and >= number of MPI ranks");
    if (cfg.steps < 0) die("--steps must be >= 0");
    if (!(cfg.dt > 0)) die("--dt must be > 0");
    if (!(cfg.eps > 0)) die("--eps must be > 0");
    if (!(cfg.theta > 0)) die("--theta must be > 0");
    if (cfg.leafcap < 1) die("--leaf must be >= 1");
    if (cfg.threads < 1) die("--threads must be >= 1");
    if (cfg.chunk < 1) die("--chunk must be >= 1");
    if (cfg.k < 1) die("--k must be >= 1");
    if (!(cfg.low < cfg.high)) die("--low must be smaller than --high");
    balanced_t balanced = BAL_DYNAMIC;
    if      (!strcmp(cfg.balanced, "dynamic")) balanced = BAL_DYNAMIC;
    else if (!strcmp(cfg.balanced, "guided"))  balanced = BAL_GUIDED;
    else if (!strcmp(cfg.balanced, "auto"))    balanced = BAL_AUTO;
    else die("--balanced must be dynamic, guided or auto");
    if (strcmp(cfg.partition, "equal") && strcmp(cfg.partition, "cost")) die("--partition must be equal or cost");
    const int part_cost = !strcmp(cfg.partition, "cost") && nranks > 1;

    omp_set_dynamic(0);
    omp_set_num_threads(cfg.threads);
    if (cfg.pin) {
        char desc[1024];
        int prc = pin_threads(rank, cfg.threads, cfg.cpus, desc, sizeof desc);
        if (!cfg.quiet) { printf("pinning %s%s\n", desc, prc ? "  [FAILED]" : ""); fflush(stdout); }
        else if (prc) fprintf(stderr, "[rank %d] warning: thread pinning failed\n", rank);
    }

    /* --------------------------------------------------------------- setup */
    sim_t s;
    memset(&s, 0, sizeof s);
    s.n = cfg.n; s.rank = rank; s.nranks = nranks;
    s.lo = (int)((long long)cfg.n * rank / nranks);
    s.hi = (int)((long long)cfg.n * (rank + 1) / nranks);
    s.eps = cfg.eps; s.eps2 = cfg.eps * cfg.eps;
    s.theta = cfg.theta; s.theta2 = cfg.theta * cfg.theta;
    s.dt = cfg.dt; s.leafcap = cfg.leafcap; s.mode = mode; s.threads = cfg.threads;
    s.near_cap = cfg.near_cap;
    s.max_own = part_cost ? cfg.n - (nranks - 1) : s.hi - s.lo;

    int *cnts = (int *)malloc(sizeof(int) * (size_t)nranks);
    int *disp = (int *)malloc(sizeof(int) * (size_t)nranks);
    for (int r = 0; r < nranks; r++) {
        int lo = (int)((long long)cfg.n * r / nranks), hi = (int)((long long)cfg.n * (r + 1) / nranks);
        disp[r] = lo; cnts[r] = hi - lo;
    }

    if (bodies_alloc(&s.b, cfg.n)) die("out of memory (bodies)");
    if (ic_generate(&s.b, cfg.ic, cfg.n, cfg.seed, cfg.spin, cfg.eps))
        die("--ic must be uniform, plummer, collide, collapse, virial or disk");
    s.b.sort_cost = part_cost;
    if (tree_init(&s.tree, cfg.n, cfg.leafcap)) die("out of memory (tree)");

    /* GPU init is per rank; make failure collective so no rank hangs */
    int init_rc = sim_init_forces(&s, !cfg.gpu_float), init_any = 0;
    MPI_Allreduce(&init_rc, &init_any, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    if (init_any) die("initialisation failed (see messages above)");

    controller_t ctl;
    if (ctl_init(&ctl, sk, cfg.chunk, cfg.k, cfg.low, cfg.high, cfg.threads, balanced)) die("out of memory (controller)");

    if (rank == 0 && !cfg.quiet) {
        printf("Hybrid MPI+OpenMP+CUDA Barnes-Hut N-body simulator\n");
        printf("ic=%s n=%d steps=%d dt=%g eps=%g theta=%g leaf=%d spin=%g\n",
               cfg.ic, cfg.n, cfg.steps, cfg.dt, cfg.eps, cfg.theta, cfg.leafcap, cfg.spin);
        printf("force=%s sched=%s balanced=%s chunk=%d k=%d low=%g high=%g | ranks=%d threads/rank=%d partition=%s\n",
               cfg.force, cfg.sched, cfg.balanced, cfg.chunk, cfg.k, cfg.low, cfg.high, nranks, cfg.threads,
               cfg.partition);
        if (mode == FORCE_GPU_DIRECT || mode == FORCE_BH_GPU)
            printf("GPU: %s (%s precision)\n", gpu_name(), cfg.gpu_float ? "single" : "double");
        fflush(stdout);
    }

    /* per-step records */
    const int S = cfg.steps;
    double *rec_lt = (double *)calloc((size_t)S + 1, sizeof(double));   /* lambda (time)        */
    double *rec_le = (double *)calloc((size_t)S + 1, sizeof(double));   /* lambda (static work) */
    double *rec_lp = (double *)calloc((size_t)S + 1, sizeof(double));   /* lambda (static pred) */
    double *rec_lc = (double *)calloc((size_t)S + 1, sizeof(double));   /* lambda at checks     */
    double *rec_tf = (double *)calloc((size_t)S + 1, sizeof(double));   /* force time           */
    double *rec_ts = (double *)calloc((size_t)S + 1, sizeof(double));   /* step time            */
    int    *rec_po = (int *)calloc((size_t)S + 1, sizeof(int));         /* policy used          */
    double *rec_E  = (double *)calloc((size_t)S + 1, sizeof(double));   /* energy (sampled)     */
    double *rec_K  = (double *)calloc((size_t)S + 1, sizeof(double));   /* kinetic (sampled)    */
    char   *rec_eh = (char *)calloc((size_t)S + 1, 1);                  /* energy sampled here  */
    if (!rec_lt || !rec_le || !rec_lp || !rec_lc || !rec_tf || !rec_ts || !rec_po || !rec_E || !rec_K || !rec_eh)
        die("out of memory (records)");
    for (int st = 0; st <= S; st++) rec_lc[st] = -1.0;

    /* ------------------------------------------------ initial forces (t=0) */
    bodies_t *b = &s.b;
    morton_sort(b);
    if (mode == FORCE_BH || mode == FORCE_BH_GPU) tree_build(&s.tree, b);
    snap_t sn = { NULL, NULL, NULL, NULL };
    if (cfg.snap[0]) {
        if (nranks * cfg.threads > 255) die("--snap supports at most 255 ranks x threads");
        if (cfg.snap_every < 1) die("--snap-every must be >= 1");
        sn.own = (unsigned char *)calloc((size_t)cfg.n, 1);
        sn.wt = (double *)calloc((size_t)nranks * (size_t)cfg.threads, sizeof(double));
        sn.buf = (float *)malloc(sizeof(float) * (size_t)(cfg.n > nranks * cfg.threads ? cfg.n : nranks * cfg.threads));
        if (!sn.own || !sn.wt || !sn.buf) die("out of memory (snapshots)");
        if (rank == 0) {
            sn.f = fopen(cfg.snap, "wb");
            if (!sn.f) die("cannot open --snap file");
            const int32_t h[3] = { cfg.n, nranks * cfg.threads, 0 };
            fwrite("NBSNAP01", 1, 8, sn.f);
            fwrite(h, 4, 3, sn.f);
        }
        s.record_owner = 1;
    }
    {
        int rc = forces_compute(&s), any = 0;
        MPI_Allreduce(&rc, &any, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
        if (any) die("force computation failed");
    }
    if (part_cost) gather_cost(&s, cnts, disp);
    rec_lt[0] = imbalance_of(s.thr_time, s.nthr);
    rec_le[0] = static_equiv_imbalance(b->cost, s.lo, s.hi, s.nthr);
    rec_lp[0] = inst_pred(&s);
    rec_tf[0] = s.t_force; rec_po[0] = ctl.pol;
    if (cfg.snap[0]) snap_frame(&sn, &s, 0, 0.0, rec_po[0], rec_lt[0], cnts, disp);
    double t_snap = 0.0;

    double ver_rms = -1, ver_max = -1;
    if (cfg.verify) {
        const int cnt = s.hi - s.lo;
        double *r = (double *)malloc(sizeof(double) * 3u * (size_t)(cnt > 0 ? cnt : 1));
        if (!r) die("out of memory (verify)");
        forces_reference_direct(&s, r, r + cnt, r + 2 * cnt);
        double loc[2] = { 0, 0 }, glob[2];
        for (int g = 0; g < cnt; g++) {
            const int i = s.lo + g;
            const double dx = b->ax[i] - r[g], dy = b->ay[i] - r[cnt + g], dz = b->az[i] - r[2 * cnt + g];
            const double ref = sqrt(r[g] * r[g] + r[cnt + g] * r[cnt + g] + r[2 * cnt + g] * r[2 * cnt + g]);
            const double e = ref > 0 ? sqrt(dx * dx + dy * dy + dz * dz) / ref : 0.0;
            loc[0] += e * e;
            if (e > loc[1]) loc[1] = e;
        }
        MPI_Allreduce(&loc[0], &glob[0], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&loc[1], &glob[1], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        ver_rms = sqrt(glob[0] / cfg.n); ver_max = glob[1];
        if (rank == 0 && !cfg.quiet) printf("VERIFY force error vs direct sum: rms_rel=%.3e max_rel=%.3e\n", ver_rms, ver_max);
        free(r);
    }

    double E0 = 0, P0 = 0, E1 = 0, P1 = 0, dEmax = 0, Escale = 1, Q0 = 0, Q1 = 0;
    if (cfg.energy) {
        total_energy(&s, &E0, &P0);
        /* energy errors are relative to |E0|, or to |W0| for (nearly) unbound systems with E0 ~ 0 */
        Escale = fabs(E0) > 0.1 * fabs(g_pe) ? fabs(E0) : fabs(g_pe);
        Q0 = g_pe != 0 ? 2.0 * g_ke / fabs(g_pe) : 0.0;
        rec_E[0] = E0; rec_K[0] = g_ke; rec_eh[0] = 1;
    }
    if (rank == 0 && !cfg.quiet && cfg.energy) { printf("t=0  E=%.10f  KE=%.6f  W=%.6f  virial 2KE/|W|=%.4f  |P|=%.3e\n", E0, g_ke, g_pe, Q0, P0); fflush(stdout); }

    /* --------------------------------------------------------- time loop */
    double t_integ = 0, t_comm = 0, t_sort = 0, t_build = 0, t_force = 0;
    s.t_gpu = 0.0;
    const double hdt = 0.5 * cfg.dt;
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_begin = wtime();

    for (int step = 1; step <= S; step++) {
        const double ts = wtime();
        double t0 = ts, t1;

        /* 1. kick + drift own block */
        #pragma omp parallel for schedule(static)
        for (int i = s.lo; i < s.hi; i++) {
            b->vx[i] += hdt * b->ax[i]; b->vy[i] += hdt * b->ay[i]; b->vz[i] += hdt * b->az[i];
            b->x[i] += cfg.dt * b->vx[i]; b->y[i] += cfg.dt * b->vy[i]; b->z[i] += cfg.dt * b->vz[i];
        }
        t1 = wtime(); t_integ += t1 - t0; t0 = t1;

        /* 2. replicate state */
        sync_state(&s, cnts, disp);
        t1 = wtime(); t_comm += t1 - t0; t0 = t1;

        /* 3. Morton sort (+ cost-zone repartition: identical on every rank, no communication) */
        morton_sort(b);
        if (part_cost) {
            partition_by_cost(b->cost, cfg.n, nranks, cnts, disp);
            s.lo = disp[rank]; s.hi = disp[rank] + cnts[rank];
        }
        t1 = wtime(); t_sort += t1 - t0; t0 = t1;

        /* 4. tree */
        if (mode == FORCE_BH || mode == FORCE_BH_GPU) tree_build(&s.tree, b);
        t1 = wtime(); t_build += t1 - t0; t0 = t1;

        /* 5. forces */
        s.record_owner = cfg.snap[0] && step % cfg.snap_every == 0;
        if (forces_compute(&s) != 0) die("force computation failed");
        t1 = wtime(); t_force += t1 - t0; t0 = t1;

        /* 6. second half-kick */
        #pragma omp parallel for schedule(static)
        for (int i = s.lo; i < s.hi; i++) {
            b->vx[i] += hdt * b->ax[i]; b->vy[i] += hdt * b->ay[i]; b->vz[i] += hdt * b->az[i];
        }
        t1 = wtime(); t_integ += t1 - t0; t0 = t1;

        /* every rank needs every body's cost for the next repartition */
        if (part_cost) { gather_cost(&s, cnts, disp); t_comm += wtime() - t0; }

        /* 7. controller */
        rec_lt[step] = imbalance_of(s.thr_time, s.nthr);
        rec_le[step] = static_equiv_imbalance(b->cost, s.lo, s.hi, s.nthr);
        rec_lp[step] = inst_pred(&s);
        rec_tf[step] = s.t_force;
        rec_po[step] = ctl.pol;
        ctl_observe(&ctl, s.thr_time, s.thr_work, b->cost, s.lo, s.hi, s.nthr);
        if (ctl_update(&ctl, step) >= 0 && sk == SCHED_ADAPTIVE && step % cfg.k == 0)
            rec_lc[step] = ctl.last_lambda;
        rec_ts[step] = wtime() - ts;
        if (s.record_owner) {                     /* snapshot I/O is excluded from all timings */
            const double t0s = wtime();
            snap_frame(&sn, &s, step, step * cfg.dt, rec_po[step], rec_lt[step], cnts, disp);
            t_snap += wtime() - t0s;
        }

        if (cfg.energy && cfg.diag > 0 && step % cfg.diag == 0 && step != S) {
            double E, P;
            total_energy(&s, &E, &P);
            if (fabs(E - E0) / Escale > dEmax) dEmax = fabs(E - E0) / Escale;
            rec_E[step] = E; rec_K[step] = g_ke; rec_eh[step] = 1;
        }
        if (rank == 0 && !cfg.quiet && cfg.print > 0 && step % cfg.print == 0) {
            printf("step %5d  t=%7.3f  step=%8.4fs  lambda=%.3f  lambda_work=%.3f  lambda_pred=%.3f  policy=%s\n",
                   step, step * cfg.dt, rec_ts[step], rec_lt[step], rec_le[step], rec_lp[step],
                   ctl_policy_name((policy_t)rec_po[step]));
            fflush(stdout);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double t_total_local = wtime() - t_begin - t_snap;

    /* ---------------------------------------------------------- results */
    if (cfg.energy) {
        total_energy(&s, &E1, &P1);
        Q1 = g_pe != 0 ? 2.0 * g_ke / fabs(g_pe) : 0.0;
        if (fabs(E1 - E0) / Escale > dEmax) dEmax = fabs(E1 - E0) / Escale;
        if (S > 0) { rec_E[S] = E1; rec_K[S] = g_ke; rec_eh[S] = 1; }
    }
    sync_state(&s, cnts, disp);                     /* full velocities for the checksum */
    const unsigned long long csum = checksum(b);

    double tl[7] = { t_total_local, t_force, s.t_gpu, t_build, t_sort, t_comm, t_integ }, tg[7];
    MPI_Reduce(tl, tg, 7, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    long long ovf = s.near_overflow, ovf_g = 0;
    MPI_Reduce(&ovf, &ovf_g, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    /* rank-level imbalance of the force phase, per step */
    double *tf_max = NULL, *tf_sum = NULL;
    if (rank == 0) {
        tf_max = (double *)calloc((size_t)S + 1, sizeof(double));
        tf_sum = (double *)calloc((size_t)S + 1, sizeof(double));
    }
    MPI_Reduce(rec_tf, tf_max, S + 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(rec_tf, tf_sum, S + 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double lt_mean = 0, lt_max = 0, le_mean = 0, lp_mean = 0, lr_mean = 0, fdyn = 0;
        for (int st = 1; st <= S; st++) {
            lt_mean += rec_lt[st]; le_mean += rec_le[st]; lp_mean += rec_lp[st];
            if (rec_lt[st] > lt_max) lt_max = rec_lt[st];
            /* fraction of steps in the balanced policy (adaptive) / in dynamic (fixed schedules) */
            if (sk == SCHED_ADAPTIVE ? rec_po[st] != POL_STATIC : rec_po[st] == POL_DYNAMIC) fdyn += 1;
            const double mean = tf_sum[st] / nranks;
            lr_mean += mean > 0 ? tf_max[st] / mean - 1.0 : 0.0;
        }
        if (S > 0) { lt_mean /= S; le_mean /= S; lp_mean /= S; lr_mean /= S; fdyn /= S; }
        const double de_rel = cfg.energy ? fabs(E1 - E0) / Escale : 0.0;
        const double t_step = S > 0 ? tg[0] / S : 0.0;

        char row[2048];
        snprintf(row, sizeof row,
                 "%s,%s,%d,%d,%g,%g,%g,%d,%s,%s,%d,%d,%g,%g,%d,%d,"
                 "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
                 "%.5f,%.5f,%.5f,%.5f,%.5f,%d,%.4f,%d,%s,"
                 "%.10e,%.10e,%.4e,%.4e,%.4e,%.4e,%.10e,%.10e,%lld,%016llx,"
                 "%s,%s,%d,%s,%d",
                 cfg.label, cfg.ic, cfg.n, S, cfg.dt, cfg.eps, cfg.theta, cfg.leafcap, cfg.force, cfg.sched,
                 cfg.chunk, cfg.k, cfg.low, cfg.high, cfg.threads, nranks,
                 tg[0], t_step, tg[1], tg[2], tg[3], tg[4], tg[5], tg[6],
                 lt_mean, lt_max, le_mean, lp_mean, lr_mean, ctl.switches, fdyn, cfg.pin,
                 (mode == FORCE_GPU_DIRECT || mode == FORCE_BH_GPU) ? (cfg.gpu_float ? "float" : "double") : "none",
                 E0, E1, de_rel, dEmax, P0, P1, ver_rms, ver_max, ovf_g, csum,
                 cfg.balanced, cfg.partition, ctl.trials,
                 ctl.chosen >= 0 ? ctl_policy_name(ctl.cand_pol[ctl.chosen]) : "none",
                 ctl.chosen >= 0 ? ctl.cand_chunk[ctl.chosen] : 0);
        static const char *HEADER =
            "label,ic,n,steps,dt,eps,theta,leaf,force,sched,chunk,k,low,high,threads,ranks,"
            "t_total,t_step,t_force,t_gpu,t_build,t_sort,t_comm,t_integ,"
            "lam_time_mean,lam_time_max,lam_work_mean,lam_pred_mean,lam_rank_mean,switches,frac_dynamic,pinned,gpu_prec,"
            "e0,e1,de_rel,de_rel_max,p0,p1,verify_rms,verify_max,near_overflow,checksum,"
            "balanced,partition,trials,chosen,chosen_chunk";

        if (!cfg.quiet) {
            printf("done: total %.4f s  (%.5f s/step)  force %.4f  gpu %.4f  build %.4f  sort %.4f  comm %.4f\n",
                   tg[0], t_step, tg[1], tg[2], tg[3], tg[4], tg[5]);
            if (cfg.energy) printf("energy: E0=%.10f E1=%.10f |dE|/%s=%.3e (max sampled %.3e)  virial %.4f -> %.4f  "
                                   "|P|: %.3e -> %.3e\n", E0, E1, Escale == fabs(E0) ? "|E0|" : "|W0|",
                                   de_rel, dEmax, Q0, Q1, P0, P1);
            printf("imbalance: lambda_time mean=%.4f max=%.4f  lambda_work mean=%.4f  lambda_pred mean=%.4f  rank lambda=%.4f  "
                   "switches=%d  dynamic fraction=%.2f\n", lt_mean, lt_max, le_mean, lp_mean, lr_mean, ctl.switches, fdyn);
            if (mode == FORCE_BH_GPU) printf("near-field list overflows (handled on CPU): %lld\n", ovf_g);
        }
        printf("RESULT ");
        {   /* RESULT key=value line built from header + row (neither contains empty fields) */
            const char *hp = HEADER, *rp = row;
            while (*hp && *rp) {
                while (*hp && *hp != ',') putchar(*hp++);
                putchar('=');
                while (*rp && *rp != ',') putchar(*rp++);
                putchar(' ');
                if (*hp == ',') hp++;
                if (*rp == ',') rp++;
            }
            printf("\n");
        }
        fflush(stdout);

        if (cfg.csv[0]) {
            FILE *chk = fopen(cfg.csv, "r");
            int exists = chk != NULL;
            if (chk) fclose(chk);
            FILE *f = fopen(cfg.csv, "a");
            if (f) {
                if (!exists) fprintf(f, "%s\n", HEADER);
                fprintf(f, "%s\n", row);
                fclose(f);
            } else fprintf(stderr, "warning: cannot open %s\n", cfg.csv);
        }
        if (cfg.trace[0]) {
            FILE *f = fopen(cfg.trace, "w");
            if (f) {
                fprintf(f, "step,time,policy,lam_time,lam_static_work,lam_static_pred,lam_check,t_force,t_step,"
                       "energy,kinetic,de_rel\n");
                for (int st = 0; st <= S; st++)
                {
                    char lc[32] = "";
                    char en[96] = ",,";                          /* empty unless sampled (--diag) */
                    if (rec_lc[st] >= 0) snprintf(lc, sizeof lc, "%.6f", rec_lc[st]);
                    if (rec_eh[st]) snprintf(en, sizeof en, "%.12e,%.12e,%.4e", rec_E[st], rec_K[st],
                                             fabs(rec_E[st] - E0) / Escale);
                    fprintf(f, "%d,%.6f,%s,%.6f,%.6f,%.6f,%s,%.6e,%.6e,%s\n", st, st * cfg.dt,
                            ctl_policy_name((policy_t)rec_po[st]), rec_lt[st], rec_le[st], rec_lp[st], lc,
                            rec_tf[st], rec_ts[st], en);
                }
                fclose(f);
            } else fprintf(stderr, "warning: cannot open %s\n", cfg.trace);
        }
        free(tf_max); free(tf_sum);
    }

    free(rec_lt); free(rec_le); free(rec_lp); free(rec_lc); free(rec_tf); free(rec_ts); free(rec_po);
    free(rec_E); free(rec_K); free(rec_eh);
    ctl_free(&ctl);
    free(cnts); free(disp);
    if (sn.f) fclose(sn.f);
    free(sn.own); free(sn.wt); free(sn.buf);
    sim_free_forces(&s);
    tree_free(&s.tree);
    bodies_free(&s.b);
    MPI_Finalize();
    return 0;
}
