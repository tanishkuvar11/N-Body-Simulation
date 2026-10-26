/* forces.c - force computation back ends and diagnostics.
 *
 *   FORCE_DIRECT      O(N^2) direct sum on the CPU (OpenMP, schedule(runtime))
 *   FORCE_BH          Barnes-Hut tree walk on the CPU (OpenMP, schedule(runtime))
 *   FORCE_GPU_DIRECT  tiled O(N^2) CUDA kernel
 *   FORCE_BH_GPU      hybrid: CPU threads walk the tree and accumulate far-field (cell)
 *                     interactions; near-field leaf ranges are collected per body and the
 *                     pairwise sums over them run in a CUDA kernel.
 *
 * Every CPU variant records, per step, the loop time of each thread and a work-unit
 * cost per body; the adaptive controller consumes both.
 */
#include "nbody.h"
#include "gpu.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int sim_init_forces(sim_t *s, int gpu_double)
{
    s->thr_time = (double *)calloc((size_t)(s->threads > 0 ? s->threads : 1), sizeof(double));
    s->thr_work = (double *)calloc((size_t)(s->threads > 0 ? s->threads : 1), sizeof(double));
    s->nthr = s->threads;
    s->near_rng = NULL; s->near_cnt = NULL; s->near_overflow = 0;
    s->owner = (unsigned char *)calloc((size_t)(s->n > 0 ? s->n : 1), 1);
    if (!s->thr_time || !s->thr_work || !s->owner) return -1;

    if (s->mode == FORCE_GPU_DIRECT || s->mode == FORCE_BH_GPU) {
        const int cnt = s->hi - s->lo;
        const int need_lists = (s->mode == FORCE_BH_GPU);
        if (need_lists && s->near_cap < 1) s->near_cap = 128;
        if (gpu_init(s->n, cnt, gpu_double, need_lists ? s->near_cap : 0) != 0) {
            fprintf(stderr, "[rank %d] GPU init failed: %s\n", s->rank, gpu_error());
            return -1;
        }
        if (need_lists) {
            s->near_rng = (int *)malloc(sizeof(int) * 2u * (size_t)s->near_cap * (size_t)(cnt > 0 ? cnt : 1));
            s->near_cnt = (int *)malloc(sizeof(int) * (size_t)(cnt > 0 ? cnt : 1));
            if (!s->near_rng || !s->near_cnt) return -1;
        }
    }
    return 0;
}

void sim_free_forces(sim_t *s)
{
    free(s->thr_time); s->thr_time = NULL;
    free(s->thr_work); s->thr_work = NULL;
    free(s->owner); s->owner = NULL;
    free(s->near_rng); s->near_rng = NULL;
    free(s->near_cnt); s->near_cnt = NULL;
    gpu_shutdown();
}

/* ------------------------------------------------------------ Barnes-Hut */
/* Tree walk for body i.  Far cells use their centre of mass; leaves are either summed
 * directly (use_gpu == 0) or recorded as body ranges for the GPU (use_gpu == 1). */
static inline void bh_body(const sim_t *s, int i, int use_gpu, int *stack, int *row, int cap,
                           int *nr_out, double *ax_o, double *ay_o, double *az_o, double *cost_o,
                           long long *overflow)
{
    const bodies_t *b = &s->b;
    const node_t *nodes = s->tree.nodes;
    const double xi = b->x[i], yi = b->y[i], zi = b->z[i];
    const double eps2 = s->eps2, th2 = s->theta2;
    double ax = 0, ay = 0, az = 0, visits = 0, inter = 0;
    int nr = 0, sp = 0;

    stack[sp++] = 0;
    while (sp) {
        const node_t *nd = &nodes[stack[--sp]];
        visits += 1.0;
        const double dx = nd->cx - xi, dy = nd->cy - yi, dz = nd->cz - zi;
        const double r2 = dx * dx + dy * dy + dz * dz;
        const int inside = (i >= nd->lo && i < nd->hi);      /* never use a cell containing the body */

        if (!inside && nd->s2 < th2 * r2) {                  /* opening criterion: s/d < theta */
            const double inv = 1.0 / sqrt(r2 + eps2);
            const double w = nd->m * inv * inv * inv;
            ax += w * dx; ay += w * dy; az += w * dz;
            inter += 1.0;
        } else if (nd->nchild == 0) {                        /* near-field leaf */
            const int a = nd->lo, e = nd->hi;
            int direct = 1;
            if (use_gpu) {
                direct = 0;
                if (nr > 0 && row[2 * (nr - 1) + 1] == a)      row[2 * (nr - 1) + 1] = e;
                else if (nr > 0 && row[2 * (nr - 1)] == e)     row[2 * (nr - 1)] = a;
                else if (nr < cap) { row[2 * nr] = a; row[2 * nr + 1] = e; nr++; }
                else {                                          /* list full: do it on the CPU */
                    direct = 1;
                    #pragma omp atomic
                    *overflow += 1;
                }
            }
            if (direct) {
                for (int j = a; j < e; j++) {
                    const double qx = b->x[j] - xi, qy = b->y[j] - yi, qz = b->z[j] - zi;
                    const double q2 = qx * qx + qy * qy + qz * qz + eps2;
                    const double inv = 1.0 / sqrt(q2);
                    const double w = b->m[j] * inv * inv * inv;
                    ax += w * qx; ay += w * qy; az += w * qz;
                }
                inter += (double)(e - a);
            }
        } else {                                             /* open the cell; push in reverse so  */
            for (int c = nd->nchild - 1; c >= 0; c--)        /* children pop in Morton order       */
                stack[sp++] = nd->child0 + c;
        }
    }
    *ax_o = ax; *ay_o = ay; *az_o = az;
    *cost_o = visits + inter + (double)nr;
    *nr_out = nr;
}

static int force_bh(sim_t *s, int use_gpu)
{
    bodies_t *b = &s->b;
    const int lo = s->lo, hi = s->hi, cap = s->near_cap;
    long long overflow = 0;
    int nthr_used = 1;

    for (int t = 0; t < s->threads; t++) s->thr_time[t] = s->thr_work[t] = 0.0;

    double t_start = wtime();
    #pragma omp parallel
    {
        int stack[STACK_MAX];
        #pragma omp master
        nthr_used = omp_get_num_threads();
        #pragma omp barrier
        double work = 0.0;
        const unsigned char me = (unsigned char)omp_get_thread_num();
        const int rec = s->record_owner;
        const double t0 = wtime();
        #pragma omp for schedule(runtime) nowait
        for (int i = lo; i < hi; i++) {
            int nr = 0;
            if (rec) s->owner[i] = me;
            int *row = use_gpu ? s->near_rng + (size_t)(i - lo) * 2u * (size_t)cap : NULL;
            bh_body(s, i, use_gpu, stack, row, cap, &nr, &b->ax[i], &b->ay[i], &b->az[i], &b->cost[i], &overflow);
            if (use_gpu) s->near_cnt[i - lo] = nr;
            work += b->cost[i];
        }
        const double t1 = wtime();
        const int tid = omp_get_thread_num();
        if (tid < s->threads) { s->thr_time[tid] = t1 - t0; s->thr_work[tid] = work; }
    }
    s->nthr = nthr_used > s->threads ? s->threads : nthr_used;
    s->t_force = wtime() - t_start;
    s->near_overflow += overflow;
    return 0;
}

static int gpu_finish_near(sim_t *s)
{
    const int lo = s->lo, cnt = s->hi - s->lo;
    if (cnt <= 0) return 0;
    double *gx = (double *)malloc(sizeof(double) * 3u * (size_t)cnt);
    if (!gx) return -1;
    double *gy = gx + cnt, *gz = gy + cnt;
    int maxnear = 0;
    for (int g = 0; g < cnt; g++) if (s->near_cnt[g] > maxnear) maxnear = s->near_cnt[g];

    const double t0 = wtime();
    int rc = gpu_near(lo, cnt, s->eps2, s->near_rng, s->near_cap, s->near_cnt, maxnear, gx, gy, gz);
    s->t_gpu += wtime() - t0;
    if (rc == 0) {
        bodies_t *b = &s->b;
        for (int g = 0; g < cnt; g++) { b->ax[lo + g] += gx[g]; b->ay[lo + g] += gy[g]; b->az[lo + g] += gz[g]; }
    }
    free(gx);
    return rc;
}

/* ----------------------------------------------------------- direct (CPU) */
static void direct_body(const bodies_t *b, int n, int i, double eps2, double *ox, double *oy, double *oz)
{
    const double xi = b->x[i], yi = b->y[i], zi = b->z[i];
    double ax = 0, ay = 0, az = 0;
    #pragma omp simd reduction(+:ax,ay,az)
    for (int j = 0; j < n; j++) {
        const double dx = b->x[j] - xi, dy = b->y[j] - yi, dz = b->z[j] - zi;
        const double r2 = dx * dx + dy * dy + dz * dz + eps2;
        const double inv = 1.0 / sqrt(r2);
        const double w = b->m[j] * inv * inv * inv;
        ax += w * dx; ay += w * dy; az += w * dz;
    }
    *ox = ax; *oy = ay; *oz = az;
}

static int force_direct(sim_t *s)
{
    bodies_t *b = &s->b;
    const int lo = s->lo, hi = s->hi, n = s->n;
    int nthr_used = 1;
    for (int t = 0; t < s->threads; t++) s->thr_time[t] = s->thr_work[t] = 0.0;

    double t_start = wtime();
    #pragma omp parallel
    {
        #pragma omp master
        nthr_used = omp_get_num_threads();
        #pragma omp barrier
        double work = 0.0;
        const unsigned char me = (unsigned char)omp_get_thread_num();
        const int rec = s->record_owner;
        const double t0 = wtime();
        #pragma omp for schedule(runtime) nowait
        for (int i = lo; i < hi; i++) {
            if (rec) s->owner[i] = me;
            direct_body(b, n, i, s->eps2, &b->ax[i], &b->ay[i], &b->az[i]);
            b->cost[i] = (double)n;
            work += (double)n;
        }
        const double t1 = wtime();
        const int tid = omp_get_thread_num();
        if (tid < s->threads) { s->thr_time[tid] = t1 - t0; s->thr_work[tid] = work; }
    }
    s->nthr = nthr_used > s->threads ? s->threads : nthr_used;
    s->t_force = wtime() - t_start;
    return 0;
}

/* -------------------------------------------------------------- GPU direct */
static int force_gpu_direct(sim_t *s)
{
    bodies_t *b = &s->b;
    const int lo = s->lo, cnt = s->hi - s->lo;
    for (int t = 0; t < s->threads; t++) s->thr_time[t] = s->thr_work[t] = 0.0;
    s->nthr = s->threads;
    s->t_force = 0.0;
    if (cnt <= 0) return 0;

    double *gx = (double *)malloc(sizeof(double) * 3u * (size_t)cnt);
    if (!gx) return -1;
    double *gy = gx + cnt, *gz = gy + cnt;
    const double t0 = wtime();
    int rc = gpu_set_positions(b->x, b->y, b->z, b->m, s->n);
    if (rc == 0) rc = gpu_direct(lo, cnt, s->eps2, gx, gy, gz);
    s->t_gpu += wtime() - t0;
    if (rc == 0)
        for (int g = 0; g < cnt; g++) {
            b->ax[lo + g] = gx[g]; b->ay[lo + g] = gy[g]; b->az[lo + g] = gz[g];
            b->cost[lo + g] = (double)s->n;
        }
    free(gx);
    return rc;
}

/* ---------------------------------------------------------------- dispatch */
int forces_compute(sim_t *s)
{
    int rc = 0;
    switch (s->mode) {
    case FORCE_BH:         rc = force_bh(s, 0); break;
    case FORCE_DIRECT:     rc = force_direct(s); break;
    case FORCE_GPU_DIRECT: rc = force_gpu_direct(s); break;
    case FORCE_BH_GPU: {
        double t0 = wtime();
        rc = gpu_set_positions(s->b.x, s->b.y, s->b.z, s->b.m, s->n);
        s->t_gpu += wtime() - t0;
        if (rc == 0) rc = force_bh(s, 1);
        if (rc == 0) rc = gpu_finish_near(s);
        break;
    }
    }
    if (rc != 0) fprintf(stderr, "[rank %d] force computation failed: %s\n", s->rank, gpu_error());
    return rc;
}

void forces_reference_direct(const sim_t *s, double *rx, double *ry, double *rz)
{
    const bodies_t *b = &s->b;
    #pragma omp parallel for schedule(static)
    for (int i = s->lo; i < s->hi; i++)
        direct_body(b, s->n, i, s->eps2, &rx[i - s->lo], &ry[i - s->lo], &rz[i - s->lo]);
}

/* ------------------------------------------------------------- diagnostics */
void energy_parts(const sim_t *s, double *ke, double *pe_local, double *px, double *py, double *pz)
{
    const bodies_t *b = &s->b;
    double k = 0, p = 0, mx = 0, my = 0, mz = 0;
    const int n = s->n;
    const double eps = s->eps, eps2 = s->eps2;

    #pragma omp parallel for schedule(static) reduction(+:k,p,mx,my,mz)
    for (int i = s->lo; i < s->hi; i++) {
        k += 0.5 * b->m[i] * (b->vx[i] * b->vx[i] + b->vy[i] * b->vy[i] + b->vz[i] * b->vz[i]);
        mx += b->m[i] * b->vx[i]; my += b->m[i] * b->vy[i]; mz += b->m[i] * b->vz[i];

        const double xi = b->x[i], yi = b->y[i], zi = b->z[i];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < n; j++) {
            const double dx = b->x[j] - xi, dy = b->y[j] - yi, dz = b->z[j] - zi;
            sum += b->m[j] / sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
        sum -= b->m[i] / eps;                    /* remove the self term (r = 0) */
        p -= 0.5 * b->m[i] * sum;                /* each pair counted twice -> 1/2 */
    }
    *ke = k; *pe_local = p; *px = mx; *py = my; *pz = mz;
}
