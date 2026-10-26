/* sched.c - adaptive OpenMP loop-schedule controller (Algorithm 1 of the synopsis),
 *           high-resolution clock.
 *
 * Imbalance metric:  lambda = max_i(t_i) / mean_i(t_i) - 1,  t_i = loop time of thread i,
 * accumulated over the k steps of the current check window (less noisy than one step).
 *
 * Every k steps the controller evaluates lambda.  Two thresholds give hysteresis:
 *   STATIC   --(lambda > high)-->  BALANCED   (DYNAMIC(chunk) by default, or GUIDED(chunk))
 *   BALANCED --(lambda < low )-->  STATIC
 *
 * Observation problem: while DYNAMIC is installed the measured thread times are balanced
 * by construction, so they cannot tell whether STATIC would now be good enough; using them
 * would flip the policy back to STATIC at every check.  The controller therefore evaluates
 * the thread times that STATIC *would* produce:
 *
 *      t_i(static) = B_i / v_i,   B_i = work units of thread i's static block,
 *                                 v_i = W_i / t_i = measured speed of thread i
 *                                       (work units it executed per second).
 *
 * Under STATIC, W_i == B_i, so this is exactly the measured loop time t_i - i.e. the
 * synopsis' metric.  Under DYNAMIC it predicts static's imbalance including both the
 * workload (clustering) and differences in core speed (e.g. P- vs E-cores).
 */
#include "nbody.h"

#include <omp.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
double wtime(void)
{
    static double inv_freq = 0.0;
    LARGE_INTEGER c;
    if (inv_freq == 0.0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        inv_freq = 1.0 / (double)f.QuadPart;
    }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * inv_freq;
}
#else
#include <time.h>
double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#endif

const char *ctl_policy_name(policy_t p)
{
    switch (p) {
    case POL_STATIC:  return "static";
    case POL_DYNAMIC: return "dynamic";
    default:          return "guided";
    }
}

void ctl_apply(const controller_t *c)
{
    switch (c->pol) {
    case POL_STATIC:  omp_set_schedule(omp_sched_static, 0); break;
    case POL_DYNAMIC: omp_set_schedule(omp_sched_dynamic, c->chunk); break;
    case POL_GUIDED:  omp_set_schedule(omp_sched_guided, c->chunk); break;
    }
}

static void reset_window(controller_t *c)
{
    for (int t = 0; t < c->maxthr; t++) c->acc_t[t] = c->acc_w[t] = c->acc_b[t] = 0.0;
    c->nobs = 0;
}

int ctl_init(controller_t *c, sched_kind_t kind, int chunk, int k, double low, double high, int maxthr,
             policy_t balanced)
{
    c->balanced = (balanced == POL_GUIDED) ? POL_GUIDED : POL_DYNAMIC;
    c->kind = kind;
    c->chunk = chunk < 1 ? 1 : chunk;
    c->k = k < 1 ? 1 : k;
    c->low = low;
    c->high = high;
    c->switches = 0;
    c->last_lambda = 0.0;
    c->maxthr = maxthr < 1 ? 1 : maxthr;
    c->nthr = c->maxthr;
    c->acc_t = (double *)calloc((size_t)c->maxthr, sizeof(double));
    c->acc_w = (double *)calloc((size_t)c->maxthr, sizeof(double));
    c->acc_b = (double *)calloc((size_t)c->maxthr, sizeof(double));
    if (!c->acc_t || !c->acc_w || !c->acc_b) return -1;
    reset_window(c);
    switch (kind) {
    case SCHED_DYNAMIC: c->pol = POL_DYNAMIC; break;
    case SCHED_GUIDED:  c->pol = POL_GUIDED;  break;
    default:            c->pol = POL_STATIC;  break;   /* static and adaptive start static */
    }
    ctl_apply(c);
    return 0;
}

void ctl_free(controller_t *c)
{
    free(c->acc_t); free(c->acc_w); free(c->acc_b);
    c->acc_t = c->acc_w = c->acc_b = NULL;
}

double imbalance_of(const double *v, int n)
{
    if (n <= 1) return 0.0;
    double mx = v[0], sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (v[i] > mx) mx = v[i];
        sum += v[i];
    }
    double mean = sum / n;
    return mean > 0.0 ? mx / mean - 1.0 : 0.0;
}

/* Work of each thread's block under schedule(static) without a chunk size: contiguous
 * blocks, the first n % T threads get one extra iteration (libgomp / OpenMP convention). */
void static_blocks(const double *cost, int lo, int hi, int nthreads, double *out)
{
    const int n = hi - lo > 0 ? hi - lo : 0;
    const int base = n / nthreads, rem = n % nthreads;
    int pos = lo;
    for (int t = 0; t < nthreads; t++) {
        const int len = base + (t < rem ? 1 : 0);
        double s = 0.0;
        for (int i = pos; i < pos + len; i++) s += cost[i];
        pos += len;
        out[t] = s;
    }
}

double static_equiv_imbalance(const double *cost, int lo, int hi, int nthreads)
{
    if (nthreads <= 1 || hi <= lo) return 0.0;
    double *blk = (double *)malloc(sizeof(double) * (size_t)nthreads);
    if (!blk) return 0.0;
    static_blocks(cost, lo, hi, nthreads, blk);
    double lam = imbalance_of(blk, nthreads);
    free(blk);
    return lam;
}

double predicted_static_imbalance(const double *block, const double *t, const double *w, int n)
{
    if (n <= 1) return 0.0;
    double ssum = 0.0;
    int sc = 0;
    for (int i = 0; i < n; i++)
        if (w[i] > 0.0 && t[i] > 0.0) { ssum += w[i] / t[i]; sc++; }
    if (sc == 0) return imbalance_of(block, n);        /* no timing data: pure work estimate */
    const double vmean = ssum / sc;
    double *pred = (double *)malloc(sizeof(double) * (size_t)n);
    if (!pred) return imbalance_of(block, n);
    for (int i = 0; i < n; i++) {
        const double v = (w[i] > 0.0 && t[i] > 0.0) ? w[i] / t[i] : vmean;
        pred[i] = block[i] / v;
    }
    double lam = imbalance_of(pred, n);
    free(pred);
    return lam;
}

void ctl_observe(controller_t *c, const double *thr_time, const double *thr_work,
                 const double *cost, int lo, int hi, int nthr)
{
    if (nthr > c->maxthr) nthr = c->maxthr;
    if (nthr < 1) nthr = 1;
    if (nthr != c->nthr) { c->nthr = nthr; reset_window(c); }
    double blk[64];
    double *b = nthr <= 64 ? blk : (double *)malloc(sizeof(double) * (size_t)nthr);
    if (!b) return;
    static_blocks(cost, lo, hi, nthr, b);
    for (int t = 0; t < nthr; t++) {
        c->acc_t[t] += thr_time[t];
        c->acc_w[t] += thr_work[t];
        c->acc_b[t] += b[t];
    }
    c->nobs++;
    if (b != blk) free(b);
}

int ctl_update(controller_t *c, int step)
{
    if (c->kind != SCHED_ADAPTIVE) { reset_window(c); return 0; }
    if (step % c->k != 0 || c->nobs == 0) return 0;

    const double lam = (c->pol == POL_STATIC)
                     ? imbalance_of(c->acc_t, c->nthr)
                     : predicted_static_imbalance(c->acc_b, c->acc_t, c->acc_w, c->nthr);
    c->last_lambda = lam;
    reset_window(c);

    policy_t next = c->pol;
    if (c->pol == POL_STATIC && lam > c->high) next = c->balanced;
    else if (c->pol == c->balanced && lam < c->low) next = POL_STATIC;

    if (next == c->pol) return 0;
    c->pol = next;
    c->switches++;
    ctl_apply(c);
    return 1;
}
