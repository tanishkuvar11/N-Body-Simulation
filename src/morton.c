/* morton.c - Z-order keys, stable radix sort, and body permutation.
 *
 * Sorting bodies along the Morton curve every step makes Morton-contiguous index
 * ranges spatial domains: block partitions over MPI ranks / OpenMP threads then
 * own compact regions of space (a "costzones"-style decomposition), so clustering
 * translates directly into unequal work per block.
 */
#include "nbody.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

static inline uint64_t part1by2(uint64_t v)
{
    v &= 0x1fffffULL;
    v = (v | (v << 32)) & 0x1f00000000ffffULL;
    v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
    v = (v | (v << 8))  & 0x100f00f00f00f00fULL;
    v = (v | (v << 4))  & 0x10c30c30c30c30c3ULL;
    v = (v | (v << 2))  & 0x1249249249249249ULL;
    return v;
}

static inline uint64_t quant(double v, double lo, double inv_side)
{
    double q = (v - lo) * inv_side * (double)(1u << MAXLEVEL);
    if (!(q > 0.0)) return 0;                       /* also catches NaN */
    uint64_t u = (uint64_t)q;
    return u > ((1ULL << MAXLEVEL) - 1) ? ((1ULL << MAXLEVEL) - 1) : u;
}

/* Stable LSD radix sort of (key, idx) over [lo,hi): input in sk/si, output in dk/di.
 * 7 passes x 9 bits cover the 63-bit key; a pass whose digit is the same for every key
 * (common: the keys of one bucket share their leading bits) is skipped. */
static void sort_range(uint64_t *dk, uint32_t *di, uint64_t *sk, uint32_t *si, int lo, int hi)
{
    enum { RB = 9, R = 1 << RB, PASSES = 7 };
    const int len = hi - lo;
    if (len <= 0) return;
    unsigned cnt[R];
    uint64_t *ak = sk, *bk = dk;
    uint32_t *ai = si, *bi = di;
    for (int p = 0; p < PASSES; p++) {
        const int sh = p * RB;
        memset(cnt, 0, sizeof cnt);
        for (int i = lo; i < hi; i++) cnt[(ak[i] >> sh) & (R - 1)]++;
        if (cnt[(ak[lo] >> sh) & (R - 1)] == (unsigned)len) continue;
        unsigned sum = (unsigned)lo;
        for (int d = 0; d < R; d++) { unsigned c = cnt[d]; cnt[d] = sum; sum += c; }
        for (int i = lo; i < hi; i++) {
            unsigned pos = cnt[(ak[i] >> sh) & (R - 1)]++;
            bk[pos] = ak[i];
            bi[pos] = ai[i];
        }
        uint64_t *tk = ak; ak = bk; bk = tk;
        uint32_t *ti = ai; ai = bi; bi = ti;
    }
    if (ak != dk) {
        memcpy(dk + lo, ak + lo, sizeof(uint64_t) * (size_t)len);
        memcpy(di + lo, ai + lo, sizeof(uint32_t) * (size_t)len);
    }
}

static int cmp_u64(const void *a, const void *b)
{
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

void morton_sort(bodies_t *b)
{
    const int n = b->n;
    if (n <= 0) return;
    int par = n >= PAR_MIN;

    /* bounding cube (min/max are exact, so the result is independent of the thread count) */
    double lx = b->x[0], ly = b->y[0], lz = b->z[0], hx = lx, hy = ly, hz = lz;
    #pragma omp parallel for schedule(static) if (par) reduction(min:lx,ly,lz) reduction(max:hx,hy,hz)
    for (int i = 1; i < n; i++) {
        if (b->x[i] < lx) lx = b->x[i];
        if (b->x[i] > hx) hx = b->x[i];
        if (b->y[i] < ly) ly = b->y[i];
        if (b->y[i] > hy) hy = b->y[i];
        if (b->z[i] < lz) lz = b->z[i];
        if (b->z[i] > hz) hz = b->z[i];
    }
    const double lo[3] = { lx, ly, lz }, hi[3] = { hx, hy, hz };
    double side = hi[0] - lo[0];
    if (hi[1] - lo[1] > side) side = hi[1] - lo[1];
    if (hi[2] - lo[2] > side) side = hi[2] - lo[2];
    side = side * (1.0 + 1e-9);
    if (!(side > 0.0)) side = 1.0;
    b->bmin[0] = lo[0]; b->bmin[1] = lo[1]; b->bmin[2] = lo[2]; b->bside = side;

    /* Stable sort of (key, idx) as a parallel sample sort: splitters from a sorted sample
     * of the keys cut the key range into one bucket per thread; a stable scatter (offsets
     * ordered bucket-major, then by thread slice) moves every key to its bucket, and each
     * bucket is radix-sorted by one thread.  The result is the unique stable order, i.e.
     * the same for every thread count. */
    const int maxt = par ? omp_get_max_threads() : 1;
    enum { SAMPLES_PER_THREAD = 64 };
    const int nsamp = SAMPLES_PER_THREAD * maxt;
    unsigned *off = (unsigned *)malloc(sizeof(unsigned) * (size_t)maxt * (size_t)maxt);
    unsigned *bstart = (unsigned *)malloc(sizeof(unsigned) * ((size_t)maxt + 1));
    uint64_t *spl = (uint64_t *)malloc(sizeof(uint64_t) * ((size_t)nsamp + 1));
    if (!off || !bstart || !spl) {                       /* fall back to one sequential sort */
        for (int i = 0; i < n; i++) {
            b->key2[i] = (part1by2(quant(b->x[i], lo[0], 1.0 / side)) << 2) |
                         (part1by2(quant(b->y[i], lo[1], 1.0 / side)) << 1) |
                          part1by2(quant(b->z[i], lo[2], 1.0 / side));
            b->idx2[i] = (uint32_t)i;
        }
        sort_range(b->key, b->idx, b->key2, b->idx2, 0, n);
        par = 0;
    }
    const int sorted = !(off && bstart && spl);
    const double inv = 1.0 / side;

    double **arr[8] = { &b->x, &b->y, &b->z, &b->vx, &b->vy, &b->vz, &b->m, &b->cost };
    const int narr = b->sort_cost ? 8 : 7;

    #pragma omp parallel if (par)
    {
        const int nt = omp_get_num_threads(), t = omp_get_thread_num();
        const int a = (int)((long long)n * t / nt), e = (int)((long long)n * (t + 1) / nt);

        if (!sorted) {
            for (int i = a; i < e; i++) {
                uint64_t qx = quant(b->x[i], lo[0], inv);
                uint64_t qy = quant(b->y[i], lo[1], inv);
                uint64_t qz = quant(b->z[i], lo[2], inv);
                b->key[i] = (part1by2(qx) << 2) | (part1by2(qy) << 1) | part1by2(qz);
            }
            #pragma omp barrier
            #pragma omp single
            {   /* nt-1 splitters: quantiles of an evenly spaced sample */
                const int ns = SAMPLES_PER_THREAD * nt;
                for (int j = 0; j < ns; j++) spl[j] = b->key[(int)(((long long)j * n) / ns)];
                qsort(spl, (size_t)ns, sizeof(uint64_t), cmp_u64);
                for (int u = 1; u < nt; u++) spl[u - 1] = spl[u * SAMPLES_PER_THREAD];
            }
            /* bucket of a key = number of splitters <= key  (equal keys share a bucket) */
            unsigned *my = off + (size_t)t * nt;
            for (int u = 0; u < nt; u++) my[u] = 0;
            for (int i = a; i < e; i++) {
                int l = 0, h = nt - 1;
                while (l < h) { int m = (l + h) >> 1; if (spl[m] <= b->key[i]) l = m + 1; else h = m; }
                my[l]++;
            }
            #pragma omp barrier
            #pragma omp single
            {
                unsigned sum = 0;
                for (int u = 0; u < nt; u++) {
                    bstart[u] = sum;
                    for (int v = 0; v < nt; v++) { unsigned c = off[(size_t)v * nt + u]; off[(size_t)v * nt + u] = sum; sum += c; }
                }
                bstart[nt] = sum;
            }
            for (int i = a; i < e; i++) {
                int l = 0, h = nt - 1;
                while (l < h) { int m = (l + h) >> 1; if (spl[m] <= b->key[i]) l = m + 1; else h = m; }
                const unsigned pos = my[l]++;
                b->key2[pos] = b->key[i];
                b->idx2[pos] = (uint32_t)i;
            }
            #pragma omp barrier
            #pragma omp for schedule(dynamic, 1)
            for (int u = 0; u < nt; u++)
                sort_range(b->key, b->idx, b->key2, b->idx2, (int)bstart[u], (int)bstart[u + 1]);
        }

        /* gather every array through the permutation in one pass, then swap buffers */
        const uint32_t *idx = b->idx;
        double *src[8], *dst[8];
        for (int k = 0; k < narr; k++) { src[k] = *arr[k]; dst[k] = b->tmp[k]; }
        for (int k = 0; k < narr; k++) {
            const double *sp = src[k];
            double *dp = dst[k];
            for (int i = a; i < e; i++) dp[i] = sp[idx[i]];
        }
        for (int i = a; i < e; i++) b->tmpi[i] = b->id[idx[i]];
        #pragma omp barrier
        #pragma omp single
        {
            for (int k = 0; k < narr; k++) { double *x = *arr[k]; *arr[k] = b->tmp[k]; b->tmp[k] = x; }
            int *ti = b->id; b->id = b->tmpi; b->tmpi = ti;
        }
    }
    free(off); free(bstart); free(spl);
}

void partition_by_cost(const double *cost, int n, int nranks, int *cnts, int *disp)
{
    double total = 0.0;
    for (int i = 0; i < n; i++) total += cost[i];
    int start = 0, i = 0;
    double acc = 0.0;                                  /* cost of bodies [0, i) */
    for (int r = 0; r < nranks; r++) {
        int end = n;
        if (r < nranks - 1) {
            if (total > 0.0) {
                const double target = total * (double)(r + 1) / nranks;
                while (i < n && acc + cost[i] <= target) acc += cost[i++];
                end = i;
                if (i < n && target - acc > acc + cost[i] - target) end = i + 1;   /* nearer cut */
            } else {
                end = (int)((long long)n * (r + 1) / nranks);
            }
            const int lo_lim = start + 1, hi_lim = n - (nranks - 1 - r);   /* >= 1 body per rank */
            if (end < lo_lim) end = lo_lim;
            if (end > hi_lim) end = hi_lim;
            while (i < end) acc += cost[i++];
            while (i > end) acc -= cost[--i];
        }
        disp[r] = start; cnts[r] = end - start;
        start = end;
    }
}
