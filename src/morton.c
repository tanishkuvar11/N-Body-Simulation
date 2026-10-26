/* morton.c - Z-order keys, stable radix sort, and body permutation.
 *
 * Sorting bodies along the Morton curve every step makes Morton-contiguous index
 * ranges spatial domains: block partitions over MPI ranks / OpenMP threads then
 * own compact regions of space (a "costzones"-style decomposition), so clustering
 * translates directly into unequal work per block.
 */
#include "nbody.h"

#include <math.h>
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

static void permute_d(double *a, double *tmp, const uint32_t *idx, int n)
{
    for (int i = 0; i < n; i++) tmp[i] = a[idx[i]];
    memcpy(a, tmp, (size_t)n * sizeof(double));
}

void morton_sort(bodies_t *b)
{
    const int n = b->n;
    if (n <= 0) return;

    /* bounding cube */
    double lo[3] = { b->x[0], b->y[0], b->z[0] }, hi[3] = { lo[0], lo[1], lo[2] };
    for (int i = 1; i < n; i++) {
        const double p[3] = { b->x[i], b->y[i], b->z[i] };
        for (int d = 0; d < 3; d++) {
            if (p[d] < lo[d]) lo[d] = p[d];
            if (p[d] > hi[d]) hi[d] = p[d];
        }
    }
    double side = hi[0] - lo[0];
    if (hi[1] - lo[1] > side) side = hi[1] - lo[1];
    if (hi[2] - lo[2] > side) side = hi[2] - lo[2];
    side = side * (1.0 + 1e-9);
    if (!(side > 0.0)) side = 1.0;
    b->bmin[0] = lo[0]; b->bmin[1] = lo[1]; b->bmin[2] = lo[2]; b->bside = side;

    const double inv = 1.0 / side;
    for (int i = 0; i < n; i++) {
        uint64_t qx = quant(b->x[i], lo[0], inv);
        uint64_t qy = quant(b->y[i], lo[1], inv);
        uint64_t qz = quant(b->z[i], lo[2], inv);
        b->key[i] = (part1by2(qx) << 2) | (part1by2(qy) << 1) | part1by2(qz);
        b->idx[i] = (uint32_t)i;
    }

    /* stable LSD radix sort of (key, idx): 6 passes x 11 bits covers the 63-bit key */
    enum { RB = 11, RADIX = 1 << RB, PASSES = 6 };
    uint64_t *k1 = b->key, *k2 = b->key2;
    uint32_t *i1 = b->idx, *i2 = b->idx2;
    for (int p = 0; p < PASSES; p++) {
        unsigned cnt[RADIX];
        memset(cnt, 0, sizeof cnt);
        const int sh = p * RB;
        for (int i = 0; i < n; i++) cnt[(k1[i] >> sh) & (RADIX - 1)]++;
        unsigned sum = 0;
        for (int d = 0; d < RADIX; d++) { unsigned c = cnt[d]; cnt[d] = sum; sum += c; }
        for (int i = 0; i < n; i++) {
            unsigned d = (unsigned)((k1[i] >> sh) & (RADIX - 1));
            unsigned pos = cnt[d]++;
            k2[pos] = k1[i];
            i2[pos] = i1[i];
        }
        uint64_t *tk = k1; k1 = k2; k2 = tk;
        uint32_t *ti = i1; i1 = i2; i2 = ti;
    }
    /* after an even number of passes the sorted data is back in b->key / b->idx */
    /* (PASSES is even, so k1 == b->key and i1 == b->idx here) */

    permute_d(b->x,  b->tmp, b->idx, n);
    permute_d(b->y,  b->tmp, b->idx, n);
    permute_d(b->z,  b->tmp, b->idx, n);
    permute_d(b->vx, b->tmp, b->idx, n);
    permute_d(b->vy, b->tmp, b->idx, n);
    permute_d(b->vz, b->tmp, b->idx, n);
    permute_d(b->m,  b->tmp, b->idx, n);
    for (int i = 0; i < n; i++) b->tmpi[i] = b->id[b->idx[i]];
    memcpy(b->id, b->tmpi, (size_t)n * sizeof(int));
}
