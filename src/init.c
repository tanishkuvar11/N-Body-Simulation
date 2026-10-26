/* init.c - body storage and initial conditions (deterministic, portable RNG). */
#include "nbody.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846

/* ------------------------------------------------------------------ memory */
static void *xalloc(size_t bytes) { return calloc(bytes ? bytes : 1, 1); }

int bodies_alloc(bodies_t *b, int n)
{
    memset(b, 0, sizeof *b);
    b->n = n;
    size_t N = (size_t)n;
    double **dp[] = { &b->x, &b->y, &b->z, &b->vx, &b->vy, &b->vz, &b->m,
                      &b->ax, &b->ay, &b->az, &b->cost,
                      &b->tmp[0], &b->tmp[1], &b->tmp[2], &b->tmp[3], &b->tmp[4], &b->tmp[5], &b->tmp[6], &b->tmp[7] };
    for (size_t i = 0; i < sizeof dp / sizeof dp[0]; i++) {
        *dp[i] = (double *)xalloc(N * sizeof(double));
        if (!*dp[i]) return -1;
    }
    b->id   = (int *)xalloc(N * sizeof(int));
    b->tmpi = (int *)xalloc(N * sizeof(int));
    b->key  = (uint64_t *)xalloc(N * sizeof(uint64_t));
    b->key2 = (uint64_t *)xalloc(N * sizeof(uint64_t));
    b->idx  = (uint32_t *)xalloc(N * sizeof(uint32_t));
    b->idx2 = (uint32_t *)xalloc(N * sizeof(uint32_t));
    if (!b->id || !b->tmpi || !b->key || !b->key2 || !b->idx || !b->idx2) return -1;
    return 0;
}

void bodies_free(bodies_t *b)
{
    free(b->x); free(b->y); free(b->z); free(b->vx); free(b->vy); free(b->vz);
    free(b->m); free(b->ax); free(b->ay); free(b->az); free(b->cost);
    for (int k = 0; k < 8; k++) free(b->tmp[k]);
    free(b->id); free(b->tmpi); free(b->key); free(b->key2); free(b->idx); free(b->idx2);
    memset(b, 0, sizeof *b);
}

/* --------------------------------------------------------------------- RNG */
typedef struct { uint64_t s[4]; int have_spare; double spare; } rng_t;

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void rng_seed(rng_t *r, uint64_t seed)
{
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) r->s[i] = splitmix64(&x);
    r->have_spare = 0;
}

static uint64_t rotl64(uint64_t v, int k) { return (v << k) | (v >> (64 - k)); }

static uint64_t rng_next(rng_t *r)   /* xoshiro256** */
{
    uint64_t *s = r->s;
    uint64_t result = rotl64(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl64(s[3], 45);
    return result;
}

/* uniform in (0,1) - never exactly 0 or 1 */
static double uni(rng_t *r) { return ((double)(rng_next(r) >> 11) + 0.5) * (1.0 / 9007199254740992.0); }

static double gauss(rng_t *r)
{
    if (r->have_spare) { r->have_spare = 0; return r->spare; }
    double u = uni(r), v = uni(r);
    double rad = sqrt(-2.0 * log(u));
    r->spare = rad * sin(2.0 * PI * v);
    r->have_spare = 1;
    return rad * cos(2.0 * PI * v);
}

static void rand_dir(rng_t *r, double *dx, double *dy, double *dz)
{
    double c = 2.0 * uni(r) - 1.0, phi = 2.0 * PI * uni(r), s = sqrt(1.0 - c * c);
    *dx = s * cos(phi); *dy = s * sin(phi); *dz = c;
}

/* --------------------------------------------------------- distributions */
/* Plummer sphere of total mass M and scale length a (Aarseth, Henon & Wielen 1974).
 * Radii are truncated at 98 % enclosed mass so a few far outliers cannot blow up
 * the bounding box.  Writes n bodies starting at index off. */
static void plummer(rng_t *r, bodies_t *b, int off, int n, double M, double a)
{
    for (int i = off; i < off + n; i++) {
        double X1 = 0.98 * uni(r);
        double rad = a / sqrt(pow(X1, -2.0 / 3.0) - 1.0);
        double dx, dy, dz;
        rand_dir(r, &dx, &dy, &dz);
        b->x[i] = rad * dx; b->y[i] = rad * dy; b->z[i] = rad * dz;

        double q, g;
        do { q = uni(r); g = 0.1 * uni(r); } while (g > q * q * pow(1.0 - q * q, 3.5));
        double vesc = sqrt(2.0 * M / a) * pow(1.0 + rad * rad / (a * a), -0.25);
        rand_dir(r, &dx, &dy, &dz);
        b->vx[i] = q * vesc * dx; b->vy[i] = q * vesc * dy; b->vz[i] = q * vesc * dz;
    }
}

/* uniform-density sphere of radius R; velocity dispersion sigma per component */
static void uniform_sphere(rng_t *r, bodies_t *b, int off, int n, double R, double sigma)
{
    for (int i = off; i < off + n; i++) {
        double px, py, pz;
        do { px = 2 * uni(r) - 1; py = 2 * uni(r) - 1; pz = 2 * uni(r) - 1; } while (px * px + py * py + pz * pz > 1.0);
        b->x[i] = R * px; b->y[i] = R * py; b->z[i] = R * pz;
        b->vx[i] = sigma * gauss(r); b->vy[i] = sigma * gauss(r); b->vz[i] = sigma * gauss(r);
    }
}

static void shift(bodies_t *b, int off, int n, double x, double y, double z, double vx, double vy, double vz)
{
    for (int i = off; i < off + n; i++) {
        b->x[i] += x; b->y[i] += y; b->z[i] += z;
        b->vx[i] += vx; b->vy[i] += vy; b->vz[i] += vz;
    }
}

int ic_generate(bodies_t *b, const char *kind, int n, uint64_t seed)
{
    rng_t r;
    rng_seed(&r, seed);
    for (int i = 0; i < n; i++) { b->m[i] = 1.0 / n; b->id[i] = i; }

    if (strcmp(kind, "uniform") == 0) {
        /* uniform sphere in homologous (Hubble) expansion v = H r with H = sqrt(2GM/R^3)
         * (parabolic, E ~ 0).  A uniform sphere with a linear velocity field stays uniform
         * (Friedmann solution), so the per-body work stays balanced for the whole run;
         * the tiny random component only seeds slow gravitational instability. */
        const double H = sqrt(2.0);
        uniform_sphere(&r, b, 0, n, 1.0, 0.005);
        for (int i = 0; i < n; i++) { b->vx[i] += H * b->x[i]; b->vy[i] += H * b->y[i]; b->vz[i] += H * b->z[i]; }
    } else if (strcmp(kind, "virial") == 0) {
        /* virialised uniform sphere (KE = -PE/2): not an equilibrium, develops a core */
        uniform_sphere(&r, b, 0, n, 1.0, sqrt(0.2));
    } else if (strcmp(kind, "collapse") == 0) {
        /* cold uniform sphere: free-fall time ~1.11, forms a dense core */
        uniform_sphere(&r, b, 0, n, 1.0, 0.02);
    } else if (strcmp(kind, "plummer") == 0) {
        plummer(&r, b, 0, n, 1.0, 1.0);
    } else if (strcmp(kind, "collide") == 0) {
        int n1 = n / 2 + (n & 1), n2 = n - n1;
        double M1 = (double)n1 / n, M2 = (double)n2 / n;
        plummer(&r, b, 0, n1, M1, 0.4);
        plummer(&r, b, n1, n2, M2, 0.4);
        shift(b, 0,  n1, -1.0, -0.15, 0.0,  0.7, 0.0, 0.0);
        shift(b, n1, n2,  1.0,  0.15, 0.0, -0.7, 0.0, 0.0);
    } else {
        return -1;
    }

    /* remove centre-of-mass position and velocity */
    double cx = 0, cy = 0, cz = 0, cvx = 0, cvy = 0, cvz = 0, M = 0;
    for (int i = 0; i < n; i++) {
        M += b->m[i];
        cx += b->m[i] * b->x[i];   cy += b->m[i] * b->y[i];   cz += b->m[i] * b->z[i];
        cvx += b->m[i] * b->vx[i]; cvy += b->m[i] * b->vy[i]; cvz += b->m[i] * b->vz[i];
    }
    shift(b, 0, n, -cx / M, -cy / M, -cz / M, -cvx / M, -cvy / M, -cvz / M);
    return 0;
}
