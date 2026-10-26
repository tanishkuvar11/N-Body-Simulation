/*
 * gpu.h - CUDA force kernels, compiled at run time with NVRTC and launched through the
 * CUDA Driver API.  Both libraries are loaded dynamically, so the executable builds and
 * runs without a CUDA toolkit; GPU modes simply report an error if the DLLs are missing.
 */
#ifndef GPU_H
#define GPU_H

/* n = total bodies, max_cnt = largest block of bodies this rank will ask for,
 * cap = max near-field ranges per body.  Returns 0 on success. */
int         gpu_init(int n, int max_cnt, int use_double, int cap);
void        gpu_shutdown(void);
const char *gpu_error(void);
const char *gpu_name(void);

/* Upload all positions and masses (needed by both kernels). */
int gpu_set_positions(const double *x, const double *y, const double *z, const double *m, int n);

/* Tiled all-pairs kernel (Nyland et al.): accelerations of bodies [lo,lo+cnt) from all n bodies. */
int gpu_direct(int lo, int cnt, double eps2, double *ax, double *ay, double *az);

/* Near-field kernel: body lo+g interacts with the body ranges [rng[(g*cap+r)*2], rng[(g*cap+r)*2+1])
 * for r < nnear[g]. */
int gpu_near(int lo, int cnt, double eps2, const int *rng, int cap, const int *nnear, int maxnear,
             double *ax, double *ay, double *az);

#endif
