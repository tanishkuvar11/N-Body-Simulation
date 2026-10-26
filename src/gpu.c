/* gpu.c - NVRTC + CUDA Driver API back end (dynamically loaded). */
#include "gpu.h"

#ifdef NO_CUDA

static const char *msg = "built without CUDA support (NO_CUDA)";
int gpu_init(int n, int max_cnt, int use_double, int cap) { (void)n; (void)max_cnt; (void)use_double; (void)cap; return -1; }
void gpu_shutdown(void) {}
const char *gpu_error(void) { return msg; }
const char *gpu_name(void) { return "none"; }
int gpu_set_positions(const double *x, const double *y, const double *z, const double *m, int n)
{ (void)x; (void)y; (void)z; (void)m; (void)n; return -1; }
int gpu_direct(int lo, int cnt, double eps2, double *ax, double *ay, double *az)
{ (void)lo; (void)cnt; (void)eps2; (void)ax; (void)ay; (void)az; return -1; }
int gpu_near(int lo, int cnt, double eps2, const int *rng, int cap, const int *nnear, int maxnear,
             double *ax, double *ay, double *az)
{ (void)lo; (void)cnt; (void)eps2; (void)rng; (void)cap; (void)nnear; (void)maxnear; (void)ax; (void)ay; (void)az; return -1; }

#else

#include <cuda.h>
#include <nvrtc.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_t;
static void *lib_sym(lib_t l, const char *s) { return (void *)GetProcAddress(l, s); }
#else
#include <dlfcn.h>
typedef void *lib_t;
static void *lib_sym(lib_t l, const char *s) { return dlsym(l, s); }
#endif

/* ---------------------------------------------------------------- kernels */
static const char *KERNEL_SRC =
"#ifndef REAL\n#define REAL double\n#endif\n"
"extern \"C\" __global__ void direct_kernel(\n"
"    const REAL* __restrict__ px, const REAL* __restrict__ py, const REAL* __restrict__ pz,\n"
"    const REAL* __restrict__ pm, int n, int lo, int cnt, REAL eps2,\n"
"    REAL* __restrict__ ox, REAL* __restrict__ oy, REAL* __restrict__ oz)\n"
"{\n"
"  extern __shared__ REAL sh[];\n"
"  REAL* sx = sh; REAL* sy = sh + blockDim.x; REAL* sz = sy + blockDim.x; REAL* sm = sz + blockDim.x;\n"
"  const int t = threadIdx.x;\n"
"  const int g = blockIdx.x * blockDim.x + t;\n"
"  const int i = lo + g;\n"
"  REAL xi = 0, yi = 0, zi = 0;\n"
"  if (g < cnt) { xi = px[i]; yi = py[i]; zi = pz[i]; }\n"
"  REAL a0 = 0, a1 = 0, a2 = 0;\n"
"  for (int base = 0; base < n; base += blockDim.x) {\n"
"    const int j = base + t;\n"
"    if (j < n) { sx[t] = px[j]; sy[t] = py[j]; sz[t] = pz[j]; sm[t] = pm[j]; }\n"
"    else       { sx[t] = 0;     sy[t] = 0;     sz[t] = 0;     sm[t] = 0;     }\n"
"    __syncthreads();\n"
"    const int lim = min((int)blockDim.x, n - base);\n"
"    #pragma unroll 8\n"
"    for (int k = 0; k < lim; k++) {\n"
"      const REAL dx = sx[k] - xi, dy = sy[k] - yi, dz = sz[k] - zi;\n"
"      const REAL r2 = dx*dx + dy*dy + dz*dz + eps2;\n"
"      const REAL inv = rsqrt(r2);\n"
"      const REAL w = sm[k] * inv * inv * inv;\n"
"      a0 += w * dx; a1 += w * dy; a2 += w * dz;\n"
"    }\n"
"    __syncthreads();\n"
"  }\n"
"  if (g < cnt) { ox[g] = a0; oy[g] = a1; oz[g] = a2; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void near_kernel(\n"
"    const REAL* __restrict__ px, const REAL* __restrict__ py, const REAL* __restrict__ pz,\n"
"    const REAL* __restrict__ pm, int lo, int cnt, REAL eps2,\n"
"    const int2* __restrict__ rng, int cap, const int* __restrict__ nnear,\n"
"    REAL* __restrict__ ox, REAL* __restrict__ oy, REAL* __restrict__ oz)\n"
"{\n"
"  const int g = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (g >= cnt) return;\n"
"  const int i = lo + g;\n"
"  const REAL xi = px[i], yi = py[i], zi = pz[i];\n"
"  REAL a0 = 0, a1 = 0, a2 = 0;\n"
"  const int nr = nnear[g];\n"
"  for (int r = 0; r < nr; r++) {\n"
"    const int2 rg = rng[(size_t)g * cap + r];\n"
"    for (int j = rg.x; j < rg.y; j++) {\n"
"      const REAL dx = px[j] - xi, dy = py[j] - yi, dz = pz[j] - zi;\n"
"      const REAL r2 = dx*dx + dy*dy + dz*dz + eps2;\n"
"      const REAL inv = rsqrt(r2);\n"
"      const REAL w = pm[j] * inv * inv * inv;\n"
"      a0 += w * dx; a1 += w * dy; a2 += w * dz;\n"
"    }\n"
"  }\n"
"  ox[g] = a0; oy[g] = a1; oz[g] = a2;\n"
"}\n";

/* ---------------------------------------------------------- function table */
#define DECL(ret, name, args) typedef ret (CUDAAPI *pfn_##name) args; static pfn_##name p_##name
DECL(CUresult, cuInit, (unsigned int));
DECL(CUresult, cuDeviceGet, (CUdevice *, int));
DECL(CUresult, cuDeviceGetName, (char *, int, CUdevice));
DECL(CUresult, cuDeviceGetAttribute, (int *, CUdevice_attribute, CUdevice));
DECL(CUresult, cuDevicePrimaryCtxRetain, (CUcontext *, CUdevice));
DECL(CUresult, cuDevicePrimaryCtxRelease, (CUdevice));
DECL(CUresult, cuCtxSetCurrent, (CUcontext));
DECL(CUresult, cuCtxSynchronize, (void));
DECL(CUresult, cuModuleLoadData, (CUmodule *, const void *));
DECL(CUresult, cuModuleUnload, (CUmodule));
DECL(CUresult, cuModuleGetFunction, (CUfunction *, CUmodule, const char *));
DECL(CUresult, cuMemAlloc, (CUdeviceptr *, size_t));
DECL(CUresult, cuMemFree, (CUdeviceptr));
DECL(CUresult, cuMemcpyHtoD, (CUdeviceptr, const void *, size_t));
DECL(CUresult, cuMemcpyDtoH, (void *, CUdeviceptr, size_t));
DECL(CUresult, cuMemcpy2D, (const CUDA_MEMCPY2D *));
DECL(CUresult, cuLaunchKernel, (CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                unsigned, CUstream, void **, void **));
DECL(CUresult, cuGetErrorString, (CUresult, const char **));

#define NDECL(ret, name, args) typedef ret (*pfn_##name) args; static pfn_##name p_##name
NDECL(nvrtcResult, nvrtcCreateProgram, (nvrtcProgram *, const char *, const char *, int,
                                        const char *const *, const char *const *));
NDECL(nvrtcResult, nvrtcDestroyProgram, (nvrtcProgram *));
NDECL(nvrtcResult, nvrtcCompileProgram, (nvrtcProgram, int, const char *const *));
NDECL(nvrtcResult, nvrtcGetProgramLogSize, (nvrtcProgram, size_t *));
NDECL(nvrtcResult, nvrtcGetProgramLog, (nvrtcProgram, char *));
NDECL(nvrtcResult, nvrtcGetCUBINSize, (nvrtcProgram, size_t *));
NDECL(nvrtcResult, nvrtcGetCUBIN, (nvrtcProgram, char *));
NDECL(nvrtcResult, nvrtcGetPTXSize, (nvrtcProgram, size_t *));
NDECL(nvrtcResult, nvrtcGetPTX, (nvrtcProgram, char *));

/* ------------------------------------------------------------------- state */
static char g_err[1024] = "";
static char g_name[256] = "none";
static int g_ready = 0, g_double = 1, g_n = 0, g_maxcnt = 0, g_cap = 0;
static CUdevice g_dev;
static CUcontext g_ctx;
static CUmodule g_mod;
static CUfunction g_fdirect, g_fnear;
static CUdeviceptr d_x, d_y, d_z, d_m, d_ox, d_oy, d_oz, d_rng, d_nn;
static void *h_conv = NULL;       /* host staging buffer for float mode / outputs */
static size_t h_conv_bytes = 0;

const char *gpu_error(void) { return g_err; }
const char *gpu_name(void) { return g_name; }

static void seterr(const char *fmt, const char *a, int code)
{
    snprintf(g_err, sizeof g_err, fmt, a, code);
}

static int cuchk(CUresult r, const char *what)
{
    if (r == CUDA_SUCCESS) return 0;
    const char *s = "?";
    if (p_cuGetErrorString) p_cuGetErrorString(r, &s);
    snprintf(g_err, sizeof g_err, "%s failed: %s (%d)", what, s, (int)r);
    return -1;
}

/* --------------------------------------------------------- dynamic loading */
#ifdef _WIN32
static lib_t try_load(const char *dir, const char *name)
{
    lib_t h;
    if (dir && *dir) SetDllDirectoryA(dir);
    h = LoadLibraryA(name);
    if (dir && *dir) SetDllDirectoryA(NULL);
    return h;
}

static void exe_dir(char *out, size_t cap)
{
    DWORD len = GetModuleFileNameA(NULL, out, (DWORD)cap);
    while (len > 0 && out[len - 1] != '\\' && out[len - 1] != '/') len--;
    out[len] = 0;
}

static lib_t load_nvrtc(void)
{
    static const char *names[] = { "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_121_0.dll",
                                   "nvrtc64_122_0.dll", "nvrtc64_112_0.dll", NULL };
    char ed[512], d1[1024], d2[1024];
    const char *env = getenv("NBODY_NVRTC_DIR");
    exe_dir(ed, sizeof ed);
    snprintf(d1, sizeof d1, "%s..\\third_party\\cuda\\nvidia\\cuda_nvrtc\\bin", ed);
    snprintf(d2, sizeof d2, "%s", ed);
    const char *dirs[] = { NULL, env, d2, d1, NULL };
    for (int i = 0; names[i]; i++)
        for (int j = 0; j < 4; j++) {
            lib_t h = try_load(dirs[j], names[i]);
            if (h) return h;
        }
    return NULL;
}
static lib_t load_cuda(void) { return try_load(NULL, "nvcuda.dll"); }
#else
static lib_t load_nvrtc(void)
{
    static const char *names[] = { "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2", NULL };
    for (int i = 0; names[i]; i++) { lib_t h = dlopen(names[i], RTLD_NOW); if (h) return h; }
    return NULL;
}
static lib_t load_cuda(void)
{
    lib_t h = dlopen("libcuda.so.1", RTLD_NOW);
    return h ? h : dlopen("libcuda.so", RTLD_NOW);
}
#endif

#define LOADSYM(lib, name, sym) do { p_##name = (pfn_##name)lib_sym(lib, sym); \
    if (!p_##name) { seterr("missing symbol %s (%d)", sym, 0); return -1; } } while (0)

static int load_symbols(lib_t cu, lib_t rt)
{
    LOADSYM(cu, cuInit, "cuInit");
    LOADSYM(cu, cuDeviceGet, "cuDeviceGet");
    LOADSYM(cu, cuDeviceGetName, "cuDeviceGetName");
    LOADSYM(cu, cuDeviceGetAttribute, "cuDeviceGetAttribute");
    LOADSYM(cu, cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    LOADSYM(cu, cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
    LOADSYM(cu, cuCtxSetCurrent, "cuCtxSetCurrent");
    LOADSYM(cu, cuCtxSynchronize, "cuCtxSynchronize");
    LOADSYM(cu, cuModuleLoadData, "cuModuleLoadData");
    LOADSYM(cu, cuModuleUnload, "cuModuleUnload");
    LOADSYM(cu, cuModuleGetFunction, "cuModuleGetFunction");
    LOADSYM(cu, cuMemAlloc, "cuMemAlloc_v2");
    LOADSYM(cu, cuMemFree, "cuMemFree_v2");
    LOADSYM(cu, cuMemcpyHtoD, "cuMemcpyHtoD_v2");
    LOADSYM(cu, cuMemcpyDtoH, "cuMemcpyDtoH_v2");
    LOADSYM(cu, cuMemcpy2D, "cuMemcpy2D_v2");
    LOADSYM(cu, cuLaunchKernel, "cuLaunchKernel");
    LOADSYM(cu, cuGetErrorString, "cuGetErrorString");
    LOADSYM(rt, nvrtcCreateProgram, "nvrtcCreateProgram");
    LOADSYM(rt, nvrtcDestroyProgram, "nvrtcDestroyProgram");
    LOADSYM(rt, nvrtcCompileProgram, "nvrtcCompileProgram");
    LOADSYM(rt, nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
    LOADSYM(rt, nvrtcGetProgramLog, "nvrtcGetProgramLog");
    LOADSYM(rt, nvrtcGetCUBINSize, "nvrtcGetCUBINSize");
    LOADSYM(rt, nvrtcGetCUBIN, "nvrtcGetCUBIN");
    LOADSYM(rt, nvrtcGetPTXSize, "nvrtcGetPTXSize");
    LOADSYM(rt, nvrtcGetPTX, "nvrtcGetPTX");
    return 0;
}

/* Compile the kernels for this device; tries a native cubin first, then portable PTX. */
static int compile_and_load(int major, int minor, int use_double)
{
    char opt_arch[64], opt_def[64];
    snprintf(opt_def, sizeof opt_def, "-DREAL=%s", use_double ? "double" : "float");

    for (int attempt = 0; attempt < 2; attempt++) {
        if (attempt == 0) snprintf(opt_arch, sizeof opt_arch, "--gpu-architecture=sm_%d%d", major, minor);
        else              snprintf(opt_arch, sizeof opt_arch, "--gpu-architecture=compute_75");
        const char *opts[] = { opt_arch, opt_def };
        nvrtcProgram prog;
        if (p_nvrtcCreateProgram(&prog, KERNEL_SRC, "nbody.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
            seterr("%s (%d)", "nvrtcCreateProgram failed", 0);
            return -1;
        }
        nvrtcResult r = p_nvrtcCompileProgram(prog, 2, opts);
        if (r != NVRTC_SUCCESS) {
            size_t ls = 0;
            p_nvrtcGetProgramLogSize(prog, &ls);
            char *log = (char *)calloc(ls + 1, 1);
            if (log) p_nvrtcGetProgramLog(prog, log);
            snprintf(g_err, sizeof g_err, "NVRTC compile failed (%s): %.800s", opt_arch, log ? log : "");
            free(log);
            p_nvrtcDestroyProgram(&prog);
            continue;                              /* try the portable PTX fallback */
        }
        size_t sz = 0;
        char *image = NULL;
        if (attempt == 0) {
            p_nvrtcGetCUBINSize(prog, &sz);
            image = (char *)malloc(sz);
            if (image) p_nvrtcGetCUBIN(prog, image);
        } else {
            p_nvrtcGetPTXSize(prog, &sz);
            image = (char *)malloc(sz);
            if (image) p_nvrtcGetPTX(prog, image);
        }
        p_nvrtcDestroyProgram(&prog);
        if (!image) { seterr("%s (%d)", "out of memory", 0); return -1; }
        CUresult cr = p_cuModuleLoadData(&g_mod, image);
        free(image);
        if (cr != CUDA_SUCCESS) {
            if (cuchk(cr, "cuModuleLoadData") != 0 && attempt == 0) continue;
            return -1;
        }
        if (cuchk(p_cuModuleGetFunction(&g_fdirect, g_mod, "direct_kernel"), "get direct_kernel")) return -1;
        if (cuchk(p_cuModuleGetFunction(&g_fnear, g_mod, "near_kernel"), "get near_kernel")) return -1;
        g_err[0] = 0;
        return 0;
    }
    return -1;
}

int gpu_init(int n, int max_cnt, int use_double, int cap)
{
    lib_t cu = load_cuda();
    if (!cu) { seterr("%s (%d)", "CUDA driver (nvcuda.dll / libcuda.so) not found", 0); return -1; }
    lib_t rt = load_nvrtc();
    if (!rt) {
        seterr("%s (%d)", "NVRTC library not found (put nvrtc64_*.dll next to the executable "
                          "or set NBODY_NVRTC_DIR)", 0);
        return -1;
    }
    if (load_symbols(cu, rt)) return -1;
    if (cuchk(p_cuInit(0), "cuInit")) return -1;
    if (cuchk(p_cuDeviceGet(&g_dev, 0), "cuDeviceGet")) return -1;
    p_cuDeviceGetName(g_name, (int)sizeof g_name, g_dev);
    if (cuchk(p_cuDevicePrimaryCtxRetain(&g_ctx, g_dev), "cuDevicePrimaryCtxRetain")) return -1;
    if (cuchk(p_cuCtxSetCurrent(g_ctx), "cuCtxSetCurrent")) return -1;

    int major = 0, minor = 0;
    p_cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_dev);
    p_cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_dev);
    if (compile_and_load(major, minor, use_double)) return -1;

    g_double = use_double; g_n = n; g_maxcnt = max_cnt; g_cap = cap;
    const size_t R = use_double ? sizeof(double) : sizeof(float);
    if (cuchk(p_cuMemAlloc(&d_x, R * (size_t)n), "alloc x")) return -1;
    if (cuchk(p_cuMemAlloc(&d_y, R * (size_t)n), "alloc y")) return -1;
    if (cuchk(p_cuMemAlloc(&d_z, R * (size_t)n), "alloc z")) return -1;
    if (cuchk(p_cuMemAlloc(&d_m, R * (size_t)n), "alloc m")) return -1;
    size_t outn = (size_t)(max_cnt > 0 ? max_cnt : 1);
    if (cuchk(p_cuMemAlloc(&d_ox, R * outn), "alloc ox")) return -1;
    if (cuchk(p_cuMemAlloc(&d_oy, R * outn), "alloc oy")) return -1;
    if (cuchk(p_cuMemAlloc(&d_oz, R * outn), "alloc oz")) return -1;
    if (cap > 0) {
        if (cuchk(p_cuMemAlloc(&d_rng, 2 * sizeof(int) * (size_t)cap * outn), "alloc near lists")) return -1;
        if (cuchk(p_cuMemAlloc(&d_nn, sizeof(int) * outn), "alloc near counts")) return -1;
    }
    h_conv_bytes = R * (size_t)(n > (int)outn ? n : (int)outn);
    h_conv = malloc(h_conv_bytes);
    if (!h_conv) { seterr("%s (%d)", "out of memory", 0); return -1; }
    g_ready = 1;
    return 0;
}

void gpu_shutdown(void)
{
    if (!g_ready) return;
    p_cuMemFree(d_x); p_cuMemFree(d_y); p_cuMemFree(d_z); p_cuMemFree(d_m);
    p_cuMemFree(d_ox); p_cuMemFree(d_oy); p_cuMemFree(d_oz);
    if (g_cap > 0) { p_cuMemFree(d_rng); p_cuMemFree(d_nn); }
    p_cuModuleUnload(g_mod);
    p_cuDevicePrimaryCtxRelease(g_dev);
    free(h_conv);
    h_conv = NULL;
    g_ready = 0;
}

/* upload a double array as REAL */
static int upload_real(CUdeviceptr dst, const double *src, int n)
{
    if (g_double) return cuchk(p_cuMemcpyHtoD(dst, src, sizeof(double) * (size_t)n), "upload");
    float *f = (float *)h_conv;
    for (int i = 0; i < n; i++) f[i] = (float)src[i];
    return cuchk(p_cuMemcpyHtoD(dst, f, sizeof(float) * (size_t)n), "upload");
}

static int download_real(double *dst, CUdeviceptr src, int cnt)
{
    if (g_double) return cuchk(p_cuMemcpyDtoH(dst, src, sizeof(double) * (size_t)cnt), "download");
    float *f = (float *)h_conv;
    if (cuchk(p_cuMemcpyDtoH(f, src, sizeof(float) * (size_t)cnt), "download")) return -1;
    for (int i = 0; i < cnt; i++) dst[i] = (double)f[i];
    return 0;
}

int gpu_set_positions(const double *x, const double *y, const double *z, const double *m, int n)
{
    if (!g_ready) return -1;
    if (n > g_n) { seterr("%s (%d)", "too many bodies for GPU buffers", n); return -1; }
    if (upload_real(d_x, x, n) || upload_real(d_y, y, n) || upload_real(d_z, z, n) || upload_real(d_m, m, n))
        return -1;
    return 0;
}

static int download_all(int cnt, double *ax, double *ay, double *az)
{
    if (cuchk(p_cuCtxSynchronize(), "kernel execution")) return -1;
    if (download_real(ax, d_ox, cnt) || download_real(ay, d_oy, cnt) || download_real(az, d_oz, cnt)) return -1;
    return 0;
}

static void set_eps(void *dst, double eps2)
{
    if (g_double) *(double *)dst = eps2; else *(float *)dst = (float)eps2;
}

int gpu_direct(int lo, int cnt, double eps2, double *ax, double *ay, double *az)
{
    if (!g_ready) return -1;
    if (cnt <= 0) return 0;
    union { double d; float f; } e;
    set_eps(&e, eps2);
    int n = g_n;
    void *args[] = { &d_x, &d_y, &d_z, &d_m, &n, &lo, &cnt, &e, &d_ox, &d_oy, &d_oz };
    const unsigned block = 256;
    const unsigned grid = ((unsigned)cnt + block - 1) / block;
    const unsigned shared = block * 4u * (unsigned)(g_double ? sizeof(double) : sizeof(float));
    if (cuchk(p_cuLaunchKernel(g_fdirect, grid, 1, 1, block, 1, 1, shared, NULL, args, NULL), "launch direct_kernel"))
        return -1;
    return download_all(cnt, ax, ay, az);
}

int gpu_near(int lo, int cnt, double eps2, const int *rng, int cap, const int *nnear, int maxnear,
             double *ax, double *ay, double *az)
{
    if (!g_ready) return -1;
    if (cnt <= 0) return 0;
    if (cap > g_cap || cnt > g_maxcnt) { seterr("%s (%d)", "near-list buffer too small", cnt); return -1; }
    if (maxnear < 0) maxnear = 0;

    if (cuchk(p_cuMemcpyHtoD(d_nn, nnear, sizeof(int) * (size_t)cnt), "upload near counts")) return -1;
    if (maxnear > 0) {
        CUDA_MEMCPY2D cp;
        memset(&cp, 0, sizeof cp);
        cp.srcMemoryType = CU_MEMORYTYPE_HOST;
        cp.srcHost = rng;
        cp.srcPitch = sizeof(int) * 2u * (size_t)cap;
        cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.dstDevice = d_rng;
        cp.dstPitch = sizeof(int) * 2u * (size_t)cap;
        cp.WidthInBytes = sizeof(int) * 2u * (size_t)maxnear;
        cp.Height = (size_t)cnt;
        if (cuchk(p_cuMemcpy2D(&cp), "upload near lists")) return -1;
    }
    union { double d; float f; } e;
    set_eps(&e, eps2);
    void *args[] = { &d_x, &d_y, &d_z, &d_m, &lo, &cnt, &e, &d_rng, &cap, &d_nn, &d_ox, &d_oy, &d_oz };
    const unsigned block = 128;
    const unsigned grid = ((unsigned)cnt + block - 1) / block;
    if (cuchk(p_cuLaunchKernel(g_fnear, grid, 1, 1, block, 1, 1, 0, NULL, args, NULL), "launch near_kernel"))
        return -1;
    return download_all(cnt, ax, ay, az);
}

#endif /* NO_CUDA */
