/* affinity.c - optional pinning of OpenMP threads to logical CPUs.
 *
 * On hybrid CPUs (e.g. Intel P-cores + E-cores) unpinned threads run at different speeds,
 * which shows up as load imbalance unrelated to the workload.  The default CPU order puts
 * one logical CPU of every physical core first, fastest cores (highest EfficiencyClass)
 * first, then the remaining SMT siblings - so up to #P-cores threads land on distinct
 * P-cores.  libgomp keeps its worker threads for later parallel regions with the same
 * thread count, so pinning once at start-up holds for the whole run.
 */
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <windows.h>
#else
#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>
#endif

#include "nbody.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXCPU 64

/* default order of logical CPUs; returns count */
static int default_order(int *order)
{
    int n = 0;
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    unsigned char *buf = (unsigned char *)malloc(len ? len : 1);
    if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore,
                                                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf, &len)) {
        int ncore = 0, eff[MAXCPU];
        unsigned long long mask[MAXCPU];
        for (DWORD off = 0; off < len && ncore < MAXCPU;) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX e = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf + off);
            if (e->Relationship == RelationProcessorCore && e->Processor.GroupMask[0].Group == 0) {
                eff[ncore] = ((const BYTE *)&e->Processor)[1];     /* EfficiencyClass (byte after Flags) */
                mask[ncore] = (unsigned long long)e->Processor.GroupMask[0].Mask;
                ncore++;
            }
            off += e->Size;
        }
        /* cores sorted by efficiency class (descending), stable */
        int idx[MAXCPU];
        for (int i = 0; i < ncore; i++) idx[i] = i;
        for (int i = 1; i < ncore; i++)
            for (int j = i; j > 0 && eff[idx[j]] > eff[idx[j - 1]]; j--) { int t = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = t; }
        /* pass 1: first logical CPU of every core; pass 2: the remaining siblings */
        for (int pass = 0; pass < 2; pass++)
            for (int ci = 0; ci < ncore; ci++) {
                unsigned long long m = mask[idx[ci]];
                int first = 1;
                for (int bit = 0; bit < MAXCPU; bit++)
                    if (m & (1ULL << bit)) {
                        if ((pass == 0) == first && n < MAXCPU) order[n++] = bit;
                        first = 0;
                    }
            }
    }
    free(buf);
    if (n == 0) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        for (DWORD i = 0; i < si.dwNumberOfProcessors && n < MAXCPU; i++) order[n++] = (int)i;
    }
#else
    long nc = sysconf(_SC_NPROCESSORS_ONLN);
    for (long i = 0; i < nc && n < MAXCPU; i++) order[n++] = (int)i;
#endif
    return n;
}

static int parse_list(const char *s, int *order)
{
    int n = 0;
    while (*s && n < MAXCPU) {
        char *end;
        long a = strtol(s, &end, 10);
        if (end == s) break;
        long b = a;
        s = end;
        if (*s == '-') { s++; b = strtol(s, &end, 10); if (end == s) break; s = end; }
        for (long v = a; v <= b && n < MAXCPU; v++) if (v >= 0 && v < MAXCPU) order[n++] = (int)v;
        if (*s == ',') s++;
    }
    return n;
}

static int set_affinity(int cpu)
{
#ifdef _WIN32
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu) ? 0 : -1;
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof set, &set);
#endif
}

int pin_threads(int rank, int threads, const char *cpus, char *desc, size_t desc_len)
{
    int order[MAXCPU];
    int n = (cpus && *cpus) ? parse_list(cpus, order) : default_order(order);
    if (n <= 0) { snprintf(desc, desc_len, "no CPUs"); return -1; }

    int fails = 0;
    int map[MAXCPU];
    #pragma omp parallel reduction(+:fails)
    {
        const int t = omp_get_thread_num();
        const int cpu = order[(rank * threads + t) % n];
        if (t < MAXCPU) map[t] = cpu;
        if (set_affinity(cpu) != 0) fails++;
    }
    size_t used = (size_t)snprintf(desc, desc_len, "rank %d:", rank);
    for (int t = 0; t < threads && t < MAXCPU && used < desc_len; t++)
        used += (size_t)snprintf(desc + used, desc_len - used, " t%d->cpu%d", t, map[t]);
    if (rank * threads + threads > n && used < desc_len)
        snprintf(desc + used, desc_len - used, " (oversubscribed: %d CPUs in list)", n);
    return fails ? -1 : 0;
}
