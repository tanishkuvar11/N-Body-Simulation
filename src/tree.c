/* tree.c - Barnes-Hut octree built top-down from Morton-sorted bodies.
 *
 * Because the bodies are sorted by Morton key, every octree cell owns a contiguous
 * index range [lo,hi).  Cells are split by binary-searching the next 3 key bits.
 * Chains of single-child cells are collapsed (the tight cell is kept), so every
 * internal node has >= 2 children and the tree has at most 2N-1 nodes.
 */
#include "nbody.h"

#include <math.h>
#include <stdlib.h>

int tree_init(tree_t *t, int n, int leafcap)
{
    t->leafcap = leafcap < 1 ? 1 : leafcap;
    t->cap = 2 * n + 16;
    t->count = 0;
    t->nodes = (node_t *)malloc((size_t)t->cap * sizeof(node_t));
    return t->nodes ? 0 : -1;
}

void tree_free(tree_t *t)
{
    free(t->nodes);
    t->nodes = NULL;
    t->cap = t->count = 0;
}

/* first index in [lo,hi) whose octant digit at `shift` is >= o (digits are non-decreasing) */
static int lower_bound_digit(const uint64_t *key, int lo, int hi, unsigned o, int shift)
{
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (((key[mid] >> shift) & 7u) < o) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static void build_node(tree_t *t, const bodies_t *b, int slot, int lo, int hi, int level)
{
    int bnd[9];
    int nonempty;

    for (;;) {
        if (hi - lo <= t->leafcap || level >= MAXLEVEL) {
            /* leaf: centre of mass over its bodies */
            double m = 0, sx = 0, sy = 0, sz = 0;
            for (int i = lo; i < hi; i++) {
                m += b->m[i];
                sx += b->m[i] * b->x[i]; sy += b->m[i] * b->y[i]; sz += b->m[i] * b->z[i];
            }
            node_t *nd = &t->nodes[slot];
            double size = b->bside / (double)(1u << level);
            nd->m = m; nd->cx = sx / m; nd->cy = sy / m; nd->cz = sz / m;
            nd->s2 = size * size;
            nd->child0 = 0; nd->nchild = 0; nd->lo = lo; nd->hi = hi;
            return;
        }
        const int shift = 3 * (MAXLEVEL - 1 - level);
        bnd[0] = lo; bnd[8] = hi;
        for (unsigned o = 1; o < 8; o++) bnd[o] = lower_bound_digit(b->key, lo, hi, o, shift);
        nonempty = 0;
        for (int o = 0; o < 8; o++) nonempty += (bnd[o + 1] > bnd[o]);
        if (nonempty > 1) break;
        level++;                                   /* single child: collapse the chain */
    }

    node_t *nd = &t->nodes[slot];
    const int c0 = t->count;
    t->count += nonempty;
    nd->child0 = c0; nd->nchild = nonempty; nd->lo = lo; nd->hi = hi;
    {
        double size = b->bside / (double)(1u << level);
        nd->s2 = size * size;
    }

    int c = 0;
    for (int o = 0; o < 8; o++)
        if (bnd[o + 1] > bnd[o]) build_node(t, b, c0 + c++, bnd[o], bnd[o + 1], level + 1);

    /* centre of mass from children (re-fetch: nodes[] does not move, but be explicit) */
    nd = &t->nodes[slot];
    double m = 0, sx = 0, sy = 0, sz = 0;
    for (int j = 0; j < nonempty; j++) {
        const node_t *ch = &t->nodes[c0 + j];
        m += ch->m; sx += ch->m * ch->cx; sy += ch->m * ch->cy; sz += ch->m * ch->cz;
    }
    nd->m = m; nd->cx = sx / m; nd->cy = sy / m; nd->cz = sz / m;
}

void tree_build(tree_t *t, const bodies_t *b)
{
    t->count = 1;
    build_node(t, b, 0, 0, b->n, 0);
}
