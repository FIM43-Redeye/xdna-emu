/* bdd_analysis.h -- Per-root analysis and format word clustering
 *
 * Core analysis functions for characterizing BDD roots. Each root in the
 * DDDMP forest represents one instruction encoding variant. Analysis
 * identifies which of the 128 bundle bits are fixed (opcode/format) vs
 * variable (operands/immediates).
 */

#ifndef BDD_ANALYSIS_H
#define BDD_ANALYSIS_H

#include <stdint.h>
#include "cudd.h"

/* 128-bit pattern storage */
typedef struct {
    uint8_t bytes[16];
} pattern128_t;

/* Cube with don't-care support: each element is 0, 1, or 2 (don't-care) */
typedef struct {
    int bits[128];
} cube128_t;

/* Per-root analysis result */
typedef struct {
    int root_index;
    int dag_size;           /* Number of BDD nodes for this root */
    int num_cubes;          /* Number of cubes (paths to 1-terminal) */
    double minterm_count;   /* Total satisfying assignments (2^don't-cares summed) */

    /* Per-bit classification: -1 = varies across cubes, 0 = always 0, 1 = always 1 */
    int fixed_bits[128];

    int num_fixed_zero;     /* Count of bits that are always 0 */
    int num_fixed_one;      /* Count of bits that are always 1 */
    int num_variable;       /* Count of bits that vary */

    /* Compact representation: fixed_mask[i] bit j = 1 means bit (i*8+j) is fixed */
    uint8_t fixed_mask[16];
    /* fixed_value[i] bit j = the fixed value (valid only where mask bit is 1) */
    uint8_t fixed_value[16];

    /* Format word: bits 0-14 fixed pattern (if all fixed), -1 if any vary */
    int format_word;
} root_analysis_t;

/* A cluster of roots sharing the same format word pattern */
typedef struct {
    int format_word;        /* The shared fixed pattern in bits 0-14 */
    int *root_indices;      /* Array of root indices in this cluster */
    int count;              /* Number of roots in cluster */
    int capacity;           /* Allocated capacity */
} format_cluster_t;

/* Forest-level analysis state */
typedef struct {
    DdManager *mgr;
    DdNode **roots;
    int num_roots;
    int num_vars;
    root_analysis_t *analyses;  /* Array of num_roots analyses (lazily filled) */
    int analyses_valid;         /* 1 if analyses[] is populated */

    format_cluster_t *clusters;
    int num_clusters;
} bdd_forest_t;

/* Load a DDDMP forest from an .ena file.
 * Returns 0 on success, -1 on error. */
int bdd_forest_load(bdd_forest_t *forest, const char *path);

/* Free all resources associated with a forest. */
void bdd_forest_free(bdd_forest_t *forest);

/* Analyze a single root: identify fixed/variable bits, count cubes.
 * Result is stored in forest->analyses[root_index].
 * Returns 0 on success, -1 on error. */
int analyze_root(bdd_forest_t *forest, int root_index);

/* Analyze all roots. Returns 0 on success, -1 on error. */
int analyze_all_roots(bdd_forest_t *forest);

/* Group roots by their format word (bits 0-14 fixed pattern).
 * Requires analyze_all_roots() to have been called first.
 * Returns 0 on success, -1 on error. */
int cluster_by_format_word(bdd_forest_t *forest);

/* Compare two forests and report differences.
 * Prints per-root comparison to stdout.
 * Returns 0 on success, -1 on error. */
int compare_forests(bdd_forest_t *a, const char *name_a,
                    bdd_forest_t *b, const char *name_b);

#endif /* BDD_ANALYSIS_H */
