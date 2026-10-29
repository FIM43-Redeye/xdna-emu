/* bdd_output.h -- Output formatting for BDD enumeration
 *
 * Handles rendering of cubes and patterns in multiple formats:
 * - hex:    Human-readable hex (e.g. "0x00001234...") with bit annotations
 * - binary: Full 128-bit binary string
 * - raw:    16-byte binary records (for piping to objdump / our decoder)
 * - json:   Machine-parseable JSON output
 */

#ifndef BDD_OUTPUT_H
#define BDD_OUTPUT_H

#include "bdd_analysis.h"

typedef enum {
    FMT_HEX,
    FMT_BINARY,
    FMT_RAW,
    FMT_JSON,
} output_format_t;

typedef struct {
    output_format_t format;
    int expand;         /* Expand don't-cares to all concrete patterns */
    int force;          /* Allow expansion even when minterm count is huge */
    long max_patterns;  /* 0 = unlimited, >0 = cap per root */
    int verbose;        /* Extra annotations */
    int quiet;          /* Minimal output */
} output_options_t;

/* Print summary table: one line per root with stats.
 * Requires analyze_all_roots() to have been called. */
void output_summary(bdd_forest_t *forest, const output_options_t *opts);

/* Print detailed characterization of a single root. */
void output_characterize(bdd_forest_t *forest, int root_index,
                         const output_options_t *opts);

/* Enumerate cubes for a single root. */
void output_enumerate(bdd_forest_t *forest, int root_index,
                      const output_options_t *opts);

/* Print minterm counts only (one line per root). */
void output_counts(bdd_forest_t *forest, const output_options_t *opts);

/* Print format word cluster table. */
void output_clusters(bdd_forest_t *forest, const output_options_t *opts);

#endif /* BDD_OUTPUT_H */
