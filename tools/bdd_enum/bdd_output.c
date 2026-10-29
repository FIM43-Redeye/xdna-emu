/* bdd_output.c -- Output formatting for BDD enumeration
 *
 * Renders BDD cubes and analysis results in multiple formats.
 * The raw format outputs 16-byte little-endian records suitable for piping
 * to llvm-objdump or our TableGen-based decoder for cross-validation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bdd_output.h"
#include "cudd.h"

/* Maximum minterm count before requiring --force for expansion */
#define EXPAND_SAFETY_LIMIT 16000000.0

/* ---- Helpers ---- */

/* Convert a cube (0/1/2 per bit) to a 128-bit hex string.
 * Don't-care bits are shown as 'x' in binary mode, '.' in hex mode. */
static void cube_to_hex(int *cube, int nvars, char *buf, int buflen)
{
    /* Build from MSB (bit 127) to LSB (bit 0) as 32 hex nybbles.
     * Each nybble = 4 bits. If any bit in a nybble is don't-care,
     * show the nybble as '.'. */
    int pos = 0;
    for (int nyb = 31; nyb >= 0 && pos < buflen - 1; nyb--) {
        int base = nyb * 4;
        int has_dc = 0;
        int val = 0;
        for (int b = 3; b >= 0; b--) {
            int bit = base + b;
            if (bit < nvars && cube[bit] == 2) has_dc = 1;
            int v = (bit < nvars) ? cube[bit] : 0;
            if (v == 2) v = 0; /* treat don't-care as 0 for value */
            val = (val << 1) | (v & 1);
        }
        if (has_dc)
            buf[pos++] = '.';
        else
            buf[pos++] = "0123456789abcdef"[val];
    }
    buf[pos] = '\0';
}

/* Convert a cube to a 128-bit binary string (MSB first).
 * Don't-care = 'x'. */
static void cube_to_binary(int *cube, int nvars, char *buf, int buflen)
{
    int pos = 0;
    for (int i = 127; i >= 0 && pos < buflen - 1; i--) {
        if (i < nvars) {
            switch (cube[i]) {
            case 0: buf[pos++] = '0'; break;
            case 1: buf[pos++] = '1'; break;
            case 2: buf[pos++] = 'x'; break;
            }
        } else {
            buf[pos++] = '0';
        }
    }
    buf[pos] = '\0';
}

/* Convert a cube to a 16-byte pattern, replacing don't-cares with 0.
 * Little-endian: byte[0] = bits 0-7, byte[15] = bits 120-127. */
static void cube_to_bytes(int *cube, int nvars, uint8_t *bytes)
{
    memset(bytes, 0, 16);
    for (int i = 0; i < nvars && i < 128; i++) {
        if (cube[i] == 1)
            bytes[i / 8] |= (1 << (i % 8));
    }
}

/* Recursively expand don't-care bits and call a callback for each pattern.
 * cube is modified in-place; dc_positions lists the variable indices of
 * don't-care bits. Returns number of patterns emitted, or -1 to stop. */
typedef int (*pattern_cb)(int *cube, int nvars, void *ctx);

static long expand_and_emit(int *cube, int nvars,
                            int *dc_positions, int num_dc, int dc_idx,
                            long max_patterns, long *emitted,
                            pattern_cb cb, void *ctx)
{
    if (max_patterns > 0 && *emitted >= max_patterns)
        return 0;

    if (dc_idx >= num_dc) {
        /* All don't-cares assigned; emit this pattern */
        if (cb(cube, nvars, ctx) < 0) return -1;
        (*emitted)++;
        return 1;
    }

    int pos = dc_positions[dc_idx];

    /* Try 0 */
    cube[pos] = 0;
    long r = expand_and_emit(cube, nvars, dc_positions, num_dc, dc_idx + 1,
                             max_patterns, emitted, cb, ctx);
    if (r < 0) return -1;

    if (max_patterns > 0 && *emitted >= max_patterns) {
        cube[pos] = 2; /* restore */
        return 0;
    }

    /* Try 1 */
    cube[pos] = 1;
    r = expand_and_emit(cube, nvars, dc_positions, num_dc, dc_idx + 1,
                        max_patterns, emitted, cb, ctx);

    cube[pos] = 2; /* restore don't-care */
    return r;
}

/* ---- Output callbacks ---- */

typedef struct {
    output_format_t format;
    int nvars;
    int root_index;
    long count;
    int first_json; /* For JSON comma handling */
} emit_ctx_t;

static int emit_pattern(int *cube, int nvars, void *vctx)
{
    emit_ctx_t *ctx = vctx;
    ctx->count++;

    switch (ctx->format) {
    case FMT_HEX: {
        char buf[64];
        cube_to_hex(cube, nvars, buf, sizeof(buf));
        printf("%s\n", buf);
        break;
    }
    case FMT_BINARY: {
        char buf[256];
        cube_to_binary(cube, nvars, buf, sizeof(buf));
        printf("%s\n", buf);
        break;
    }
    case FMT_RAW: {
        uint8_t bytes[16];
        cube_to_bytes(cube, nvars, bytes);
        fwrite(bytes, 1, 16, stdout);
        break;
    }
    case FMT_JSON: {
        uint8_t bytes[16];
        cube_to_bytes(cube, nvars, bytes);
        if (!ctx->first_json) printf(",\n");
        ctx->first_json = 0;
        printf("    {\"root\":%d,\"bytes\":[", ctx->root_index);
        for (int i = 0; i < 16; i++) {
            if (i) printf(",");
            printf("%d", bytes[i]);
        }
        printf("]}");
        break;
    }
    }
    return 0;
}

/* ---- Summary output ---- */

void output_summary(bdd_forest_t *forest, const output_options_t *opts)
{
    if (!forest->analyses_valid) return;

    if (!opts->quiet) {
        printf("=== BDD Forest Summary ===\n");
        printf("Roots: %d, Variables: %d\n\n", forest->num_roots, forest->num_vars);
    }

    if (opts->format == FMT_JSON) {
        printf("{\"roots\":%d,\"variables\":%d,\"analyses\":[\n", forest->num_roots, forest->num_vars);
    }

    if (opts->format != FMT_JSON && !opts->quiet) {
        printf("%-6s  %-8s  %-7s  %-15s  %-5s  %-5s  %-5s  %s\n",
               "Root", "DAG", "Cubes", "Minterms", "Fix1", "Fix0", "Var", "FmtWord");
        printf("%-6s  %-8s  %-7s  %-15s  %-5s  %-5s  %-5s  %s\n",
               "----", "---", "-----", "--------", "----", "----", "---", "-------");
    }

    for (int i = 0; i < forest->num_roots; i++) {
        root_analysis_t *a = &forest->analyses[i];

        if (opts->format == FMT_JSON) {
            if (i > 0) printf(",\n");
            printf("  {\"root\":%d,\"dag\":%d,\"cubes\":%d,\"minterms\":%.0f,"
                   "\"fixed_one\":%d,\"fixed_zero\":%d,\"variable\":%d,"
                   "\"format_word\":%d}",
                   i, a->dag_size, a->num_cubes, a->minterm_count,
                   a->num_fixed_one, a->num_fixed_zero, a->num_variable,
                   a->format_word);
        } else {
            char fw_buf[16];
            if (a->format_word >= 0)
                snprintf(fw_buf, sizeof(fw_buf), "0x%04x", a->format_word);
            else
                snprintf(fw_buf, sizeof(fw_buf), "varies");

            printf("%-6d  %-8d  %-7d  %-15.0f  %-5d  %-5d  %-5d  %s\n",
                   i, a->dag_size, a->num_cubes, a->minterm_count,
                   a->num_fixed_one, a->num_fixed_zero, a->num_variable,
                   fw_buf);
        }
    }

    if (opts->format == FMT_JSON)
        printf("\n]}\n");
    else
        printf("\n");
}

/* ---- Characterize output ---- */

void output_characterize(bdd_forest_t *forest, int root_index,
                         const output_options_t *opts)
{
    root_analysis_t *a = &forest->analyses[root_index];

    if (opts->format == FMT_JSON) {
        printf("{\"root\":%d,\"dag\":%d,\"cubes\":%d,\"minterms\":%.0f,\n",
               root_index, a->dag_size, a->num_cubes, a->minterm_count);
        printf(" \"fixed_one\":%d,\"fixed_zero\":%d,\"variable\":%d,\n",
               a->num_fixed_one, a->num_fixed_zero, a->num_variable);
        printf(" \"format_word\":%d,\n", a->format_word);

        /* Fixed mask and value as hex */
        printf(" \"fixed_mask\":\"");
        for (int i = 15; i >= 0; i--) printf("%02x", a->fixed_mask[i]);
        printf("\",\n \"fixed_value\":\"");
        for (int i = 15; i >= 0; i--) printf("%02x", a->fixed_value[i]);
        printf("\",\n");

        /* Per-bit detail */
        printf(" \"bits\":[");
        for (int i = 0; i < 128; i++) {
            if (i) printf(",");
            printf("%d", a->fixed_bits[i]);
        }
        printf("]}\n");
        return;
    }

    printf("=== Root %d Characterization ===\n\n", root_index);
    printf("DAG size:     %d nodes\n", a->dag_size);
    printf("Cubes:        %d\n", a->num_cubes);
    printf("Minterms:     %.0f (2^%.1f)\n",
           a->minterm_count, log2(a->minterm_count));
    printf("Fixed 1-bits: %d\n", a->num_fixed_one);
    printf("Fixed 0-bits: %d\n", a->num_fixed_zero);
    printf("Variable:     %d\n", a->num_variable);

    if (a->format_word >= 0)
        printf("Format word:  0x%04x (15 bits)\n", a->format_word);
    else
        printf("Format word:  varies (some bits 0-14 are not fixed)\n");

    /* Print bit map: show fixed pattern with dots for variable bits */
    printf("\nBit pattern (MSB..LSB, 128 bits):\n  ");
    for (int i = 127; i >= 0; i--) {
        switch (a->fixed_bits[i]) {
        case 0:  putchar('0'); break;
        case 1:  putchar('1'); break;
        case -1: putchar('.'); break;
        }
        /* Group by bytes with spaces, and by 32-bit words with double space */
        if (i > 0 && i % 32 == 0) printf("  ");
        else if (i > 0 && i % 8 == 0) printf(" ");
    }
    printf("\n");

    /* Annotate bit ranges */
    printf("\nBit  0-14  (format word): ");
    for (int i = 14; i >= 0; i--) {
        switch (a->fixed_bits[i]) {
        case 0:  putchar('0'); break;
        case 1:  putchar('1'); break;
        case -1: putchar('.'); break;
        }
    }
    printf("\n");

    /* Show fixed value as hex (for mask comparison with ISB) */
    printf("\nFixed mask  (hex, MSB first): ");
    for (int i = 15; i >= 0; i--) printf("%02x", a->fixed_mask[i]);
    printf("\n");
    printf("Fixed value (hex, MSB first): ");
    for (int i = 15; i >= 0; i--) printf("%02x", a->fixed_value[i]);
    printf("\n");

    /* List variable bit ranges (contiguous runs) */
    if (opts->verbose) {
        printf("\nVariable bit ranges:\n");
        int in_range = 0, range_start = 0;
        for (int i = 0; i < 128; i++) {
            if (a->fixed_bits[i] == -1) {
                if (!in_range) { range_start = i; in_range = 1; }
            } else {
                if (in_range) {
                    if (range_start == i - 1)
                        printf("  [%d]\n", range_start);
                    else
                        printf("  [%d..%d] (%d bits)\n",
                               range_start, i - 1, i - range_start);
                    in_range = 0;
                }
            }
        }
        if (in_range) {
            if (range_start == 127)
                printf("  [127]\n");
            else
                printf("  [%d..127] (%d bits)\n",
                       range_start, 128 - range_start);
        }
    }
}

/* ---- Cube enumeration ---- */

void output_enumerate(bdd_forest_t *forest, int root_index,
                      const output_options_t *opts)
{
    root_analysis_t *a = &forest->analyses[root_index];
    DdNode *root = forest->roots[root_index];

    /* Safety check for --expand: skip if --max limits the output */
    if (opts->expand && a->minterm_count > EXPAND_SAFETY_LIMIT
        && !opts->force && opts->max_patterns <= 0) {
        fprintf(stderr,
                "Error: root %d has %.0f minterms (%.1f bits of freedom).\n"
                "  Expansion would generate too many patterns.\n"
                "  Use --force to override, or use --max N to limit output.\n",
                root_index, a->minterm_count, log2(a->minterm_count));
        return;
    }

    emit_ctx_t ctx = {
        .format = opts->format,
        .nvars = forest->num_vars,
        .root_index = root_index,
        .count = 0,
        .first_json = 1,
    };

    if (opts->format == FMT_JSON)
        printf("{\"root\":%d,\"patterns\":[\n", root_index);

    DdGen *gen;
    int *cube;
    CUDD_VALUE_TYPE value;

    Cudd_ForeachCube(forest->mgr, root, gen, cube, value) {
        if (opts->max_patterns > 0 && ctx.count >= opts->max_patterns)
            break;

        if (opts->expand) {
            /* Find don't-care positions */
            int dc_pos[128];
            int num_dc = 0;
            for (int i = 0; i < forest->num_vars && i < 128; i++) {
                if (cube[i] == 2)
                    dc_pos[num_dc++] = i;
            }

            if (num_dc == 0) {
                /* No don't-cares; emit directly */
                emit_pattern(cube, forest->num_vars, &ctx);
            } else {
                /* Expand all don't-care combinations */
                long emitted = ctx.count;
                expand_and_emit(cube, forest->num_vars,
                                dc_pos, num_dc, 0,
                                opts->max_patterns, &emitted,
                                emit_pattern, &ctx);
                ctx.count = emitted;
            }
        } else {
            emit_pattern(cube, forest->num_vars, &ctx);
        }
    }

    if (opts->format == FMT_JSON)
        printf("\n],\"count\":%ld}\n", ctx.count);
    else if (opts->format != FMT_RAW && !opts->quiet)
        fprintf(stderr, "Root %d: %ld %s emitted\n",
                root_index, ctx.count,
                opts->expand ? "patterns" : "cubes");
}

/* ---- Count-only output ---- */

void output_counts(bdd_forest_t *forest, const output_options_t *opts)
{
    if (opts->format == FMT_JSON) printf("[");

    for (int i = 0; i < forest->num_roots; i++) {
        DdNode *root = forest->roots[i];
        double count = Cudd_CountMinterm(forest->mgr, root, forest->num_vars);

        if (opts->format == FMT_JSON) {
            if (i > 0) printf(",");
            printf("{\"root\":%d,\"minterms\":%.0f}", i, count);
        } else {
            printf("%d\t%.0f\n", i, count);
        }
    }

    if (opts->format == FMT_JSON) printf("]\n");
}

/* ---- Cluster output ---- */

void output_clusters(bdd_forest_t *forest, const output_options_t *opts)
{
    if (!forest->clusters || forest->num_clusters == 0) return;

    if (opts->format == FMT_JSON) {
        printf("{\"clusters\":[\n");
        for (int i = 0; i < forest->num_clusters; i++) {
            format_cluster_t *cl = &forest->clusters[i];
            if (i > 0) printf(",\n");
            printf("  {\"format_word\":%d,\"roots\":[", cl->format_word);
            for (int j = 0; j < cl->count; j++) {
                if (j) printf(",");
                printf("%d", cl->root_indices[j]);
            }
            printf("]}");
        }
        printf("\n]}\n");
        return;
    }

    printf("=== Format Word Clusters ===\n");
    printf("%d unique format word patterns\n\n", forest->num_clusters);

    printf("%-10s  %-15s  %-6s  %s\n",
           "FmtWord", "Binary (14..0)", "Roots", "Example roots");
    printf("%-10s  %-15s  %-6s  %s\n",
           "-------", "--------------", "-----", "-------------");

    for (int i = 0; i < forest->num_clusters; i++) {
        format_cluster_t *cl = &forest->clusters[i];

        char fw_str[16];
        char bin_str[20];

        if (cl->format_word >= 0) {
            snprintf(fw_str, sizeof(fw_str), "0x%04x", cl->format_word);
            /* Binary representation of format word, MSB first */
            for (int b = 14; b >= 0; b--)
                bin_str[14 - b] = (cl->format_word & (1 << b)) ? '1' : '0';
            bin_str[15] = '\0';
        } else {
            snprintf(fw_str, sizeof(fw_str), "<varies>");
            snprintf(bin_str, sizeof(bin_str), "...............");
        }

        /* Show first few root indices as examples */
        char examples[80];
        int pos = 0;
        int show = (cl->count < 5) ? cl->count : 5;
        for (int j = 0; j < show && pos < (int)sizeof(examples) - 10; j++) {
            if (j > 0) pos += snprintf(examples + pos, sizeof(examples) - pos, ", ");
            pos += snprintf(examples + pos, sizeof(examples) - pos, "%d",
                            cl->root_indices[j]);
        }
        if (cl->count > show)
            snprintf(examples + pos, sizeof(examples) - pos, ", ...");

        printf("%-10s  %s  %-6d  %s\n", fw_str, bin_str, cl->count, examples);
    }
    printf("\n");
}
