/* bdd_analysis.c -- Per-root BDD analysis and format word clustering
 *
 * The core analysis loop uses Cudd_ForeachCube to iterate over all cubes
 * (paths to the 1-terminal) of each root BDD. Each cube assigns each of
 * the 128 variables to 0, 1, or 2 (don't-care). By tracking which bits
 * are constant across ALL cubes of a root, we identify:
 *   - Fixed-1 bits: opcode/format bits that are always set
 *   - Fixed-0 bits: opcode/format bits that are always clear
 *   - Variable bits: operand fields, immediates, register selectors
 *
 * Format word clustering groups roots by their fixed pattern in bits 0-14,
 * which we know from ISB analysis to be the VLIW format selector. Each
 * cluster corresponds to one bundle template (slot assignment).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#include "bdd_analysis.h"
#include "dddmp.h"

/* ---- Forest loading ---- */

int bdd_forest_load(bdd_forest_t *forest, const char *path)
{
    memset(forest, 0, sizeof(*forest));

    /* Create CUDD manager with 128 variables */
    forest->mgr = Cudd_Init(128, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);
    if (!forest->mgr) {
        fprintf(stderr, "Error: Cudd_Init failed\n");
        return -1;
    }

    /* Load the DDDMP forest. DDDMP_VAR_MATCHIDS uses the .ids header to
     * map BDD variable indices, which for our files is identity (0..127). */
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open %s: %s\n", path, strerror(errno));
        Cudd_Quit(forest->mgr);
        return -1;
    }

    /* Dddmp_cuddBddArrayLoad signature (CUDD 3.0):
     *   int nroots = Dddmp_cuddBddArrayLoad(
     *       DdManager*, RootMatchType, char **rootmatchnames,
     *       VarMatchType, char **varmatchnames, int *varmatchauxids,
     *       int *varcomposeids, int mode, char *file, FILE *fp,
     *       DdNode ***pproots);
     * Returns number of roots loaded; root array via pproots. */
    DdNode **roots = NULL;
    int nroots = Dddmp_cuddBddArrayLoad(
        forest->mgr,
        DDDMP_ROOT_MATCHLIST,   /* Load all roots */
        NULL,                   /* rootmatchnames (NULL = load all) */
        DDDMP_VAR_MATCHIDS,    /* Match variables by ID */
        NULL,                   /* varmatchnames (unused with MATCHIDS) */
        NULL,                   /* varmatchauxids (unused) */
        NULL,                   /* varcomposeids (unused) */
        DDDMP_MODE_BINARY,
        NULL,                   /* filename (unused, using fp) */
        fp,
        &roots
    );

    fclose(fp);

    if (!roots || nroots <= 0) {
        fprintf(stderr, "Error: Dddmp_cuddBddArrayLoad failed (nroots=%d)\n",
                nroots);
        Cudd_Quit(forest->mgr);
        return -1;
    }

    forest->roots = roots;
    forest->num_roots = nroots;
    forest->num_vars = Cudd_ReadSize(forest->mgr);

    /* Allocate analysis array (zeroed = not yet computed) */
    forest->analyses = calloc(nroots, sizeof(root_analysis_t));
    if (!forest->analyses) {
        fprintf(stderr, "Error: allocation failed for %d analyses\n", nroots);
        /* Free root references */
        for (int i = 0; i < nroots; i++) {
            if (forest->roots[i])
                Cudd_RecursiveDeref(forest->mgr, forest->roots[i]);
        }
        free(forest->roots);
        Cudd_Quit(forest->mgr);
        return -1;
    }

    return 0;
}

void bdd_forest_free(bdd_forest_t *forest)
{
    if (!forest) return;

    /* Free cluster data */
    if (forest->clusters) {
        for (int i = 0; i < forest->num_clusters; i++)
            free(forest->clusters[i].root_indices);
        free(forest->clusters);
    }

    free(forest->analyses);

    /* Deref all root BDDs */
    if (forest->roots && forest->mgr) {
        for (int i = 0; i < forest->num_roots; i++) {
            if (forest->roots[i])
                Cudd_RecursiveDeref(forest->mgr, forest->roots[i]);
        }
        free(forest->roots);
    }

    if (forest->mgr)
        Cudd_Quit(forest->mgr);

    memset(forest, 0, sizeof(*forest));
}

/* ---- Per-root analysis ---- */

int analyze_root(bdd_forest_t *forest, int root_index)
{
    if (root_index < 0 || root_index >= forest->num_roots) {
        fprintf(stderr, "Error: root index %d out of range\n", root_index);
        return -1;
    }

    DdNode *root = forest->roots[root_index];
    if (!root) {
        fprintf(stderr, "Error: root %d is NULL\n", root_index);
        return -1;
    }

    root_analysis_t *a = &forest->analyses[root_index];
    a->root_index = root_index;
    a->dag_size = Cudd_DagSize(root);
    a->minterm_count = Cudd_CountMinterm(forest->mgr, root, forest->num_vars);

    /* Use BDD cofactoring to determine fixed/variable bits efficiently.
     *
     * For each variable xi, we compute the cofactors f|xi=0 and f|xi=1:
     *   - If f|xi=1 == 0 (constant false), xi is always 0 in satisfying assignments
     *   - If f|xi=0 == 0 (constant false), xi is always 1 in satisfying assignments
     *   - Otherwise xi is variable (both values appear in some assignment)
     *
     * Each cofactor is O(|BDD|), so 128 cofactors is O(128 * |BDD|) -- much
     * faster than Cudd_ForeachCube which enumerates exponentially many paths.
     *
     * We also count cubes for informational purposes using Cudd_CountPath,
     * which counts the number of paths to the 1-terminal (= number of cubes
     * that Cudd_ForeachCube would yield). */
    DdNode *zero = Cudd_ReadLogicZero(forest->mgr);
    a->num_cubes = (int)Cudd_CountPathsToNonZero(root);

    /* Classify each bit using cofactoring */
    a->num_fixed_zero = 0;
    a->num_fixed_one = 0;
    a->num_variable = 0;
    memset(a->fixed_mask, 0, sizeof(a->fixed_mask));
    memset(a->fixed_value, 0, sizeof(a->fixed_value));

    for (int i = 0; i < 128; i++) {
        int byte_idx = i / 8;
        int bit_idx = i % 8;

        if (i >= forest->num_vars) {
            /* Variable doesn't exist in BDD; treat as fixed 0 */
            a->fixed_bits[i] = 0;
            a->num_fixed_zero++;
            a->fixed_mask[byte_idx] |= (1 << bit_idx);
            continue;
        }

        DdNode *var = Cudd_bddIthVar(forest->mgr, i);
        DdNode *cof_1 = Cudd_Cofactor(forest->mgr, root, var);
        Cudd_Ref(cof_1);
        DdNode *not_var = Cudd_Not(var);
        DdNode *cof_0 = Cudd_Cofactor(forest->mgr, root, not_var);
        Cudd_Ref(cof_0);

        if (cof_1 == zero) {
            /* f|xi=1 == 0: xi is always 0 in satisfying assignments */
            a->fixed_bits[i] = 0;
            a->num_fixed_zero++;
            a->fixed_mask[byte_idx] |= (1 << bit_idx);
        } else if (cof_0 == zero) {
            /* f|xi=0 == 0: xi is always 1 in satisfying assignments */
            a->fixed_bits[i] = 1;
            a->num_fixed_one++;
            a->fixed_mask[byte_idx] |= (1 << bit_idx);
            a->fixed_value[byte_idx] |= (1 << bit_idx);
        } else {
            /* Both cofactors are non-zero: variable bit */
            a->fixed_bits[i] = -1;
            a->num_variable++;
        }

        Cudd_RecursiveDeref(forest->mgr, cof_0);
        Cudd_RecursiveDeref(forest->mgr, cof_1);
    }

    /* Extract format word: bits 0-14. Value is -1 if any of those bits vary. */
    a->format_word = 0;
    for (int i = 0; i < 15; i++) {
        if (a->fixed_bits[i] == -1) {
            a->format_word = -1;
            break;
        }
        if (a->fixed_bits[i] == 1)
            a->format_word |= (1 << i);
    }

    return 0;
}

int analyze_all_roots(bdd_forest_t *forest)
{
    for (int i = 0; i < forest->num_roots; i++) {
        if (analyze_root(forest, i) != 0)
            return -1;
        /* Progress for large forests */
        if ((i + 1) % 1000 == 0)
            fprintf(stderr, "  Analyzed %d / %d roots...\n",
                    i + 1, forest->num_roots);
    }
    forest->analyses_valid = 1;
    if (forest->num_roots >= 1000)
        fprintf(stderr, "  Analyzed %d / %d roots.\n",
                forest->num_roots, forest->num_roots);
    return 0;
}

/* ---- Format word clustering ---- */

/* Comparison function for sorting clusters by format word */
static int cluster_cmp(const void *a, const void *b)
{
    const format_cluster_t *ca = a;
    const format_cluster_t *cb = b;
    return ca->format_word - cb->format_word;
}

int cluster_by_format_word(bdd_forest_t *forest)
{
    if (!forest->analyses_valid) {
        fprintf(stderr, "Error: must call analyze_all_roots() first\n");
        return -1;
    }

    /* Free any existing clusters */
    if (forest->clusters) {
        for (int i = 0; i < forest->num_clusters; i++)
            free(forest->clusters[i].root_indices);
        free(forest->clusters);
        forest->clusters = NULL;
        forest->num_clusters = 0;
    }

    /* Temporary: collect unique format words. We use a simple approach since
     * the number of distinct format words is bounded by 2^15 = 32768 but
     * in practice will be much smaller (likely < 100). */
    int cap = 256;
    format_cluster_t *clusters = calloc(cap, sizeof(format_cluster_t));
    int nclusters = 0;

    for (int i = 0; i < forest->num_roots; i++) {
        int fw = forest->analyses[i].format_word;

        /* Find existing cluster for this format word */
        int found = -1;
        for (int c = 0; c < nclusters; c++) {
            if (clusters[c].format_word == fw) {
                found = c;
                break;
            }
        }

        if (found < 0) {
            /* New cluster */
            if (nclusters >= cap) {
                cap *= 2;
                clusters = realloc(clusters, cap * sizeof(format_cluster_t));
            }
            found = nclusters++;
            clusters[found].format_word = fw;
            clusters[found].count = 0;
            clusters[found].capacity = 64;
            clusters[found].root_indices = malloc(64 * sizeof(int));
        }

        /* Add root to cluster */
        format_cluster_t *cl = &clusters[found];
        if (cl->count >= cl->capacity) {
            cl->capacity *= 2;
            cl->root_indices = realloc(cl->root_indices,
                                       cl->capacity * sizeof(int));
        }
        cl->root_indices[cl->count++] = i;
    }

    /* Sort clusters by format word value */
    qsort(clusters, nclusters, sizeof(format_cluster_t), cluster_cmp);

    forest->clusters = clusters;
    forest->num_clusters = nclusters;
    return 0;
}

/* ---- Cross-forest comparison ---- */

int compare_forests(bdd_forest_t *a, const char *name_a,
                    bdd_forest_t *b, const char *name_b)
{
    fprintf(stderr, "Analyzing both forests...\n");

    if (analyze_all_roots(a) != 0) return -1;
    if (analyze_all_roots(b) != 0) return -1;

    printf("=== Forest Comparison ===\n\n");
    printf("%-30s  %s\n", "Attribute", "Value");
    printf("%-30s  %s\n", "-----", "-----");
    printf("%-30s  %d\n", name_a, a->num_roots);
    printf("%-30s  %d\n", name_b, b->num_roots);
    printf("\n");

    /* Compare by format word distribution */
    if (cluster_by_format_word(a) != 0) return -1;
    if (cluster_by_format_word(b) != 0) return -1;

    printf("Format word clusters:\n");
    printf("  %-20s: %d unique format words\n", name_a, a->num_clusters);
    printf("  %-20s: %d unique format words\n", name_b, b->num_clusters);
    printf("\n");

    /* Find format words in A but not B, and vice versa */
    printf("Format words unique to %s:\n", name_a);
    int unique_a = 0;
    for (int i = 0; i < a->num_clusters; i++) {
        int fw = a->clusters[i].format_word;
        int found = 0;
        for (int j = 0; j < b->num_clusters; j++) {
            if (b->clusters[j].format_word == fw) { found = 1; break; }
        }
        if (!found) {
            if (fw >= 0)
                printf("  0x%04x (%d roots)\n", fw, a->clusters[i].count);
            else
                printf("  <variable> (%d roots)\n", a->clusters[i].count);
            unique_a++;
        }
    }
    if (unique_a == 0) printf("  (none)\n");

    printf("\nFormat words unique to %s:\n", name_b);
    int unique_b = 0;
    for (int j = 0; j < b->num_clusters; j++) {
        int fw = b->clusters[j].format_word;
        int found = 0;
        for (int i = 0; i < a->num_clusters; i++) {
            if (a->clusters[i].format_word == fw) { found = 1; break; }
        }
        if (!found) {
            if (fw >= 0)
                printf("  0x%04x (%d roots)\n", fw, b->clusters[j].count);
            else
                printf("  <variable> (%d roots)\n", b->clusters[j].count);
            unique_b++;
        }
    }
    if (unique_b == 0) printf("  (none)\n");

    /* Shared format words with root count comparison */
    printf("\nShared format words (roots in A / roots in B):\n");
    for (int i = 0; i < a->num_clusters; i++) {
        int fw = a->clusters[i].format_word;
        for (int j = 0; j < b->num_clusters; j++) {
            if (b->clusters[j].format_word == fw) {
                if (fw >= 0)
                    printf("  0x%04x: %d / %d\n",
                           fw, a->clusters[i].count, b->clusters[j].count);
                else
                    printf("  <variable>: %d / %d\n",
                           a->clusters[i].count, b->clusters[j].count);
                break;
            }
        }
    }

    return 0;
}
