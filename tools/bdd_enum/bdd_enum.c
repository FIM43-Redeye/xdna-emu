/* bdd_enum.c -- BDD-based AIE2 instruction encoding enumeration tool
 *
 * Loads CUDD/DDDMP BDD forests (.ena files) from the aietools ISG directory
 * and provides analysis of instruction encoding patterns. Each BDD root
 * represents one instruction encoding variant; the 128 BDD variables map to
 * the 128 bits of a VLIW instruction bundle.
 *
 * This tool bridges the ISB's hierarchical ISG coordinates with physical
 * bundle bit positions, enabling format template derivation and exhaustive
 * decoder validation.
 *
 * Usage: bdd_enum [OPTIONS] <ena-file> [second-ena-file]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>

#include "bdd_analysis.h"
#include "bdd_output.h"
#include "dddmp.h"

/* ---- CLI mode selection ---- */

typedef enum {
    MODE_SUMMARY,       /* Default: per-root stats + format word clusters */
    MODE_CHARACTERIZE,  /* Detailed single-root analysis */
    MODE_ENUMERATE,     /* Dump cubes for one root */
    MODE_ENUMERATE_ALL, /* Dump cubes for all roots */
    MODE_COUNT,         /* Just minterm counts per root */
    MODE_COMPARE,       /* Diff two ENA files */
} run_mode_t;

/* ---- Usage ---- */

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [OPTIONS] <ena-file> [second-ena-file]\n"
        "\n"
        "Modes:\n"
        "  --summary              Per-root stats + format word clusters (default)\n"
        "  --characterize ROOT    Detailed single-root analysis\n"
        "  --enumerate ROOT       Dump all cubes for one root\n"
        "  --enumerate-all        Dump cubes for all roots\n"
        "  --count                Just minterm counts per root\n"
        "  --compare              Diff two ENA files (requires two arguments)\n"
        "\n"
        "Options:\n"
        "  --format hex|binary|raw|json   Output format (default: hex)\n"
        "  --expand               Expand don't-cares to concrete 128-bit patterns\n"
        "  --force                Allow expansion even when minterm count is huge\n"
        "  --max N                Limit to N patterns per root\n"
        "  --roots RANGE          Process subset (e.g. \"0-99\" or \"42\")\n"
        "  --quiet                Minimal output\n"
        "  --verbose              Extra annotations\n"
        "  -h, --help             Show this help\n"
        "\n"
        "Examples:\n"
        "  %s me_das.ena                          # Summary of all 13K+ roots\n"
        "  %s --characterize 0 me_das.ena         # Analyze root 0\n"
        "  %s --enumerate 42 --format hex me.ena  # Dump root 42 cubes\n"
        "  %s --compare me.ena me_das.ena         # Diff compiler vs assembler\n",
        prog, prog, prog, prog, prog);
}

/* ---- Root range parsing ---- */

typedef struct {
    int start;      /* -1 = all */
    int end;        /* inclusive */
} root_range_t;

static int parse_root_range(const char *s, root_range_t *range)
{
    char *dash = strchr(s, '-');
    if (dash && dash != s) {
        /* "A-B" form */
        *dash = '\0';
        range->start = atoi(s);
        range->end = atoi(dash + 1);
        *dash = '-';
    } else {
        /* Single number */
        range->start = atoi(s);
        range->end = range->start;
    }
    if (range->start < 0 || range->end < range->start) {
        fprintf(stderr, "Error: invalid range '%s'\n", s);
        return -1;
    }
    return 0;
}

/* ---- Main ---- */

int main(int argc, char *argv[])
{
    run_mode_t mode = MODE_SUMMARY;
    int target_root = -1;
    output_options_t opts = {
        .format = FMT_HEX,
        .expand = 0,
        .force = 0,
        .max_patterns = 0,
        .verbose = 0,
        .quiet = 0,
    };
    root_range_t range = { .start = -1, .end = -1 };

    static struct option long_opts[] = {
        { "summary",        no_argument,       NULL, 'S' },
        { "characterize",   required_argument, NULL, 'c' },
        { "enumerate",      required_argument, NULL, 'e' },
        { "enumerate-all",  no_argument,       NULL, 'E' },
        { "count",          no_argument,       NULL, 'C' },
        { "compare",        no_argument,       NULL, 'D' },
        { "format",         required_argument, NULL, 'f' },
        { "expand",         no_argument,       NULL, 'x' },
        { "force",          no_argument,       NULL, 'F' },
        { "max",            required_argument, NULL, 'm' },
        { "roots",          required_argument, NULL, 'r' },
        { "quiet",          no_argument,       NULL, 'q' },
        { "verbose",        no_argument,       NULL, 'v' },
        { "help",           no_argument,       NULL, 'h' },
        { NULL, 0, NULL, 0 }
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "hqv", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'S': mode = MODE_SUMMARY; break;
        case 'c':
            mode = MODE_CHARACTERIZE;
            target_root = atoi(optarg);
            break;
        case 'e':
            mode = MODE_ENUMERATE;
            target_root = atoi(optarg);
            break;
        case 'E': mode = MODE_ENUMERATE_ALL; break;
        case 'C': mode = MODE_COUNT; break;
        case 'D': mode = MODE_COMPARE; break;
        case 'f':
            if (strcmp(optarg, "hex") == 0) opts.format = FMT_HEX;
            else if (strcmp(optarg, "binary") == 0) opts.format = FMT_BINARY;
            else if (strcmp(optarg, "raw") == 0) opts.format = FMT_RAW;
            else if (strcmp(optarg, "json") == 0) opts.format = FMT_JSON;
            else {
                fprintf(stderr, "Error: unknown format '%s'\n", optarg);
                return 1;
            }
            break;
        case 'x': opts.expand = 1; break;
        case 'F': opts.force = 1; break;
        case 'm': opts.max_patterns = atol(optarg); break;
        case 'r':
            if (parse_root_range(optarg, &range) != 0) return 1;
            break;
        case 'q': opts.quiet = 1; break;
        case 'v': opts.verbose = 1; break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 1;
        }
    }

    /* Need at least one ENA file */
    if (optind >= argc) {
        fprintf(stderr, "Error: no ENA file specified\n");
        usage(argv[0]);
        return 1;
    }

    const char *ena_path = argv[optind];
    const char *ena_path2 = (optind + 1 < argc) ? argv[optind + 1] : NULL;

    if (mode == MODE_COMPARE && !ena_path2) {
        fprintf(stderr, "Error: --compare requires two ENA files\n");
        return 1;
    }

    /* Load primary forest */
    bdd_forest_t forest = {0};
    if (!opts.quiet)
        fprintf(stderr, "Loading %s...\n", ena_path);

    if (bdd_forest_load(&forest, ena_path) != 0) {
        fprintf(stderr, "Error: failed to load %s\n", ena_path);
        return 1;
    }

    if (!opts.quiet) {
        fprintf(stderr, "  %d roots, %d variables\n",
                forest.num_roots, forest.num_vars);
    }

    /* Validate root index for single-root modes */
    if (target_root >= 0 && target_root >= forest.num_roots) {
        fprintf(stderr, "Error: root %d out of range (0..%d)\n",
                target_root, forest.num_roots - 1);
        bdd_forest_free(&forest);
        return 1;
    }

    /* Apply root range if specified */
    int r_start, r_end;
    if (range.start >= 0) {
        r_start = range.start;
        r_end = (range.end < forest.num_roots) ? range.end : forest.num_roots - 1;
    } else {
        r_start = 0;
        r_end = forest.num_roots - 1;
    }

    /* Dispatch by mode */
    int rc = 0;
    switch (mode) {
    case MODE_SUMMARY:
        if (analyze_all_roots(&forest) != 0) { rc = 1; break; }
        if (cluster_by_format_word(&forest) != 0) { rc = 1; break; }
        output_summary(&forest, &opts);
        output_clusters(&forest, &opts);
        break;

    case MODE_CHARACTERIZE:
        if (analyze_root(&forest, target_root) != 0) { rc = 1; break; }
        output_characterize(&forest, target_root, &opts);
        break;

    case MODE_ENUMERATE:
        if (analyze_root(&forest, target_root) != 0) { rc = 1; break; }
        output_enumerate(&forest, target_root, &opts);
        break;

    case MODE_ENUMERATE_ALL:
        for (int i = r_start; i <= r_end; i++) {
            if (analyze_root(&forest, i) != 0) { rc = 1; break; }
            output_enumerate(&forest, i, &opts);
        }
        break;

    case MODE_COUNT:
        output_counts(&forest, &opts);
        break;

    case MODE_COMPARE: {
        bdd_forest_t forest2 = {0};
        if (!opts.quiet)
            fprintf(stderr, "Loading %s...\n", ena_path2);
        if (bdd_forest_load(&forest2, ena_path2) != 0) {
            fprintf(stderr, "Error: failed to load %s\n", ena_path2);
            rc = 1;
            break;
        }
        if (!opts.quiet) {
            fprintf(stderr, "  %d roots, %d variables\n",
                    forest2.num_roots, forest2.num_vars);
        }
        rc = compare_forests(&forest, ena_path, &forest2, ena_path2);
        bdd_forest_free(&forest2);
        break;
    }
    }

    bdd_forest_free(&forest);
    return rc;
}
