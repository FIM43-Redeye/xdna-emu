#!/bin/bash
# bdd-validate.sh -- Exhaustive decoder validation via BDD enumeration
#
# Generates one concrete 128-bit pattern per BDD root from the aietools ENA
# file and pipes them through our TableGen decoder. The output is a report
# showing which encodings we can/cannot decode.
#
# Usage:
#   scripts/bdd-validate.sh                     # Default: me_das.ena (13K roots)
#   scripts/bdd-validate.sh me.ena              # Compiler BDD (3K roots)
#   scripts/bdd-validate.sh me_chess.ena        # Chess BDD
#
# Output goes to reports/bdd-validate-<ena>-<timestamp>.txt
# Summary is printed to stderr (visible in terminal).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BDD_ENUM="$PROJECT_DIR/tools/bdd_enum/build/bdd_enum"
VALIDATOR="$PROJECT_DIR/target/release/examples/bdd_validate"
ISG_DIR="${ISG_DIR:-/home/triple/npu-work/amd-unified-software/aietools/data/aie_ml/lib/isg}"

# Which ENA file to use
ENA_NAME="${1:-me_das.ena}"
ENA_FILE="$ISG_DIR/$ENA_NAME"

if [ ! -f "$ENA_FILE" ]; then
    echo "Error: ENA file not found: $ENA_FILE" >&2
    echo "Available files:" >&2
    ls "$ISG_DIR"/*.ena 2>/dev/null | while read f; do echo "  $(basename "$f")" >&2; done
    exit 1
fi

# Output directory and report file
REPORT_DIR="$PROJECT_DIR/reports"
mkdir -p "$REPORT_DIR"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
BASENAME="$(basename "$ENA_NAME" .ena)"
REPORT="$REPORT_DIR/bdd-validate-${BASENAME}-${TIMESTAMP}.txt"

echo "=== BDD Decoder Validation ===" >&2
echo "ENA file:  $ENA_FILE" >&2
echo "Report:    $REPORT" >&2
echo "" >&2

# Step 1: Build bdd_enum if needed
if [ ! -x "$BDD_ENUM" ]; then
    echo "Building bdd_enum..." >&2
    nice -n 19 make -C "$PROJECT_DIR/tools/bdd_enum" 2>&1 | tail -3 >&2
fi

# Step 2: Build validator if needed
if [ ! -x "$VALIDATOR" ]; then
    echo "Building bdd_validate..." >&2
    (cd "$PROJECT_DIR" && nice -n 19 cargo build --release --example bdd_validate 2>&1 | tail -3 >&2)
fi

# Step 3: Run the pipeline
# bdd_enum outputs one 16-byte pattern per root in raw format.
# The validator reads them sequentially, root 0 = first record.
echo "Running validation pipeline..." >&2
echo "" >&2

nice -n 19 "$BDD_ENUM" --enumerate-all --format raw --expand --max 1 \
    "$ENA_FILE" 2>/dev/null | \
    "$VALIDATOR" > "$REPORT" 2>&1

# The validator writes the per-root table to stdout (captured in $REPORT)
# and the summary to stderr (also captured since we used 2>&1).
# Re-print summary to terminal.
echo "" >&2
echo "Report written to: $REPORT" >&2
echo "Lines: $(wc -l < "$REPORT")" >&2

# Extract and show just the summary section
echo "" >&2
sed -n '/^=====/,$ p' "$REPORT" >&2
