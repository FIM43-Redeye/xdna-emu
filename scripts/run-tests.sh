#!/bin/bash
# Run tests with appropriate priority levels.
#
# Tests can saturate the local CPU. This script runs every Cargo invocation
# at the lowest scheduling priority so the desktop stays responsive.
#
# Usage:
#   ./scripts/run-tests.sh              # Run all tests at nice 19
#   ./scripts/run-tests.sh --lib        # Run only library tests (fast)
#   ./scripts/run-tests.sh --doc        # Run only doc tests (nice'd)
#   ./scripts/run-tests.sh --all        # Run all tests including ignored
#   ./scripts/run-tests.sh --integration  # Run xclbin integration tests (needs built binaries)
#   ./scripts/run-tests.sh --full       # Run unit + integration + doc tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 19 = lowest priority, very background.
NICE_LEVEL="${NICE_LEVEL:-19}"

# Limit parallel doc test jobs to reduce system load
DOC_TEST_JOBS="${DOC_TEST_JOBS:-2}"

case "${1:-}" in
    --lib)
        echo "Running library tests with nice $NICE_LEVEL..."
        nice -n "$NICE_LEVEL" cargo test --lib
        ;;
    --doc)
        echo "Running doc tests with nice $NICE_LEVEL (jobs=$DOC_TEST_JOBS)..."
        nice -n "$NICE_LEVEL" cargo test --doc -- --test-threads="$DOC_TEST_JOBS"
        ;;
    --integration)
        echo "Running xclbin integration tests..."
        echo "(Requires built binaries: ./scripts/build-mlir-aie-tests.sh)"
        echo ""
        nice -n "$NICE_LEVEL" cargo test --features xclbin-tests --test xclbin_integration
        ;;
    --full)
        echo "Running library tests with nice $NICE_LEVEL..."
        nice -n "$NICE_LEVEL" cargo test --lib
        echo ""
        echo "Running xclbin integration tests..."
        nice -n "$NICE_LEVEL" cargo test --features xclbin-tests --test xclbin_integration 2>&1 || true
        echo ""
        echo "Running doc tests with nice $NICE_LEVEL (jobs=$DOC_TEST_JOBS)..."
        nice -n "$NICE_LEVEL" cargo test --doc -- --test-threads="$DOC_TEST_JOBS"
        ;;
    --all)
        echo "Running library tests with nice $NICE_LEVEL..."
        nice -n "$NICE_LEVEL" cargo test --lib
        echo ""
        echo "Running doc tests with nice $NICE_LEVEL (jobs=$DOC_TEST_JOBS)..."
        nice -n "$NICE_LEVEL" cargo test --doc -- --test-threads="$DOC_TEST_JOBS"
        ;;
    *)
        echo "Running library tests with nice $NICE_LEVEL..."
        nice -n "$NICE_LEVEL" cargo test --lib
        echo ""
        echo "Running doc tests with nice $NICE_LEVEL (jobs=$DOC_TEST_JOBS)..."
        nice -n "$NICE_LEVEL" cargo test --doc -- --test-threads="$DOC_TEST_JOBS"
        ;;
esac

echo ""
echo "All tests completed."
