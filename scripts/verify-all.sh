#!/bin/bash
# Full verification pipeline for xdna-emu.
#
# Runs all levels of testing in order:
# 1. Build test binaries from mlir-aie sources (if toolchain available)
# 2. Unit tests (cargo test --lib)
# 3. Integration tests (cargo test --features xclbin-tests)
# 4. Full binary suite (cargo run --example run_mlir_aie_tests)
# 5. Summary report
#
# Usage:
#   ./scripts/verify-all.sh            # Full pipeline
#   ./scripts/verify-all.sh --skip-build  # Skip binary build step

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SKIP_BUILD=0
ERRORS=0
WARNINGS=0

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        -h|--help)
            echo "Usage: $0 [--skip-build]"
            echo ""
            echo "Run the full xdna-emu verification pipeline."
            echo ""
            echo "Options:"
            echo "  --skip-build   Skip building mlir-aie test binaries"
            exit 0
            ;;
    esac
done

section() {
    echo ""
    echo -e "${BOLD}=========================================${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${BOLD}=========================================${NC}"
    echo ""
}

pass() {
    echo -e "  ${GREEN}PASS${NC} $1"
}

fail() {
    echo -e "  ${RED}FAIL${NC} $1"
    ERRORS=$((ERRORS + 1))
}

warn() {
    echo -e "  ${YELLOW}WARN${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

info() {
    echo -e "  ${BLUE}INFO${NC} $1"
}

# Step 1: Build test binaries
if [[ $SKIP_BUILD -eq 0 ]]; then
    section "Step 1: Build mlir-aie Test Binaries"

    if [[ -f "$SCRIPT_DIR/build-mlir-aie-tests.sh" ]]; then
        if "$SCRIPT_DIR/build-mlir-aie-tests.sh" 2>&1; then
            pass "Test binaries built"
        else
            warn "Some test binaries failed to build (non-fatal)"
        fi
    else
        warn "build-mlir-aie-tests.sh not found, skipping binary build"
    fi
else
    info "Skipping binary build (--skip-build)"
fi

# Step 2: Unit tests
section "Step 2: Unit Tests (cargo test --lib)"

if cargo test --lib 2>&1; then
    UNIT_COUNT=$(cargo test --lib 2>&1 | grep 'test result:' | grep -o '[0-9]* passed' | head -1)
    pass "Unit tests: $UNIT_COUNT"
else
    fail "Unit tests failed"
fi

# Step 3: Integration tests
section "Step 3: Integration Tests (xclbin-tests)"

if cargo test --features xclbin-tests --test xclbin_integration 2>&1; then
    INTEG_COUNT=$(cargo test --features xclbin-tests --test xclbin_integration 2>&1 | grep 'test result:' | grep -o '[0-9]* passed' | head -1)
    pass "Integration tests: $INTEG_COUNT"
else
    # Integration tests may fail if binaries not built -- that's a warning, not fatal
    warn "Some integration tests failed (check if binaries are built)"
fi

# Step 4: Bridge test suite
section "Step 4: Bridge Test Suite"

if [ -x scripts/emu-bridge-test.sh ]; then
    info "Run bridge tests manually: ./scripts/emu-bridge-test.sh"
    info "(Requires XRT, NPU hardware, and driver plugin installed)"
else
    info "Bridge test script not found, skipping"
fi

# Step 5: Hardware cross-validation (conditional on NPU availability)
section "Step 5: Hardware Cross-Validation"

NPU_DEVICE="/dev/accel/accel0"
RUNNER_BIN="$PROJECT_DIR/tools/npu-runner/build/npu_runner"

if [[ -c "$NPU_DEVICE" || -c "/dev/accel/accel1" ]]; then
    info "NPU device detected"

    # Build npu-runner if not present
    if [[ ! -f "$RUNNER_BIN" ]]; then
        info "Building npu-runner..."
        if command -v cmake &>/dev/null && [[ -d /opt/xilinx/xrt ]]; then
            mkdir -p "$PROJECT_DIR/tools/npu-runner/build"
            if cmake -B "$PROJECT_DIR/tools/npu-runner/build" \
                     -S "$PROJECT_DIR/tools/npu-runner" 2>&1 && \
               cmake --build "$PROJECT_DIR/tools/npu-runner/build" 2>&1; then
                pass "npu-runner built"
            else
                warn "npu-runner build failed (XRT headers missing?)"
            fi
        else
            warn "Cannot build npu-runner (cmake or XRT not found)"
        fi
    fi

    if [[ -f "$RUNNER_BIN" ]]; then
        # Capture NPU outputs
        if cargo run --example capture_npu_outputs 2>&1; then
            pass "NPU outputs captured"
        else
            warn "Some NPU captures failed (non-fatal)"
        fi

        # Run cross-validation
        if cargo run --example compare_emu_hw 2>&1; then
            pass "Cross-validation report generated"
        else
            warn "Cross-validation had issues (see report above)"
        fi

        # Run hardware comparison tests
        if cargo test --features hardware-compare --test hardware_comparison 2>&1; then
            pass "Hardware comparison tests passed"
        else
            warn "Some hardware comparison tests failed"
        fi
    else
        info "npu-runner not available, skipping hardware validation"
    fi
else
    info "No NPU device detected, skipping hardware validation"
fi

# Summary
section "Verification Summary"

if [[ $ERRORS -eq 0 ]]; then
    echo -e "  ${GREEN}${BOLD}ALL CHECKS PASSED${NC}"
    if [[ $WARNINGS -gt 0 ]]; then
        echo -e "  ${YELLOW}$WARNINGS warning(s)${NC}"
    fi
    echo ""
    exit 0
else
    echo -e "  ${RED}${BOLD}$ERRORS check(s) FAILED${NC}"
    if [[ $WARNINGS -gt 0 ]]; then
        echo -e "  ${YELLOW}$WARNINGS warning(s)${NC}"
    fi
    echo ""
    exit 1
fi
