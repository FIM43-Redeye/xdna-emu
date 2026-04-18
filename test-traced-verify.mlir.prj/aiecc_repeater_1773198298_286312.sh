#!/bin/bash
set -e
# Repeater script for: routing
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
'aie.masterset' op targets same destination South: 2 as another connect or masterset operation
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='test-traced-verify.mlir.prj/aiecc_failure_1773198298_286312.mlir'
PASS_PIPELINE='builtin.module(aie.device(aie-create-pathfinder-flows))'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
