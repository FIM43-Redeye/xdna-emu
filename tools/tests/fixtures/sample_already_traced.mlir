//
// Minimal already-traced AIE design for mlir-trace-inject idempotency tests.
//
// This is a superset of sample_untraced.mlir: the same tiles, buffer, core, and
// runtime_sequence, but with one aie.trace op inserted in the device body.
//
// The idempotency guard in mlir-trace-inject.py must detect this trace op and
// refuse with exit code 2 rather than double-injecting.
//
// Syntax verified by round-tripping through Module.parse() -- the printer
// normalises spacing (id = 1, broadcast = 15, etc.) which is reflected here.
//
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.trace @trace_existing(%tile_0_2) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id = 1 type = core
      aie.trace.event <"INSTR_VECTOR">
      aie.trace.start broadcast = 15
      aie.trace.stop broadcast = 14
    }
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<16xi32>
    aie.core(%tile_0_2) {
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
    }
  }
}
