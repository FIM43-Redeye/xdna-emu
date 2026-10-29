//
// Multi-device fixture: @base has tiles only (like ctrl_packet_reconfig
// overlays), @main has tiles + runtime_sequence. Trace-inject must put the
// trace decls in @main (where the runtime_sequence lives), NOT @base -- if
// it injects into @base, aiecc compile on --device-name=base will fail with
// "aie.trace ops found but no runtime_sequence defined".
//
module {
  aie.device(npu1_1col) @base {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
  }
  aie.device(npu1_1col) @main {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<16xi32>
    aie.core(%tile_0_2) {
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
    }
  }
}
