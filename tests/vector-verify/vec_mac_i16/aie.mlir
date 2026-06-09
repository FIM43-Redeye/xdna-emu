//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Half-B vec_mac_i16 capture design (GENERATED). Single compute tile (0,2) reads
// row-major A (512 elems) and B (1024 elems) from DDR via two shim DMAs, runs a
// batch of native mmul tiles, writes C (2048 elems) back. Direct shim<->core
// objectfifos (no memtile relay).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @mac_i16(memref<512xi16>, memref<1024xi16>, memref<2048xi32>)
        attributes {link_with = "mac.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<512xi16>>
    aie.objectfifo @inB(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<1024xi16>>
    aie.objectfifo @outC(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<2048xi32>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<512xi16>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<512xi16>> -> memref<512xi16>
      %sb = aie.objectfifo.acquire @inB(Consume, 1) : !aie.objectfifosubview<memref<1024xi16>>
      %b = aie.objectfifo.subview.access %sb[0] : !aie.objectfifosubview<memref<1024xi16>> -> memref<1024xi16>
      %so = aie.objectfifo.acquire @outC(Produce, 1) : !aie.objectfifosubview<memref<2048xi32>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<2048xi32>> -> memref<2048xi32>

      func.call @mac_i16(%a, %b, %o) : (memref<512xi16>, memref<1024xi16>, memref<2048xi32>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @inB(Consume, 1)
      aie.objectfifo.release @outC(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<512xi16>, %b: memref<1024xi16>, %c: memref<2048xi32>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<512xi16>
      aiex.npu.dma_memcpy_nd(%b[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, metadata = @inB} : memref<1024xi16>
      aiex.npu.dma_memcpy_nd(%c[0, 0, 0, 0][1, 1, 1, 2048][0, 0, 0, 1]) {id = 2 : i64, metadata = @outC} : memref<2048xi32>
      aiex.npu.dma_wait {symbol = @outC}
    }
  }
}
