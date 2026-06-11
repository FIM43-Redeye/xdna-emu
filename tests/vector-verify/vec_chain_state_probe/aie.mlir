//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
// Chained-state isolation probe (GENERATED). One 1536-element bf16 input
// [a|c|b], one 512-element output; out = add(add(a,c), b).
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1_1col) {
    func.func private @chain_probe(memref<1536xbf16>, memref<512xbf16>)
        attributes {link_with = "kernel.o"}

    %shim = aie.tile(0, 0)
    %core = aie.tile(0, 2)

    aie.objectfifo @inA(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<1536xbf16>>
    aie.objectfifo @outO(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<512xbf16>>

    %core_0_2 = aie.core(%core) {
      %sa = aie.objectfifo.acquire @inA(Consume, 1) : !aie.objectfifosubview<memref<1536xbf16>>
      %a = aie.objectfifo.subview.access %sa[0] : !aie.objectfifosubview<memref<1536xbf16>> -> memref<1536xbf16>
      %so = aie.objectfifo.acquire @outO(Produce, 1) : !aie.objectfifosubview<memref<512xbf16>>
      %o = aie.objectfifo.subview.access %so[0] : !aie.objectfifosubview<memref<512xbf16>> -> memref<512xbf16>

      func.call @chain_probe(%a, %o) : (memref<1536xbf16>, memref<512xbf16>) -> ()

      aie.objectfifo.release @inA(Consume, 1)
      aie.objectfifo.release @outO(Produce, 1)
      aie.end
    }

    aie.runtime_sequence @sequence(%a: memref<1536xbf16>, %o: memref<512xbf16>) {
      aiex.npu.dma_memcpy_nd(%a[0, 0, 0, 0][1, 1, 1, 1536][0, 0, 0, 1]) {id = 0 : i64, metadata = @inA} : memref<1536xbf16>
      aiex.npu.dma_memcpy_nd(%o[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1]) {id = 1 : i64, metadata = @outO} : memref<512xbf16>
      aiex.npu.dma_wait {symbol = @outO}
    }
  }
}
