//! DMA chain -> raw aie.mlir lowering (spike-verified forms).
//!
//! Emits the no-core shim loopback topology proven in
//! `build/experiments/dma-spike/shim_memtile_pass/aie.mlir` (and its strided /
//! scatter siblings): DDR(in) -> shim MM2S -> memtile linear passthrough ->
//! shim S2MM -> DDR(out). The fuzzed access pattern rides the runtime_sequence
//! `aiex.dma_configure_task` BDs; the memtile is a plain static passthrough.
use super::chain::{BdPattern, Direction, DmaChain, Engine};

/// Render a pattern's access list: `, [<size = a, stride = b>, ...]` or empty.
fn pattern_list(p: &BdPattern) -> String {
    if p.sizes.len() == 1 && p.strides == vec![1] && p.pad_before.is_empty() {
        return String::new(); // pure linear: no list
    }
    let dims: Vec<String> = p
        .sizes
        .iter()
        .zip(&p.strides)
        .map(|(s, st)| format!("<size = {s}, stride = {st}>"))
        .collect();
    format!(", [{}]", dims.join(", "))
}

pub fn lower_chain(chain: &DmaChain) -> String {
    match chain.engine {
        Engine::Shim => lower_shim(chain),
        Engine::Memtile => lower_memtile(chain),
    }
}

fn lower_shim(chain: &DmaChain) -> String {
    let elem = chain.dtype.mlir_elem();
    let in_words = chain.in_words();
    let out_words = chain.out_words();

    // Shared per-transfer region length L (all transfers' in_elems are equal by
    // construction; assert so a generator regression surfaces here, not in aiecc).
    let l = chain.transfers[0].in_elems;
    assert!(
        chain.transfers.iter().all(|t| t.in_elems == l),
        "shim lowering requires a shared per-transfer region length; got {:?}",
        chain.transfers.iter().map(|t| t.in_elems).collect::<Vec<_>>()
    );

    let mut m = String::new();
    m.push_str("module {\n");
    m.push_str("  aie.device(npu1_1col) {\n");
    m.push_str("    %tile_0_0 = aie.tile(0, 0)\n");
    m.push_str("    %tile_0_1 = aie.tile(0, 1)\n");
    m.push_str("    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)\n");
    m.push_str("    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)\n");

    // Memtile linear passthrough (no aie.core): S2MM into a buffer, MM2S back out.
    m.push_str("    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {\n");
    m.push_str(&format!(
        "      %buf = aie.buffer(%tile_0_1) {{sym_name = \"mt_buf\"}} : memref<{l}x{elem}>\n"
    ));
    m.push_str("      %prod = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = \"mt_prod\"}\n");
    m.push_str("      %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = \"mt_cons\"}\n");
    m.push_str("      %0 = aie.dma_start(S2MM, 0, ^s2mm, ^mm2s_entry)\n");
    m.push_str("    ^s2mm:\n");
    m.push_str("      aie.use_lock(%prod, AcquireGreaterEqual, 1)\n");
    m.push_str(&format!("      aie.dma_bd(%buf : memref<{l}x{elem}>, 0, {l})\n"));
    m.push_str("      aie.use_lock(%cons, Release, 1)\n");
    m.push_str("      aie.next_bd ^s2mm\n");
    m.push_str("    ^mm2s_entry:\n");
    m.push_str("      %1 = aie.dma_start(MM2S, 0, ^mm2s, ^end)\n");
    m.push_str("    ^mm2s:\n");
    m.push_str("      aie.use_lock(%cons, AcquireGreaterEqual, 1)\n");
    m.push_str(&format!("      aie.dma_bd(%buf : memref<{l}x{elem}>, 0, {l})\n"));
    m.push_str("      aie.use_lock(%prod, Release, 1)\n");
    m.push_str("      aie.next_bd ^mm2s\n");
    m.push_str("    ^end:\n");
    m.push_str("      aie.end\n");
    m.push_str("    }\n");

    // Runtime sequence: per transfer, a recv (S2MM -> %out) and a send (MM2S
    // from %in). The fuzzed pattern rides whichever engine matches the
    // transfer's direction: gather (Mm2s) -> send BD; scatter (S2mm) -> recv BD.
    m.push_str(&format!(
        "    aie.runtime_sequence(%in: memref<{in_words}x{elem}>, %out: memref<{out_words}x{elem}>) {{\n"
    ));

    for (k, t) in chain.transfers.iter().enumerate() {
        let (rpat, spat) = match t.dir {
            // gather: pattern on MM2S send-from-%in; recv linear.
            Direction::Mm2s => (String::new(), pattern_list(&t.pattern)),
            // scatter: pattern on S2MM recv-to-%out; send linear.
            Direction::S2mm => (pattern_list(&t.pattern), String::new()),
        };
        let in_off = t.in_off;
        let out_off = t.out_off;
        // Shim never pads, so RLEN/SLEN both equal the shared region L.
        let bd_id = k % 8; // MM2S bd_id cycles 0..=7

        // recv (S2MM, to %out)
        m.push_str(&format!("      %recv{k} = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {{\n"));
        m.push_str(&format!(
            "        aie.dma_bd(%out : memref<{out_words}x{elem}>, {out_off}, {l}{rpat}) {{bd_id = 8 : i32}}\n"
        ));
        m.push_str("        aie.end\n");
        m.push_str("      } {issue_token = true}\n");
        m.push_str(&format!("      aiex.dma_start_task(%recv{k})\n"));

        // send (MM2S, from %in)
        m.push_str(&format!("      %t{k} = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {{\n"));
        m.push_str(&format!(
            "        aie.dma_bd(%in : memref<{in_words}x{elem}>, {in_off}, {l}{spat}) {{bd_id = {bd_id} : i32}}\n"
        ));
        m.push_str("        aie.end\n");
        m.push_str("      } {issue_token = true}\n");
        m.push_str(&format!("      aiex.dma_start_task(%t{k})\n"));

        m.push_str(&format!("      aiex.dma_await_task(%t{k})\n"));
        m.push_str(&format!("      aiex.dma_await_task(%recv{k})\n"));
    }

    m.push_str("    }\n"); // runtime_sequence
    m.push_str("  }\n"); // device
    m.push_str("}\n"); // module
    m
}

/// Task 7 fills this in; for now return a minimal placeholder so the crate
/// compiles. Task 5 only exercises the shim path.
fn lower_memtile(_chain: &DmaChain) -> String {
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::dma::gen::generate;

    fn compile_one(key: &str, seed: u64) {
        let tools = match crate::fuzzer::core::toolchain::ToolPaths::discover() {
            Ok(t) => t,
            Err(_) => return, // no toolchain -> skip
        };
        let c = generate(seed, key);
        let dir = std::env::temp_dir().join(format!("dma_lower_{}_{}", seed, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("aie.mlir"), lower_chain(&c)).unwrap();
        let r = crate::fuzzer::core::toolchain::compile_dma_mlir(&tools, &dir);
        assert!(r.is_ok(), "{key} seed{seed}: {r:?}\n{}", lower_chain(&c));
        assert!(dir.join("aie.xclbin").exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[ignore = "requires toolchain; run with --ignored"]
    fn shim_linear_compiles() {
        compile_one("linear/shim/mm2s/I32", 1);
    }

    #[test]
    #[ignore = "requires toolchain; run with --ignored"]
    fn shim_strided2d_compiles() {
        compile_one("strided2d/shim/mm2s/I32", 2);
    }

    #[test]
    fn shim_mlir_has_device_and_runtime_sequence() {
        let c = generate(1, "linear/shim/mm2s/I32");
        let m = lower_chain(&c);
        assert!(m.contains("aie.device(npu1_1col)"));
        assert!(m.contains("aie.runtime_sequence"));
        assert!(m.contains("aie.memtile_dma(%tile_0_1)"));
        assert!(!m.contains("aie.core"), "shim path must have no core ELF");
    }
}
