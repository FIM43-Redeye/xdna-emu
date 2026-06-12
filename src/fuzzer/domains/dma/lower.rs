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
        // A memtile chain (3b `chain` feature) is N>=2 transfers carried as a
        // multi-BD next_bd chain with double-buffer locks; all other memtile
        // cases are single-transfer (3a).
        Engine::Memtile if chain.transfers.len() >= 2 => lower_memtile_chain(chain),
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

    // Packet routing is device-level. A packet chain is N=1 (gen enforces purity):
    // the single MM2S transfer carries pattern.packet = Some((id, ty)) and routes
    // via aie.packet_flow on the shim->memtile DMA:0 port instead of the circuit
    // aie.flow. Detect it here.
    let packet = chain.transfers.iter().find_map(|t| t.pattern.packet.map(|(id, ty)| (id, ty)));

    let mut m = String::new();
    m.push_str("module {\n");
    m.push_str("  aie.device(npu1_1col) {\n");
    m.push_str("    %tile_0_0 = aie.tile(0, 0)\n");
    m.push_str("    %tile_0_1 = aie.tile(0, 1)\n");
    if let Some((id, _ty)) = packet {
        // Packet rides the shim->memtile DMA:0 port: replace that circuit flow with
        // a packet_flow (both on one static path is contradictory). The reverse
        // memtile->shim leg stays circuit.
        m.push_str("    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)\n");
        m.push_str(&format!("    aie.packet_flow({id}) {{\n"));
        m.push_str("      aie.packet_source<%tile_0_0, DMA : 0>\n");
        m.push_str("      aie.packet_dest<%tile_0_1, DMA : 0>\n");
        m.push_str("    }\n");
    } else {
        m.push_str("    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)\n");
        m.push_str("    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)\n");
    }

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

        // send (MM2S, from %in). Packet rides the configure_task OP SIGNATURE
        // (4th arg), not the inner dma_bd (spike-verified on the shim).
        let pkt_arg = match t.pattern.packet {
            Some((id, ty)) => format!(", <pkt_type = {ty}, pkt_id = {id}>"),
            None => String::new(),
        };
        m.push_str(&format!("      %t{k} = aiex.dma_configure_task(%tile_0_0, MM2S, 0{pkt_arg}) {{\n"));
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

/// Render `, [<const_pad_before = b, const_pad_after = a>, ...]` or empty.
fn pad_list(p: &BdPattern) -> String {
    if p.pad_before.is_empty() {
        return String::new();
    }
    let dims: Vec<String> = p
        .pad_before
        .iter()
        .zip(&p.pad_after)
        .map(|(b, a)| format!("<const_pad_before = {b}, const_pad_after = {a}>"))
        .collect();
    format!(", [{}]", dims.join(", "))
}

/// Memtile topology: the fuzzed access pattern rides a raw `aie.memtile_dma`
/// static BD; the shim runs linear transfers in the runtime_sequence.
///
/// Single-transfer by construction (gen forces N=1 for memtile -- see the guard
/// in `gen::generate`). The transfer's direction decides which memtile BD (S2MM
/// or MM2S) carries the fuzzed n-D pattern; the other memtile BD is linear.
///
/// Three emission forms compose on the memtile MM2S BD:
/// - n-D access pattern (strided2d/3d/4d/transpose/overlap): rides the dma_bd
///   `[<size, stride>...]` list, verified vs `strided4d_memtile/aie.mlir`.
/// - packet (MM2S-only by gating): a top-level `aie.packet_flow` replaces the
///   memtile->shim circuit flow, and the BD carries a `{packet = ...}` attribute,
///   verified vs `packet_mm2s/aie.mlir`.
/// - padding (MM2S-only by gating): the BD carries a `[<const_pad_before, ...>]`
///   list and the padded `out_elems` length; the S2MM recv stays linear at the
///   un-padded `in_elems`, verified vs `memtile_pad2d/aie.mlir`.
fn lower_memtile(chain: &DmaChain) -> String {
    assert_eq!(
        chain.transfers.len(),
        1,
        "memtile lowering is single-transfer (gen forces N=1); got {}",
        chain.transfers.len()
    );
    let t = &chain.transfers[0];
    let elem = chain.dtype.mlir_elem();
    let in_words = chain.in_words();
    let out_words = chain.out_words();

    // Direction routes the fuzzed pattern onto the matching memtile BD; the
    // opposite memtile BD stays linear (empty access list). Padding rides only
    // the MM2S BD (gating guarantees pad => MM2S), where it grows the streamed
    // length to the padded total `out_elems`; the S2MM recv always lands the
    // un-padded `in_elems` of data into the buffer.
    let pat = pattern_list(&t.pattern);
    let pad = pad_list(&t.pattern);
    let (s2mm_bd, mm2s_bd) = match t.dir {
        // Fuzzed pattern (+ optional padding) on MM2S; S2MM linear, un-padded.
        Direction::Mm2s => (format!("0, {}", t.in_elems), format!("0, {}{pat}{pad}", t.out_elems)),
        // Fuzzed pattern on S2MM; MM2S linear. Padding never occurs here.
        Direction::S2mm => (format!("0, {}{pat}", t.in_elems), format!("0, {}", t.out_elems)),
    };

    // Packet (memtile MM2S-only by gating) rides the dma_bd ATTRIBUTE, not the
    // op signature; it also turns the memtile->shim circuit flow into a
    // packet_flow (both on one static path is contradictory).
    let packet = t.pattern.packet;
    let mm2s_attr = match packet {
        Some((id, ty)) => format!(" {{packet = #aie.packet_info<pkt_type = {ty}, pkt_id = {id}>}}"),
        None => String::new(),
    };

    let mut m = String::new();
    m.push_str("module {\n");
    m.push_str("  aie.device(npu1_1col) {\n");
    m.push_str("    %tile_0_0 = aie.tile(0, 0)\n");
    m.push_str("    %tile_0_1 = aie.tile(0, 1)\n");
    m.push_str("    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)\n");
    if let Some((id, _ty)) = packet {
        // Packet rides the memtile->shim DMA:0 port: replace that circuit flow
        // with a packet_flow. The forward shim->memtile leg stays circuit.
        m.push_str(&format!("    aie.packet_flow({id}) {{\n"));
        m.push_str("      aie.packet_source<%tile_0_1, DMA : 0>\n");
        m.push_str("      aie.packet_dest<%tile_0_0, DMA : 0>\n");
        m.push_str("    }\n");
    } else {
        m.push_str("    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)\n");
    }

    // Memtile static block: S2MM recv into a buffer, MM2S read back out. One of
    // the two BDs carries the fuzzed pattern (per direction above).
    m.push_str("    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {\n");
    m.push_str(&format!(
        "      %buf = aie.buffer(%tile_0_1) {{sym_name = \"mt_buf\"}} : memref<{out_words}x{elem}>\n"
    ));
    m.push_str("      %prod = aie.lock(%tile_0_1, 0) {init = 1 : i32, sym_name = \"mt_prod\"}\n");
    m.push_str("      %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = \"mt_cons\"}\n");
    m.push_str("      %0 = aie.dma_start(S2MM, 0, ^s2mm, ^mm2s_entry)\n");
    m.push_str("    ^s2mm:\n");
    m.push_str("      aie.use_lock(%prod, AcquireGreaterEqual, 1)\n");
    m.push_str(&format!("      aie.dma_bd(%buf : memref<{out_words}x{elem}>, {s2mm_bd})\n"));
    m.push_str("      aie.use_lock(%cons, Release, 1)\n");
    m.push_str("      aie.next_bd ^s2mm\n");
    m.push_str("    ^mm2s_entry:\n");
    m.push_str("      %1 = aie.dma_start(MM2S, 0, ^mm2s, ^end)\n");
    m.push_str("    ^mm2s:\n");
    m.push_str("      aie.use_lock(%cons, AcquireGreaterEqual, 1)\n");
    m.push_str(&format!("      aie.dma_bd(%buf : memref<{out_words}x{elem}>, {mm2s_bd}){mm2s_attr}\n"));
    m.push_str("      aie.use_lock(%prod, Release, 1)\n");
    m.push_str("      aie.next_bd ^mm2s\n");
    m.push_str("    ^end:\n");
    m.push_str("      aie.end\n");
    m.push_str("    }\n");

    // Runtime sequence: shim runs plain linear transfers (DDR in -> memtile,
    // memtile -> DDR out). The reshuffle lives entirely on the memtile BD.
    m.push_str(&format!(
        "    aie.runtime_sequence(%in: memref<{in_words}x{elem}>, %out: memref<{out_words}x{elem}>) {{\n"
    ));
    m.push_str("      %recv = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {\n");
    m.push_str(&format!(
        "        aie.dma_bd(%out : memref<{out_words}x{elem}>, 0, {out_words}) {{bd_id = 8 : i32}}\n"
    ));
    m.push_str("        aie.end\n");
    m.push_str("      } {issue_token = true}\n");
    m.push_str("      aiex.dma_start_task(%recv)\n");
    m.push_str("      %t0 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {\n");
    m.push_str(&format!(
        "        aie.dma_bd(%in : memref<{in_words}x{elem}>, 0, {in_words}) {{bd_id = 0 : i32}}\n"
    ));
    m.push_str("        aie.end\n");
    m.push_str("      } {issue_token = true}\n");
    m.push_str("      aiex.dma_start_task(%t0)\n");
    m.push_str("      aiex.dma_await_task(%t0)\n");
    m.push_str("      aiex.dma_await_task(%recv)\n");
    m.push_str("    }\n"); // runtime_sequence
    m.push_str("  }\n"); // device
    m.push_str("}\n"); // module
    m
}

/// Memtile multi-BD `next_bd` chain (3b `chain` feature). N>=2 transfers ride one
/// memtile_dma block as a chain of distinct BDs on each channel, with a
/// double-buffer lock pair (producer init=N) -- the proven form from the spike's
/// `chain_memtile_clean`. The patterned channel is the chain's direction; the
/// opposite channel runs N linear BDs. The shim sends N linear regions in and
/// drains N*region linearly out; the reshuffle lives entirely on the memtile BDs.
fn lower_memtile_chain(chain: &DmaChain) -> String {
    let elem = chain.dtype.mlir_elem();
    let in_words = chain.in_words();
    let out_words = chain.out_words();
    let n = chain.transfers.len();
    // All chain transfers share one region length (no padding in a chain).
    let region = chain.transfers[0].in_elems;
    assert!(
        chain.transfers.iter().all(|t| t.in_elems == region && t.out_elems == region),
        "memtile chain requires equal un-padded regions; got {:?}",
        chain.transfers.iter().map(|t| (t.in_elems, t.out_elems)).collect::<Vec<_>>()
    );

    let mut m = String::new();
    m.push_str("module {\n");
    m.push_str("  aie.device(npu1_1col) {\n");
    m.push_str("    %tile_0_0 = aie.tile(0, 0)\n");
    m.push_str("    %tile_0_1 = aie.tile(0, 1)\n");
    m.push_str("    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)\n");
    m.push_str("    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)\n");

    m.push_str("    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {\n");
    for k in 0..n {
        m.push_str(&format!(
            "      %buf{k} = aie.buffer(%tile_0_1) {{sym_name = \"mt_buf{k}\"}} : memref<{region}x{elem}>\n"
        ));
    }
    // Double-buffer locks: producer starts at N (all buffers free), consumer at 0.
    m.push_str(&format!(
        "      %prod = aie.lock(%tile_0_1, 0) {{init = {n} : i32, sym_name = \"mt_prod\"}}\n"
    ));
    m.push_str("      %cons = aie.lock(%tile_0_1, 1) {init = 0 : i32, sym_name = \"mt_cons\"}\n");

    // S2MM chain: ^s2mm0 -> ^s2mm1 -> ... -> ^s2mm0. Each fills its buffer,
    // acquiring prod and releasing cons.
    m.push_str("      %0 = aie.dma_start(S2MM, 0, ^s2mm0, ^mm2s_entry)\n");
    for k in 0..n {
        let t = &chain.transfers[k];
        let pat = if t.dir == Direction::S2mm {
            pattern_list(&t.pattern)
        } else {
            String::new()
        };
        let next = (k + 1) % n;
        m.push_str(&format!("    ^s2mm{k}:\n"));
        m.push_str("      aie.use_lock(%prod, AcquireGreaterEqual, 1)\n");
        m.push_str(&format!("      aie.dma_bd(%buf{k} : memref<{region}x{elem}>, 0, {region}{pat})\n"));
        m.push_str("      aie.use_lock(%cons, Release, 1)\n");
        m.push_str(&format!("      aie.next_bd ^s2mm{next}\n"));
    }

    // MM2S chain: ^mm2s0 -> ^mm2s1 -> ... -> ^mm2s0. Each drains its buffer,
    // acquiring cons and releasing prod.
    m.push_str("    ^mm2s_entry:\n");
    m.push_str("      %1 = aie.dma_start(MM2S, 0, ^mm2s0, ^end)\n");
    for k in 0..n {
        let t = &chain.transfers[k];
        let pat = if t.dir == Direction::Mm2s {
            pattern_list(&t.pattern)
        } else {
            String::new()
        };
        let next = (k + 1) % n;
        m.push_str(&format!("    ^mm2s{k}:\n"));
        m.push_str("      aie.use_lock(%cons, AcquireGreaterEqual, 1)\n");
        m.push_str(&format!("      aie.dma_bd(%buf{k} : memref<{region}x{elem}>, 0, {region}{pat})\n"));
        m.push_str("      aie.use_lock(%prod, Release, 1)\n");
        m.push_str(&format!("      aie.next_bd ^mm2s{next}\n"));
    }
    m.push_str("    ^end:\n");
    m.push_str("      aie.end\n");
    m.push_str("    }\n");

    // Runtime sequence: one linear recv of all N regions, N linear sends (one per
    // region). The reshuffle is entirely on the memtile BDs above.
    m.push_str(&format!(
        "    aie.runtime_sequence(%in: memref<{in_words}x{elem}>, %out: memref<{out_words}x{elem}>) {{\n"
    ));
    m.push_str("      %recv = aiex.dma_configure_task(%tile_0_0, S2MM, 0) {\n");
    m.push_str(&format!(
        "        aie.dma_bd(%out : memref<{out_words}x{elem}>, 0, {out_words}) {{bd_id = 8 : i32}}\n"
    ));
    m.push_str("        aie.end\n");
    m.push_str("      } {issue_token = true}\n");
    m.push_str("      aiex.dma_start_task(%recv)\n");
    for k in 0..n {
        let off = k * region;
        let bd_id = k % 8;
        let token = if k == n - 1 { " {issue_token = true}" } else { "" };
        m.push_str(&format!("      %t{k} = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {{\n"));
        m.push_str(&format!(
            "        aie.dma_bd(%in : memref<{in_words}x{elem}>, {off}, {region}) {{bd_id = {bd_id} : i32}}\n"
        ));
        m.push_str("        aie.end\n");
        m.push_str(&format!("      }}{token}\n"));
        m.push_str(&format!("      aiex.dma_start_task(%t{k})\n"));
    }
    m.push_str(&format!("      aiex.dma_await_task(%t{})\n", n - 1));
    m.push_str("      aiex.dma_await_task(%recv)\n");
    m.push_str("    }\n"); // runtime_sequence
    m.push_str("  }\n"); // device
    m.push_str("}\n"); // module
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuzzer::domains::dma::gen::generate;

    #[test]
    fn memtile_chain_emits_multibd_double_buffer() {
        let c = generate(3, "chain/memtile/mm2s/I32");
        let n = c.transfers.len();
        assert!(n >= 2, "chain must be multi-transfer, got {n}");
        let m = lower_chain(&c);
        assert!(m.contains(&format!("init = {n} : i32")), "producer lock init must equal N:\n{m}");
        assert!(m.contains("^s2mm1:") && m.contains("^mm2s1:"), "needs >=2 chained BDs per channel:\n{m}");
        assert!(
            m.contains("aie.next_bd ^s2mm0") && m.contains("aie.next_bd ^mm2s0"),
            "chain must loop back to BD0:\n{m}"
        );
        assert!(m.contains("%buf0") && m.contains("%buf1"), "one buffer per chained region");
    }

    #[test]
    #[ignore = "requires toolchain; run with --ignored"]
    fn memtile_chain_compiles() {
        compile_one("chain/memtile/mm2s/I16", 3);
    }

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
    #[ignore = "requires toolchain; run with --ignored"]
    fn shim_strided3d_compiles() {
        compile_one("strided3d/shim/mm2s/I32", 3);
    }

    #[test]
    #[ignore]
    fn shim_transpose_compiles() {
        compile_one("transpose/shim/mm2s/I16", 4);
    }

    #[test]
    #[ignore]
    fn shim_overlap_compiles() {
        compile_one("overlap/shim/s2mm/I8", 5);
    }

    #[test]
    #[ignore]
    fn shim_packet_compiles() {
        compile_one("packet/shim/mm2s/I32", 6);
    }

    #[test]
    #[ignore]
    fn shim_scatter_compiles() {
        compile_one("strided2d/shim/s2mm/I32", 7);
    }

    #[test]
    fn shim_packet_on_op_signature() {
        let c = generate(5, "packet/shim/mm2s/I32");
        let m = lower_chain(&c);
        assert!(
            m.contains("aiex.dma_configure_task(%tile_0_0, MM2S, 0, <pkt_type ="),
            "shim packet on op sig:\n{m}"
        );
        assert!(m.contains("aie.packet_flow("), "needs a packet_flow decl");
    }

    #[test]
    fn memtile_mlir_uses_static_block() {
        let c = generate(10, "strided2d/memtile/mm2s/I32");
        let m = lower_chain(&c);
        assert!(m.contains("aie.memtile_dma(%tile_0_1)"));
        assert!(m.contains("aie.dma_start(MM2S"));
        assert!(m.contains("aie.next_bd"));
        assert!(!m.contains("aie.core"));
    }

    #[test]
    #[ignore = "requires toolchain; run with --ignored"]
    fn memtile_linear_compiles() {
        compile_one("linear/memtile/mm2s/I32", 20);
    }
    #[test]
    #[ignore]
    fn memtile_strided_compiles() {
        compile_one("strided2d/memtile/mm2s/I32", 21);
    }
    #[test]
    #[ignore]
    fn memtile_strided_s2mm_compiles() {
        compile_one("strided2d/memtile/s2mm/I16", 22);
    }

    #[test]
    fn memtile_packet_on_bd_attribute() {
        let c = generate(30, "packet/memtile/mm2s/I32");
        let m = lower_chain(&c);
        assert!(m.contains("{packet = #aie.packet_info<pkt_type ="), "memtile packet on bd attr:\n{m}");
        assert!(m.contains("aie.packet_flow("));
    }

    #[test]
    fn memtile_padding_uses_const_pad_and_grows_output() {
        let c = generate(31, "padboth/memtile/mm2s/I8");
        let m = lower_chain(&c);
        assert!(m.contains("const_pad_before"), "padding list missing:\n{m}");
        assert!(c.transfers.iter().any(|t| t.out_elems > t.in_elems), "padded output must grow");
    }

    #[test]
    #[ignore]
    fn memtile_strided3d_compiles() {
        compile_one("strided3d/memtile/mm2s/I32", 32);
    }
    #[test]
    #[ignore]
    fn memtile_strided4d_compiles() {
        compile_one("strided4d/memtile/mm2s/I32", 33);
    }
    #[test]
    #[ignore]
    fn memtile_transpose_compiles() {
        compile_one("transpose/memtile/s2mm/I16", 34);
    }
    #[test]
    #[ignore]
    fn memtile_overlap_compiles() {
        compile_one("overlap/memtile/mm2s/I8", 35);
    }
    #[test]
    #[ignore]
    fn memtile_packet_compiles() {
        compile_one("packet/memtile/mm2s/I32", 36);
    }
    #[test]
    #[ignore]
    fn memtile_padbefore_compiles() {
        compile_one("padbefore/memtile/mm2s/I8", 37);
    }
    #[test]
    #[ignore]
    fn memtile_padafter_compiles() {
        compile_one("padafter/memtile/mm2s/I16", 38);
    }
    #[test]
    #[ignore]
    fn memtile_padboth_compiles() {
        compile_one("padboth/memtile/mm2s/I32", 39);
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
