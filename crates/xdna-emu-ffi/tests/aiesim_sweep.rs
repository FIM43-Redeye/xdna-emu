// SPDX-License-Identifier: MIT
//
// Generic single-kernel runner for the aiesim breadth sweep. Unlike aiesim_e2e.rs
// (which asserts add_256's known answer), this runs an ARBITRARY kernel through
// the aiesim backend and reports whether it executes cleanly -- the "how broadly
// does the backend handle real kernels" signal -- plus an output sample.
//
// Buffer seeding is GENERIC by default (every arg gets in[i]=i), which is wrong
// for kernels whose arguments carry structured input -- notably control-packet
// kernels, where one arg is a stream of control packets (header+parity+payload).
// Such kernels are described in a per-kernel input spec (tests/aiesim/
// kernel-inputs.json, path via AIESIM_SWEEP_SPEC): the spec overrides seeding
// per-argument and, where an `expect` is given, the runner verifies outputs and
// emits `SWEEP correct=PASS|MISMATCH`. Kernels with no spec entry keep the
// generic behavior and the opportunistic add_<N> check the sweep script applies.
//
// Per-kernel isolation + timeout is the sweep SCRIPT's job (scripts/aiesim-sweep.sh):
// each kernel runs in its own process under `timeout`, so a hang or crash in one
// kernel is recorded, not fatal to the sweep.
//
// #[ignore] -- needs aietools libs + the bridge .so + XDNA_BACKEND=aiesim. Driven
// by AIESIM_KERNEL_DIR (the dir holding aie.xclbin + insts.bin) and, optionally,
// AIESIM_KERNEL_NAME (else derived from the dir) + AIESIM_SWEEP_SPEC.
//
// Output is machine-parseable: lines beginning "SWEEP " that the script tallies.

use std::collections::HashMap;
use std::ffi::CString;

use serde::Deserialize;

use xdna_emu::{
    xdna_emu_alloc_buffer, xdna_emu_create, xdna_emu_destroy, xdna_emu_execute_npu_instructions,
    xdna_emu_load_xclbin, xdna_emu_read_host_memory, xdna_emu_run, xdna_emu_set_max_cycles,
    xdna_emu_write_host_memory, XdnaEmuResult,
};

// Generic buffer set. The runtime sequence's DdrPatch ops reference args by index;
// arg_idx N binds to the Nth alloc_buffer call. Allocate enough to cover any
// reasonable kernel, each large enough for any test transfer. The allocator
// page-bumps by size, so distinct 256 KB buffers never overlap.
const NBUF: usize = 8;
const BUFSZ: u64 = 256 * 1024;
const SAMPLE_WORDS: usize = 8;

// ---------------------------------------------------------------------------
// Per-kernel input/expectation spec (schema documented in the JSON's _readme).
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct SpecFile {
    #[serde(default)]
    kernels: HashMap<String, KernelSpec>,
}

#[derive(Deserialize)]
struct KernelSpec {
    #[serde(default)]
    args: Vec<ArgSpec>,
}

#[derive(Deserialize)]
struct ArgSpec {
    idx: usize,
    role: Role,
    #[serde(default)]
    packets: Vec<PacketOp>,
    #[serde(default)]
    expect: Option<Expect>,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum Role {
    CtrlIn,
    Output,
    Generic,
}

#[derive(Deserialize)]
struct PacketOp {
    op: OpKind,
    addr: String,
    #[serde(default)]
    data: Option<DataField>,
    #[serde(default)]
    stream_id: Option<u32>,
    #[serde(default)]
    beats: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum OpKind {
    Write,
    Read,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DataField {
    One(u32),
    Many(Vec<u32>),
}

#[derive(Deserialize)]
struct Expect {
    base: u32,
    count: usize,
    #[serde(default)]
    stride: Option<u32>,
    #[serde(default)]
    modulo: Option<u32>,
}

// Even-popcount parity in bit 31 (mirrors mlir-aie test.cpp parity(): popcount
// even -> returns true -> sets bit 31, so the full word carries odd parity).
fn parity_bit(n: u32) -> u32 {
    if n.count_ones() % 2 == 0 {
        1
    } else {
        0
    }
}

fn parse_addr(s: &str) -> u32 {
    let t = s.trim();
    let h = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X"));
    match h {
        Some(hex) => u32::from_str_radix(hex, 16).expect("hex addr"),
        None => t.parse().expect("decimal addr"),
    }
}

// Encode high-level packet ops to the u32 word stream that lands in the ctrl_in
// buffer. Header: stream_id<<24 | op<<22 | (beats-1)<<20 | addr, parity in bit 31.
//   write -> op 0, beats = number of data words; emits [header, data...]
//   read  -> op 1, beats = logical response beats; emits [header]
fn encode_packets(ops: &[PacketOp]) -> Vec<u32> {
    let mut out = Vec::new();
    for p in ops {
        let addr = parse_addr(&p.addr);
        let stream_id = p.stream_id.unwrap_or(0);
        match p.op {
            OpKind::Write => {
                let data: Vec<u32> = match &p.data {
                    Some(DataField::One(x)) => vec![*x],
                    Some(DataField::Many(v)) => v.clone(),
                    None => Vec::new(),
                };
                let beats_field = (data.len().max(1) as u32) - 1;
                let mut h = stream_id << 24 | 0u32 << 22 | beats_field << 20 | addr;
                h |= parity_bit(h) << 31;
                out.push(h);
                out.extend_from_slice(&data);
            }
            OpKind::Read => {
                let beats = p.beats.unwrap_or(1).max(1);
                let beats_field = beats - 1;
                let mut h = stream_id << 24 | 1u32 << 22 | beats_field << 20 | addr;
                h |= parity_bit(h) << 31;
                out.push(h);
            }
        }
    }
    out
}

fn expected_vec(e: &Expect) -> Vec<u32> {
    let stride = e.stride.unwrap_or(1);
    (0..e.count)
        .map(|i| {
            let k = match e.modulo {
                Some(m) if m != 0 => (i as u32) % m,
                _ => i as u32,
            };
            e.base + stride * k
        })
        .collect()
}

// Resolve kernel name: explicit env wins; else the basename of AIESIM_KERNEL_DIR's
// parent (.../npu-xrt/<name>/<compiler>).
fn kernel_name(dir: &str) -> String {
    if let Ok(n) = std::env::var("AIESIM_KERNEL_NAME") {
        if !n.is_empty() {
            return n;
        }
    }
    std::path::Path::new(dir)
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn load_kernel_spec(name: &str) -> Option<KernelSpec> {
    let path = std::env::var("AIESIM_SWEEP_SPEC").ok()?;
    let text = std::fs::read_to_string(&path).ok()?;
    let mut file: SpecFile = match serde_json::from_str(&text) {
        Ok(f) => f,
        Err(e) => {
            println!("SWEEP spec_error=parse:{e}");
            return None;
        }
    };
    file.kernels.remove(name)
}

#[test]
#[ignore = "needs aietools + bridge .so + XDNA_BACKEND=aiesim; driven by scripts/aiesim-sweep.sh"]
fn sweep_one_kernel_aiesim() {
    let dir = std::env::var("AIESIM_KERNEL_DIR").expect("set AIESIM_KERNEL_DIR");
    let xclbin = format!("{dir}/aie.xclbin");
    let insts_path = format!("{dir}/insts.bin");
    let name = kernel_name(&dir);
    let spec = load_kernel_spec(&name);

    // Emit one machine-parseable field per stage; the script greps these.
    let field = |k: &str, v: &str| println!("SWEEP {k}={v}");
    if spec.is_some() {
        field("spec", &name);
    }

    let insts = match std::fs::read(&insts_path) {
        Ok(b) => b,
        Err(e) => {
            field("fatal", &format!("read_insts:{e}"));
            return;
        }
    };

    // Precompute the per-arg seed plan from the spec (idx -> role/packets).
    let arg_spec =
        |idx: usize| -> Option<&ArgSpec> { spec.as_ref().and_then(|s| s.args.iter().find(|a| a.idx == idx)) };

    unsafe {
        let h = xdna_emu_create();
        if h.is_null() {
            field("create", "fail");
            return;
        }
        field("create", "ok");

        let c_xclbin = CString::new(xclbin.as_str()).unwrap();
        let rc = xdna_emu_load_xclbin(h, c_xclbin.as_ptr(), std::ptr::null_mut());
        field("load", &format!("{rc:?}"));
        if rc != XdnaEmuResult::Success {
            xdna_emu_destroy(h);
            field("done", "load_failed");
            return;
        }

        // Generic seed pattern: in[i] = i.
        let words = (BUFSZ / 4) as u32;
        let mut generic = Vec::with_capacity(BUFSZ as usize);
        for i in 0..words {
            generic.extend_from_slice(&i.to_le_bytes());
        }

        // Allocate the buffer set, seeding each per the spec (ctrl_in -> encoded
        // packets, output -> zeros, generic/unlisted -> i-pattern).
        let mut addrs = Vec::with_capacity(NBUF);
        for idx in 0..NBUF {
            let a = xdna_emu_alloc_buffer(h, BUFSZ);
            if a == 0 {
                field("alloc", "fail");
                xdna_emu_destroy(h);
                return;
            }
            match arg_spec(idx).map(|s| &s.role) {
                Some(Role::CtrlIn) => {
                    let pkts = encode_packets(&arg_spec(idx).unwrap().packets);
                    let mut buf = vec![0u8; BUFSZ as usize];
                    for (i, w) in pkts.iter().enumerate() {
                        buf[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
                    }
                    let _ = xdna_emu_write_host_memory(h, a, buf.as_ptr(), BUFSZ);
                }
                Some(Role::Output) => {
                    let zeros = vec![0u8; BUFSZ as usize];
                    let _ = xdna_emu_write_host_memory(h, a, zeros.as_ptr(), BUFSZ);
                }
                _ => {
                    let _ = xdna_emu_write_host_memory(h, a, generic.as_ptr(), BUFSZ);
                }
            }
            addrs.push(a);
        }
        field("alloc", &format!("{NBUF}"));

        let rc = xdna_emu_execute_npu_instructions(h, insts.as_ptr(), insts.len() as u64);
        field("exec", &format!("{rc:?}"));

        // Bounded settling window: these DMA/GMIO kernels mostly complete during
        // exec_npu (the dma_wait blocks), and never fire plio_complete, so an
        // unbounded run would grind the full default budget. run() flushes
        // already-computed output back regardless.
        let _ = xdna_emu_set_max_cycles(h, 2_000);
        let status = xdna_emu_run(h);
        field("run", &format!("{:?}", status.result));
        field("halt", &format!("{:?}", status.halt_reason));
        field("cycles", &format!("{}", status.cycles_executed));

        // Report which buffers the kernel altered vs the seeded pattern, with a
        // sample. A buffer that differs from in[i]=i is (probably) an output.
        for (idx, &a) in addrs.iter().enumerate() {
            let mut out = vec![0u8; SAMPLE_WORDS * 4];
            let rc = xdna_emu_read_host_memory(h, a, out.as_mut_ptr(), (SAMPLE_WORDS * 4) as u64);
            if rc != XdnaEmuResult::Success {
                continue;
            }
            let got: Vec<u32> =
                out.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let altered = got.iter().enumerate().any(|(i, &v)| v != i as u32);
            if altered {
                field(&format!("buf{idx}"), &format!("{got:?}"));
            }
        }

        // Spec-driven correctness: verify every output arg that carries an expect.
        // Emits `SWEEP correct=PASS|MISMATCH` (the script prefers this over its
        // opportunistic add_<N> grep). Kernels without expects emit nothing here.
        if let Some(ks) = spec.as_ref() {
            let checks: Vec<(usize, &Expect)> = ks
                .args
                .iter()
                .filter(|a| a.idx < NBUF)
                .filter_map(|a| a.expect.as_ref().map(|e| (a.idx, e)))
                .collect();
            if !checks.is_empty() {
                let mut ok = true;
                for (idx, e) in &checks {
                    let want = expected_vec(e);
                    let nbytes = want.len() * 4;
                    let mut buf = vec![0u8; nbytes];
                    let rc = xdna_emu_read_host_memory(h, addrs[*idx], buf.as_mut_ptr(), nbytes as u64);
                    let got: Vec<u32> =
                        buf.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                    let pass = rc == XdnaEmuResult::Success && got == want;
                    if !pass {
                        ok = false;
                        field(&format!("mismatch_arg{idx}"), &format!("got={got:?} want={want:?}"));
                    }
                }
                field("correct", if ok { "PASS" } else { "MISMATCH" });
            }
        }

        xdna_emu_destroy(h);
        field("done", "ran");
    }
}

// Pure-logic guards for the spec path -- no hardware/aietools needed, so they
// run in the normal `cargo test` suite and pin the encoder + committed spec.
#[cfg(test)]
mod spec_tests {
    use super::*;

    #[test]
    fn encodes_base_ctrl_packet_stream() {
        // Must reproduce add_one_ctrl_packet/test.cpp's bo_ctrlIn word-for-word.
        let json = r#"[
            {"op":"write","addr":"0x1F000","data":2},
            {"op":"write","addr":"0x1F020","data":2},
            {"op":"read","addr":"0x440","stream_id":2,"beats":4},
            {"op":"read","addr":"0x450","stream_id":2,"beats":4}
        ]"#;
        let ops: Vec<PacketOp> = serde_json::from_str(json).unwrap();
        let words = encode_packets(&ops);
        assert_eq!(words, vec![0x0001_F000, 2, 0x8001_F020, 2, 0x8270_0440, 0x0270_0450]);
    }

    #[test]
    fn expected_vec_plain_and_modulo() {
        let e: Expect = serde_json::from_str(r#"{"base":7,"count":8}"#).unwrap();
        assert_eq!(expected_vec(&e), (7u32..15).collect::<Vec<u32>>());

        let m: Expect = serde_json::from_str(r#"{"base":7,"count":32,"modulo":8}"#).unwrap();
        let got = expected_vec(&m);
        assert_eq!(got.len(), 32);
        assert_eq!(&got[0..8], &[7, 8, 9, 10, 11, 12, 13, 14]);
        assert_eq!(&got[8..16], &[7, 8, 9, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn committed_spec_parses_and_covers_ctrl_kernels() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/aiesim/kernel-inputs.json");
        let text = std::fs::read_to_string(path).unwrap();
        let file: SpecFile = serde_json::from_str(&text).unwrap();
        for k in ["add_one_ctrl_packet", "add_one_ctrl_packet_4_cores", "add_one_ctrl_packet_col_overlay"] {
            assert!(file.kernels.contains_key(k), "spec missing {k}");
        }
    }
}
