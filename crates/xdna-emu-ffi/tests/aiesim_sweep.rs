// SPDX-License-Identifier: MIT
//
// Generic single-kernel runner for the aiesim breadth sweep. Unlike aiesim_e2e.rs
// (which asserts add_256's known answer), this runs an ARBITRARY kernel through
// the aiesim backend and reports whether it executes cleanly -- the "how broadly
// does the backend handle real kernels" signal -- plus an output sample. It
// asserts nothing about numerical correctness (we don't have per-kernel golden
// data here); the value is the pass/hang/crash breakdown across the corpus.
//
// Per-kernel isolation + timeout is the sweep SCRIPT's job (scripts/aiesim-sweep.sh):
// each kernel runs in its own process under `timeout`, so a hang or crash in one
// kernel is recorded, not fatal to the sweep.
//
// #[ignore] -- needs aietools libs + the bridge .so + XDNA_BACKEND=aiesim. Driven
// by AIESIM_KERNEL_DIR (the dir holding aie.xclbin + insts.bin).
//
// Output is machine-parseable: lines beginning "SWEEP " that the script tallies.

use std::ffi::CString;

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

#[test]
#[ignore = "needs aietools + bridge .so + XDNA_BACKEND=aiesim; driven by scripts/aiesim-sweep.sh"]
fn sweep_one_kernel_aiesim() {
    let dir = std::env::var("AIESIM_KERNEL_DIR").expect("set AIESIM_KERNEL_DIR");
    let xclbin = format!("{dir}/aie.xclbin");
    let insts_path = format!("{dir}/insts.bin");

    // Emit one machine-parseable field per stage; the script greps these.
    let field = |k: &str, v: &str| println!("SWEEP {k}={v}");

    let insts = match std::fs::read(&insts_path) {
        Ok(b) => b,
        Err(e) => {
            field("fatal", &format!("read_insts:{e}"));
            return;
        }
    };

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

        // Allocate the generic buffer set and seed every buffer with in[i]=i so
        // input args carry a known pattern; output args get overwritten.
        let mut addrs = Vec::with_capacity(NBUF);
        let words = (BUFSZ / 4) as u32;
        let mut pattern = Vec::with_capacity(BUFSZ as usize);
        for i in 0..words {
            pattern.extend_from_slice(&i.to_le_bytes());
        }
        for _ in 0..NBUF {
            let a = xdna_emu_alloc_buffer(h, BUFSZ);
            if a == 0 {
                field("alloc", "fail");
                xdna_emu_destroy(h);
                return;
            }
            let _ = xdna_emu_write_host_memory(h, a, pattern.as_ptr(), BUFSZ);
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

        xdna_emu_destroy(h);
        field("done", "ran");
    }
}
