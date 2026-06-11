// SPDX-License-Identifier: MIT
//
// Drives the CLEAN single-tile re-author of vector-fuzzer seed 6159 (the bf16
// chained-NaN divergence) through a runtime backend (interpreter by default,
// aiesim bridge under XDNA_BACKEND=aiesim) and reads lane 29 of slice 4.
//
// The interpreter gives 0xFF81 (collapse); real NPU1 silicon gives 0xFF8C
// (computed significand). This probe asks whether AMD's own AIE2 model (aiesim)
// reproduces 0xFF8C. The CLEAN kernel (tools/gen_seed6159_clean.py -- direct
// shim<->core, no memtile/trace) runs through the FFI/aiesim path without the
// memtile objectfifo-lock wedge the fuzzer-banked xclbin hits.
//
// #[ignore] -- needs the clean xclbin built + (for aiesim) aietools libs + bridge
// .so + XDNA_BACKEND=aiesim env. Run inline:
//   ./scripts/emu-bridge-test.sh --peano-only --compile vec_seed6159_clean   # builds the xclbin
//   # interpreter sanity (expect 0xFF81):
//   cargo test -p xdna-emu-ffi --features aiesim --test aiesim_bf16_nan_probe -- --ignored --nocapture
//   # aiesim (expect 0xFF8C if AMD's model reproduces it):
//   source toolchain-build/activate-npu-env.sh
//   export LD_LIBRARY_PATH="$XILINX_VITIS_AIETOOLS/lib/lnx64.o:$PWD/aiesim-bridge/build:$LD_LIBRARY_PATH"
//   export XDNA_BACKEND=aiesim
//   export XDNA_AIESIM_BRIDGE=$PWD/aiesim-bridge/build/libxdna_aiesim_bridge.so
//   export XDNA_AIESIM_DEVICE_JSON=$NPU_WORK_DIR/amd-unified-software/aietools/data/aie_ml/devices/VC2802.json
//   export XDNA_AIESIM_NATIVE_GEOMETRY=1
//   export XDNA_AIESIM_VCD=$PWD/build/experiments/special-value-dense/seed6159_clean.vcd  # optional register dump
//   cargo test -p xdna-emu-ffi --features aiesim --test aiesim_bf16_nan_probe -- --ignored --nocapture

use std::ffi::CString;

use xdna_emu::{
    xdna_emu_alloc_buffer, xdna_emu_create, xdna_emu_destroy, xdna_emu_execute_npu_instructions,
    xdna_emu_load_xclbin, xdna_emu_read_host_memory, xdna_emu_run, xdna_emu_set_max_cycles,
    xdna_emu_write_host_memory, XdnaEmuResult,
};

const BUF_BYTES: u64 = 512; // 128 i32 = 256 u16 = 8 slices of 32 bf16

fn build_dir() -> String {
    std::env::var("CLEAN_BUILD_DIR").unwrap_or_else(|_| {
        "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/vec_seed6159_clean/peano".to_string()
    })
}

fn pool_path() -> String {
    std::env::var("AIESIM_POOL_BIN").unwrap_or_else(|_| {
        "/home/triple/npu-work/xdna-emu/build/experiments/bf16-nan-replay/pools/pool_seed_6159.bin"
            .to_string()
    })
}

#[test]
#[ignore = "needs clean xclbin built; for aiesim also bridge .so + XDNA_BACKEND=aiesim env"]
fn seed6159_clean_bf16_nan() {
    let dir = build_dir();
    let xclbin = format!("{dir}/aie.xclbin");
    let insts =
        std::fs::read(format!("{dir}/insts.bin")).expect("read insts.bin (build the clean xclbin first)");
    let mut pool = std::fs::read(pool_path()).expect("read pool");
    pool.resize(BUF_BYTES as usize, 0);

    let backend = std::env::var("XDNA_BACKEND").unwrap_or_else(|_| "interpreter".into());
    eprintln!("backend = {backend}");

    unsafe {
        let h = xdna_emu_create();
        assert!(!h.is_null(), "create returned null");

        let c_xclbin = CString::new(xclbin.as_str()).unwrap();
        let rc = xdna_emu_load_xclbin(h, c_xclbin.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, XdnaEmuResult::Success, "load_xclbin {xclbin}");

        // Clean kernel runtime-sequence args: 0=in, 1=out.
        let addr_in = xdna_emu_alloc_buffer(h, BUF_BYTES);
        let addr_out = xdna_emu_alloc_buffer(h, BUF_BYTES);
        assert!(addr_in != 0 && addr_out != 0, "alloc_buffer failed");

        let rc = xdna_emu_write_host_memory(h, addr_in, pool.as_ptr(), BUF_BYTES);
        assert_eq!(rc, XdnaEmuResult::Success, "write input pool");

        let rc = xdna_emu_execute_npu_instructions(h, insts.as_ptr(), insts.len() as u64);
        assert_eq!(rc, XdnaEmuResult::Success, "execute_npu_instructions");

        let _ = xdna_emu_set_max_cycles(h, 200_000);
        let status = xdna_emu_run(h);
        eprintln!(
            "run: result={:?} halt={:?} cycles={}",
            status.result, status.halt_reason, status.cycles_executed
        );

        let mut out = vec![0u8; BUF_BYTES as usize];
        let rc = xdna_emu_read_host_memory(h, addr_out, out.as_mut_ptr(), BUF_BYTES);
        assert_eq!(rc, XdnaEmuResult::Success, "read output");
        xdna_emu_destroy(h);

        let lane = |sl: usize, i: usize| -> u16 {
            let off = sl * 64 + i * 2;
            u16::from_le_bytes([out[off], out[off + 1]])
        };
        let nz = out.iter().filter(|&&b| b != 0).count();
        eprintln!("output nonzero bytes: {nz}/{}", out.len());
        let l29_s4 = lane(4, 29);
        eprintln!("=== seed 6159 (clean) via {backend} ===");
        eprintln!("slice4 lane29 = 0x{l29_s4:04X}  (interpreter 0xFF81, real NPU1 0xFF8C)");
        match l29_s4 {
            0xFF8C => eprintln!(">>> REPRODUCES HARDWARE (0xFF8C)"),
            0xFF81 => eprintln!(">>> collapse (0xFF81) -- same as interpreter datapath"),
            other => eprintln!(">>> 0x{other:04X} (unexpected)"),
        }
        eprint!("slice4 lanes: ");
        for i in 0..32 {
            eprint!("{:04X} ", lane(4, i));
        }
        eprintln!();
        // Sanity: with the interpreter backend this must be the known 0xFF81.
        if backend == "interpreter" {
            assert_eq!(l29_s4, 0xFF81, "interpreter clean re-author should match the seed's EMU value");
        }
    }
}
