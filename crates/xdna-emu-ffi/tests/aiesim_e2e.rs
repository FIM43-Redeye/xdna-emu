// SPDX-License-Identifier: MIT
//
// End-to-end harness for the aiesim backend (Fork A: NPU1 xclbin replayed onto
// the Versal AIE-ML cluster through the coordinate remap). Drives the xdna_emu_*
// FFI exactly as the XRT plugin does, with XDNA_BACKEND=aiesim, against a real
// NPU1 kernel and asserts the known-answer output.
//
// #[ignore] -- needs aietools libs + the bridge .so + the env below; cannot run
// in the sandbox. Run inline on the dev box:
//
//   source toolchain-build/activate-npu-env.sh
//   export LD_LIBRARY_PATH="$XILINX_VITIS_AIETOOLS/lib/lnx64.o:\
//     $(pwd)/xdna-emu/aiesim-bridge/build:$LD_LIBRARY_PATH"
//   export XDNA_BACKEND=aiesim
//   export XDNA_AIESIM_BRIDGE=$(pwd)/xdna-emu/aiesim-bridge/build/libxdna_aiesim_bridge.so
//   export XDNA_AIESIM_DEVICE_JSON=$NPU_WORK_DIR/amd-unified-software/aietools/\
//     data/aie_ml/devices/VC2802.json
//   cargo test -p xdna-emu-ffi --features aiesim --test aiesim_e2e -- --ignored --nocapture
//
// Kernel: add_256_using_dma_op_no_double_buffering (out[i] = in[i] + 256, 64
// int32, single compute tile (0,2)). Runtime-sequence args, in DdrPatch arg
// order: 0 = inA (input), 1 = inB (unused), 2 = out (output).

use std::ffi::CString;

use xdna_emu::{
    xdna_emu_alloc_buffer, xdna_emu_create, xdna_emu_destroy, xdna_emu_execute_npu_instructions,
    xdna_emu_load_xclbin, xdna_emu_read_host_memory, xdna_emu_run, xdna_emu_set_max_cycles,
    xdna_emu_write_host_memory, XdnaEmuResult,
};

const N: usize = 64;
const BYTES: u64 = (N * 4) as u64;
const ADDEND: u32 = 256;

fn kernel_dir() -> String {
    std::env::var("AIESIM_E2E_KERNEL_DIR").unwrap_or_else(|_| {
        "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/\
         add_256_using_dma_op_no_double_buffering/chess"
            .to_string()
    })
}

#[test]
#[ignore = "needs aietools + bridge .so + XDNA_BACKEND=aiesim env; run inline on the dev box"]
fn add_256_end_to_end_aiesim() {
    // Progress/result log, flushed per step to a dedicated file. The cluster's
    // stdout is block-buffered and lost on abort, so stdout/stderr can't be
    // trusted to show where we got; this file always survives. Path overridable.
    let log_path =
        std::env::var("AIESIM_E2E_LOG").unwrap_or_else(|_| "/tmp/claude-1000/e2e_steps.log".to_string());
    let _ = std::fs::write(&log_path, "");
    let step = |msg: &str| {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
            let _ = writeln!(f, "{msg}");
            let _ = f.flush();
        }
    };

    let dir = kernel_dir();
    let xclbin = format!("{dir}/aie.xclbin");
    let insts_path = format!("{dir}/insts.bin");
    let insts = std::fs::read(&insts_path).unwrap_or_else(|e| panic!("read {insts_path}: {e}"));

    unsafe {
        step("create: begin");
        let h = xdna_emu_create();
        assert!(!h.is_null(), "create returned null (XDNA_BACKEND=aiesim + bridge env set?)");
        step("create: ok");

        let c_xclbin = CString::new(xclbin.as_str()).unwrap();
        step("load_xclbin: begin");
        let rc = xdna_emu_load_xclbin(h, c_xclbin.as_ptr(), std::ptr::null_mut());
        step(&format!("load_xclbin: rc={rc:?}"));
        assert_eq!(rc, XdnaEmuResult::Success, "load_xclbin {xclbin}");

        // Register the three runtime-sequence buffers in arg order so DdrPatch
        // arg_idx 0/1/2 -> inA/inB/out. alloc_buffer registers each in turn.
        let addr_in = xdna_emu_alloc_buffer(h, BYTES);
        let _addr_mid = xdna_emu_alloc_buffer(h, BYTES); // inB (unused by the kernel)
        let addr_out = xdna_emu_alloc_buffer(h, BYTES);
        step(&format!("alloc: in=0x{addr_in:x} mid=0x{_addr_mid:x} out=0x{addr_out:x}"));
        assert!(addr_in != 0 && _addr_mid != 0 && addr_out != 0, "alloc_buffer failed");

        // Input: in[i] = i.
        let mut input = Vec::with_capacity(N * 4);
        for i in 0..N as u32 {
            input.extend_from_slice(&i.to_le_bytes());
        }
        let rc = xdna_emu_write_host_memory(h, addr_in, input.as_ptr(), BYTES);
        step(&format!("write_input: rc={rc:?}"));
        assert_eq!(rc, XdnaEmuResult::Success, "write input");

        step("exec_npu: begin");
        let rc = xdna_emu_execute_npu_instructions(h, insts.as_ptr(), insts.len() as u64);
        step(&format!("exec_npu: rc={rc:?}"));
        assert_eq!(rc, XdnaEmuResult::Success, "execute_npu_instructions");

        // The kernel already completed during exec_npu (the @data_out dma_wait
        // blocked until the whole pipeline finished). This GMIO/DMA kernel never
        // fires plio_complete, so an unbounded run would grind the full 100M-cyc
        // default budget; cap it to a short settling window. run() flushes the
        // already-computed output back from the bridge DDR regardless.
        let _ = xdna_emu_set_max_cycles(h, 2_000);
        step("run: begin (max_cycles=2000)");
        let status = xdna_emu_run(h);
        step(&format!(
            "run: result={:?} halt={:?} cycles={}",
            status.result, status.halt_reason, status.cycles_executed
        ));
        assert_eq!(status.result, XdnaEmuResult::Success, "run (halt={:?})", status.halt_reason);

        let mut out = vec![0u8; N * 4];
        let rc = xdna_emu_read_host_memory(h, addr_out, out.as_mut_ptr(), BYTES);
        step(&format!("read_output: rc={rc:?}"));
        assert_eq!(rc, XdnaEmuResult::Success, "read output");

        let got: Vec<u32> = out.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        step(&format!("out[0..8] = {:?}", &got[..8]));
        let mut mismatches = 0;
        for (i, &v) in got.iter().enumerate() {
            let want = i as u32 + ADDEND;
            if v != want {
                if mismatches < 8 {
                    step(&format!("MISMATCH out[{i}] = {v} (want {want})"));
                }
                mismatches += 1;
            }
        }
        xdna_emu_destroy(h);
        step(&format!("RESULT: {} ({mismatches}/{N} wrong)", if mismatches == 0 { "PASS" } else { "FAIL" }));
        assert_eq!(mismatches, 0, "{mismatches}/{N} elements wrong");
        println!("[e2e] PASS: add_256 end-to-end via aiesim backend");
    }
}
