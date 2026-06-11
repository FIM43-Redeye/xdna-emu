// SPDX-License-Identifier: MIT
//
// Classifier for the seed-6159 chained-NaN effect. Runs the ISOLATED single-add
// probe (tools/gen_h4_isolated_probe.py) through a runtime backend. The kernel is
// out = aie::add(v3, p3) fed h4's EXACT 32-lane input vectors with NO preceding
// h0..h3 history.
//
//   lane29 == 0xFF8C  -> trigger is CROSS-LANE vector content (modelable from inputs)
//   lane29 == 0xFF81  -> trigger is PRECEDING-INSTRUCTION float-unit state
//
// #[ignore] -- run inline like aiesim_bf16_nan_probe.rs. Env for the aiesim leg
// is identical (XDNA_BACKEND=aiesim + bridge .so + native NPU1 device JSON).
//   H4_BUILD_DIR=mlir-aie/build/test/npu-xrt/vec_h4_isolated_probe/peano
//   H4_INPUT_BIN=build/experiments/special-value-dense/h4probe_input.bin

use std::ffi::CString;

use xdna_emu::{
    xdna_emu_alloc_buffer, xdna_emu_create, xdna_emu_destroy, xdna_emu_execute_npu_instructions,
    xdna_emu_load_xclbin, xdna_emu_read_host_memory, xdna_emu_run, xdna_emu_set_max_cycles,
    xdna_emu_write_host_memory, XdnaEmuResult,
};

const IN_BYTES: u64 = 128; // 32 i32 = 64 u16 (a|b)
const OUT_BYTES: u64 = 64; // 16 i32 = 32 u16 (r)

fn build_dir() -> String {
    std::env::var("H4_BUILD_DIR").unwrap_or_else(|_| {
        "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/vec_h4_isolated_probe/peano".to_string()
    })
}

fn input_path() -> String {
    std::env::var("H4_INPUT_BIN").unwrap_or_else(|_| {
        "/home/triple/npu-work/xdna-emu/build/experiments/special-value-dense/h4probe_input.bin".to_string()
    })
}

#[test]
#[ignore = "needs isolated-probe xclbin built; for aiesim also bridge .so + XDNA_BACKEND=aiesim env"]
fn h4_isolated_classify() {
    let dir = build_dir();
    let xclbin = format!("{dir}/aie.xclbin");
    let insts = std::fs::read(format!("{dir}/insts.bin")).expect("read insts.bin");
    let mut input = std::fs::read(input_path()).expect("read h4probe_input.bin");
    input.resize(IN_BYTES as usize, 0);

    let backend = std::env::var("XDNA_BACKEND").unwrap_or_else(|_| "interpreter".into());
    eprintln!("backend = {backend}");

    unsafe {
        let h = xdna_emu_create();
        assert!(!h.is_null());

        let c_xclbin = CString::new(xclbin.as_str()).unwrap();
        let rc = xdna_emu_load_xclbin(h, c_xclbin.as_ptr(), std::ptr::null_mut());
        assert_eq!(rc, XdnaEmuResult::Success, "load_xclbin {xclbin}");

        // runtime-sequence args: 0=in, 1=out.
        let addr_in = xdna_emu_alloc_buffer(h, IN_BYTES);
        let addr_out = xdna_emu_alloc_buffer(h, OUT_BYTES);
        assert!(addr_in != 0 && addr_out != 0);

        let rc = xdna_emu_write_host_memory(h, addr_in, input.as_ptr(), IN_BYTES);
        assert_eq!(rc, XdnaEmuResult::Success);

        let rc = xdna_emu_execute_npu_instructions(h, insts.as_ptr(), insts.len() as u64);
        assert_eq!(rc, XdnaEmuResult::Success);

        let _ = xdna_emu_set_max_cycles(h, 200_000);
        let status = xdna_emu_run(h);
        eprintln!(
            "run: result={:?} halt={:?} cycles={}",
            status.result, status.halt_reason, status.cycles_executed
        );

        let mut out = vec![0u8; OUT_BYTES as usize];
        let rc = xdna_emu_read_host_memory(h, addr_out, out.as_mut_ptr(), OUT_BYTES);
        assert_eq!(rc, XdnaEmuResult::Success);
        xdna_emu_destroy(h);

        let lane = |i: usize| -> u16 { u16::from_le_bytes([out[i * 2], out[i * 2 + 1]]) };
        let l29 = lane(29);
        eprintln!("=== h4-isolated via {backend} ===");
        eprintln!("lane29 = 0x{l29:04X}");
        match l29 {
            0xFF8C => eprintln!(">>> 0xFF8C: CROSS-LANE vector context is the trigger"),
            0xFF81 => eprintln!(">>> 0xFF81: PRECEDING-INSTRUCTION state is the trigger"),
            other => eprintln!(">>> 0x{other:04X} (unexpected)"),
        }
        eprint!("all lanes: ");
        for i in 0..32 {
            eprint!("{:04X} ", lane(i));
        }
        eprintln!();
        if backend == "interpreter" {
            assert_eq!(l29, 0xFF81, "interpreter isolated add should collapse to 0xFF81");
        }
    }
}
