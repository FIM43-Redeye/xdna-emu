// SPDX-License-Identifier: MIT
//
// End-to-end harness for the aiesim backend exercising CONTROL PACKETS -- the
// distinct mechanism where a packet routed to a tile's TileControl port carries
// register write/read operations (op in the packet header), rather than ordinary
// data. Mirrors mlir-aie's add_one_ctrl_packet/test.cpp.
//
// Why this test exists separately from the breadth sweep: the generic sweep seeds
// EVERY buffer with in[i]=i, but arg1 here is the control-packet stream and must
// contain specific packet headers (stream-id/op/beats/address + parity bit). Seed
// it with 0,1,2,... and the headers are malformed -> the control read responses
// never come back -> dma_wait @ctrl0 hangs. That hang is a harness artifact, not
// an emulator bug. This test feeds the REAL packets and asserts the known answer,
// proving whether the cluster routes+executes control packets correctly.
//
// Kernel: add_one_ctrl_packet (single compute tile (0,2)). Runtime-sequence args,
// in DdrPatch arg order: 0 = ctrlOut (control read response), 1 = ctrlIn (the
// control packets), 2 = out (DMA output of output_buffer).
//
//   out[i]     == 7 + i   (input init i+3, +1 four times, streamed via pkt flow 0x3)
//   ctrlOut[i] == 8 + i   (other_buffer = input+1, read back via control read pkts)
//
// #[ignore] -- needs aietools libs + the bridge .so + XDNA_BACKEND=aiesim env.
//
//   cargo test -p xdna-emu-ffi --features aiesim --test aiesim_ctrl_packet \
//     -- --ignored --nocapture

use std::ffi::CString;

use xdna_emu::{
    xdna_emu_alloc_buffer, xdna_emu_create, xdna_emu_destroy, xdna_emu_execute_npu_instructions,
    xdna_emu_load_xclbin, xdna_emu_read_host_memory, xdna_emu_run, xdna_emu_set_max_cycles,
    xdna_emu_write_host_memory, XdnaEmuResult,
};

const BYTES: u64 = 64 * 4; // OUT_SIZE int32, matches test.cpp

fn kernel_dir() -> String {
    std::env::var("AIESIM_CTRL_KERNEL_DIR").unwrap_or_else(|_| {
        "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_ctrl_packet/chess".to_string()
    })
}

// Mirror test.cpp's parity(): popcount, return (popcount % 2)==0 as the bit value.
// Even popcount -> 1 (sets header bit 31), so the full word carries odd parity.
fn parity_bit(n: u32) -> u32 {
    if n.count_ones() % 2 == 0 {
        1
    } else {
        0
    }
}

// Build the exact control-packet stream test.cpp writes into bo_ctrlIn.
fn make_ctrl_packets() -> Vec<u32> {
    let mut v = Vec::new();

    // Two WRITE packets: set lock0 (0x1F000) and lock2 (0x1F020) to value 2.
    // beats = 1-1 = 0, operation = 0 (write), stream_id = 0.
    for k in 0..2u32 {
        let address = 0x0001_F000u32 + k * 0x20;
        let mut header = 0u32 << 24 | 0u32 << 22 | 0u32 << 20 | address;
        header |= parity_bit(header) << 31;
        v.push(header);
        v.push(2); // data = 2
    }

    // Two READ packets: read 8 words from other_buffer at 0x440 (4 words each).
    // operation = 0x1 (read), stream_id = 0x2, beats = 3 (4 beats).
    for i in 0..2u32 {
        let address = 0x440u32 + i * 4 * 4;
        let mut header = 0x2u32 << 24 | 0x1u32 << 22 | 3u32 << 20 | address;
        header |= parity_bit(header) << 31;
        v.push(header);
    }
    v
}

#[test]
#[ignore = "needs aietools + bridge .so + XDNA_BACKEND=aiesim env; run inline on the dev box"]
fn add_one_ctrl_packet_end_to_end_aiesim() {
    let log_path = std::env::var("AIESIM_CTRL_LOG")
        .unwrap_or_else(|_| "/tmp/claude-1000/ctrl_pkt_steps.log".to_string());
    let _ = std::fs::write(&log_path, "");
    let step = |msg: &str| {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
            let _ = writeln!(f, "{msg}");
            let _ = f.flush();
        }
        println!("{msg}");
    };

    let dir = kernel_dir();
    let xclbin = format!("{dir}/aie.xclbin");
    let insts_path = format!("{dir}/insts.bin");
    let insts = std::fs::read(&insts_path).unwrap_or_else(|e| panic!("read {insts_path}: {e}"));

    let packets = make_ctrl_packets();
    step(&format!("ctrl packets = {:#010x?}", packets));

    unsafe {
        let h = xdna_emu_create();
        assert!(!h.is_null(), "create returned null (XDNA_BACKEND=aiesim + bridge env set?)");

        let c_xclbin = CString::new(xclbin.as_str()).unwrap();
        let rc = xdna_emu_load_xclbin(h, c_xclbin.as_ptr(), std::ptr::null_mut());
        step(&format!("load_xclbin: rc={rc:?}"));
        assert_eq!(rc, XdnaEmuResult::Success, "load_xclbin {xclbin}");

        // Buffers in DdrPatch arg order: 0=ctrlOut, 1=ctrlIn, 2=out.
        let addr_ctrl_out = xdna_emu_alloc_buffer(h, BYTES);
        let addr_ctrl_in = xdna_emu_alloc_buffer(h, BYTES);
        let addr_out = xdna_emu_alloc_buffer(h, BYTES);
        assert!(addr_ctrl_out != 0 && addr_ctrl_in != 0 && addr_out != 0, "alloc_buffer failed");
        step(&format!("alloc: ctrlOut=0x{addr_ctrl_out:x} ctrlIn=0x{addr_ctrl_in:x} out=0x{addr_out:x}"));

        // Seed ctrlIn with the real packets; zero the rest.
        let mut ctrl_in = vec![0u8; BYTES as usize];
        for (i, w) in packets.iter().enumerate() {
            ctrl_in[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        let rc = xdna_emu_write_host_memory(h, addr_ctrl_in, ctrl_in.as_ptr(), BYTES);
        assert_eq!(rc, XdnaEmuResult::Success, "write ctrlIn");

        step("exec_npu: begin");
        let rc = xdna_emu_execute_npu_instructions(h, insts.as_ptr(), insts.len() as u64);
        step(&format!("exec_npu: rc={rc:?}"));
        assert_eq!(rc, XdnaEmuResult::Success, "execute_npu_instructions");

        // aiesim path: exec_npu already drove the DMAs, run() just flushes, so a
        // tiny settling budget suffices. The interpreter path executes the whole
        // kernel inside run(), needing ~hundreds of k cycles -- override via env.
        let max_cycles: u64 = std::env::var("AIESIM_CTRL_MAX_CYCLES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2_000);
        let _ = xdna_emu_set_max_cycles(h, max_cycles);
        let status = xdna_emu_run(h);
        step(&format!(
            "run: result={:?} halt={:?} cycles={}",
            status.result, status.halt_reason, status.cycles_executed
        ));

        let read_words = |addr: u64| -> Vec<u32> {
            let mut buf = vec![0u8; 8 * 4];
            let rc = xdna_emu_read_host_memory(h, addr, buf.as_mut_ptr(), 8 * 4);
            assert_eq!(rc, XdnaEmuResult::Success, "read 0x{addr:x}");
            buf.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
        };

        let out = read_words(addr_out);
        let ctrl_out = read_words(addr_ctrl_out);
        step(&format!("out[0..8]     = {out:?}"));
        step(&format!("ctrlOut[0..8] = {ctrl_out:?}"));

        let mut errors = 0;
        for i in 0..8 {
            if out[i] != 7 + i as u32 {
                step(&format!("MISMATCH out[{i}] = {} (want {})", out[i], 7 + i));
                errors += 1;
            }
            if ctrl_out[i] != 8 + i as u32 {
                step(&format!("MISMATCH ctrlOut[{i}] = {} (want {})", ctrl_out[i], 8 + i));
                errors += 1;
            }
        }

        xdna_emu_destroy(h);
        step(&format!("RESULT: {} ({errors} errors)", if errors == 0 { "PASS" } else { "FAIL" }));
        assert_eq!(errors, 0, "{errors} mismatches");
    }
}
