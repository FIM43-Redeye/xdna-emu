//! Verify the LE-bundle squash fix end-to-end: run seed_18's Peano xclbin
//! through the in-process emulator and confirm the output now matches the
//! every-4th-element-zeroed pattern that real hardware produces (BUG-B).
//!
//! Usage: cargo run --example check_le_squash -- build/fuzz/seed_18 128

use std::path::PathBuf;

use xdna_emu::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use xdna_emu::testing::xclbin_suite::{XclbinSuite, XclbinTest};

fn fuzz_spec(size: usize, dt: ElementType) -> BufferSpec {
    BufferSpec {
        buffers: vec![
            BufferDef {
                name: "buf_in".into(),
                group_id: 3,
                size_elements: size,
                element_type: dt,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Sequential { start: 1, step: 1 },
            },
            BufferDef {
                name: "buf_scratch".into(),
                group_id: 4,
                size_elements: size,
                element_type: dt,
                direction: BufferDir::Input,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_out".into(),
                group_id: 5,
                size_elements: size,
                element_type: dt,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
            BufferDef {
                name: "buf_trace".into(),
                group_id: 6,
                size_elements: 262_144,
                element_type: ElementType::I32,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            },
        ],
        multi_kernel: false,
    }
}

fn main() {
    let dir = PathBuf::from(std::env::args().nth(1).expect("usage: <case_dir> <size> <i8|i16|i32>"));
    let size: usize = std::env::args().nth(2).expect("size").parse().unwrap();
    let (dt, bytes) = match std::env::args().nth(3).as_deref() {
        Some("i16") => (ElementType::I16, 2usize),
        Some("i32") => (ElementType::I32, 4usize),
        _ => (ElementType::I8, 1usize),
    };
    let nbytes = size * bytes;

    let xclbin = dir.join("aie.xclbin");
    let test = XclbinTest::from_path(&xclbin).with_buffer_spec(fuzz_spec(size, dt));
    let suite = XclbinSuite::new();
    let (_outcome, raw_output, _trace) = suite.run_single_with_trace(&test);
    let mut emu = raw_output.expect("emu output");
    emu.truncate(nbytes);

    let npu_path = dir.join("npu_output.bin");
    let mut npu = std::fs::read(&npu_path).unwrap_or_default();
    npu.truncate(nbytes);

    // Decode first 16 elements for display.
    let decode = |b: &[u8]| -> Vec<i64> {
        b.chunks_exact(bytes)
            .take(16)
            .map(|c| match bytes {
                1 => c[0] as i8 as i64,
                2 => i16::from_le_bytes([c[0], c[1]]) as i64,
                _ => i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64,
            })
            .collect()
    };
    println!("EMU (post-fix), first 16: {:?}", decode(&emu));
    if !npu.is_empty() {
        println!("HW  (saved),    first 16: {:?}", decode(&npu));
        let matches = emu == npu;
        println!("EMU == HW (saved npu_output.bin): {}  [{} bytes]", matches, nbytes);
        std::process::exit(if matches { 0 } else { 1 });
    }
}
