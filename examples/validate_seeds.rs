//! Batch in-process validation: for every build/fuzz/seed_* with a saved
//! npu_output.bin (the true hardware output, written on mismatch), run the
//! kernel through the in-process emulator and compare byte-for-byte against
//! hardware. No HW needed -- this re-uses the captured HW outputs.
//!
//! Usage: cargo run --release --example validate_seeds [-- build/fuzz]

use std::path::{Path, PathBuf};

use xdna_emu::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
use xdna_emu::testing::xclbin_suite::{XclbinSuite, XclbinTest};

fn fuzz_spec(size: usize, dt: ElementType) -> BufferSpec {
    let mk = |name: &str, gid: u32, dir, pat| BufferDef {
        name: name.into(),
        group_id: gid,
        size_elements: size,
        element_type: dt,
        direction: dir,
        input_pattern: pat,
    };
    BufferSpec {
        buffers: vec![
            mk("buf_in", 3, BufferDir::Input, InputPattern::Sequential { start: 1, step: 1 }),
            mk("buf_scratch", 4, BufferDir::Input, InputPattern::Zeros),
            mk("buf_out", 5, BufferDir::Output, InputPattern::Zeros),
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

/// Parse `// dtype=i8 size=128 ...` from the generated aie.mlir header.
fn parse_dtype_size(dir: &Path) -> Option<(ElementType, usize, usize)> {
    let text = std::fs::read_to_string(dir.join("aie.mlir")).ok()?;
    let line = text.lines().find(|l| l.contains("dtype="))?;
    let mut dt = None;
    let mut size = None;
    for tok in line.split_whitespace() {
        if let Some(v) = tok.strip_prefix("dtype=") {
            dt = match v {
                "i8" => Some((ElementType::I8, 1usize)),
                "i16" => Some((ElementType::I16, 2)),
                "i32" => Some((ElementType::I32, 4)),
                _ => None,
            };
        } else if let Some(v) = tok.strip_prefix("size=") {
            size = v.parse::<usize>().ok();
        }
    }
    let (dt, bytes) = dt?;
    Some((dt, size?, bytes))
}

fn main() {
    let root = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "build/fuzz".into()));
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("read build/fuzz")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("npu_output.bin").exists() && p.join("aie.xclbin").exists())
        .collect();
    dirs.sort();

    let mut matched = 0usize;
    let mut mism = 0usize;
    let mut skipped = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for dir in &dirs {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        let Some((dt, size, bytes)) = parse_dtype_size(dir) else {
            skipped += 1;
            continue;
        };
        let nbytes = size * bytes;

        let test = XclbinTest::from_path(&dir.join("aie.xclbin")).with_buffer_spec(fuzz_spec(size, dt));
        let suite = XclbinSuite::new();
        let (_o, raw, _t) = suite.run_single_with_trace(&test);
        let Some(mut emu) = raw else {
            skipped += 1;
            continue;
        };
        emu.truncate(nbytes);
        let mut npu = std::fs::read(dir.join("npu_output.bin")).unwrap_or_default();
        npu.truncate(nbytes);

        if emu == npu {
            matched += 1;
        } else {
            mism += 1;
            // First differing element for a quick signature.
            let first = (0..nbytes / bytes)
                .find(|&i| emu[i * bytes..(i + 1) * bytes] != npu[i * bytes..(i + 1) * bytes]);
            mismatches.push(format!("{} (dtype {:?}, first diff elem {:?})", name, dt, first));
        }
    }

    println!("validated {} seeds with saved HW output:", dirs.len());
    println!("  EMU == HW : {}", matched);
    println!("  EMU != HW : {}", mism);
    println!("  skipped   : {}", skipped);
    if !mismatches.is_empty() {
        println!("remaining EMU != HW:");
        for m in &mismatches {
            println!("  {}", m);
        }
    }
}
