//! SP-5b R1 emu inject-and-recover loop (#140): run the R1 xclbin in-process
//! twice (injected + zero constants) and assert the Python differencing bridge
//! recovers the injected {d_v, intra_contrast} exactly. Plumbing, not physics.
#[cfg(test)]
mod tests {
    use crate::testing::test_cpp_parser::{BufferDef, BufferDir, BufferSpec, ElementType, InputPattern};
    use crate::testing::xclbin_suite::{XclbinSuite, XclbinTest};
    use std::io::Write;
    use std::process::Command;
    use xdna_archspec::types::BroadcastTiming;

    /// sp5_skew_r1.py's runtime_sequence has exactly one memref arg (the
    /// completion-sink output tensor, group_id 3 -- Q=0, no DDR input). The
    /// in-process runner has no test.cpp to parse a BufferSpec from (this
    /// kernel is trace-only), so we build the spec by hand from the kernel
    /// source's own REPS/OBJ constants (32 * 64 i32 elements).
    fn r1_buffer_spec() -> BufferSpec {
        const REPS: usize = 32;
        const OBJ: usize = 64;
        BufferSpec {
            buffers: vec![BufferDef {
                name: "outTensor".to_string(),
                group_id: 3,
                size_elements: REPS * OBJ,
                element_type: ElementType::I32,
                direction: BufferDir::Output,
                input_pattern: InputPattern::Zeros,
            }],
            multi_kernel: false,
        }
    }

    #[test]
    #[ignore] // requires the Task-4 xclbin built under mlir-aie/build/...
    fn r1_emu_recover_matches_injected() {
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let xclbin = manifest.join("../mlir-aie/build/test/npu-xrt/sp5_skew_r1/chess/aie.xclbin");
        let geom = manifest.join("../mlir-aie/test/npu-xrt/sp5_skew_r1/geometry.json");
        if !xclbin.exists() {
            eprintln!("SKIP r1_emu_recover_matches_injected: chess xclbin not built at {}", xclbin.display());
            return;
        }
        let (d_v, core_off, mem_off) = (3u8, 4u8, 2u8); // contrast = core-mem = +2
        let injected = BroadcastTiming {
            per_hop_horizontal: 0,
            per_hop_vertical: d_v,
            intra_tile_core_offset: core_off,
            intra_tile_mem_offset: mem_off,
            calibrated: false,
        };
        let suite = XclbinSuite::new();
        let dir = tempfile::tempdir().unwrap();

        let run = |bt: Option<BroadcastTiming>, name: &str| -> std::path::PathBuf {
            let test = XclbinTest::from_path(&xclbin)
                .with_buffer_spec(r1_buffer_spec())
                .with_broadcast_timing_override(bt);
            let (outcome, _out, trace) = suite.run_single_with_trace(&test);
            let bytes =
                trace.unwrap_or_else(|| panic!("in-process run produced no trace (outcome={:?})", outcome));
            let p = dir.path().join(name);
            std::fs::File::create(&p).unwrap().write_all(&bytes).unwrap();
            p
        };
        let inj = run(Some(injected.clone()), "injected.bin");
        let zero = run(None, "zero.bin");

        let status = Command::new("python")
            .arg(manifest.join("tools/calibration/skew/r1_emu_recover.py"))
            .arg(&inj)
            .arg(&zero)
            .arg(&geom)
            .arg(d_v.to_string())
            .arg(((core_off as i32) - (mem_off as i32)).to_string())
            .env("PYTHONPATH", manifest.join("tools"))
            .status()
            .expect("run r1_emu_recover.py");
        assert!(status.success(), "recovered != injected (see harness JSON on stdout above)");
    }
}
