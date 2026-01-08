//! Run all xclbin tests from mlir-aie npu-xrt directory.
//!
//! This is a quick test runner to see how many mlir-aie tests pass.

use std::path::PathBuf;
use std::io::{self, Write};
use xdna_emu::testing::xclbin_suite::{XclbinSuite, XclbinTest, TestOutcome};

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("error"),
    )
    .init();

    let mlir_aie_path = PathBuf::from("/home/triple/npu-work/mlir-aie");
    let npu_xrt_path = mlir_aie_path.join("build/test/npu-xrt");

    if !npu_xrt_path.exists() {
        eprintln!("Error: mlir-aie npu-xrt tests not found at {}", npu_xrt_path.display());
        eprintln!("Make sure mlir-aie is built with tests enabled.");
        std::process::exit(1);
    }

    println!("Discovering tests in {}...", npu_xrt_path.display());

    let mut suite = match XclbinSuite::discover(&npu_xrt_path) {
        Ok(s) => s.with_max_cycles(1_000_000), // 1M cycle timeout (reduced for speed)
        Err(e) => {
            eprintln!("Failed to discover tests: {}", e);
            std::process::exit(1);
        }
    };

    // Also look in subdirectories (e.g., adjacent_memtile_access/two_memtiles/)
    if let Ok(entries) = std::fs::read_dir(&npu_xrt_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Check subdirectories for xclbin files
                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub_entry in sub_entries.flatten() {
                        let sub_path = sub_entry.path();
                        if sub_path.is_dir() {
                            for xclbin_name in &["aie.xclbin", "final.xclbin"] {
                                let xclbin_path = sub_path.join(xclbin_name);
                                if xclbin_path.exists() {
                                    suite.add_test(XclbinTest::from_path(&xclbin_path));
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let total = suite.test_count();
    println!("Found {} tests\n", total);

    // Run tests one at a time with progress
    let tests: Vec<_> = suite.tests().to_vec();
    let mut passed = 0;
    let mut failed = 0;
    let mut unknown_count = 0;
    let mut timeout_count = 0;
    let mut load_error = 0;

    for (i, test) in tests.iter().enumerate() {
        print!("[{:2}/{}] {:40} ... ", i + 1, total, &test.name[..test.name.len().min(40)]);
        io::stdout().flush().unwrap();

        // Debug: check for insts.bin and ELFs
        let elf_files = test.find_elf_files();
        if elf_files.is_empty() {
            eprint!("(NO ELFs) ");
        } else {
            eprint!("({} ELFs) ", elf_files.len());
        }
        if test.find_insts_bin().is_none() {
            eprint!("(NO insts.bin) ");
        }

        let outcome = suite.run_single(test);

        match &outcome {
            TestOutcome::Pass { cycles } => {
                println!("PASS ({} cycles)", cycles);
                passed += 1;
            }
            TestOutcome::Fail { message, cycles } => {
                println!("FAIL ({} cycles)", cycles);
                println!("      {}", message);
                failed += 1;
            }
            TestOutcome::UnknownOpcode { details, cycles } => {
                println!("UNKNOWN ({} cycles)", cycles);
                println!("      {:?}", details);
                suite.record_unknown(details.clone(), &test.name);
                unknown_count += 1;
            }
            TestOutcome::Timeout { cycles } => {
                println!("TIMEOUT ({} cycles)", cycles);
                timeout_count += 1;
            }
            TestOutcome::LoadError { message } => {
                println!("LOAD ERROR");
                println!("      {}", message);
                load_error += 1;
            }
        }
    }

    println!("\n{:=<60}", "");
    println!("=== SUMMARY ===");
    println!("Total:      {}", total);
    println!("Passed:     {} ({:.1}%)", passed, 100.0 * passed as f64 / total.max(1) as f64);
    println!("Failed:     {}", failed);
    println!("Unknown:    {}", unknown_count);
    println!("Timeout:    {}", timeout_count);
    println!("Load Error: {}", load_error);

    // Show unknown opcodes if any
    let collector = suite.collector();
    let unknowns = collector.by_impact();
    if !unknowns.is_empty() {
        println!("\n=== UNKNOWN OPCODES (by impact) ===");
        for stats in unknowns.iter().take(10) {
            let op = &stats.first;
            let mnemonic = op.mnemonic.as_deref().unwrap_or("unknown");
            println!("  {:?} opcode 0x{:04X} '{}' (hits: {}, tests: {})",
                     op.slot, op.opcode, mnemonic, stats.count, stats.tests.len());
        }
    }
}
