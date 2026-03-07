//! NPU Architecture Graph -- standalone extraction and query tool.

use std::env;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: npu-graph <command> [options]");
        eprintln!("Commands:");
        eprintln!("  extract  --arch <aie|aie2|aie2p> [source options]");
        eprintln!("  query    --model <path.json> <query>");
        eprintln!("  diff     --a <model_a.json> --b <model_b.json>");
        process::exit(1);
    }

    match args[1].as_str() {
        "extract" => {
            eprintln!("extract: not yet implemented");
            process::exit(1);
        }
        "query" => {
            eprintln!("query: not yet implemented");
            process::exit(1);
        }
        "diff" => {
            eprintln!("diff: not yet implemented");
            process::exit(1);
        }
        other => {
            eprintln!("Unknown command: {}", other);
            process::exit(1);
        }
    }
}
