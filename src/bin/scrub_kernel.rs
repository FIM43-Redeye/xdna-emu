//! Print the scrub-kernel artifact (pattern, table, kernel source, exclusions) as JSON.
//!
//! The harness reads the checked-in copy; a unit test keeps it current:
//!   cargo run --bin scrub-kernel > tools/harness/scrub_kernel.json

fn main() {
    let doc = xdna_emu::interpreter::state::scrub::artifact_json();
    println!("{}", serde_json::to_string_pretty(&doc).expect("serialize"));
}
