//! One-shot generator for the committed Axis-2 artifacts. Re-run after any
//! change that alters coverage rollups; commit the regenerated files.
fn main() {
    use xdna_archspec::coverage::artifacts::{
        render_architecture_index, render_comprehension, render_implementation_gaps, render_perishable,
        render_subsystem_index,
    };
    use xdna_archspec::types::Architecture;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let dir = root.join("docs/coverage/aie2");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("perishable-queue.md"), render_perishable(Architecture::Aie2)).unwrap();
    std::fs::write(dir.join("comprehension-gaps.md"), render_comprehension(Architecture::Aie2)).unwrap();
    std::fs::write(dir.join("architecture-index.md"), render_architecture_index(Architecture::Aie2)).unwrap();
    std::fs::write(dir.join("subsystem-index.md"), render_subsystem_index(Architecture::Aie2)).unwrap();
    std::fs::write(dir.join("implementation-gaps.md"), render_implementation_gaps(Architecture::Aie2))
        .unwrap();
    eprintln!(
        "wrote docs/coverage/aie2/{{perishable-queue,comprehension-gaps,architecture-index,subsystem-index,implementation-gaps}}.md"
    );
}
