//! Thread-local executed-vector-op recorder for the vector fuzzer.
//!
//! The fuzzer emits C++ chains; compilers can fold ops away. Correctness
//! credit for a fuzzed stage comes from its silicon-matched output slice;
//! this recorder is the anti-folding sentinel and audit record: it proves
//! the emulator actually executed vector semantics, and captures the set of
//! (SemanticOp, ElementType, mode) keys executed per case for coverage
//! banking. Keys are descriptive, not tied to any dispatch-table naming.
//!
//! Disarmed by default; `record()` costs one thread-local Option check when
//! unarmed, so the hook in [`super::VectorAlu::execute`] is free for normal
//! emulation.

use std::cell::RefCell;
use std::collections::HashSet;

use crate::interpreter::bundle::ElementType;
use xdna_archspec::aie2::isa::SemanticOp;

thread_local! {
    static RECORDED: RefCell<Option<HashSet<String>>> = const { RefCell::new(None) };
}

/// Arm the recorder for the current thread, clearing any previous set.
pub fn arm() {
    RECORDED.with(|r| *r.borrow_mut() = Some(HashSet::new()));
}

/// Take the recorded keys (sorted, deduplicated) and disarm.
///
/// Returns `None` if the recorder was never armed on this thread.
pub fn take() -> Option<Vec<String>> {
    RECORDED.with(|r| {
        r.borrow_mut().take().map(|set| {
            let mut keys: Vec<String> = set.into_iter().collect();
            keys.sort();
            keys
        })
    })
}

/// Record an executed vector op. No-op unless armed.
///
/// Key format: `{semantic:?}/{et:?}/m{mode}` with `NoEt` when the op had no
/// element type. `mode` is `(sat << 4) | rnd` for SRS-config-sensitive ops,
/// 0 otherwise.
pub fn record(semantic: SemanticOp, et: Option<ElementType>, mode: u8) {
    RECORDED.with(|r| {
        if let Some(set) = r.borrow_mut().as_mut() {
            let et_str = match et {
                Some(et) => format!("{et:?}"),
                None => "NoEt".to_string(),
            };
            set.insert(format!("{semantic:?}/{et_str}/m{mode}"));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unarmed_record_noop_take_none() {
        // Never armed on this thread: record is a no-op, take returns None.
        record(SemanticOp::Add, Some(ElementType::Int32), 0);
        assert_eq!(take(), None);
    }

    #[test]
    fn test_arm_record_take_roundtrip_sorted_dedup() {
        arm();
        record(SemanticOp::Sub, Some(ElementType::Int16), 0);
        record(SemanticOp::Add, Some(ElementType::Int32), 0);
        record(SemanticOp::Add, Some(ElementType::Int32), 0); // dup
        record(SemanticOp::Srs, None, 0x13);

        let keys = take().expect("armed recorder must return Some");
        assert_eq!(
            keys,
            vec!["Add/Int32/m0".to_string(), "Srs/NoEt/m19".to_string(), "Sub/Int16/m0".to_string(),]
        );
    }

    #[test]
    fn test_take_disarms() {
        arm();
        record(SemanticOp::Add, Some(ElementType::Int32), 0);
        assert!(take().is_some());
        // Disarmed now: further records ignored, take is None again.
        record(SemanticOp::Sub, Some(ElementType::Int32), 0);
        assert_eq!(take(), None);
    }
}
