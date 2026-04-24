//! Semantics: CdoRaw -> DeviceOp lowering.
//!
//! Half 1 of Subsystem 8 Stage 8b: this module is a pass-through.
//! It accepts CdoRaw and returns CdoRaw unchanged, preserving the
//! parser/device-state interface while the CDO file is split into
//! layers. Half 2 rewrites `lower` to emit DeviceOp (arch-generic
//! device-facing ops), consulting &ArchHandle for address decoding
//! and BD field parsing.
//!
//! See docs/superpowers/specs/2026-04-23-subsys8-parser-design.md
//! §Stage 8b for the two-halves rationale.

use super::syntax::CdoRaw;

/// Pass-through in Half 1 of Stage 8b.
///
/// Half 2 replaces this with a proper lowering that consults an
/// ArchHandle and emits a device-facing op stream.
pub fn lower(raw: CdoRaw) -> CdoRaw {
    raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantics_pass_through_returns_input_unchanged() {
        // CdoRaw does not derive PartialEq, so we verify pass-through
        // via a unit-variant `matches!` on EndMark. The behavioral
        // contract for Half 1 is simply: lower(x) returns x. A richer
        // test will land in Half 2 when `lower` does real work.
        let input = CdoRaw::EndMark;
        let output = lower(input);
        assert!(matches!(output, CdoRaw::EndMark));

        // Sanity check with a parameterized variant: values survive
        // the pass-through.
        let input = CdoRaw::Write { address: 0x1234, value: 0xDEAD_BEEF };
        let output = lower(input);
        match output {
            CdoRaw::Write { address, value } => {
                assert_eq!(address, 0x1234);
                assert_eq!(value, 0xDEAD_BEEF);
            }
            _ => panic!("expected Write variant to survive pass-through"),
        }
    }
}
