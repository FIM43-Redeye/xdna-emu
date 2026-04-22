//! AIE2 VMAC crossbar routing data + evaluation entry points.
//!
//! Moved from `src/interpreter/execute/vmac_routing.rs` as part of
//! Subsystem 7 Part B (audit item 1). Pure static data probed from
//! the AMD C++ ISS: 789 active m-bits, 15,808 route entries
//! (PRMX tables), 26 PRMY pmode expansions.
//!
//! The routing data is purely arch-specific: any AIE1 or AIE2P port
//! would ship its own similar-sized probed table alongside. The
//! evaluation algorithms (`eval_prmx`, `eval_prmy`) are arch-generic
//! shift-and-mask logic that operates on whatever table data is
//! present -- so only the data moves; consumers in xdna-emu reach
//! them through this crate boundary.

#[allow(unused, unused_parens)]
mod routing;

pub use routing::{eval_prmx, eval_prmy};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_prmx_callable_smoke() {
        // Smoke test: the eval_prmx entry point exists, has the
        // expected signature, and returns without panic on an
        // all-zero input. Content-level correctness is exercised
        // by vmac_hw.rs tests on the xdna-emu side.
        let input = [0u8; 128];
        let mbits = [0u64; 13];
        let out = eval_prmx(&input, &mbits);
        // All-zero m-bits select nothing; output is implementation-
        // defined but the call must return a 512-byte array.
        assert_eq!(out.len(), 512);
    }

    #[test]
    fn eval_prmy_callable_smoke() {
        // Same smoke test for the Y-permute entry point.
        let input = [0u8; 128];
        let out = eval_prmy(&input, 0);
        assert_eq!(out.len(), 512);
    }
}
