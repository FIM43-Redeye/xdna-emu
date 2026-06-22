//! Static recovery of lock acquire/release ops from a Chess-compiled compute-core ELF.
//!
//! # Purpose
//!
//! Provides `recover_lock_ops`, which linearly decodes the `.text` section of a
//! compute-core ELF and identifies every lock acquire/release via the Chess
//! `JL -> ACQ/REL helper` call idiom.  The recovered ops feed the
//! `program_path` predicate (issue #140) that identifies through-core dataflow
//! relays.
//!
//! # Idiom
//!
//! Chess compiles lock operations as:
//! 1. A `JL <helper>` (Call with Immediate target) to one of two helper
//!    stubs that contain the lone `LockAcquire` / `LockRelease` instructions.
//! 2. In the delay-slot window following the call (up to 6 bundles), a
//!    `Copy dest=ScalarReg(0) sources=[Immediate(N)]` sets `r0` to
//!    `N = 48 + local_lock_id`.
//!
//! Helper addresses are found by semantic scan (the lone bundles containing
//! `LockAcquire` / `LockRelease`), never hardcoded.
//!
//! # Soundness
//!
//! If no `Copy -> r0` with a recognisable immediate is found in the delay-slot
//! window, the call is silently skipped (safe false-negative).  We never guess.
//!
//! # Coverage
//!
//! Covers Chess-compiled objectFIFO / simple-elementwise kernels.
//! Peano coverage is documented follow-up.

use crate::interpreter::bundle::Operand;
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::traits::Decoder;
use xdna_archspec::aie2::isa::SemanticOp;

/// Raw lock-id base: Chess passes `48 + local_id` in `r0`. Local id = raw − 48.
const LOCK_ID_BASE: i32 = 48;

/// Delay-slot window size (number of bundles to scan after the call bundle,
/// inclusive).  AIE2 has 5 branch delay slots; 7 bundles total is conservative.
const DELAY_SLOT_WINDOW: usize = 7;

// ── Public types ───────────────────────────────────────────────────────────────

/// Whether a lock was acquired or released at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreLockKind {
    Acquire,
    Release,
}

/// A single lock acquire or release recovered from static analysis of `.text`.
#[derive(Debug, Clone)]
pub struct CoreLockOp {
    /// Local lock id (0-3 for a compute tile; N − 48 from the Chess `r0` idiom).
    pub lock_id: u8,
    /// Whether this is an acquire or release.
    pub kind: CoreLockKind,
    /// PC of the `JL <helper>` call instruction.
    pub pc: u32,
}

/// Recovered lock usage for a compute-core ELF.
///
/// Extended by later tasks (buffer contacts, function boundary, ordering).
pub struct CoreLockUsage {
    /// All lock ops, in PC order.
    pub locks: Vec<CoreLockOp>,
}

// ── Core algorithm ─────────────────────────────────────────────────────────────

/// Recover all lock acquire/release ops from a `.text` byte slice.
///
/// - `text`:      raw `.text` bytes (from `AieElf::text_section()`).
/// - `text_base`: virtual address of the first byte (from `AieElf::text_address()`).
/// - `dec`:       instruction decoder (caller creates once, reuses across calls).
///
/// Returns ops in PC order.  Unresolvable call sites are silently skipped.
pub fn recover_lock_ops(text: &[u8], text_base: u32, dec: &InstructionDecoder) -> Vec<CoreLockOp> {
    // ── Pass 1: linear decode ──────────────────────────────────────────────────
    let bundles = linear_decode(text, text_base, dec);

    // ── Pass 2: semantic scan for helper addresses ─────────────────────────────
    let (acq_addr, rel_addr) = find_helper_addresses(&bundles);

    // If neither helper is present this ELF has no lock ops.
    if acq_addr.is_none() && rel_addr.is_none() {
        return Vec::new();
    }

    // ── Pass 3: resolve each call site ────────────────────────────────────────
    let mut ops = Vec::new();
    for (i, (pc, bundle)) in bundles.iter().enumerate() {
        // Find a Call slot with an Immediate target pointing at a helper.
        let call_target = bundle.active_slots().find_map(|s| {
            if s.semantic != Some(SemanticOp::Call) {
                return None;
            }
            s.sources.iter().find_map(|src| {
                if let Operand::Immediate(v) = src {
                    let t = *v as u32;
                    if Some(t) == acq_addr || Some(t) == rel_addr {
                        Some(t)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        });

        let Some(target) = call_target else { continue };
        let kind = if Some(target) == acq_addr {
            CoreLockKind::Acquire
        } else {
            CoreLockKind::Release
        };

        // Resolve r0 from the delay-slot window.
        if let Some(lock_id) = resolve_r0_in_window(&bundles, i) {
            ops.push(CoreLockOp { lock_id, kind, pc: *pc });
        }
        // Unresolvable => skip (soundness rule: false-negative over false-positive).
    }

    ops
}

// ── Internal helpers ───────────────────────────────────────────────────────────

/// Bundle list entry: (pc, bundle).
type BundleList = Vec<(u32, crate::interpreter::bundle::VliwBundle)>;

/// Linearly decode `.text` into a list of `(pc, VliwBundle)`.
///
/// On a decode error, advances by the smallest valid instruction size (2) and
/// continues.  This mirrors what hardware would do if it encountered an
/// unrecognised encoding.
fn linear_decode(text: &[u8], text_base: u32, dec: &InstructionDecoder) -> BundleList {
    let mut bundles = Vec::new();
    let mut off = 0usize;

    while off < text.len() {
        let pc = text_base + off as u32;
        let rest = &text[off..];

        // Use instruction_size first so we always advance correctly even when
        // decode fails (the size result comes from format detection, which
        // tolerates more than full decode).
        let size = match dec.instruction_size(rest) {
            Ok(s) => s as usize,
            Err(_) => {
                off += 2; // minimum bundle size, keep going
                continue;
            }
        };
        if size == 0 {
            break; // shouldn't happen, guard against infinite loop
        }

        match dec.decode(rest, pc) {
            Ok(bundle) => {
                bundles.push((pc, bundle));
            }
            Err(_) => {
                // Decode error: skip this bundle, advance by size.
            }
        }

        off += size;
    }

    bundles
}

/// Scan all decoded bundles for the lone `LockAcquire` / `LockRelease` slots.
///
/// Returns `(acq_addr, rel_addr)` — the PCs of the bundles that contain
/// those semantics (the Chess helper stubs).  Either may be `None` if absent.
fn find_helper_addresses(bundles: &BundleList) -> (Option<u32>, Option<u32>) {
    let mut acq_addr = None;
    let mut rel_addr = None;

    for (pc, bundle) in bundles {
        for slot in bundle.active_slots() {
            match slot.semantic {
                Some(SemanticOp::LockAcquire) => acq_addr = Some(*pc),
                Some(SemanticOp::LockRelease) => rel_addr = Some(*pc),
                _ => {}
            }
        }
    }

    (acq_addr, rel_addr)
}

/// Resolve the local lock id from `r0` in the delay-slot window of a call.
///
/// Scans `bundles[call_idx ..]` for up to `DELAY_SLOT_WINDOW` bundles,
/// stopping early at the next `Call` or `Ret` (not counting the call bundle
/// itself).  Returns the **last** `Copy dest=ScalarReg(0) sources=[Immediate(N)]`
/// found; `local_id = N − LOCK_ID_BASE`.
///
/// Returns `None` if no such Copy is found (unresolvable → caller skips).
fn resolve_r0_in_window(bundles: &BundleList, call_idx: usize) -> Option<u8> {
    let end = (call_idx + DELAY_SLOT_WINDOW).min(bundles.len());
    let mut last_imm: Option<i32> = None;

    for (slot_idx, (_, bundle)) in bundles[call_idx..end].iter().enumerate() {
        // For bundles after the call itself, stop at the next call or return.
        if slot_idx > 0 {
            let has_ctrl = bundle
                .active_slots()
                .any(|s| matches!(s.semantic, Some(SemanticOp::Call) | Some(SemanticOp::Ret)));
            if has_ctrl {
                break;
            }
        }

        for slot in bundle.active_slots() {
            if slot.semantic != Some(SemanticOp::Copy) {
                continue;
            }
            if slot.dest != Some(Operand::ScalarReg(0)) {
                continue;
            }
            if let Some(Operand::Immediate(n)) = slot.sources.first() {
                last_imm = Some(*n);
            }
        }
    }

    let n = last_imm?;
    let local = n - LOCK_ID_BASE;
    if local >= 0 {
        Some(local as u8)
    } else {
        None // negative offset from base — skip
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::AieElf;
    use std::collections::BTreeSet;

    /// Path to the add_one_using_dma Chess-compiled core ELF.
    const ELF_PATH: &str = concat!(
        "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/",
        "add_one_using_dma/chess/aie_arch.mlir.prj/main_core_0_2.elf"
    );

    /// Load `.text` bytes and base address from a core ELF.
    ///
    /// Returns `None` if the file is absent (so the test skips cleanly).
    fn load_core_text(path: &str) -> Option<(Vec<u8>, u32)> {
        let data = std::fs::read(path).ok()?;
        let elf = AieElf::parse(&data).ok()?;
        let text = elf.text_section()?;
        let base = elf.text_address()?;
        Some((text.to_vec(), base))
    }

    #[test]
    fn recovers_add_one_lock_ops() {
        let Some((text, base)) = load_core_text(ELF_PATH) else {
            println!("SKIP: ELF fixture not present at {}", ELF_PATH);
            return;
        };

        let dec = InstructionDecoder::load_cached();
        let ops = recover_lock_ops(&text, base, &dec);

        assert!(!ops.is_empty(), "expected lock ops but recovered none");

        let acq: BTreeSet<u8> = ops
            .iter()
            .filter(|o| matches!(o.kind, CoreLockKind::Acquire))
            .map(|o| o.lock_id)
            .collect();
        let rel: BTreeSet<u8> = ops
            .iter()
            .filter(|o| matches!(o.kind, CoreLockKind::Release))
            .map(|o| o.lock_id)
            .collect();

        assert_eq!(acq, BTreeSet::from([1u8, 2u8]), "acquires mismatch");
        assert_eq!(rel, BTreeSet::from([0u8, 3u8]), "releases mismatch");
    }
}
