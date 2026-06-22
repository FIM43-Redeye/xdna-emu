//! Static recovery of lock and buffer-contact ops from a Chess-compiled compute-core ELF.
//!
//! # Purpose
//!
//! Provides two public analysis functions over a decoded bundle stream:
//!
//! - [`recover_lock_ops`]: identifies every lock acquire/release via the Chess
//!   `JL -> ACQ/REL helper` call idiom.
//! - [`recover_buffer_accesses`]: identifies every load/store that contacts an
//!   input or output buffer, via a forward abstract-value pass over registers
//!   and stack slots.
//!
//! Both functions consume the bundle stream produced by [`decode_all`], which
//! linearises a `.text` byte slice into `(pc, VliwBundle)` pairs.
//!
//! # Lock Idiom (Chess)
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
//! # Buffer-Contact Pass
//!
//! Chess-compiled objectFIFO kernels move buffer base addresses through a
//! chain: `MOVXM imm -> reg -> ST reg,[sp,#off] -> LDA p,[sp,#off] ->
//! load/store via p`.  A naive pointer-register tracker fails because the
//! pointer never holds the base before the spill — only a modifier or scalar
//! register does.
//!
//! The abstract-value pass tracks a `Val = Known(i32) | Unknown` lattice over
//! all register classes *and* stack slots, so it follows the full chain.
//! PointerReg(255) (sp) and ModifierReg(8) (dn0) are excluded only as
//! address-pointer candidates in contact detection; they remain valid spill
//! value sources (excluding them as values would lose the in1 contact).
//!
//! # Soundness
//!
//! - Lock recovery: if no `Copy -> r0` with a recognisable immediate is found
//!   in the delay-slot window, the call is silently skipped (safe false-negative).
//! - Buffer recovery: single linear forward pass, no join-on-merge widening.
//!   At control-flow merges the predecessor's Known value is kept, so the pass
//!   can over-approximate contacts (spurious contacts only add entries filtered
//!   downstream by the lock∩buffer∩ordering intersection).  This is intentional.
//!
//! # Coverage
//!
//! Chess-compiled objectFIFO / simple-elementwise kernels.
//! Peano coverage is documented follow-up.

use std::collections::HashMap;

use crate::interpreter::bundle::{Operand, VliwBundle};
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

// ── Public types: buffer access ────────────────────────────────────────────────

/// A single load or store that contacts a buffer, recovered from static analysis
/// of `.text`.
///
/// `local_off` is the low 16 bits of the resolved base address (`addr & 0xFFFF`),
/// matching the tile-local address layout used by DMA BDs.
#[derive(Debug, Clone)]
pub struct CoreBufAccess {
    /// Low 16 bits of the resolved base address (buffer's tile-local offset).
    pub local_off: u32,
    /// `true` for a store (write to the buffer), `false` for a load (read).
    pub is_store: bool,
    /// PC of the load/store bundle.
    pub pc: u32,
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
    let bundles = decode_all(text, text_base, dec);

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

// ── Public decode helper ───────────────────────────────────────────────────────

/// Linearly decode a `.text` byte slice into a list of `(pc, VliwBundle)`.
///
/// On a decode error the bundle is skipped and the walk advances by the size
/// reported by `instruction_size` (which tolerates more than full decode).
/// On a size error the walk advances by 2 (minimum bundle size) and continues.
///
/// This is the shared decode pass consumed by both `recover_lock_ops` and
/// `recover_buffer_accesses`.  Callers that need both can call `decode_all`
/// once and pass the result to each analysis function.
pub fn decode_all(text: &[u8], text_base: u32, dec: &InstructionDecoder) -> Vec<(u32, VliwBundle)> {
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

// ── Buffer-contact recovery ────────────────────────────────────────────────────

/// Abstract lattice value for the register + stack-slot tracking pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Val {
    /// Register/slot holds a statically known immediate.
    Known(i32),
    /// Unknown or multiply-defined value.
    Unknown,
}

/// Compact key for the abstract-value map: `(class_tag, number)`.
///
/// `class_tag`: 0 = ScalarReg, 1 = PointerReg, 2 = ModifierReg.
/// This matches the three register classes that can carry buffer bases in
/// Chess-compiled kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RegKey(u8, u8);

/// Extract a `RegKey` from an `Operand` for the classes that can carry pointer
/// values (ScalarReg, PointerReg, ModifierReg).  Returns `None` for everything
/// else (Immediate, Memory, VectorReg, …).
fn reg_key(op: &Operand) -> Option<RegKey> {
    match op {
        Operand::ScalarReg(r) => Some(RegKey(0, *r)),
        Operand::PointerReg(r) => Some(RegKey(1, *r)),
        Operand::ModifierReg(r) => Some(RegKey(2, *r)),
        _ => None,
    }
}

/// Recover all buffer load/store contacts from a pre-decoded bundle stream.
///
/// Runs a single forward abstract-value pass over `(pc, VliwBundle)` pairs,
/// tracking a `Known(i32) | Unknown` lattice over all register classes and
/// stack slots.  Each load or store whose pointer-register base is `Known`
/// (and is not `sp` / `PointerReg(255)`) is recorded as a [`CoreBufAccess`].
///
/// Stack spills (`ST reg,[sp,#off]`) and reloads (`LDA reg,[sp,#off]`) are
/// propagated through the lattice but are NOT recorded as buffer contacts.
///
/// # The dn0 / sp exclusion
///
/// `PointerReg(255)` (sp) and `ModifierReg(8)` (dn0) are excluded **only**
/// from buffer-pointer *address candidates* in contact detection.  They remain
/// valid spill/copy *value* sources — the Chess in1 flow loads the base into
/// `dn0`, spills it via `ST dn0,[sp,-48]`, reloads into `p7` via `LDA
/// p7,[sp,-48]`, and then issues loads via `p7`.  Dropping `dn0` as a value
/// source would lose the entire in1 contact chain.
///
/// # Soundness
///
/// Single linear forward pass with no join-on-merge widening.  At control-flow
/// merges the predecessor's `Known` is kept rather than widened to `Unknown`.
/// This can produce spurious contacts (false positives), which is acceptable
/// because downstream intersection (lock-pair ∩ buffer-contact ∩ ordering)
/// filters them, and over-approximation is superset-safe.
pub fn recover_buffer_accesses(bundles: &[(u32, VliwBundle)]) -> Vec<CoreBufAccess> {
    // Abstract state: registers and stack slots.
    let mut regs: HashMap<RegKey, Val> = HashMap::new();
    // Stack map: stack-slot offset (i16) -> Val.
    let mut stack: HashMap<i16, Val> = HashMap::new();
    let mut contacts: Vec<CoreBufAccess> = Vec::new();

    for (pc, bundle) in bundles {
        let pc = *pc;
        for slot in bundle.active_slots() {
            match slot.semantic {
                // ── Copy (MOVXM / mov): propagate known values ─────────────
                Some(SemanticOp::Copy) => {
                    let Some(dest_op) = &slot.dest else { continue };
                    let Some(dk) = reg_key(dest_op) else { continue };

                    // Immediate source: direct known-value assignment.
                    if let Some(Operand::Immediate(v)) = slot.sources.first() {
                        regs.insert(dk, Val::Known(*v));
                        continue;
                    }

                    // Register source: copy-propagate.
                    // NOTE: ModifierReg(8)/dn0 and PointerReg(255)/sp are
                    // valid VALUE sources here — they are only excluded from
                    // contact-detection (address use), not from value tracking.
                    if let Some(src_op) = slot.sources.first() {
                        if let Some(sk) = reg_key(src_op) {
                            let v = regs.get(&sk).copied().unwrap_or(Val::Unknown);
                            regs.insert(dk, v);
                            continue;
                        }
                    }

                    // Any other Copy form: destination goes Unknown.
                    regs.insert(dk, Val::Unknown);
                }

                // ── Store: spill to stack OR buffer contact ─────────────────
                Some(SemanticOp::Store) => {
                    // Detect spill: ST val_reg,[sp,#off]
                    // In the spike-verified operand shape:
                    //   sources = [val_reg, Memory{base:255, off}]
                    let mem_sp = slot.sources.iter().find_map(|s| {
                        if let Operand::Memory { base: 255, offset } = s {
                            Some(*offset)
                        } else {
                            None
                        }
                    });

                    if let Some(stack_off) = mem_sp {
                        // Spill: propagate val_reg's value into the stack slot.
                        // val_reg is the first non-Memory source.
                        let val_src = slot.sources.iter().find_map(|s| reg_key(s));
                        let v = val_src.and_then(|k| regs.get(&k).copied()).unwrap_or(Val::Unknown);
                        stack.insert(stack_off, v);
                        // Spill is NOT a buffer contact — skip to next slot.
                        continue;
                    }

                    // Buffer contact: Store via PointerReg(p), p != 255 (sp).
                    // Spike-verified shape: dest = Some(PointerReg(p)) for
                    // post-increment stores; PointerReg(p) also appears in
                    // sources (both roles reference the same register).
                    if let Some(Operand::PointerReg(p)) = &slot.dest {
                        let p = *p;
                        if p != 255 {
                            let key = RegKey(1, p);
                            if let Some(Val::Known(addr)) = regs.get(&key).copied() {
                                contacts.push(CoreBufAccess {
                                    local_off: (addr as u32) & 0xFFFF,
                                    is_store: true,
                                    pc,
                                });
                            }
                        }
                    }
                }

                // ── Load: reload from stack OR buffer contact ───────────────
                Some(SemanticOp::Load) => {
                    let Some(dest_op) = &slot.dest else { continue };

                    // Detect stack reload: LDA reg,[sp,#off]
                    let mem_sp = slot.sources.iter().find_map(|s| {
                        if let Operand::Memory { base: 255, offset } = s {
                            Some(*offset)
                        } else {
                            None
                        }
                    });

                    if let Some(stack_off) = mem_sp {
                        // Reload: propagate stack slot value into dest register.
                        if let Some(dk) = reg_key(dest_op) {
                            let v = stack.get(&stack_off).copied().unwrap_or(Val::Unknown);
                            regs.insert(dk, v);
                        }
                        // Reload is NOT a buffer contact — skip to next slot.
                        continue;
                    }

                    // Buffer contact: Load via PointerReg(p), p != 255 (sp).
                    // Spike-verified shape: sources = [PointerReg(p)] for
                    // post-increment loads.
                    let ptr_src = slot.sources.iter().find_map(|s| {
                        if let Operand::PointerReg(p) = s {
                            if *p != 255 {
                                Some(*p)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });

                    if let Some(p) = ptr_src {
                        let key = RegKey(1, p);
                        if let Some(Val::Known(addr)) = regs.get(&key).copied() {
                            contacts.push(CoreBufAccess {
                                local_off: (addr as u32) & 0xFFFF,
                                is_store: false,
                                pc,
                            });
                        }
                        // Whether or not we emitted a contact, the load also
                        // updates the dest register.  For a post-increment load
                        // (the common form in Chess objectFIFO kernels), the
                        // pointer advances by the element stride — unknown to us
                        // statically without modelling post-modify arithmetic.
                        // Mark the dest Unknown.
                        if let Some(dk) = reg_key(dest_op) {
                            regs.insert(dk, Val::Unknown);
                        }
                        continue;
                    }

                    // Any other Load form: dest goes Unknown.
                    if let Some(dk) = reg_key(dest_op) {
                        regs.insert(dk, Val::Unknown);
                    }
                }

                // ── Any other op that writes a tracked register ─────────────
                // (Add, PointerAdd, PointerMov, …): dest → Unknown.
                _ => {
                    if let Some(dest_op) = &slot.dest {
                        if let Some(dk) = reg_key(dest_op) {
                            regs.insert(dk, Val::Unknown);
                        }
                    }
                }
            }
        }
    }

    contacts
}

/// Scan all decoded bundles for the lone `LockAcquire` / `LockRelease` slots.
///
/// Returns `(acq_addr, rel_addr)` — the PCs of the bundles that contain
/// those semantics (the Chess helper stubs).  Either may be `None` if absent.
fn find_helper_addresses(bundles: &[(u32, VliwBundle)]) -> (Option<u32>, Option<u32>) {
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
fn resolve_r0_in_window(bundles: &[(u32, VliwBundle)], call_idx: usize) -> Option<u8> {
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

    /// Buffer-contact oracle for add_one_using_dma (Chess-compiled).
    ///
    /// Verifies:
    /// 1. At least one non-store contact at in1 buffer addresses (0x400 or 0x420).
    /// 2. At least one store contact at out1 buffer addresses (0x440 or 0x460).
    /// 3. No pointer-spill bundles misclassified as buffer contacts:
    ///    the spill ST pX,[sp,#off] bundles at PC 0xEA and 0x1B4 must NOT appear.
    ///
    /// Addresses 0x400/0x420/0x440/0x460 appear only in test assertions;
    /// the analysis code derives them from the ELF, never hardcodes them.
    #[test]
    fn recovers_add_one_buffer_contacts() {
        let Some((text, base)) = load_core_text(ELF_PATH) else {
            println!("SKIP: ELF fixture not present at {}", ELF_PATH);
            return;
        };

        let dec = InstructionDecoder::load_cached();
        let bundles = decode_all(&text, base, &dec);
        let acc = recover_buffer_accesses(&bundles);

        // Must find at least one load at an in1 buffer address.
        assert!(
            acc.iter()
                .any(|a| !a.is_store && (a.local_off == 0x400 || a.local_off == 0x420)),
            "expected at least one in1 load (local_off 0x400 or 0x420); got: {:?}",
            acc.iter().filter(|a| !a.is_store).map(|a| a.local_off).collect::<Vec<_>>()
        );

        // Must find at least one store at an out1 buffer address.
        assert!(
            acc.iter().any(|a| a.is_store && (a.local_off == 0x440 || a.local_off == 0x460)),
            "expected at least one out1 store (local_off 0x440 or 0x460); got: {:?}",
            acc.iter().filter(|a| a.is_store).map(|a| a.local_off).collect::<Vec<_>>()
        );

        // Pointer-spill bundles (ST pX,[sp,#off]) must NOT appear as contacts.
        // PCs 0xEA and 0x1B4 are the Chess-emitted spill sites in add_one.
        assert!(
            !acc.iter().any(|a| a.pc == 0xEA || a.pc == 0x1B4),
            "pointer-spill bundles misclassified as buffer contacts: {:?}",
            acc.iter().filter(|a| a.pc == 0xEA || a.pc == 0x1B4).collect::<Vec<_>>()
        );
    }
}
