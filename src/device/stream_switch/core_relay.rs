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

/// Recovered lock and buffer-contact usage for a compute-core ELF, bounded to
/// the entry function (helpers and init stubs excluded).
pub struct CoreLockUsage {
    /// All lock ops within the entry function, in PC order.
    pub locks: Vec<CoreLockOp>,
    /// All buffer load/store contacts within the entry function, in PC order.
    pub accesses: Vec<CoreBufAccess>,
    /// One-past-last PC of the entry function (exclusive upper bound).
    ///
    /// Set to the PC of the first `Ret` that terminates the top-level entry
    /// function.  Lock ops and buffer accesses with `pc >= fn_end` are dropped.
    pub fn_end: u32,
    /// Start PCs of every decoded bundle in the scoped range `[text_base, fn_end)`,
    /// in ascending order.
    ///
    /// Used by [`relay_ordered`] / `compute_retire_pc` to measure the release's
    /// delay-slot retire boundary in true bundle-stride (the dense decoded-bundle
    /// stream), rather than the sparse event-only universe.
    pub bundle_pcs: Vec<u32>,
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

// ── Entry-point analysis ───────────────────────────────────────────────────────

/// Analyse a compute-core ELF and return the combined lock + buffer-access
/// usage, bounded to the **entry function**.
///
/// Function scoping uses a Ret-based fallback heuristic (the ELF object is not
/// available here — only the raw `.text` bytes are passed in):
///
/// 1. Decode the full `.text` via [`decode_all`].
/// 2. Collect all lock calls (same scan as [`recover_lock_ops`]).
/// 3. The first `Ret` bundle **at or after** the last lock call's bundle index
///    is the entry-function return.  Everything after that belongs to helper
///    stubs or init code and is excluded.
/// 4. Run [`recover_buffer_accesses`] over the in-scope prefix.
///
/// `fn_end` in the returned [`CoreLockUsage`] is the PC of the first Ret that
/// retires from the entry function (exclusive upper bound for both lock ops and
/// buffer accesses).
///
/// # Note on the ELF symbol table
///
/// `analyze_core_program` takes only `(text, text_base, dec)` — it does not
/// hold the `AieElf` object.  Plumbing the ELF object would require changing
/// every call site.  The Ret-based heuristic is sufficient for Chess-compiled
/// objectFIFO kernels: helpers always follow the body, and the last lock call
/// always precedes the body's Ret.  If a kernel genuinely has no lock calls the
/// function falls back to the very first Ret in `.text` as `fn_end`.
pub fn analyze_core_program(text: &[u8], text_base: u32, dec: &InstructionDecoder) -> CoreLockUsage {
    // ── Decode everything once ─────────────────────────────────────────────
    let all_bundles = decode_all(text, text_base, dec);

    // ── Find helper addresses (needed to identify lock-call bundle indices) ─
    let (acq_addr, rel_addr) = find_helper_addresses(&all_bundles);

    // ── Find the last lock-call bundle index ───────────────────────────────
    // We walk the same call-site detection logic as recover_lock_ops, but we
    // only need the index of the last matching bundle.
    let last_lock_bundle_idx = if acq_addr.is_none() && rel_addr.is_none() {
        None
    } else {
        let mut last: Option<usize> = None;
        for (i, (_, bundle)) in all_bundles.iter().enumerate() {
            let is_lock_call = bundle.active_slots().any(|s| {
                s.semantic == Some(SemanticOp::Call)
                    && s.sources.iter().any(|src| {
                        if let Operand::Immediate(v) = src {
                            let t = *v as u32;
                            Some(t) == acq_addr || Some(t) == rel_addr
                        } else {
                            false
                        }
                    })
            });
            if is_lock_call {
                last = Some(i);
            }
        }
        last
    };

    // ── Find fn_end: first Ret at or after the last lock call ─────────────
    //
    // For kernels with no lock calls, use the very first Ret (conservative).
    // For kernels with lock calls, the Ret that follows them is the body's
    // return; any Ret before them would be in a helper that precedes the body
    // (not typical in Chess-compiled kernels, but handled safely).
    let start_search = last_lock_bundle_idx.unwrap_or(0);
    let fn_end_pc: u32 = all_bundles[start_search..]
        .iter()
        .find_map(|(pc, bundle)| {
            let has_ret = bundle.active_slots().any(|s| s.semantic == Some(SemanticOp::Ret));
            if has_ret {
                Some(*pc)
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            // No Ret found at all (shouldn't happen for valid ELFs) — use end
            // of text as a safe upper bound.
            text_base + text.len() as u32
        });

    // ── Slice to the in-scope prefix ──────────────────────────────────────
    let scoped_bundles: Vec<(u32, VliwBundle)> =
        all_bundles.into_iter().take_while(|(pc, _)| *pc < fn_end_pc).collect();

    // ── Recover lock ops (scoped) ─────────────────────────────────────────
    // IMPORTANT: helper stubs (ACQ/REL) live AFTER fn_end, so
    // find_helper_addresses over scoped_bundles would return (None, None).
    // We must reuse the helper addresses found in the full-ELF scan (acq_addr,
    // rel_addr) when resolving call sites in the scoped prefix.
    let mut locks: Vec<CoreLockOp> = Vec::new();
    for (i, (pc, bundle)) in scoped_bundles.iter().enumerate() {
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

        if let Some(lock_id) = resolve_r0_in_window(&scoped_bundles, i) {
            locks.push(CoreLockOp { lock_id, kind, pc: *pc });
        }
    }

    // ── Recover buffer accesses (scoped) ──────────────────────────────────
    let accesses = recover_buffer_accesses(&scoped_bundles);

    // ── Collect scoped bundle start PCs (for bundle-stride retire) ────────────
    let bundle_pcs: Vec<u32> = scoped_bundles.iter().map(|(pc, _)| *pc).collect();

    CoreLockUsage { locks, accesses, fn_end: fn_end_pc, bundle_pcs }
}

/// Ordering proxy: confirms that within the entry function the data-relay
/// protocol follows the correct acquire → load → store → release order.
///
/// Returns `true` iff there exist:
/// - An `Acquire(l_in)` at `pc_a`
/// - An `in1` LOAD (`local_off ∈ in_off`) at `pc_l`
/// - An `out1` STORE (`local_off ∈ out_off`) at `pc_s`
/// - A `Release(l_out)` at `pc_r`
///
/// satisfying:
/// ```text
/// pc_a < pc_l < pc_s  AND  pc_s <= retire(pc_r)
/// ```
///
/// where `retire(pc_r)` is the PC of the bundle that comes immediately after
/// the ≤`DELAY_SLOT_WINDOW − 1` delay-slot bundles following the release call
/// bundle at `pc_r`.
///
/// # Why the delay-slot retire boundary matters
///
/// A Chess `JL <rel_helper>` (the release call) may *issue* before the final
/// out1 stores that live in its delay slots.  Using `pc_s < pc_r` (the naive
/// check) would falsely reject valid orderings where stores are emitted in the
/// call's delay slots.  Instead we allow `pc_s <= retire(pc_r)`, accepting
/// stores that are fully retired before the call's architectural commit.
///
/// # Scoping assumption
///
/// This function assumes `u` was produced by [`analyze_core_program`], which
/// already scopes `locks`, `accesses`, and `bundle_pcs` to `[text_base, fn_end)`.
/// No additional bounds check is performed here; callers must not pass a
/// `CoreLockUsage` constructed with PCs outside the intended function range.
pub fn relay_ordered(u: &CoreLockUsage, l_in: u8, l_out: u8, in_off: &[u32], out_off: &[u32]) -> bool {
    // Collect candidate PCs.
    let acq_pcs: Vec<u32> = u
        .locks
        .iter()
        .filter(|op| op.kind == CoreLockKind::Acquire && op.lock_id == l_in)
        .map(|op| op.pc)
        .collect();

    let rel_ops: Vec<&CoreLockOp> = u
        .locks
        .iter()
        .filter(|op| op.kind == CoreLockKind::Release && op.lock_id == l_out)
        .collect();

    let load_pcs: Vec<u32> = u
        .accesses
        .iter()
        .filter(|a| !a.is_store && in_off.contains(&a.local_off))
        .map(|a| a.pc)
        .collect();

    let store_pcs: Vec<u32> = u
        .accesses
        .iter()
        .filter(|a| a.is_store && out_off.contains(&a.local_off))
        .map(|a| a.pc)
        .collect();

    // Try all combinations; succeed on the first valid one.
    for &pc_a in &acq_pcs {
        for rel_op in &rel_ops {
            let pc_r = rel_op.pc;

            // retire(pc_r): first bundle PC after the delay-slot window
            // (computed via bundle-stride u.bundle_pcs; see compute_retire_pc).
            let retire_pc = compute_retire_pc(u, pc_r);

            for &pc_l in &load_pcs {
                for &pc_s in &store_pcs {
                    // Core ordering predicate.
                    if pc_a < pc_l && pc_l < pc_s && pc_s <= retire_pc {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Compute `retire(pc_r)`: the PC of the bundle immediately after the ≤6
/// delay-slot bundles that follow a release call at `pc_r`.
///
/// Strategy: use the dense decoded-bundle start PCs in `u.bundle_pcs` (the full
/// scoped bundle stream, not just event PCs).  Find `pc_r`'s index in that
/// sorted list and return `bundle_pcs[idx + DELAY_SLOT_WINDOW]` — the first
/// bundle outside the `[pc_r .. pc_r + 6 delay slots]` window.
///
/// `DELAY_SLOT_WINDOW = 7` means the window covers the call bundle itself
/// (index `idx`) plus up to 6 delay-slot bundles (indices `idx+1 .. idx+6`),
/// matching the `bundles[call_idx .. call_idx + DELAY_SLOT_WINDOW]` range
/// used by `resolve_r0_in_window`.  Retire is therefore the bundle at
/// `idx + DELAY_SLOT_WINDOW`.
///
/// If there are fewer than `DELAY_SLOT_WINDOW` bundles remaining after `pc_r`
/// (i.e., the release is near the end of the function), returns `u32::MAX`,
/// which conservatively accepts any trailing store.
fn compute_retire_pc(u: &CoreLockUsage, pc_r: u32) -> u32 {
    // u.bundle_pcs is the sorted list of all decoded-bundle start PCs within
    // the scoped range [text_base, fn_end).  Using it — not the sparse event
    // universe — ensures the retire boundary is measured in true bundle-stride.
    let pcs = &u.bundle_pcs;

    // Find the index of the release call bundle.
    let idx = match pcs.binary_search(&pc_r) {
        Ok(i) => i,
        Err(_) => return u32::MAX, // pc_r not in decoded stream — conservative accept
    };

    // retire = first bundle after the call + ≤6 delay-slot window.
    // Mirrors resolve_r0_in_window: that function walks bundles[call_idx..call_idx+7],
    // so the retire boundary is the bundle at call_idx + DELAY_SLOT_WINDOW.
    let retire_idx = idx + DELAY_SLOT_WINDOW;
    if retire_idx < pcs.len() {
        pcs[retire_idx]
    } else {
        u32::MAX // window extends past end of function — accept any trailing store
    }
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

/// Aggregate lock-order fact for the instruction-event layer.
///
/// Returns `(min acquire PC, min release PC)` iff the first acquire precedes the
/// first release in program order, else `None` (safe false-negative). Aggregate
/// across lock IDs because the `INSTR_LOCK_ACQUIRE_REQ` / `INSTR_LOCK_RELEASE_REQ`
/// trace events are lock-ID-agnostic (they fire on every acquire / release). The
/// orientation is DERIVED from the ELF-decoded PCs, not assumed.
pub fn aggregate_lock_order(usage: &CoreLockUsage) -> Option<(u32, u32)> {
    let min_acq = usage
        .locks
        .iter()
        .filter(|o| o.kind == CoreLockKind::Acquire)
        .map(|o| o.pc)
        .min()?;
    let min_rel = usage
        .locks
        .iter()
        .filter(|o| o.kind == CoreLockKind::Release)
        .map(|o| o.pc)
        .min()?;
    (min_acq < min_rel).then_some((min_acq, min_rel))
}

// ── CoreLockRelay edge builder ─────────────────────────────────────────────────

/// Emit intra-tile through-core relay edges for a single compute tile.
///
/// A CoreLockRelay edge claims **structural data-contact under producer/consumer
/// lock ordering**: the compute core program had the opportunity to relay bytes
/// from an S2MM input buffer to an MM2S output buffer, as witnessed by the
/// three-way intersection:
///
/// 1. **Lock-pairing**: the S2MM channel's BD releases a data-ready lock
///    (`release_value > 0`) that the core ACQUIRES, AND the core RELEASES a
///    lock that the MM2S channel's BD acquires.
/// 2. **Buffer-contact**: the core's ELF contains a LOAD whose `local_off` lies
///    within the S2MM channel's buffer range, AND a STORE whose `local_off` lies
///    within the MM2S channel's buffer range.
/// 3. **Ordering**: `relay_ordered` confirms acquire→load→store→release ordering
///    with delay-slot-aware retire boundary.
///
/// This is **NOT value-dependence** — the trace oracle cannot witness value flow
/// through the core. The edge claims only structural opportunity.
///
/// # Orientation
///
/// `src = S2MM master DMA port (writer)` → `dst = MM2S slave DMA port (reader)`.
/// The reverse (back-pressure) is never emitted. See `dma_lock_pair_edges` for
/// the same convention.
///
/// # Coverage
///
/// Chess-compiled objectFIFO passthrough / simple-elementwise kernels.
/// Peano coverage is a documented follow-on. Unresolvable lock id, buffer
/// pointer, or ordering → no edge (safe false-negative).
pub fn core_lock_relay_edges(
    tile: &crate::device::tile::Tile,
    dma: &crate::device::dma::DmaEngine,
    s2mm_count: usize,
    usage: &CoreLockUsage,
) -> Vec<crate::device::stream_switch::route_graph::RouteEdge> {
    use crate::device::dma::engine::LockTarget;
    use crate::device::dma::ChannelType;
    use crate::device::stream_switch::route_graph::{EdgeKind, PortDir, PortRef, RouteEdge};

    let col = tile.col;
    let row = tile.row;

    // Helper: resolve a raw cross-tile lock id to its Own-tile local index.
    // Only same-tile (Own) locks are considered; cross-tile is out of scope.
    let own_local = |raw: u8| -> Option<u8> {
        match dma.resolve_lock_id(raw) {
            Some(LockTarget::Own(id)) => Some(id),
            _ => None,
        }
    };

    // Helper: collect the half-open byte range [base_addr, base_addr + length)
    // for each BD in a flat channel's configured BD chain.
    // Returns (local_offset_set, range_list): local_offset_set for fast contains
    // checks (base_addr & 0xFFFF per BD), range_list for use in relay_ordered.
    let start_bd_field = crate::device::stream_switch::route_graph::start_bd_field_for(tile);
    let channel_buf_info = |flat_ch: usize| -> (Vec<u32>, Vec<u64>) {
        let bds =
            crate::device::stream_switch::route_graph::channel_bd_chain(tile, dma, start_bd_field, flat_ch);
        let mut offsets = Vec::new();
        let mut addrs = Vec::new();
        for bd_id in bds {
            let Some(bd) = dma.get_bd(bd_id) else { continue };
            if bd.length > 0 {
                let local_off = (bd.base_addr & 0xFFFF) as u32;
                offsets.push(local_off);
                addrs.push(bd.base_addr);
            }
        }
        (offsets, addrs)
    };

    // Helper: find the first BD in a flat channel's chain that has a release
    // with release_value > 0 and return the resolved Own-local lock id.
    let first_release_lock = |flat_ch: usize| -> Option<u8> {
        let bds =
            crate::device::stream_switch::route_graph::channel_bd_chain(tile, dma, start_bd_field, flat_ch);
        for bd_id in bds {
            let Some(bd) = dma.get_bd(bd_id) else { continue };
            if bd.release_value > 0 {
                if let Some(raw) = bd.release_lock {
                    if let Some(local) = own_local(raw) {
                        return Some(local);
                    }
                }
            }
        }
        None
    };

    // Helper: find the first BD in a flat channel's chain that has an acquire
    // lock and return the resolved Own-local lock id.
    let first_acquire_lock = |flat_ch: usize| -> Option<u8> {
        let bds =
            crate::device::stream_switch::route_graph::channel_bd_chain(tile, dma, start_bd_field, flat_ch);
        for bd_id in bds {
            let Some(bd) = dma.get_bd(bd_id) else { continue };
            if let Some(raw) = bd.acquire_lock {
                if let Some(local) = own_local(raw) {
                    return Some(local);
                }
            }
        }
        None
    };

    let num_channels = dma.num_channels();
    let mut edges: Vec<RouteEdge> = Vec::new();
    let mut seen_pairs: std::collections::HashSet<(u8, u8)> = std::collections::HashSet::new();

    for s2mm_flat in 0..num_channels {
        if ChannelType::from_channel_index(s2mm_flat, s2mm_count) != ChannelType::S2MM {
            continue;
        }
        // S2MM channel releases l_in (the data-ready lock the core acquires).
        let Some(l_in) = first_release_lock(s2mm_flat) else {
            continue;
        };
        // Core must acquire l_in.
        if !usage
            .locks
            .iter()
            .any(|op| op.kind == CoreLockKind::Acquire && op.lock_id == l_in)
        {
            continue;
        }
        let (in_offsets, _in_addrs) = channel_buf_info(s2mm_flat);
        if in_offsets.is_empty() {
            continue;
        }

        let s2mm_ch = s2mm_flat as u8;
        let Some(src_port) = tile.stream_switch.dma_master(s2mm_ch) else {
            continue;
        };

        for mm2s_flat in 0..num_channels {
            if ChannelType::from_channel_index(mm2s_flat, s2mm_count) != ChannelType::MM2S {
                continue;
            }
            // MM2S channel acquires l_out (the lock the core releases).
            let Some(l_out) = first_acquire_lock(mm2s_flat) else {
                continue;
            };
            // Core must release l_out.
            if !usage
                .locks
                .iter()
                .any(|op| op.kind == CoreLockKind::Release && op.lock_id == l_out)
            {
                continue;
            }
            let (out_offsets, _out_addrs) = channel_buf_info(mm2s_flat);
            if out_offsets.is_empty() {
                continue;
            }

            // Buffer-contact check: must have a LOAD at a local_off in in_offsets
            // AND a STORE at a local_off in out_offsets.
            let has_load = usage.accesses.iter().any(|a| !a.is_store && in_offsets.contains(&a.local_off));
            let has_store = usage.accesses.iter().any(|a| a.is_store && out_offsets.contains(&a.local_off));
            if !has_load || !has_store {
                continue;
            }

            // Ordering check: acquire(l_in) → load(in_buf) → store(out_buf) → retire(release(l_out)).
            if !relay_ordered(usage, l_in, l_out, &in_offsets, &out_offsets) {
                continue;
            }

            let mm2s_ch = (mm2s_flat - s2mm_count) as u8;
            let Some(dst_port) = tile.stream_switch.dma_slave(mm2s_ch) else {
                continue;
            };

            if !seen_pairs.insert((src_port.index, dst_port.index)) {
                continue;
            }
            edges.push(RouteEdge {
                src: PortRef {
                    col,
                    row,
                    port: src_port.index,
                    dir: PortDir::Master,
                    kind: src_port.port_type.as_kind_str().to_owned(),
                },
                dst: PortRef {
                    col,
                    row,
                    port: dst_port.index,
                    dir: PortDir::Slave,
                    kind: dst_port.port_type.as_kind_str().to_owned(),
                },
                kind: EdgeKind::CoreLockRelay,
            });
        }
    }

    edges
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

    /// Synthetic `CoreLockUsage` where the out-buffer STORE precedes the in-buffer
    /// LOAD (violating `pc_l < pc_s`).  Proves that `relay_ordered` rejects the
    /// load-after-store mis-ordering independently of the retire boundary.
    fn bad_order_usage() -> CoreLockUsage {
        // Store appears BEFORE the load — violates pc_l < pc_s.
        CoreLockUsage {
            locks: vec![
                CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x100 },
                CoreLockOp { lock_id: 3, kind: CoreLockKind::Release, pc: 0x130 },
            ],
            accesses: vec![
                CoreBufAccess { local_off: 0x440, is_store: true, pc: 0x110 }, // store FIRST
                CoreBufAccess { local_off: 0x400, is_store: false, pc: 0x120 }, // load AFTER
            ],
            fn_end: 0x200,
            bundle_pcs: vec![
                0x100, 0x110, 0x120, 0x130, 0x140, 0x150, 0x160, 0x170, 0x180, 0x190, 0x1A0, 0x1B0, 0x1C0,
                0x1D0, 0x1E0, 0x1F0,
            ],
        }
    }

    /// Synthetic `CoreLockUsage` where acquire/load/store ordering is correct
    /// (`pc_a < pc_l < pc_s`) but the out-buffer STORE is placed AFTER the strict
    /// bundle-stride retire boundary of the release.  Proves that the retire clause
    /// of `relay_ordered` actually rejects — not the `pc_l < pc_s` check.
    ///
    /// Layout:
    /// ```text
    /// bundle_pcs: [0x100, 0x110, 0x120, 0x130, ..., 0x1A0]  (17 bundles, stride 0x10)
    /// 0x100  Acquire(lock=1)                      idx=0
    /// 0x110  Load  local_off=0x400               idx=1
    /// 0x120  Release(lock=3)   pc_r              idx=2
    ///        retire = bundle_pcs[2 + 7] = bundle_pcs[9] = 0x190
    /// 0x1A0  Store local_off=0x440               idx=10  (> 0x190 = retire)
    /// ```
    /// `pc_a(0x100) < pc_l(0x110) < pc_s(0x1A0)` holds, but `pc_s(0x1A0) > retire(0x190)`.
    fn retire_boundary_bad_usage() -> CoreLockUsage {
        // Dense bundle stream: 17 bundles at 0x100..=0x1A0, stride 0x10.
        let bundle_pcs: Vec<u32> = (0u32..17).map(|i| 0x100 + i * 0x10).collect();
        CoreLockUsage {
            locks: vec![
                CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x100 },
                CoreLockOp { lock_id: 3, kind: CoreLockKind::Release, pc: 0x120 },
            ],
            accesses: vec![
                CoreBufAccess { local_off: 0x400, is_store: false, pc: 0x110 }, // load
                CoreBufAccess { local_off: 0x440, is_store: true, pc: 0x1A0 },  // store past retire
            ],
            fn_end: 0x200,
            bundle_pcs,
        }
    }

    /// Ordering proxy: confirms the delay-slot-aware acquire→load→store→release
    /// ordering for the add_one_using_dma Chess-compiled ELF.
    ///
    /// In1 buffer: local_off 0x400 / 0x420 (both unrolled loop iterations).
    /// Out1 buffer: local_off 0x440 / 0x460.
    /// Lock pair: Acquire(1) → Release(3).
    ///
    /// The test verifies both the positive (real ELF) and negative (synthetic
    /// bad ordering) cases.  The ELF fixture is skipped if absent.
    #[test]
    fn add_one_relay_ordered() {
        let Some((text, base)) = load_core_text(ELF_PATH) else {
            println!("SKIP: ELF fixture not present at {}", ELF_PATH);
            return;
        };

        let dec = InstructionDecoder::load_cached();
        let u = analyze_core_program(&text, base, &dec);

        // fn_end must be set and non-zero.
        assert!(u.fn_end > base, "fn_end should be past text_base");
        // fn_end (the Ret PC) must be before the helper stubs.
        // The helpers start at 0x330; fn_end should be at 0x31A or similar.
        // We check it's below 0x330 without hardcoding the exact PC.
        assert!(
            u.fn_end < base + 0x330,
            "fn_end={:#x} should precede helper stubs (base+0x330={:#x})",
            u.fn_end,
            base + 0x330
        );

        // Scoped lock ops must still contain the expected acquire/release ids.
        let acq_ids: BTreeSet<u8> = u
            .locks
            .iter()
            .filter(|o| matches!(o.kind, CoreLockKind::Acquire))
            .map(|o| o.lock_id)
            .collect();
        let rel_ids: BTreeSet<u8> = u
            .locks
            .iter()
            .filter(|o| matches!(o.kind, CoreLockKind::Release))
            .map(|o| o.lock_id)
            .collect();
        assert_eq!(acq_ids, BTreeSet::from([1u8, 2u8]), "scoped acquires mismatch");
        assert_eq!(rel_ids, BTreeSet::from([0u8, 3u8]), "scoped releases mismatch");

        // Scoped accesses must include in1 loads and out1 stores.
        assert!(
            u.accesses
                .iter()
                .any(|a| !a.is_store && (a.local_off == 0x400 || a.local_off == 0x420)),
            "no in1 load in scoped accesses"
        );
        assert!(
            u.accesses
                .iter()
                .any(|a| a.is_store && (a.local_off == 0x440 || a.local_off == 0x460)),
            "no out1 store in scoped accesses"
        );

        // Positive ordering check.
        assert!(
            relay_ordered(&u, 1, 3, &[0x400, 0x420], &[0x440, 0x460]),
            "add_one acq1..load..store..rel3 ordered"
        );

        // Negative check: a usage with store before load must be rejected.
        assert!(
            !relay_ordered(&bad_order_usage(), 1, 3, &[0x400], &[0x440]),
            "bad_order_usage should NOT pass relay_ordered"
        );

        // Retire-boundary negative: acquire/load/store ordering is correct
        // (pc_a < pc_l < pc_s), but the store falls strictly outside the
        // bundle-stride retire window of the release.  Proves the retire
        // clause of relay_ordered actually fires, not just the pc_l < pc_s guard.
        //
        // retire(pc_r=0x120) = bundle_pcs[idx(0x120) + 7] = bundle_pcs[2+7] = bundle_pcs[9] = 0x190.
        // pc_s = 0x1A0 > 0x190 = retire → relay_ordered must return false.
        assert!(
            !relay_ordered(&retire_boundary_bad_usage(), 1, 3, &[0x400], &[0x440]),
            "store past retire boundary should NOT pass relay_ordered"
        );
    }

    #[test]
    fn aggregate_lock_order_acquire_before_release() {
        let usage = CoreLockUsage {
            locks: vec![
                CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x134 },
                CoreLockOp { lock_id: 0, kind: CoreLockKind::Release, pc: 0x1b4 },
            ],
            accesses: vec![],
            fn_end: 0x300,
            bundle_pcs: vec![],
        };
        assert_eq!(aggregate_lock_order(&usage), Some((0x134, 0x1b4)));
    }

    #[test]
    fn aggregate_lock_order_uses_min_pcs_across_lock_ids() {
        // Aggregate: min acquire (0x134 on lock 1) precedes min release (0x1b4 on lock 0),
        // even though no single lock has both an acquire and a release.
        let usage = CoreLockUsage {
            locks: vec![
                CoreLockOp { lock_id: 2, kind: CoreLockKind::Acquire, pc: 0x150 },
                CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x134 },
                CoreLockOp { lock_id: 3, kind: CoreLockKind::Release, pc: 0x2c0 },
                CoreLockOp { lock_id: 0, kind: CoreLockKind::Release, pc: 0x1b4 },
            ],
            accesses: vec![],
            fn_end: 0x300,
            bundle_pcs: vec![],
        };
        assert_eq!(aggregate_lock_order(&usage), Some((0x134, 0x1b4)));
    }

    #[test]
    fn aggregate_lock_order_none_when_release_first() {
        let usage = CoreLockUsage {
            locks: vec![
                CoreLockOp { lock_id: 0, kind: CoreLockKind::Release, pc: 0x100 },
                CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x200 },
            ],
            accesses: vec![],
            fn_end: 0x300,
            bundle_pcs: vec![],
        };
        assert_eq!(aggregate_lock_order(&usage), None);
    }

    #[test]
    fn aggregate_lock_order_none_when_kind_missing() {
        let usage = CoreLockUsage {
            locks: vec![CoreLockOp { lock_id: 1, kind: CoreLockKind::Acquire, pc: 0x100 }],
            accesses: vec![],
            fn_end: 0x300,
            bundle_pcs: vec![],
        };
        assert_eq!(aggregate_lock_order(&usage), None);
    }
}
