//! Fill-loop fast-path: collapse a large zero-overhead `loop` whose body is a
//! contiguous memory fill (a compiler-emitted `memset`/`memclr`) into one bulk
//! operation, instead of interpreting every iteration. This recognizer produces
//! byte-identical architectural state (the
//! filled memory, the advanced pointer, LCOUNT=0, PC=LEND) as the per-iteration
//! interpreter would, in O(range) native work.
//!
//! Recognized shape (store-first contiguous fill), body = exactly two instrs:
//!   LBEG: s{8,16,32}i  val, [ptr + 0]    ; store width w = 1/2/4
//!         addi/addi.n  ptr, ptr, w        ; advance SAME ptr by exactly w
//!   LEND: (loop back-edge)
//! Anything else is not recognized (`None`) and the caller grinds normally.

use super::{Access, Cpu, Step};
use crate::firmware::mmio::Bus;
use crate::firmware::xtensa::decode::{self, Op};

/// Only collapse loops with at least this many remaining iterations; below it,
/// grinding is cheap and not worth the recognizer's setup + validation walk.
pub(super) const MIN_ITERS: u32 = 1024;

/// If the CPU is poised at a fill loop's back-edge (`pc == lbeg`) with a large
/// remaining trip count and a recognized contiguous-fill body, execute the whole
/// fill and return `Some(Step::Ran)`. Returns `None` (caller grinds normally) if
/// the pattern is not recognized or any page in the fill range faults on a
/// `Store` translation.
pub(super) fn try_fill_loop(cpu: &mut Cpu, bus: &mut Bus) -> Option<Step> {
    // Cheap gate: only at a loop start with architectural back-edges enabled
    // and a large remaining trip count.
    if cpu.pc != cpu.regs.lbeg || cpu.regs.lcount < MIN_ITERS || cpu.regs.excm() {
        return None;
    }
    let lbeg = cpu.regs.lbeg;
    let lend = cpu.regs.lend;

    // Body must be exactly {store [ptr+0]; addi ptr,ptr,w}.
    let (w, val_reg, ptr_reg) = decode_fill_body(cpu, bus, lbeg, lend)?;

    // The value register must be constant across iterations (the body only
    // modifies ptr_reg via the addi); a store of the pointer itself is not a
    // constant fill.
    if val_reg == ptr_reg {
        return None;
    }

    // N = remaining iterations = LCOUNT + 1 (this pass plus the LCOUNT back-
    // edges still to come). u64 so N*w cannot overflow.
    let n = cpu.regs.lcount as u64 + 1;
    let start = cpu.regs.read_ar(ptr_reg);
    if start % w as u32 != 0 {
        return None; // unaligned base: leave to the normal (possibly faulting) path
    }
    let total = n * w as u64; // total bytes to fill

    // Fill pattern = low w bytes of the value register, little-endian store order.
    let val = cpu.regs.read_ar(val_reg);
    let pat_full = val.to_le_bytes();
    let pattern = &pat_full[..w as usize];

    // Single forward pass: translate each page-bounded chunk via the NON-raising
    // MMU path (`cpu.mmu.translate`, not `cpu.translate`) and fill it. Using the
    // MMU-level call has two purposes: (1) it never raises, so a speculative
    // probe cannot corrupt pc/epc1/exccause; (2) translating each page exactly
    // once matches grinding's one-autorefill-per-page TLB churn (a two-phase
    // validate-then-fill walk would double it). On a fault we reproduce the
    // exact state grinding would have at the faulting store -- prior bytes
    // filled, ptr at the fault address, lcount decremented to the remaining
    // count -- and raise the real exception via `cpu.translate` (pc is still
    // lbeg here, so epc1 = lbeg, matching the faulting store instruction). The
    // fast-path is thus byte-identical to per-iteration interpretation,
    // including the fault path.
    //
    // Scope note: TLB *contents* (which slot holds which page) and
    // `autorefill_idx` are micro-architectural, observable only via privileged
    // TLB-introspection ops the firmware does not execute around a memset; this
    // fast-path matches grinding on all architecturally-observable state
    // (memory, registers, PC, LCOUNT, exceptions), which is the guarantee.
    let mut off = 0u64;
    while off < total {
        let vaddr = start.wrapping_add(off as u32);
        match cpu.mmu.translate(bus, vaddr, 1 /*store*/, 0) {
            Ok(t) => {
                let psize = t.page_size as u64;
                let page_left = psize - (vaddr as u64 & (psize - 1));
                let chunk = page_left.min(total - off);
                // `data_fill` owns the region/LOCAL_DATA_END sub-splitting, so
                // the low window no longer needs a bypass here.
                bus.data_fill(t.paddr, pattern, chunk as usize);
                off += chunk;
            }
            Err(_) => {
                // Grinding faults at the store for this iteration. Its state:
                //   memory [start, vaddr) already filled (chunks above);
                //   ptr_reg = vaddr; lcount = N-1 - (bytes_filled / w).
                // pc is still lbeg, so the raising translate sets epc1 = lbeg
                // and excvaddr = vaddr, exactly as the faulting store would.
                cpu.regs.write_ar(ptr_reg, vaddr);
                cpu.regs.lcount = (n - 1 - off / w as u64) as u32;
                let step = cpu
                    .translate(bus, vaddr, Access::Store)
                    .expect_err("mmu.translate just faulted at this vaddr");
                return Some(step);
            }
        }
    }

    // No fault: pointer advanced by total, loop exhausted, PC at LEND (the
    // loop's fall-through exit). val_reg and all other regs unchanged.
    cpu.regs.write_ar(ptr_reg, start.wrapping_add(total as u32));
    cpu.regs.lcount = 0;
    cpu.pc = lend;
    Some(Step::Ran)
}

/// Decode the loop body `[lbeg, lend)`; if it is exactly a store-then-advance
/// contiguous fill, return `(width, val_reg, ptr_reg)`. `None` otherwise (incl.
/// stride != width, non-zero store offset, mismatched pointer register, or an
/// unmapped body byte).
fn decode_fill_body(cpu: &mut Cpu, bus: &mut Bus, lbeg: u32, lend: u32) -> Option<(u8, u8, u8)> {
    // Instruction 1: the store, offset 0.
    let (op1, len1) = decode_at(cpu, bus, lbeg)?;
    let (w, val_reg, ptr_reg) = match op1 {
        Op::S8i { t, s, imm } if imm == 0 => (1u8, t, s),
        Op::S16i { t, s, imm } if imm == 0 => (2u8, t, s),
        Op::S32i { t, s, imm } | Op::S32iN { t, s, imm } if imm == 0 => (4u8, t, s),
        _ => return None,
    };
    // Instruction 2: the pointer advance, in place, by exactly the store width.
    let pc2 = lbeg.wrapping_add(len1 as u32);
    let (op2, len2) = decode_at(cpu, bus, pc2)?;
    let ok_advance = match op2 {
        Op::Addi { t, s, imm } | Op::AddiN { t, s, imm } => t == s && t == ptr_reg && imm == w as i32,
        _ => false,
    };
    if !ok_advance {
        return None;
    }
    // The body must be EXACTLY these two instructions -- nothing between the
    // advance and LEND.
    if pc2.wrapping_add(len2 as u32) != lend {
        return None;
    }
    Some((w, val_reg, ptr_reg))
}

/// Translate + decode one instruction at virtual `pc`, using the same length
/// rule as `Cpu::step`. Probes translation via the NON-raising `Mmu::translate`
/// (so an unmapped body byte during speculative recognition cannot corrupt
/// pc/epc1/exccause); returns `None` on any unmapped byte or an Unknown op, and
/// the caller then leaves the loop to the normal (raising) fetch path.
fn decode_at(cpu: &mut Cpu, bus: &mut Bus, pc: u32) -> Option<(Op, u8)> {
    let phys0 = cpu.mmu.translate(bus, pc, 2 /*fetch*/, 0).ok()?.paddr;
    let b0 = bus.inst_load8(phys0);
    let op0 = b0 & 0xF;
    // Match step()'s length rule: 0xE/0xF are 8-byte FLIX bundles, narrow .n
    // ops 2 bytes, else 3. (A FLIX bundle won't be a fill-loop body, so this
    // path bails on it either way; kept consistent so decode() sees full bytes.)
    let need = if op0 == 0xE || op0 == 0xF {
        8
    } else if (0x8..=0xD).contains(&op0) {
        2
    } else {
        3
    };
    let mut buf = [b0, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8];
    for i in 1..need {
        let p = cpu.mmu.translate(bus, pc.wrapping_add(i as u32), 2 /*fetch*/, 0).ok()?.paddr;
        buf[i] = bus.inst_load8(p);
    }
    let d = decode::decode(&buf[..need], pc);
    if matches!(d.op, Op::Unknown { .. }) {
        return None;
    }
    Some((d.op, d.len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware::mmio::Bus;

    // Assemble a fill loop at `base` (a code+data address that must be in the
    // RAM aperture so stores land): loop a4,LEND; s8i a3,[a5+0]; addi.n a5,a5,1.
    // Returns (bytes, lend). Encodings are the real ones verified elsewhere:
    //   loop a4, off   -> 76 84 <off8>   (BRI8, LEND = pc + 4 + off)
    // Rather than hand-encode, drive the loop via the registers directly and a
    // real `loop` instruction fetched from memory; simplest is to set LBEG/LEND/
    // LCOUNT ourselves and place only the 2-instruction body in memory, then
    // call try_fill_loop at pc==LBEG. That exercises exactly what step() does.

    /// Place `s8i a3,[a5+0]; addi.n a5,a5,1` at `pc` in the bus (RAM), return
    /// the address just past the body (= LEND). The byte patterns are verified
    /// (not merely asserted after the fact) by `fill_body_bytes_decode_as_expected`
    /// below: same RRI8 nibble layout as `decode::mem::tests::decodes_s8i`
    /// (`82 44 2c` -> S8i{t:8,s:4,imm:0x2c}) gives `s8i a3,a5,0` = `32 45 00`;
    /// same narrow-addi.n layout as `decode::arith::tests::decodes_addi_n`
    /// (`1b 22` -> AddiN{t:2,s:2,imm:1}) gives `addi.n a5,a5,1` = `1b 55`.
    fn place_byte_fill_body(bus: &mut Bus, pc: u32) -> u32 {
        // s8i a3,a5,0: RRI8, byte0=(t<<4)|0x2, byte1=(r<<4)|s, byte2=imm8.
        // t=3,s=5,r=0x4(S8i),imm=0 -> 32 45 00.
        bus.data_store8(pc, 0x32);
        bus.data_store8(pc + 1, 0x45);
        bus.data_store8(pc + 2, 0x00);
        // addi.n a5,a5,1: narrow op0=0xB, byte0=(imm_sel<<4)|0xB,
        // byte1=(t<<4)|s. t=5,s=5,imm_sel=1(raw 1 -> imm 1) -> 1b 55.
        bus.data_store8(pc + 3, 0x1b);
        bus.data_store8(pc + 4, 0x55);
        pc + 5
    }

    /// Judgment-point verification (required by the task brief): decode the
    /// exact bytes `place_byte_fill_body` writes and assert they yield the
    /// intended ops, so the fast-path tests below cannot silently pass against
    /// a wrong body encoding.
    #[test]
    fn fill_body_bytes_decode_as_expected() {
        let d1 = decode::decode(&[0x32, 0x45, 0x00], 0);
        assert!(matches!(d1.op, Op::S8i { t: 3, s: 5, imm: 0 }), "got {:?}", d1.op);
        assert_eq!(d1.len, 3);

        let d2 = decode::decode(&[0x1b, 0x55], 0);
        assert!(matches!(d2.op, Op::AddiN { t: 5, s: 5, imm: 1 }), "got {:?}", d2.op);
        assert_eq!(d2.len, 2);
    }

    #[test]
    fn fastpath_matches_grind_byte_fill() {
        // Two runs of the SAME loop -- one fast-pathed, one ground -- must leave
        // identical RAM and registers.
        const CODE: u32 = 0x08b0_0000; // RAM aperture, so the body decodes AND
                                       // stores land in real backing.
        const DEST: u32 = 0x08b0_4000;
        const N: u32 = 5000;

        let run = |fast: bool| -> (Vec<u8>, u32, u32) {
            let mut cpu = Cpu::new(CODE);
            cpu.fastpath_enabled = fast;
            // Identity-map the low RAM region for fetch+load+store. Use the same
            // way/attr the interp test helper uses; if new_test_cpu maps only a
            // narrow page, map a wider region here so DEST+N is covered.
            cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
            cpu.mmu.write_tlb(true, (CODE & 0xfff0_0000) | 0x3, (CODE & 0xfff0_0000) | 4);
            let mut bus = Bus::new(vec![]);
            let lend = place_byte_fill_body(&mut bus, CODE);
            // Set up the loop state as `Op::Loop` would, poised at LBEG.
            cpu.pc = CODE;
            cpu.regs.lbeg = CODE;
            cpu.regs.lend = lend;
            cpu.regs.lcount = N - 1; // remaining iters = N
            cpu.regs.write_ar(5, DEST); // ptr
            cpu.regs.write_ar(3, 0xab); // fill byte
                                        // Step until the loop exits (pc past lend or a Wait/Unknown).
            for _ in 0..(N * 4 + 16) {
                if cpu.pc == lend {
                    break;
                }
                match cpu.step(&mut bus) {
                    Step::Ran | Step::Exception { .. } => {}
                    other => panic!("unexpected {other:?}"),
                }
            }
            let filled: Vec<u8> = (DEST..DEST + N).map(|a| bus.data_load8(a)).collect();
            (filled, cpu.regs.read_ar(5), cpu.regs.lcount)
        };

        let (fast_mem, fast_ptr, fast_lc) = run(true);
        let (grind_mem, grind_ptr, grind_lc) = run(false);
        assert_eq!(fast_mem, grind_mem, "filled memory must be byte-identical");
        assert!(fast_mem.iter().all(|&b| b == 0xab), "all bytes are the fill value");
        assert_eq!(fast_ptr, grind_ptr, "final pointer identical");
        assert_eq!(fast_ptr, DEST + N, "pointer advanced by N*1");
        assert_eq!(fast_lc, 0);
        assert_eq!(grind_lc, 0);
    }

    /// Place `s{16,32}i a3,a5,0; addi.n a5,a5,w` (`w` = 2 or 4) at `pc` in the
    /// bus (RAM), return the address just past the body (= LEND). The byte
    /// patterns are verified (not merely asserted after the fact) by
    /// `fill_body_bytes_decode_as_expected_word_and_half` below: same RRI8
    /// nibble layout as `decode::mem::tests::decodes_s32i`/`decodes_s16i`
    /// (`r`=0x6 for S32i, 0x5 for S16i, `t`=3, `s`=5, `imm8`=0) gives `s32i
    /// a3,a5,0` = `32 65 00` and `s16i a3,a5,0` = `32 55 00`; same narrow-addi.n
    /// layout as `decode::arith::tests::decodes_addi_n` (raw imm nibble ==
    /// imm value for nibble != 0) gives `addi.n a5,a5,4` = `4b 55` and
    /// `addi.n a5,a5,2` = `2b 55`.
    fn place_width_fill_body(bus: &mut Bus, pc: u32, width: u8) -> u32 {
        let (r, imm_sel): (u32, u32) = match width {
            4 => (0x6, 0x4), // S32i, addi.n imm=4
            2 => (0x5, 0x2), // S16i, addi.n imm=2
            _ => panic!("test helper only supports width 2 or 4, got {width}"),
        };
        // s{16,32}i a3,a5,0: RRI8, byte0=(t<<4)|0x2, byte1=(r<<4)|s, byte2=imm8.
        // t=3,s=5,imm8=0.
        bus.data_store8(pc, 0x32);
        bus.data_store8(pc + 1, (r << 4) | 5);
        bus.data_store8(pc + 2, 0x00);
        // addi.n a5,a5,w: narrow op0=0xB, byte0=(imm_sel<<4)|0xB, byte1=(t<<4)|s.
        // t=5,s=5,imm_sel=w (raw nibble w -> imm w, since w != 0).
        bus.data_store8(pc + 3, (imm_sel << 4) | 0xb);
        bus.data_store8(pc + 4, 0x55);
        pc + 5
    }

    /// Judgment-point verification (required by the task brief): decode the
    /// exact bytes `place_width_fill_body` writes for both the 32-bit and
    /// 16-bit variants and assert they yield the intended ops, so
    /// `fastpath_matches_grind_width_variants` below cannot silently pass
    /// against a wrong body encoding.
    #[test]
    fn fill_body_bytes_decode_as_expected_word_and_half() {
        // width=4: s32i a3,a5,0 -> 32 65 00; addi.n a5,a5,4 -> 4b 55.
        let d1 = decode::decode(&[0x32, 0x65, 0x00], 0);
        assert!(matches!(d1.op, Op::S32i { t: 3, s: 5, imm: 0 }), "got {:?}", d1.op);
        assert_eq!(d1.len, 3);
        let d2 = decode::decode(&[0x4b, 0x55], 0);
        assert!(matches!(d2.op, Op::AddiN { t: 5, s: 5, imm: 4 }), "got {:?}", d2.op);
        assert_eq!(d2.len, 2);

        // width=2: s16i a3,a5,0 -> 32 55 00; addi.n a5,a5,2 -> 2b 55.
        let d3 = decode::decode(&[0x32, 0x55, 0x00], 0);
        assert!(matches!(d3.op, Op::S16i { t: 3, s: 5, imm: 0 }), "got {:?}", d3.op);
        assert_eq!(d3.len, 3);
        let d4 = decode::decode(&[0x2b, 0x55], 0);
        assert!(matches!(d4.op, Op::AddiN { t: 5, s: 5, imm: 2 }), "got {:?}", d4.op);
        assert_eq!(d4.len, 2);
    }

    #[test]
    fn fastpath_matches_grind_width_variants() {
        // Same structure as `fastpath_matches_grind_byte_fill`, generalized to
        // the two other widths the recognizer supports: a 32-bit word fill
        // (S32i) and a 16-bit half fill (S16i). Each width's fast-path output
        // must be byte-identical to grinding, and (Important-3's key
        // correctness point) the advance immediate must equal the store
        // width with the pointer register advanced in place -- exercised here
        // by actually running both widths, not just asserting the encoding.
        const CODE: u32 = 0x08b0_0000; // RAM aperture, so the body decodes AND
                                       // stores land in real backing.
        const DEST: u32 = 0x08b0_4000; // width-aligned (4- and 2-aligned).
        const N: u32 = 5000;
        const PATTERN: u32 = 0xdead_beef;

        for &width in &[4u32, 2u32] {
            let run = |fast: bool| -> (Vec<u8>, u32, u32) {
                let mut cpu = Cpu::new(CODE);
                cpu.fastpath_enabled = fast;
                // Identity-map the low RAM region for fetch+load+store, same
                // as the byte-fill test.
                cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
                cpu.mmu.write_tlb(true, (CODE & 0xfff0_0000) | 0x3, (CODE & 0xfff0_0000) | 4);
                let mut bus = Bus::new(vec![]);
                let lend = place_width_fill_body(&mut bus, CODE, width as u8);
                // Set up the loop state as `Op::Loop` would, poised at LBEG.
                cpu.pc = CODE;
                cpu.regs.lbeg = CODE;
                cpu.regs.lend = lend;
                cpu.regs.lcount = N - 1; // remaining iters = N
                cpu.regs.write_ar(5, DEST); // ptr
                cpu.regs.write_ar(3, PATTERN); // fill value (truncated to width by the store)
                                               // Step until the loop exits (pc past lend or a Wait/Unknown).
                for _ in 0..(N * 4 + 16) {
                    if cpu.pc == lend {
                        break;
                    }
                    match cpu.step(&mut bus) {
                        Step::Ran | Step::Exception { .. } => {}
                        other => panic!("unexpected {other:?}"),
                    }
                }
                let byte_len = N * width;
                let filled: Vec<u8> = (DEST..DEST + byte_len).map(|a| bus.data_load8(a)).collect();
                if width == 4 {
                    // Word-fill readback: the filled region must read back as
                    // the 32-bit fill value repeated across the range.
                    for i in 0..N {
                        assert_eq!(
                            bus.data_load32(DEST + i * 4),
                            PATTERN,
                            "fast={fast}, word {i} must equal the fill pattern"
                        );
                    }
                }
                (filled, cpu.regs.read_ar(5), cpu.regs.lcount)
            };

            let (fast_mem, fast_ptr, fast_lc) = run(true);
            let (grind_mem, grind_ptr, grind_lc) = run(false);
            assert_eq!(fast_mem, grind_mem, "width {width}: filled memory must be byte-identical");
            assert_eq!(fast_ptr, grind_ptr, "width {width}: final pointer identical");
            assert_eq!(fast_ptr, DEST + N * width, "width {width}: pointer advanced by N*w");
            assert_eq!(fast_lc, 0, "width {width}: fast LCOUNT exhausted");
            assert_eq!(grind_lc, 0, "width {width}: grind LCOUNT exhausted");
        }
    }

    #[test]
    fn small_loop_is_not_fast_pathed() {
        // Below MIN_ITERS the recognizer declines (returns None).
        let mut cpu = Cpu::new(0x08b0_0000);
        let mut bus = Bus::new(vec![]);
        let lend = place_byte_fill_body(&mut bus, 0x08b0_0000);
        cpu.pc = 0x08b0_0000;
        cpu.regs.lbeg = 0x08b0_0000;
        cpu.regs.lend = lend;
        cpu.regs.lcount = 10; // < MIN_ITERS
        assert!(try_fill_loop(&mut cpu, &mut bus).is_none());
    }

    #[test]
    fn fill_loop_is_not_fast_pathed_in_exception_mode() {
        // PS.EXCM suppresses architectural zero-overhead loop back-edges, so
        // collapsing the remaining iterations here would execute work that the
        // scalar interpreter (and hardware) must not repeat.
        const CODE: u32 = 0x08b0_0000;
        const DEST: u32 = 0x08b0_4000;
        let mut cpu = Cpu::new(CODE);
        cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
        cpu.mmu.write_tlb(true, (CODE & 0xfff0_0000) | 0x3, (CODE & 0xfff0_0000) | 4);
        let mut bus = Bus::new(vec![]);
        let lend = place_byte_fill_body(&mut bus, CODE);
        cpu.regs.lbeg = CODE;
        cpu.regs.lend = lend;
        cpu.regs.lcount = MIN_ITERS;
        cpu.regs.write_ar(5, DEST);
        cpu.regs.write_ar(3, 0xab);
        cpu.regs.set_excm();

        assert!(try_fill_loop(&mut cpu, &mut bus).is_none());
        assert_eq!(cpu.pc, CODE);
        assert_eq!(cpu.regs.lcount, MIN_ITERS);
        assert_eq!(cpu.regs.read_ar(5), DEST);
        assert_eq!(bus.data_load8(DEST), 0, "declined fastpath must not touch memory");
    }

    #[test]
    fn fastpath_local_window_fill_matches_grind() {
        // A non-zero byte fill whose DEST is in the low window (< LOCAL_DATA_END)
        // must fill local_data, byte-identical fast vs grind. Body stays in RAM so
        // it is fetchable; only the fill target is local.
        const CODE: u32 = 0x08b0_0000; // RAM: body fetchable
        const DEST: u32 = 0x0020_0000; // low window (< 0x04000000)
        const N: u32 = 5000;

        let run = |fast: bool| -> (Vec<u8>, u32, u32) {
            let mut cpu = Cpu::new(CODE);
            // Real boot (`Firmware::load_m2c`) always runs with varway56=true: way-6
            // entry 0 identity-maps 0..0x1fffffff (attr 3, RWX), which covers DEST.
            // Translation is now authoritative for D-side accesses too (Task 4), so
            // the low-window fill target needs a DTLB mapping, not just the fetch page.
            cpu.mmu = crate::firmware::xtensa::mmu::Mmu::new_with_varway56(true);
            cpu.fastpath_enabled = fast;
            // Map only the body (fetch) region; the local DEST is covered by way-6 above.
            cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
            let mut bus = Bus::new(vec![]);
            let lend = place_byte_fill_body(&mut bus, CODE);
            cpu.pc = CODE;
            cpu.regs.lbeg = CODE;
            cpu.regs.lend = lend;
            cpu.regs.lcount = N - 1;
            cpu.regs.write_ar(5, DEST); // ptr
            cpu.regs.write_ar(3, 0xab); // fill byte
            for _ in 0..(N * 4 + 16) {
                if cpu.pc == lend {
                    break;
                }
                match cpu.step(&mut bus) {
                    Step::Ran | Step::Exception { .. } => {}
                    other => panic!("unexpected {other:?}"),
                }
            }
            let filled: Vec<u8> = (DEST..DEST + N).map(|a| bus.load_local8(a)).collect();
            (filled, cpu.regs.read_ar(5), cpu.regs.lcount)
        };

        let (fast_mem, fast_ptr, fast_lc) = run(true);
        let (grind_mem, grind_ptr, grind_lc) = run(false);
        assert_eq!(fast_mem, grind_mem, "local fill must be byte-identical fast vs grind");
        assert!(fast_mem.iter().all(|&b| b == 0xab));
        assert_eq!(fast_ptr, grind_ptr);
        assert_eq!(fast_ptr, DEST + N);
        assert_eq!(fast_lc, 0);
        assert_eq!(grind_lc, 0);
    }

    #[test]
    fn fastpath_fill_spanning_local_boundary_matches_grind() {
        // A byte fill starting below LOCAL_DATA_END and running across it: the
        // local portion fills local_data; the portion at/above 0x04000000 lands in
        // the Array aperture (dropped, reads back 0). Fast == grind on both sides.
        const CODE: u32 = 0x08b0_0000;
        // Start 0x800 below the boundary; N carries it 0x800 past into the array.
        const DEST: u32 = crate::firmware::mmio::LOCAL_DATA_END - 0x800;
        const N: u32 = 0x1000; // 0x800 local + 0x800 array
        const BOUNDARY: u32 = crate::firmware::mmio::LOCAL_DATA_END;

        let run = |fast: bool| -> (Vec<u8>, Vec<u8>, u32, u32) {
            let mut cpu = Cpu::new(CODE);
            // Real boot (`Firmware::load_m2c`) always runs with varway56=true: way-6
            // entry 0 identity-maps 0..0x1fffffff (attr 3, RWX), which is what makes
            // the array-aperture side of this fill a real (stubbed, write-dropping)
            // access rather than an MMU miss. Without this, the array-side store
            // would fault at the very first array byte instead of exercising the
            // aperture-drop path this test targets.
            cpu.mmu = crate::firmware::xtensa::mmu::Mmu::new_with_varway56(true);
            cpu.fastpath_enabled = fast;
            cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
            let mut bus = Bus::new(vec![]);
            let lend = place_byte_fill_body(&mut bus, CODE);
            cpu.pc = CODE;
            cpu.regs.lbeg = CODE;
            cpu.regs.lend = lend;
            cpu.regs.lcount = N - 1;
            cpu.regs.write_ar(5, DEST);
            cpu.regs.write_ar(3, 0xcd);
            for _ in 0..(N * 4 + 16) {
                if cpu.pc == lend {
                    break;
                }
                match cpu.step(&mut bus) {
                    Step::Ran | Step::Exception { .. } => {}
                    other => panic!("unexpected {other:?}"),
                }
            }
            let local: Vec<u8> = (DEST..BOUNDARY).map(|a| bus.load_local8(a)).collect();
            let array: Vec<u8> = (BOUNDARY..DEST + N).map(|a| bus.data_load8(a)).collect();
            (local, array, cpu.regs.read_ar(5), cpu.regs.lcount)
        };

        let (f_local, f_array, f_ptr, f_lc) = run(true);
        let (g_local, g_array, g_ptr, g_lc) = run(false);
        assert_eq!(f_local, g_local, "local side identical");
        assert_eq!(f_array, g_array, "array side identical");
        assert!(f_local.iter().all(|&b| b == 0xcd), "local side filled");
        assert!(f_array.iter().all(|&b| b == 0), "array side dropped (reads 0)");
        assert_eq!(f_ptr, g_ptr);
        assert_eq!(f_ptr, DEST + N, "pointer advanced across the boundary");
        assert_eq!(f_lc, g_lc);
        assert_eq!(f_lc, 0);
    }

    #[test]
    fn fastpath_nonzero_straddle_no_dram_leak_above_boundary() {
        // A non-zero byte fill crossing LOCAL_DATA_END must, fast AND grind, leave
        // local_data ABOVE the boundary untouched (the array side is dropped, not
        // mis-routed into DRAM). The pre-existing spanning test reads the array side
        // via the region path and cannot see a DRAM leak; this one reads local_data.
        const CODE: u32 = 0x08b0_0000;
        const DEST: u32 = crate::firmware::mmio::LOCAL_DATA_END - 0x800;
        const N: u32 = 0x1000; // 0x800 local + 0x800 array
        const BOUNDARY: u32 = crate::firmware::mmio::LOCAL_DATA_END;

        let run = |fast: bool| -> Vec<u8> {
            let mut cpu = Cpu::new(CODE);
            cpu.mmu = crate::firmware::xtensa::mmu::Mmu::new_with_varway56(true);
            cpu.fastpath_enabled = fast;
            cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
            let mut bus = Bus::new(vec![]);
            let lend = place_byte_fill_body(&mut bus, CODE);
            cpu.pc = CODE;
            cpu.regs.lbeg = CODE;
            cpu.regs.lend = lend;
            cpu.regs.lcount = N - 1;
            cpu.regs.write_ar(5, DEST);
            cpu.regs.write_ar(3, 0xcd);
            for _ in 0..(N * 4 + 16) {
                if cpu.pc == lend {
                    break;
                }
                match cpu.step(&mut bus) {
                    Step::Ran | Step::Exception { .. } => {}
                    o => panic!("{o:?}"),
                }
            }
            // local_data across and ABOVE the boundary.
            (DEST..DEST + N).map(|a| bus.load_local8(a)).collect()
        };
        let fast = run(true);
        let grind = run(false);
        assert_eq!(fast, grind, "DRAM state identical fast vs grind, including above the boundary");
        // Below the boundary: filled; at/above: DRAM untouched (0), not 0xcd.
        let split = (BOUNDARY - DEST) as usize;
        assert!(fast[..split].iter().all(|&b| b == 0xcd), "DRAM side filled");
        assert!(fast[split..].iter().all(|&b| b == 0), "no 0xcd leaked into DRAM above the boundary");
    }

    #[test]
    fn fault_mid_fill_matches_grind() {
        // The fault-replication arm: a fill whose range starts in a mapped page
        // and runs off the end of the mapping must, in the fast-path, reproduce
        // EXACTLY the state grinding produces at the faulting store -- the
        // filled prefix, the pointer at the fault address, the decremented
        // LCOUNT, and an identical Step::Exception (cause + epc1 + excvaddr).
        //
        // Setup: one 1 MB way-4 identity page at 0x08b00000..0x08c00000 covers
        // the loop body (at CODE) and the fill destination, which is placed near
        // the top so the byte fill crosses the 0x08c00000 boundary into unmapped
        // space after 0x1000 bytes. N=0x2000 (> MIN_ITERS) so the fast-path
        // fires; the fault lands mid-fill.
        const CODE: u32 = 0x08b0_0000;
        const DEST: u32 = 0x08bf_f000; // last page of the mapped 1 MB region
        const FAULT_AT: u32 = 0x08c0_0000; // first unmapped address
        const N: u32 = 0x2000;
        const FILLED: u32 = FAULT_AT - DEST; // 0x1000 bytes fill before the fault

        // Run one pass; return (exception cause, epc1, excvaddr, ptr, lcount,
        // filled prefix bytes). Steps until the first Step::Exception.
        let run = |fast: bool| -> (u32, u32, u32, u32, u32, Vec<u8>) {
            let mut cpu = Cpu::new(CODE);
            cpu.fastpath_enabled = fast;
            cpu.mmu.write_tlb(false, (CODE & 0xfff0_0000) | 0x1, (CODE & 0xfff0_0000) | 4);
            cpu.mmu.write_tlb(true, (CODE & 0xfff0_0000) | 0x3, (CODE & 0xfff0_0000) | 4);
            let mut bus = Bus::new(vec![]);
            let lend = place_byte_fill_body(&mut bus, CODE);
            cpu.pc = CODE;
            cpu.regs.lbeg = CODE;
            cpu.regs.lend = lend;
            cpu.regs.lcount = N - 1; // remaining iters = N
            cpu.regs.write_ar(5, DEST); // ptr
            cpu.regs.write_ar(3, 0xab); // fill byte
            let mut cause = None;
            for _ in 0..(N * 4 + 16) {
                match cpu.step(&mut bus) {
                    Step::Ran => {}
                    Step::Exception { cause: c, .. } => {
                        cause = Some(c);
                        break;
                    }
                    other => panic!("unexpected {other:?}"),
                }
            }
            let filled: Vec<u8> = (DEST..FAULT_AT).map(|a| bus.data_load8(a)).collect();
            (
                cause.expect("the fill must fault running off the mapped page"),
                cpu.epc1,
                cpu.excvaddr,
                cpu.regs.read_ar(5),
                cpu.regs.lcount,
                filled,
            )
        };

        let (f_cause, f_epc1, f_vaddr, f_ptr, f_lc, f_mem) = run(true);
        let (g_cause, g_epc1, g_vaddr, g_ptr, g_lc, g_mem) = run(false);

        // Fast-path and grind agree on every architecturally-observable value.
        assert_eq!(f_cause, g_cause, "exception cause must match grinding");
        assert_eq!(f_epc1, g_epc1, "epc1 must match grinding");
        assert_eq!(f_epc1, CODE, "epc1 is the faulting store's pc (== lbeg)");
        assert_eq!(f_vaddr, g_vaddr, "excvaddr must match grinding");
        assert_eq!(f_vaddr, FAULT_AT, "excvaddr is the first unmapped address");
        assert_eq!(f_ptr, g_ptr, "final pointer must match grinding");
        assert_eq!(f_ptr, FAULT_AT, "ptr sits at the faulting store's address");
        assert_eq!(f_lc, g_lc, "LCOUNT must match grinding");
        assert_eq!(f_lc, N - 1 - FILLED, "LCOUNT decremented by the completed iterations");
        assert_eq!(f_mem, g_mem, "the filled prefix must be byte-identical");
        assert!(f_mem.iter().all(|&b| b == 0xab), "prefix is the fill value");
        assert_eq!(f_mem.len(), FILLED as usize, "exactly the pre-fault bytes were filled");
    }
}
