//! Xtensa instruction decoder. Base ISA + the windowed-call ops the firmware
//! uses. DERIVED FROM THE TOOLCHAIN: every opcode here has a test vector taken
//! from the real firmware, disassembled by xtensa-lx106-elf-objdump (base) or
//! the captured Ghidra listing.txt (windowed ops -- lx106 objdump cannot
//! decode these; verified empirically, it prints `excw` regardless of the
//! actual bits).
//!
//! Split into per-category modules (`arith`/`mem`/`branch`/`control`/
//! `system`) as a scaffold for M2a's additional opcodes. `Op`/`Decoded` stay
//! here (an enum can't span files); this module keeps the instruction-format
//! skeleton (op0 dispatch, narrow/wide length, field extraction) and
//! delegates leaf construction to each category's `decode_*` helpers, which
//! return `None` when the bits aren't theirs so the next category (or
//! `Op::Unknown`) gets a turn.

mod arith;
mod branch;
mod control;
mod mem;
mod system;

/// A decoded Xtensa instruction: one variant per implemented opcode, each
/// carrying its decoded operands (register indices as `u8`, immediates as
/// `i32`/`u32`, branch/call targets as absolute `u32`). `Unknown` covers
/// anything the decoder doesn't (yet) implement or can't recognize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Entry {
        s: u8,
        imm: u32,
    },
    Call8 {
        target: u32,
    },
    /// Register-indirect windowed call, size 8 (`callx8 as`). Target comes
    /// from AR[s] at execute time; like `call8` it sets PS.CALLINC=2.
    Callx8 {
        s: u8,
    },
    /// Windowed return (`retw`). Restores the window using the call-size in
    /// `a0[31:30]` and returns to `a0[29:0]`.
    Retw,
    /// Narrow (2-byte) windowed return (`retw.n`); same semantics as `retw`.
    RetwN,
    L32iN {
        t: u8,
        s: u8,
        imm: u32,
    },
    MovN {
        t: u8,
        s: u8,
    },
    MoviN {
        t: u8,
        imm: i32,
    },
    Movi {
        t: u8,
        imm: i32,
    },
    L32i {
        t: u8,
        s: u8,
        imm: u32,
    },
    L32r {
        t: u8,
        target: u32,
    },
    /// Narrow (2-byte) store, `s32i.n at,as,imm4*4` -- sibling of `L32iN`
    /// (same imm4*4 pre-scaling). Verified: `69 c7` -> s32i.n a6,a7,0x30.
    S32iN {
        t: u8,
        s: u8,
        imm: u32,
    },
    /// 8-bit zero-extending load, `l8ui at,as,imm8` (scale 0). RRI8 `r==0x0`.
    /// Verified: `22 02 00` -> l8ui a2,a2,0x0.
    L8ui {
        t: u8,
        s: u8,
        imm: u32,
    },
    /// 8-bit store (low byte of AR[t]), `s8i at,as,imm8` (scale 0). RRI8
    /// `r==0x4`. Verified: `82 44 2c` -> s8i a8,a4,0x2c.
    S8i {
        t: u8,
        s: u8,
        imm: u32,
    },
    /// 32-bit store, `s32i at,as,imm8*4` -- sibling of `L32i` (same imm8*4
    /// pre-scaling). RRI8 `r==0x6`. Verified: `22 61 07` -> s32i a2,a1,0x1c.
    S32i {
        t: u8,
        s: u8,
        imm: u32,
    },
    /// 16-bit zero-extending load, `l16ui at,as,imm8*2` (scale 1). RRI8
    /// `r==0x1`. Verified: `32 13 02` -> l16ui a3,a3,0x4.
    L16ui {
        t: u8,
        s: u8,
        imm: u32,
    },
    /// 16-bit store (low halfword of AR[t]), `s16i at,as,imm8*2` (scale 1).
    /// RRI8 `r==0x5`. Verified: `42 57 02` -> s16i a4,a7,0x4.
    S16i {
        t: u8,
        s: u8,
        imm: u32,
    },
    /// 16-bit sign-extending load, `l16si at,as,imm8*2` (scale 1). RRI8
    /// `r==0x9`. Verified: `22 92 00` -> l16si a2,a2,0x0.
    L16si {
        t: u8,
        s: u8,
        imm: u32,
    },
    /// Store-release, `s32ri at,as,imm8*4` (scale 2, same field layout as
    /// `S32i`). RRI8 `r==0xF`. `xtensa-lx106-elf-objdump` can't decode this
    /// opcode (the lx106 core lacks the Interrupt option it belongs to) and
    /// prints `excw` for it regardless of the actual bits (confirmed by
    /// sweeping every `r` value 0x0..=0xF against objdump: 0x3/0x7/0x8/0xB/
    /// 0xE/0xF all print `excw`) -- verified instead via the captured Ghidra
    /// `listing.txt`: `a2 ff 86` -> s32ri a10,a15,0x218. Kept as a distinct
    /// `Op` from `S32i` (matching the real ISA's distinct encoding), though
    /// the release ordering has no effect in this single-threaded
    /// interpreter -- see `interp::mem::exec`.
    S32ri {
        t: u8,
        s: u8,
        imm: u32,
    },
    Or {
        r: u8,
        s: u8,
        t: u8,
    },
    Extui {
        r: u8,
        t: u8,
        shiftimm: u8,
        maskimm: u8,
    },
    /// `add ar,as,at`, RRR `op1=0,op2=8`. Verified: `20 26 80` -> add
    /// a2,a6,a2.
    Add {
        r: u8,
        s: u8,
        t: u8,
    },
    /// Narrow `add.n ar,as,at`, `op0=0xA` -- same fields/positions as the
    /// wide `Add` (r/s/t all confirmed distinct in the oracle vector).
    /// Verified: `7a e8` -> add.n a14,a8,a7.
    AddN {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `addi at,as,imm8`, RRI8 `r==0xC`. `imm` is `sign_extend8(imm8)`.
    /// Verified: `62 cf 27` -> addi a6,a15,0x27 (39).
    Addi {
        t: u8,
        s: u8,
        imm: i32,
    },
    /// Narrow `addi.n at,as,imm`, `op0=0xB`. The immediate is NOT a plain
    /// nibble: raw encoded value 0 means -1, values 1-15 mean themselves
    /// (there is no encoding for +0). Field positions confirmed by sweeping
    /// every register nibble independently against objdump (byte1 high
    /// nibble = dest `t`, byte1 low nibble = src `s`, byte0 high nibble =
    /// the immediate selector -- NOT the same nibble->field mapping as
    /// `mov.n`, a different narrow format). Verified: `1b 22` -> addi.n
    /// a2,a2,1.
    AddiN {
        t: u8,
        s: u8,
        imm: i32,
    },
    /// `addmi at,as,imm8`, RRI8 `r==0xD`. `imm` is already
    /// `sign_extend8(imm8) << 8` (the final value to add, matching this
    /// decoder's convention of storing pre-scaled immediates). Verified:
    /// `22 d7 02` -> addmi a2,a7,0x200 (raw byte 0x02 -> sign_extend8=2,
    /// <<8 = 0x200).
    Addmi {
        t: u8,
        s: u8,
        imm: i32,
    },
    /// `sub ar,as,at`, RRR `op1=0,op2=0xC`. Verified: `90 44 c0` -> sub
    /// a4,a4,a9.
    Sub {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `addx2 ar,as,at` = `(AR[s]<<1)+AR[t]`, RRR `op1=0,op2=9`. Verified:
    /// `20 32 90` -> addx2 a3,a2,a2.
    Addx2 {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `addx4 ar,as,at` = `(AR[s]<<2)+AR[t]`, RRR `op1=0,op2=0xA`. Verified:
    /// `20 22 a0` -> addx4 a2,a2,a2.
    Addx4 {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `addx8 ar,as,at` = `(AR[s]<<3)+AR[t]`, RRR `op1=0,op2=0xB`. Verified:
    /// `50 52 b0` -> addx8 a5,a2,a5.
    Addx8 {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `subx8 ar,as,at` = `(AR[s]<<3)-AR[t]` (minuend is the shifted `s`),
    /// RRR `op1=0,op2=0xF`. Only `subx8` appears in the firmware (no
    /// subx2/subx4). Verified: `80 98 f0` -> subx8 a9,a8,a8.
    Subx8 {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `and ar,as,at`, RRR `op1=0,op2=1`. Verified: `30 72 10` -> and
    /// a7,a2,a3.
    And {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `xor ar,as,at`, RRR `op1=0,op2=3`. Verified: `80 85 30` -> xor
    /// a8,a5,a8.
    Xor {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `neg ar,at` = `-AR[t]` (two's complement), RRR `op1=0,op2=6`, `s==0`
    /// -- `s` is a fixed selector nibble here, not a register (disambiguates
    /// from `Abs`, `s==1`, sharing the same `(op1,op2)`). Verified: `70 80
    /// 60` -> neg a8,a7 (s field is 0).
    Neg {
        r: u8,
        t: u8,
    },
    /// `abs ar,at` = `|AR[t]|` (signed), RRR `op1=0,op2=6`, `s==1` -- see
    /// [`Op::Neg`] for the disambiguation. Verified: `20 21 60` -> abs a2,a2
    /// (s field is 1).
    Abs {
        r: u8,
        t: u8,
    },
    /// `moveqz ar,as,at`: `if AR[t]==0 { AR[r]=AR[s] }`, RRR `op1=3,op2=8`.
    /// Verified: `40 f2 83` -> moveqz a15,a2,a4.
    Moveqz {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `movnez ar,as,at`: `if AR[t]!=0 { AR[r]=AR[s] }`, RRR `op1=3,op2=9`.
    /// Verified: `50 a3 93` -> movnez a10,a3,a5.
    Movnez {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `min ar,as,at`: signed minimum, RRR `op1=3,op2=4`. (`max`, signed, is
    /// `op2=5` and absent from the firmware.) Verified: `20 2a 43` -> min
    /// a2,a10,a2 (Ghidra listing.txt oracle -- `xtensa-lx106-elf-objdump`
    /// prints `excw` for this opcode, same gap as `s32ri` in task 2).
    Min {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `minu ar,as,at`: unsigned minimum, RRR `op1=3,op2=6`. Verified: `90
    /// 94 63` -> minu a9,a4,a9 (Ghidra listing.txt oracle, see [`Op::Min`]).
    Minu {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `maxu ar,as,at`: unsigned maximum, RRR `op1=3,op2=7`. Verified: `40
    /// 57 73` -> maxu a5,a7,a4 (Ghidra listing.txt oracle, see [`Op::Min`]).
    Maxu {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `slli ar,as,sa` = `AR[s] << sa` (`sa` 1..31), RRR `op1=1,op2∈{0,1}`.
    /// The 5-bit shift count is NOT a plain nibble: it's `32 - imm5` where
    /// `imm5 = ((op2&1)<<4)|t` -- `t` supplies the low 4 bits, the LSB of
    /// `op2` the top bit. `imm` here is already the resolved final shift
    /// count (not the raw `imm5`), matching this decoder's pre-resolved-
    /// immediate convention (cf. `Extui`). Verified by an exhaustive objdump
    /// sweep (the firmware's own vector aliases dest/src so it alone can't
    /// pin the roles) and cross-checked against 1346 real `slli` instances
    /// in the Ghidra listing (100% match). Vector: `20 33 01` -> slli
    /// a3,a3,0x1e (t=2,op2=0 -> imm5=2 -> shift=30).
    Slli {
        r: u8,
        s: u8,
        imm: u8,
    },
    /// `srli ar,at,imm4` = `AR[t] >> imm4` (logical, `imm4` 0..15), RRR
    /// `op1=1,op2=4`. Unlike `Slli`, the source register is `t` (not `s`) --
    /// `s` instead carries the (plain, no split needed) 4-bit shift count.
    /// Verified by objdump sweep + cross-checked against 193 real instances
    /// (100% match). Vector: `40 42 41` -> srli a4,a4,0x2.
    Srli {
        r: u8,
        t: u8,
        imm: u8,
    },
    /// `srai ar,at,sa` = `(AR[t] as i32) >> sa` (arithmetic, `sa` 0..31), RRR
    /// `op1=1,op2∈{2,3}`. Same source-is-`t` layout as `Srli`; the 5-bit `sa`
    /// is `((op2&1)<<4)|s` (split across `s` and the LSB of `op2`, mirroring
    /// `Slli`'s split but on `s` instead of `t`, and without the `32-` complement).
    /// Verified by objdump sweep + cross-checked against 49 real instances
    /// (100% match). Vector: `a0 38 31` -> srai a3,a10,0x18 (s=8,op2=3 ->
    /// imm5 = 16|8 = 24).
    Srai {
        r: u8,
        t: u8,
        imm: u8,
    },
    /// `sll ar,as` = `AR[s] << SAR_shift` (see [`RegFile`](super::regfile::RegFile)
    /// docs / `interp::arith::exec` for the `ssl`/`sll` SAR relationship), RRR
    /// `op1=1,op2=0xA`. `t` is a fixed selector nibble here (must be 0, not a
    /// register) -- confirmed by sweep: any nonzero `t` decodes to `excw`
    /// (an invalid encoding), matching the pattern already used by
    /// `Neg`/`Abs`'s selector nibble. Verified by objdump sweep + cross-
    /// checked against 160 real instances (100% match). Vector: `00 33 a1`
    /// -> sll a3,a3 (t=0).
    Sll {
        r: u8,
        s: u8,
    },
    /// `srl ar,at` = `AR[t] >> SAR_shift` (logical), RRR `op1=1,op2=9`. `s`
    /// is the fixed selector nibble here (must be 0) -- the `Sll`/`Srl` pair
    /// swap which of `s`/`t` is the selector vs. the source register.
    /// Verified by objdump sweep + cross-checked against 107 real instances
    /// (100% match). Vector: `a0 d0 91` -> srl a13,a10 (s=0).
    Srl {
        r: u8,
        t: u8,
    },
    /// `src ar,as,at`: funnel shift right -- concatenate `AR[s]` (high) :
    /// `AR[t]` (low) into a 64-bit value, shift right by `SAR`, take the low
    /// 32 bits. RRR `op1=1,op2=8`; the only op in this shift family with all
    /// three register roles in their usual `r`=dest/`s`/`t` positions (no
    /// selector nibble). Verified by objdump sweep + cross-checked against
    /// 19 real instances (100% match). Vector: `80 28 81` -> src a2,a8,a8.
    Src {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `ssl as`: `SAR = 32 - (AR[s] & 31)` (sets up a left shift by `AR[s]`,
    /// consumed by `Sll`). RRR `op1=0,op2=4,r=1` -- within this op2=4 group,
    /// `r` is itself a sub-opcode selector (not a register), distinguishing
    /// `Ssr`(r=0)/`Ssl`(r=1)/`Ssai`(r=4)/`Nsau`(r=15) among others not
    /// implemented here (ssa8l/ssa8b/rer/wer/nsa). `t` must be 0 (a further
    /// selector nibble; nonzero decodes to `excw`). Verified by objdump
    /// sweep + cross-checked against 154 real instances (100% match).
    /// Vector: `00 14 40` -> ssl a4.
    Ssl {
        s: u8,
    },
    /// `ssr as`: `SAR = AR[s] & 31` (sets up a right shift by `AR[s]`,
    /// consumed by `Srl`). RRR `op1=0,op2=4,r=0` -- see [`Op::Ssl`] for the
    /// `r`-as-selector layout shared by this group. Verified by objdump
    /// sweep + cross-checked against 103 real instances (100% match).
    /// Vector: `00 06 40` -> ssr a6.
    Ssr {
        s: u8,
    },
    /// `ssai imm`: `SAR = imm` (imm 0..31, a plain immediate, not derived
    /// from a register). RRR `op1=0,op2=4,r=4` -- see [`Op::Ssl`] for the
    /// `r`-as-selector layout; `imm = ((t&1)<<4)|s` (`t` restricted to 0/1,
    /// supplying the top bit; any other `t` decodes to `excw`). Verified by
    /// objdump sweep + the one real instance in the firmware (`10 40 40` ->
    /// ssai 0x10, t=1,s=0 -> imm=16).
    Ssai {
        imm: u8,
    },
    /// `sext ar,as,imm` (imm 7..22): sign-extend `AR[s]` from bit `imm`
    /// (i.e. treat bit `imm` as the new sign bit). RRR `op1=3,op2=2`; `imm =
    /// t + 7` (`t` is the raw 4-bit field, 0..15, offset by the format's
    /// fixed 7-bit floor). `xtensa-lx106-elf-objdump` can't decode this
    /// opcode (prints `excw`, the lx106 core lacks the Boolean/extended
    /// option it belongs to, same gap as `s32ri`/`min` in earlier tasks) --
    /// verified instead via the Ghidra `listing.txt` oracle: every one of
    /// its 17 real instances in the firmware uses `imm=7` (`t=0`), so the
    /// `+7` floor is confirmed but the field's full 7..22 range only by the
    /// format's known bit width (4 bits), not by direct example. Vector:
    /// `00 98 23` -> sext a9,a8,7.
    Sext {
        r: u8,
        s: u8,
        imm: u8,
    },
    /// `nsau ar,as`: normalize-shift-amount-unsigned -- count of leading
    /// zero bits of `AR[s]` (32 when `AR[s]==0`). RRR `op1=0,op2=4,r=15` --
    /// see [`Op::Ssl`] for the `r`-as-selector layout. Unlike every other op
    /// in this file, the DEST register here is `t`, not `r` (`r` is fixed at
    /// 15, the sub-op selector) -- confirmed by objdump sweep with distinct
    /// t/s registers, and cross-checked against 6 real instances (100%
    /// match). Vector: `20 f2 40` -> nsau a2,a2 (t=2,s=2, aliased in this
    /// particular instance).
    Nsau {
        t: u8,
        s: u8,
    },
    /// `mull ar,as,at` = low 32 bits of `AR[s] * AR[t]` (sign-agnostic: the
    /// low word of a 32x32 product is identical whether the inputs are
    /// treated as signed or unsigned). RRR `op1=2,op2=8`. Verified via BOTH
    /// oracles: `xtensa-lx106-elf-objdump` decodes it directly (`50 73 82`
    /// -> mull a7,a3,a5) AND `xtensa-modules.c`'s op1=2 decode table lists
    /// op2=8 -> opcode 461 (mull), confirming the `(op1,op2)` pair
    /// independently.
    Mull {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `mul16s ar,as,at` = `sign_extend16(AR[s]) * sign_extend16(AR[t])`
    /// (16x16 signed multiply, full 32-bit result). RRR `op1=1,op2=0xD`.
    /// Verified via BOTH oracles: objdump decodes it directly (`30 22 d1`
    /// -> mul16s a2,a2,a3) AND xtensa-modules.c's op1=1 table lists op2=13
    /// -> opcode 297 (mul16s).
    Mul16s {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `mul16u ar,as,at` = `(AR[s]&0xFFFF) * (AR[t]&0xFFFF)` (16x16 unsigned
    /// multiply). RRR `op1=1,op2=0xC`. Verified via BOTH oracles: objdump
    /// decodes it directly (`d0 23 c1` -> mul16u a2,a3,a13) AND
    /// xtensa-modules.c's op1=1 table lists op2=12 -> opcode 296 (mul16u).
    Mul16u {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `quou ar,as,at` = `AR[s] / AR[t]` (unsigned). RRR `op1=2,op2=0xC`.
    /// `xtensa-lx106-elf-objdump` can't decode this opcode (prints `excw`;
    /// the lx106 core lacks the 32-bit integer Divide option, the same class
    /// of gap as `s32ri`/`min`/`sext`) -- verified instead against
    /// `xtensa-modules.c`'s op1=2 decode table: op2=12 -> opcode 457 (quou);
    /// register roles (`arr` dest, `ars`/`art` sources, in that order)
    /// confirmed by the shared `Iclass_xt_iclass_div_args` operand-order
    /// table (`{arr,'o'},{ars,'i'},{art,'i'}` -- the same convention as every
    /// other RRR op in this file). `quos` (signed quotient) is absent from
    /// the firmware. Vector: `f0 2e c2` -> quou a2,a14,a15.
    ///
    /// **Divide by zero**: real Xtensa hardware (and QEMU's
    /// `target/xtensa/translate.c`, confirmed by reading its source) raises
    /// an actual `INTEGER_DIVIDE_BY_ZERO_CAUSE` architectural exception
    /// BEFORE the divide executes (`gen_zero_check` /
    /// `XTENSA_OP_DIVIDE_BY_ZERO`) -- contrary to this task's originating
    /// brief, which assumed Xtensa "does not trap" on this case. The
    /// general-exception-raise machinery (EXCCAUSE, the non-window vector)
    /// isn't modeled yet (scoped to a later M2a task; only the
    /// window-overflow/underflow vector exists so far, see
    /// `interp::raise_window_exception`). Until it lands, `quou`/`remu`/
    /// `rems` guard the zero-divisor case explicitly (a bare Rust `/0` or
    /// `%0` panics) and return 0 rather than panic, logging a `warn!` -- a
    /// deliberately visible placeholder, not an attempt to model the real
    /// hardware fault.
    Quou {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `remu ar,as,at` = `AR[s] % AR[t]` (unsigned). RRR `op1=2,op2=0xE`.
    /// Same objdump gap as [`Op::Quou`] (the lx106 core lacks the Divide
    /// option) -- verified against xtensa-modules.c: op1=2,op2=14 -> opcode
    /// 459 (remu). Divide-by-zero policy: see [`Op::Quou`]. Vector: `70 52
    /// e2` -> remu a5,a2,a7.
    Remu {
        r: u8,
        s: u8,
        t: u8,
    },
    /// `rems ar,as,at` = `(AR[s] as i32) % (AR[t] as i32)` (signed
    /// remainder; sign follows the dividend `AR[s]` -- Rust's `%` on `i32`
    /// already matches this truncating-division convention). RRR
    /// `op1=2,op2=0xF`. Same objdump gap as [`Op::Quou`] -- verified against
    /// xtensa-modules.c: op1=2,op2=15 -> opcode 460 (rems). Divide-by-zero
    /// policy: see [`Op::Quou`]. A SECOND guard is needed beyond the zero
    /// divisor: Rust's plain `%` on `i32` panics for `i32::MIN % -1` in
    /// every build profile (confirmed empirically -- the host `idiv`
    /// instruction can't compute it), even though the true architectural
    /// remainder is 0; `interp::arith::exec` uses `wrapping_rem` there
    /// instead, which returns 0 without panicking. Vector: `50 6a f2` ->
    /// rems a6,a10,a5.
    Rems {
        r: u8,
        s: u8,
        t: u8,
    },
    Witlb {
        t: u8,
        s: u8,
    },
    /// Write DTLB entry (`wdtlb at, as`) -- the data-side sibling of `witlb`,
    /// same operand layout. Boot uses it in the MMU-setup sequence.
    Wdtlb {
        t: u8,
        s: u8,
    },
    /// Invalidate ITLB entry (`iitlb as`); single register operand.
    Iitlb {
        s: u8,
    },
    /// Invalidate DTLB entry (`idtlb as`); single register operand.
    Idtlb {
        s: u8,
    },
    /// Write special register (`wsr.<sr> at`): `sr` is the 8-bit special-
    /// register number, `t` the source AR. Boot writes the MMU-config SRs
    /// (ITLBCFG/DTLBCFG/PTEVADDR); the interpreter routes the modeled ones
    /// (PS/VECBASE/WINDOWBASE/...) and logs the rest.
    Wsr {
        sr: u8,
        t: u8,
    },
    Isync,
    /// Data-memory synchronization barrier (`dsync`); no modeled pipeline
    /// effect, treated as a logged no-op like `isync`.
    Dsync,
    /// Jump register (`jx as`): unconditional jump to the address in AR[s].
    Jx {
        s: u8,
    },
    Unknown {
        word: u32,
    },
}

/// The result of decoding one instruction: the operation, plus how many
/// bytes it occupied in the instruction stream (1 for an undecodable op0
/// selector, 2 for narrow `.n` ops, 3 for standard ops).
#[derive(Debug)]
pub struct Decoded {
    pub op: Op,
    pub len: u8,
}

/// Sign-extends the low `bits` bits of `value` to i32, by shifting the field
/// into the top of the word and back down arithmetically. Shared by the
/// category decoders that carry a signed immediate (`arith::decode_rri8`'s
/// `movi`, `control::decode_calln`'s `call8`).
fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

/// Decodes the instruction at `bytes[0..]`. `pc` resolves PC-relative targets
/// (`l32r`, `call8`). Reads at most 3 bytes, never panics: a short slice or an
/// unrecognized opcode decodes to `Op::Unknown`.
pub fn decode(bytes: &[u8], pc: u32) -> Decoded {
    let Some(&b0) = bytes.first() else {
        return Decoded { op: Op::Unknown { word: 0 }, len: 0 };
    };
    // op0 = low nibble of byte0. Narrow (.n) ops live in 0x8..=0xD (2 bytes);
    // 0x0..=0x7 select a standard 3-byte format. 0xE/0xF aren't valid
    // Xtensa format selectors at all -- treat as a single undecodable byte
    // rather than assuming a (possibly wrong) 3-byte length.
    let op0 = b0 & 0xF;
    if op0 == 0xE || op0 == 0xF {
        return Decoded { op: Op::Unknown { word: b0 as u32 }, len: 1 };
    }
    let narrow = (0x8..=0xD).contains(&op0);
    let need = if narrow { 2 } else { 3 };
    if bytes.len() < need {
        let word = bytes.iter().enumerate().fold(0u32, |w, (i, &b)| w | ((b as u32) << (8 * i)));
        return Decoded { op: Op::Unknown { word }, len: bytes.len() as u8 };
    }
    let b1 = bytes[1];

    if narrow {
        let n1 = (b0 >> 4) & 0xF; // byte0 high nibble
        let n2 = b1 & 0xF; // byte1 low nibble
        let n3 = (b1 >> 4) & 0xF; // byte1 high nibble
        let op = mem::decode_narrow(op0, n1, n2, n3)
            .or_else(|| arith::decode_narrow(op0, n1, n2, n3))
            .or_else(|| control::decode_narrow(op0, n1, n2, n3))
            .unwrap_or(Op::Unknown { word: (b0 as u32) | ((b1 as u32) << 8) });
        return Decoded { op, len: 2 };
    }

    let b2 = bytes[2];
    let word = (b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16);
    // Nibbles common to the RRR/RRI8/RI16 formats (n1 = "t", n2 = "s",
    // n3 = "r", n4 = op1, n5 = op2). CALLN (op0=5) doesn't align to this --
    // its "n" field is only the low 2 bits of n1 -- so it's handled with its
    // own extraction below.
    let n1 = (b0 >> 4) & 0xF;
    let n2 = b1 & 0xF;
    let n3 = (b1 >> 4) & 0xF;
    let n4 = b2 & 0xF;
    let n5 = (b2 >> 4) & 0xF;

    let op = match op0 {
        // RI16 format: l32r at, <literal>.
        0x1 => mem::decode_ri16(n1, word, pc).unwrap_or(Op::Unknown { word }),
        // RRI8 format (LSAI group): r (n3) selects the sub-op.
        0x2 => mem::decode_rri8(n3, n1, n2, b2)
            .or_else(|| arith::decode_rri8(n3, n1, n2, b2))
            .unwrap_or(Op::Unknown { word }),
        // RRR format (op1 = n4, op2 = n5 select the specific op).
        0x0 => arith::decode_rrr(n4, n5, n3, n2, n1, word)
            .or_else(|| system::decode_rrr(n4, n5, n3, n2, n1, word))
            .or_else(|| control::decode_rrr(n4, n5, n3, n2, n1, word))
            .unwrap_or(Op::Unknown { word }),
        // CALLN format: n (bits 5:4 of byte0) selects call size; only CALL8
        // (n==2) is implemented.
        0x5 => control::decode_calln(b0, word, pc).unwrap_or(Op::Unknown { word }),
        // entry as, imm12*8 (frame size, always a multiple of 8).
        0x6 => control::decode_entry_fmt(n1, n2, n3, b2).unwrap_or(Op::Unknown { word }),
        _ => Op::Unknown { word },
    };
    Decoded { op, len: 3 }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These test the instruction-format skeleton itself (op0 handling, short
    // slices) rather than any one category's leaf op, so they stay here
    // rather than moving into a category file.

    #[test]
    fn unknown_opcode_is_reported_not_panicked() {
        // 0xff byte region (padding) must not panic.
        let d = decode(&[0xff, 0xff, 0xff], 0);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }

    #[test]
    fn undecodable_op0_reports_single_byte_length() {
        // op0 0xE/0xF aren't valid Xtensa format selectors (0x0..=0x7 select
        // the six 3-byte formats, 0x8..=0xD the narrow 2-byte ones) -- a
        // single undecodable byte, not a false 3-byte instruction.
        let d = decode(&[0xff, 0xff, 0xff], 0);
        assert_eq!(d.len, 1);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
        let d = decode(&[0xee, 0x00, 0x00], 0);
        assert_eq!(d.len, 1);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }

    #[test]
    fn short_slice_does_not_panic() {
        // A single byte of what would otherwise be a 2- or 3-byte
        // instruction must decode to Unknown, never index out of bounds.
        let d = decode(&[0x48], 0);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
        let d = decode(&[], 0);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }
}
