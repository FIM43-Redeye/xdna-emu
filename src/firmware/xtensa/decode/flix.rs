//! FLIX (VLIW) bundle decode for the management core's two 8-byte formats.
//!
//! `op0` nibble 0xe selects `xt_format1` (three slots) and 0xf selects
//! `xt_format2` (two slots). Each slot is decoded independently from its own
//! bits, so the decoder is a set of per-slot descriptor tables (opcode
//! mask/value plus operand field layouts) and a small assembler from decoded
//! operands to [`Op`].
//!
//! # Provenance
//!
//! Every encoding fact here (slot extents, opcode bit patterns, operand field
//! positions, immediate scalings, the `xt_format2` validity rules) was derived
//! by black-box probing of `xtdis`, a disassembler that links GNU binutils
//! libisa with this core's configuration. No binutils source, generated ISA
//! tables, or other Xtensa decoder/simulator source was consulted. The probing
//! was:
//!
//! 1. Flip each bundle bit of a decodable bundle and record which slot's
//!    extracted bits (and mnemonic) change; this gave the slot extents below.
//! 2. For each slot, decode every possible slot value (2^24, 2^20 and 2^16,
//!    with the other slots zero) and group by mnemonic. Each mnemonic's value
//!    set was an exact mask/value cube (member count equal to 2^free bits), and
//!    everything not claimed by a mnemonic decoded as `excw`.
//! 3. For each mnemonic, flip each free slot bit across sampled members and
//!    record which operand's raw field moved and which raw bit, giving the
//!    field layouts; then enumerate each narrow operand's raw values to
//!    recover the raw-to-printed mapping (sign extension, scaling, offsets).
//! 4. For `xt_format2`, sweep the upper bundle bits to find which of them gate
//!    validity and what the second slot is, since its encodings are wider
//!    than its 24 extracted bits.
//!
//! # Layout
//!
//! `xt_format1`: slot0 = bundle bits 4..28 (24 bits), slot1 = bits 28..48
//! (20 bits), slot2 = bits 48..64 (16 bits). Slots have different opcode
//! spaces: slot0 is the widest (loads/stores, `l32r`, ALU), slot1 carries
//! ALU plus `j`/`jx`, slot2 only 16-bit narrow-style ALU ops.
//!
//! `xt_format2`: slot0 = bits 4..28 and decodes exactly like `xt_format1`
//! slot0. Slot1 only ever holds the `*.w18` branch family (none of which is
//! supported) or an inert op; which one is determined by bits 54..63, so a
//! real slot1 rejects the whole bundle. Bit 63 set makes the bundle
//! undecodable. This decoder supports a single real op per `xt_format2`
//! bundle, so a real slot0 op alongside a real slot1 op is also `Unknown`.
//!
//! Inert slots (`excw`, or the slot's designated `nop`) are the complement of
//! the tables below: any slot value matching no descriptor is inert. A
//! recognised op outside the supported mnemonic set rejects the bundle
//! (`Mn::Unsupported`), as does a reserved form (`slli` by 32).

use super::{branch, mem, sign_extend, Op};
use Interp::{Plus, Raw, Shl, Signed, SmallAddend, SmallImm, Sub32};

/// Decodes an `xt_format1` (`op0 == 0xe`) bundle. `bytes` >= 8 (caller
/// guarantees); `pc` anchors pc-relative operands.
pub(super) fn decode_format1(bytes: &[u8], pc: u32) -> Op {
    let word = bundle_word(bytes);
    let slots = [
        decode_slot(SLOT0, (word >> 4) as u32 & 0xff_ffff, pc),
        decode_slot(SLOT1, (word >> 28) as u32 & 0xf_ffff, pc),
        decode_slot(SLOT2, (word >> 48) as u32 & 0xffff, pc),
    ];
    let mut ops = Vec::new();
    for slot in slots {
        match slot {
            Slot::Inert => {}
            Slot::Real(op) => ops.push(op),
            Slot::Reject => return unknown(bytes),
        }
    }
    match ops.len() {
        0 => Op::Nop,
        1 => ops.pop().expect("one op"),
        _ => Op::Flix1 { ops },
    }
}

/// Decodes an `xt_format2` (`op0 == 0xf`) bundle. `bytes` >= 8 (caller
/// guarantees); `pc` anchors pc-relative operands.
pub(super) fn decode_format2(bytes: &[u8], pc: u32) -> Op {
    let word = bundle_word(bytes);
    if word >> 63 != 0 || format2_slot1_is_real(word) {
        return unknown(bytes);
    }
    match decode_slot(SLOT0, (word >> 4) as u32 & 0xff_ffff, pc) {
        Slot::Inert => Op::Nop,
        Slot::Real(op) => op,
        Slot::Reject => unknown(bytes),
    }
}

fn bundle_word(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("caller guarantees 8 bytes"))
}

fn unknown(bytes: &[u8]) -> Op {
    Op::Unknown { word: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) }
}

/// Whether `xt_format2`'s second slot holds a real (non-inert) op. Its opcode
/// is bits 54..63; the `*.w18` branch family occupies 2..=6 and the even
/// values up to 40 outright, and the compare-with-zero branches (42..=48)
/// only when bits 28..32 are clear. Everything else is `excw`/`nop`.
fn format2_slot1_is_real(word: u64) -> bool {
    let opcode = (word >> 54) & 0x1ff;
    match opcode {
        2..=6 => true,
        8..=40 => opcode % 2 == 0,
        42 | 44 | 46 | 48 => (word >> 28) & 0xf == 0,
        _ => false,
    }
}

/// Result of decoding one slot.
enum Slot {
    /// `excw`/`nop`: no architectural effect, dropped from the bundle.
    Inert,
    Real(Op),
    /// A recognised op outside the supported set, or a reserved form.
    Reject,
}

fn decode_slot(table: &[Entry], bits: u32, pc: u32) -> Slot {
    let Some(entry) = table.iter().find(|e| bits & e.mask == e.value) else {
        return Slot::Inert;
    };
    let mut values = [0u32; 4];
    for (value, operand) in values.iter_mut().zip(entry.ops) {
        *value = operand.extract(bits);
    }
    match assemble(entry.mn, &values, pc) {
        Some(op) => Slot::Real(op),
        None => Slot::Reject,
    }
}

/// How a raw operand field becomes the printed value.
#[derive(Clone, Copy)]
enum Interp {
    Raw,
    /// Sign-extend from `bits`, then shift left.
    Signed(u32, u32),
    Shl(u32),
    Plus(i32),
    /// `32 - raw`: shift amount of `slli` (raw 0 is the reserved 32).
    Sub32,
    /// Raw 0 means -1, 1..=15 themselves.
    SmallAddend,
    /// 7-bit field where 96..=127 mean -32..=-1.
    SmallImm,
}

/// An operand field: up to two slot-bit ranges `(offset, width)`, the first
/// range supplying the low raw bits (width 0 = unused).
#[derive(Clone, Copy)]
struct Operand {
    parts: [(u8, u8); 2],
    how: Interp,
}

const fn field(lo: u8, width: u8, how: Interp) -> Operand {
    Operand { parts: [(lo, width), (0, 0)], how }
}

const fn reg(lo: u8) -> Operand {
    field(lo, 4, Interp::Raw)
}

const fn split(low: (u8, u8), high: (u8, u8), how: Interp) -> Operand {
    Operand { parts: [low, high], how }
}

impl Operand {
    fn extract(&self, slot_bits: u32) -> u32 {
        let (mut raw, mut at) = (0u32, 0u32);
        for &(lo, width) in &self.parts {
            raw |= ((slot_bits >> lo) & ((1u32 << width) - 1)) << at;
            at += width as u32;
        }
        match self.how {
            Interp::Raw => raw,
            Interp::Signed(bits, shl) => (sign_extend(raw, bits) << shl) as u32,
            Interp::Shl(n) => raw << n,
            Interp::Plus(c) => raw.wrapping_add(c as u32),
            Interp::Sub32 => 32 - raw,
            Interp::SmallAddend => {
                if raw == 0 {
                    u32::MAX
                } else {
                    raw
                }
            }
            Interp::SmallImm => {
                if raw >= 96 {
                    raw.wrapping_sub(128)
                } else {
                    raw
                }
            }
        }
    }
}

/// One opcode: a slot value matches when `bits & mask == value`. Operands are
/// in the order the oracle prints them, which is the order [`assemble`] reads.
struct Entry {
    mask: u32,
    value: u32,
    mn: Mn,
    ops: &'static [Operand],
}

/// Supported mnemonics (named as their `Op` variants) plus `Unsupported` for
/// recognised ops this emulator does not model.
#[derive(Clone, Copy)]
enum Mn {
    Add,
    Sub,
    Addx2,
    Addx4,
    Addx8,
    Subx8,
    And,
    Or,
    Xor,
    Min,
    Minu,
    Maxu,
    Moveqz,
    Movnez,
    Mull,
    Mul16u,
    Mul16s,
    Src,
    Neg,
    Abs,
    Ssr,
    Ssl,
    Ssai,
    Nsau,
    Slli,
    Srai,
    Srli,
    Sll,
    Srl,
    Sext,
    Extui,
    L8ui,
    L16ui,
    L16si,
    L32i,
    S8i,
    S16i,
    S32i,
    Movi,
    MoviN,
    Addi,
    AddiN,
    Addmi,
    L32r,
    MovN,
    J,
    Jx,
    Unsupported,
}

/// Builds the `Op` for a matched entry from its operand values (in print
/// order). `None` rejects the bundle.
fn assemble(mn: Mn, v: &[u32; 4], pc: u32) -> Option<Op> {
    let (a, b, c, d) = (v[0] as u8, v[1] as u8, v[2] as u8, v[3] as u8);
    Some(match mn {
        Mn::Add => Op::Add { r: a, s: b, t: c },
        Mn::Sub => Op::Sub { r: a, s: b, t: c },
        Mn::Addx2 => Op::Addx2 { r: a, s: b, t: c },
        Mn::Addx4 => Op::Addx4 { r: a, s: b, t: c },
        Mn::Addx8 => Op::Addx8 { r: a, s: b, t: c },
        Mn::Subx8 => Op::Subx8 { r: a, s: b, t: c },
        Mn::And => Op::And { r: a, s: b, t: c },
        Mn::Or => Op::Or { r: a, s: b, t: c },
        Mn::Xor => Op::Xor { r: a, s: b, t: c },
        Mn::Min => Op::Min { r: a, s: b, t: c },
        Mn::Minu => Op::Minu { r: a, s: b, t: c },
        Mn::Maxu => Op::Maxu { r: a, s: b, t: c },
        Mn::Moveqz => Op::Moveqz { r: a, s: b, t: c },
        Mn::Movnez => Op::Movnez { r: a, s: b, t: c },
        Mn::Mull => Op::Mull { r: a, s: b, t: c },
        Mn::Mul16u => Op::Mul16u { r: a, s: b, t: c },
        Mn::Mul16s => Op::Mul16s { r: a, s: b, t: c },
        Mn::Src => Op::Src { r: a, s: b, t: c },
        Mn::Neg => Op::Neg { r: a, t: b },
        Mn::Abs => Op::Abs { r: a, t: b },
        Mn::Ssr => Op::Ssr { s: a },
        Mn::Ssl => Op::Ssl { s: a },
        Mn::Ssai => Op::Ssai { imm: a },
        Mn::Nsau => Op::Nsau { t: a, s: b },
        // Raw 0 encodes the reserved shift by 32.
        Mn::Slli if v[2] >= 32 => return None,
        Mn::Slli => Op::Slli { r: a, s: b, imm: c },
        Mn::Srai => Op::Srai { r: a, t: b, imm: c },
        Mn::Srli => Op::Srli { r: a, t: b, imm: c },
        Mn::Sll => Op::Sll { r: a, s: b },
        Mn::Srl => Op::Srl { r: a, t: b },
        Mn::Sext => Op::Sext { r: a, s: b, imm: c },
        Mn::Extui => Op::Extui { r: a, t: b, shiftimm: c, maskimm: d },
        Mn::L8ui => Op::L8ui { t: a, s: b, imm: v[2] },
        Mn::L16ui => Op::L16ui { t: a, s: b, imm: v[2] },
        Mn::L16si => Op::L16si { t: a, s: b, imm: v[2] },
        Mn::L32i => Op::L32i { t: a, s: b, imm: v[2] },
        Mn::S8i => Op::S8i { t: a, s: b, imm: v[2] },
        Mn::S16i => Op::S16i { t: a, s: b, imm: v[2] },
        Mn::S32i => Op::S32i { t: a, s: b, imm: v[2] },
        Mn::Movi => Op::Movi { t: a, imm: v[1] as i32 },
        Mn::MoviN => Op::MoviN { t: a, imm: v[1] as i32 },
        Mn::Addi => Op::Addi { t: a, s: b, imm: v[2] as i32 },
        Mn::AddiN => Op::AddiN { t: a, s: b, imm: v[2] as i32 },
        Mn::Addmi => Op::Addmi { t: a, s: b, imm: v[2] as i32 },
        // The 16-bit literal offset is encoded as in the base RI16 format, so
        // reuse its target formula (it reads the offset from word bits 8..24).
        Mn::L32r => return mem::decode_ri16(a, v[1] << 8, pc),
        Mn::MovN => Op::MovN { t: a, s: b },
        Mn::J => Op::J { target: branch::pc_rel_target(pc, v[0], 18) },
        Mn::Jx => Op::Jx { s: a },
        Mn::Unsupported => return None,
    })
}

/// slot0 opcode table; unlisted slot values are inert.
static SLOT0: &[Entry] = &[
    Entry { mask: 0xfff0f0, value: 0x060010, mn: Mn::Abs, ops: &[reg(8), reg(0)] },
    Entry { mask: 0xfff000, value: 0x080000, mn: Mn::Add, ops: &[reg(8), reg(4), reg(0)] },
    Entry {
        mask: 0xf00f00,
        value: 0x200c00,
        mn: Mn::Addi,
        ops: &[reg(0), reg(4), field(12, 8, Signed(8, 0))],
    },
    Entry {
        mask: 0xf00f00,
        value: 0x200d00,
        mn: Mn::Addmi,
        ops: &[reg(0), reg(4), field(12, 8, Signed(8, 8))],
    },
    Entry { mask: 0xfff000, value: 0x090000, mn: Mn::Addx2, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x0a0000, mn: Mn::Addx4, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x0b0000, mn: Mn::Addx8, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x010000, mn: Mn::And, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x033000, mn: Mn::Unsupported, ops: &[] }, // clamps
    Entry {
        mask: 0xf0e000,
        value: 0x004000,
        mn: Mn::Extui,
        ops: &[reg(8), reg(0), split((4, 4), (12, 1), Raw), field(16, 4, Plus(1))],
    },
    Entry {
        mask: 0xf00f00,
        value: 0x200900,
        mn: Mn::L16si,
        ops: &[reg(0), reg(4), field(12, 8, Shl(1))],
    },
    Entry {
        mask: 0xf00f00,
        value: 0x200100,
        mn: Mn::L16ui,
        ops: &[reg(0), reg(4), field(12, 8, Shl(1))],
    },
    Entry {
        mask: 0xf00f00,
        value: 0x200200,
        mn: Mn::L32i,
        ops: &[reg(0), reg(4), field(12, 8, Shl(2))],
    },
    Entry { mask: 0xf00000, value: 0x100000, mn: Mn::L32r, ops: &[reg(0), field(4, 16, Raw)] },
    Entry {
        mask: 0xf00f00,
        value: 0x200000,
        mn: Mn::L8ui,
        ops: &[reg(0), reg(4), field(12, 8, Raw)],
    },
    Entry { mask: 0xfff000, value: 0x053000, mn: Mn::Unsupported, ops: &[] }, // max
    Entry { mask: 0xfff000, value: 0x073000, mn: Mn::Maxu, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x043000, mn: Mn::Min, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x063000, mn: Mn::Minu, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xffef00, value: 0x006000, mn: Mn::MovN, ops: &[reg(0), reg(4)] },
    Entry { mask: 0xfff000, value: 0x083000, mn: Mn::Moveqz, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x0b3000, mn: Mn::Unsupported, ops: &[] }, // movgez
    Entry {
        mask: 0xf00f00,
        value: 0x200a00,
        mn: Mn::Movi,
        ops: &[reg(0), split((12, 8), (4, 4), Signed(12, 0))],
    },
    Entry { mask: 0xfff000, value: 0x0a3000, mn: Mn::Unsupported, ops: &[] }, // movltz
    Entry { mask: 0xfff000, value: 0x093000, mn: Mn::Movnez, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x0d1000, mn: Mn::Mul16s, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x0c1000, mn: Mn::Mul16u, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x082000, mn: Mn::Mull, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff0f0, value: 0x060000, mn: Mn::Neg, ops: &[reg(8), reg(0)] },
    Entry { mask: 0xffff00, value: 0x040e00, mn: Mn::Unsupported, ops: &[] }, // nsa
    Entry { mask: 0xffff00, value: 0x040f00, mn: Mn::Nsau, ops: &[reg(0), reg(4)] },
    Entry { mask: 0xfff000, value: 0x020000, mn: Mn::Or, ops: &[reg(8), reg(4), reg(0)] },
    Entry {
        mask: 0xf00f00,
        value: 0x200500,
        mn: Mn::S16i,
        ops: &[reg(0), reg(4), field(12, 8, Shl(1))],
    },
    Entry {
        mask: 0xf00f00,
        value: 0x200600,
        mn: Mn::S32i,
        ops: &[reg(0), reg(4), field(12, 8, Shl(2))],
    },
    Entry {
        mask: 0xf00f00,
        value: 0x200400,
        mn: Mn::S8i,
        ops: &[reg(0), reg(4), field(12, 8, Raw)],
    },
    Entry {
        mask: 0xfff000,
        value: 0x023000,
        mn: Mn::Sext,
        ops: &[reg(8), reg(4), field(0, 4, Plus(7))],
    },
    Entry { mask: 0xfff00f, value: 0x0a1000, mn: Mn::Sll, ops: &[reg(8), reg(4)] },
    Entry {
        mask: 0xfef000,
        value: 0x001000,
        mn: Mn::Slli,
        ops: &[reg(8), reg(4), split((0, 4), (16, 1), Sub32)],
    },
    Entry { mask: 0xfff0f0, value: 0x0b1000, mn: Mn::Unsupported, ops: &[] }, // sra
    Entry {
        mask: 0xfef000,
        value: 0x021000,
        mn: Mn::Srai,
        ops: &[reg(8), reg(0), split((4, 4), (16, 1), Raw)],
    },
    Entry { mask: 0xfff000, value: 0x081000, mn: Mn::Src, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff0f0, value: 0x091000, mn: Mn::Srl, ops: &[reg(8), reg(0)] },
    Entry { mask: 0xfff000, value: 0x041000, mn: Mn::Srli, ops: &[reg(8), reg(0), reg(4)] },
    Entry { mask: 0xffff0f, value: 0x040300, mn: Mn::Unsupported, ops: &[] }, // ssa8b
    Entry { mask: 0xffff0f, value: 0x040200, mn: Mn::Unsupported, ops: &[] }, // ssa8l
    Entry {
        mask: 0xffff0e,
        value: 0x040400,
        mn: Mn::Ssai,
        ops: &[split((4, 4), (0, 1), Raw)],
    },
    Entry { mask: 0xffff0f, value: 0x040100, mn: Mn::Ssl, ops: &[reg(4)] },
    Entry { mask: 0xffff0f, value: 0x040000, mn: Mn::Ssr, ops: &[reg(4)] },
    Entry { mask: 0xfff000, value: 0x0c0000, mn: Mn::Sub, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x0d0000, mn: Mn::Unsupported, ops: &[] }, // subx2
    Entry { mask: 0xfff000, value: 0x0e0000, mn: Mn::Unsupported, ops: &[] }, // subx4
    Entry { mask: 0xfff000, value: 0x0f0000, mn: Mn::Subx8, ops: &[reg(8), reg(4), reg(0)] },
    Entry { mask: 0xfff000, value: 0x030000, mn: Mn::Xor, ops: &[reg(8), reg(4), reg(0)] },
];

/// slot1 opcode table; unlisted slot values are inert.
static SLOT1: &[Entry] = &[
    Entry { mask: 0x0ff000, value: 0x092000, mn: Mn::Add, ops: &[reg(4), reg(8), reg(0)] },
    Entry {
        mask: 0x0f0000,
        value: 0x060000,
        mn: Mn::Addi,
        ops: &[reg(0), reg(8), split((4, 4), (12, 4), Signed(8, 0))],
    },
    Entry {
        mask: 0x0f0000,
        value: 0x070000,
        mn: Mn::Addmi,
        ops: &[reg(0), reg(8), split((4, 4), (12, 4), Signed(8, 8))],
    },
    Entry { mask: 0x0ff000, value: 0x094000, mn: Mn::Addx2, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x098000, mn: Mn::Addx4, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x093000, mn: Mn::Addx8, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x095000, mn: Mn::And, ops: &[reg(4), reg(8), reg(0)] },
    Entry {
        mask: 0x0e0000,
        value: 0x040000,
        mn: Mn::Extui,
        ops: &[reg(4), reg(0), field(12, 5, Raw), field(8, 4, Plus(1))],
    },
    Entry { mask: 0x0c0000, value: 0x0c0000, mn: Mn::J, ops: &[field(0, 18, Raw)] },
    Entry { mask: 0x0ff0ff, value: 0x0a3010, mn: Mn::Jx, ops: &[reg(8)] },
    Entry { mask: 0x0ff0f0, value: 0x0a3000, mn: Mn::MovN, ops: &[reg(0), reg(8)] },
    Entry { mask: 0x0ff000, value: 0x096000, mn: Mn::Moveqz, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x097000, mn: Mn::Unsupported, ops: &[] }, // movgez
    Entry {
        mask: 0x0f0000,
        value: 0x080000,
        mn: Mn::Movi,
        ops: &[reg(0), field(4, 12, Signed(12, 0))],
    },
    Entry { mask: 0x0ff000, value: 0x099000, mn: Mn::Unsupported, ops: &[] }, // movltz
    Entry { mask: 0x0ff000, value: 0x09a000, mn: Mn::Movnez, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x09c000, mn: Mn::Mul16s, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x09b000, mn: Mn::Mul16u, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x09d000, mn: Mn::Mull, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0fff00, value: 0x0a5000, mn: Mn::Neg, ops: &[reg(4), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x09e000, mn: Mn::Or, ops: &[reg(4), reg(8), reg(0)] },
    Entry {
        mask: 0x0ff000,
        value: 0x09f000,
        mn: Mn::Sext,
        ops: &[reg(4), reg(8), field(0, 4, Plus(7))],
    },
    Entry { mask: 0x0ff00f, value: 0x0a6000, mn: Mn::Sll, ops: &[reg(4), reg(8)] },
    Entry {
        mask: 0x0fe000,
        value: 0x090000,
        mn: Mn::Slli,
        ops: &[reg(4), reg(8), split((0, 4), (12, 1), Sub32)],
    },
    Entry { mask: 0x0fff00, value: 0x0a5100, mn: Mn::Unsupported, ops: &[] }, // sra
    Entry {
        mask: 0x0fe000,
        value: 0x0a0000,
        mn: Mn::Srai,
        ops: &[reg(4), reg(0), field(8, 5, Raw)],
    },
    Entry { mask: 0x0ff000, value: 0x0a2000, mn: Mn::Src, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0fff00, value: 0x0a5200, mn: Mn::Srl, ops: &[reg(4), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x0a4000, mn: Mn::Srli, ops: &[reg(4), reg(0), reg(8)] },
    Entry { mask: 0x0ff0ff, value: 0x0a3020, mn: Mn::Ssl, ops: &[reg(8)] },
    Entry { mask: 0x0ff000, value: 0x0a8000, mn: Mn::Sub, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x0ff000, value: 0x0b0000, mn: Mn::Xor, ops: &[reg(4), reg(8), reg(0)] },
];

/// slot2 opcode table; unlisted slot values are inert.
static SLOT2: &[Entry] = &[
    Entry { mask: 0x00ff00, value: 0x00d000, mn: Mn::Abs, ops: &[reg(4), reg(0)] },
    Entry { mask: 0x00f000, value: 0x002000, mn: Mn::Add, ops: &[reg(4), reg(8), reg(0)] },
    Entry {
        mask: 0x00f000,
        value: 0x003000,
        mn: Mn::AddiN,
        ops: &[reg(4), reg(8), field(0, 4, SmallAddend)],
    },
    Entry { mask: 0x00f000, value: 0x004000, mn: Mn::Addx2, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x00f000, value: 0x005000, mn: Mn::Addx4, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x00f000, value: 0x006000, mn: Mn::And, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x00f0f0, value: 0x00c080, mn: Mn::MovN, ops: &[reg(0), reg(8)] },
    Entry {
        mask: 0x00f080,
        value: 0x00c000,
        mn: Mn::MoviN,
        ops: &[reg(8), field(0, 7, SmallImm)],
    },
    Entry { mask: 0x00ff00, value: 0x00d100, mn: Mn::Neg, ops: &[reg(4), reg(0)] },
    Entry { mask: 0x00f000, value: 0x007000, mn: Mn::Or, ops: &[reg(4), reg(8), reg(0)] },
    Entry {
        mask: 0x00f000,
        value: 0x008000,
        mn: Mn::Sext,
        ops: &[reg(4), reg(8), field(0, 4, Plus(7))],
    },
    Entry { mask: 0x00ff00, value: 0x00d200, mn: Mn::Unsupported, ops: &[] }, // sra
    Entry {
        mask: 0x00e000,
        value: 0x00e000,
        mn: Mn::Srai,
        ops: &[reg(4), reg(0), field(8, 5, Raw)],
    },
    Entry { mask: 0x00ff00, value: 0x00d400, mn: Mn::Srl, ops: &[reg(4), reg(0)] },
    Entry { mask: 0x00f000, value: 0x009000, mn: Mn::Srli, ops: &[reg(4), reg(0), reg(8)] },
    Entry { mask: 0x00f000, value: 0x00a000, mn: Mn::Sub, ops: &[reg(4), reg(8), reg(0)] },
    Entry { mask: 0x00f000, value: 0x00b000, mn: Mn::Xor, ops: &[reg(4), reg(8), reg(0)] },
];

#[cfg(test)]
#[path = "flix_tests.rs"]
mod tests;
