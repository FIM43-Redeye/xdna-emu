//! Legacy per-field decode types (`OperandField::operand_type`) and the name-derived addressing
//! mode and memory width of an encoding. The build script computes them
//! (`build_helpers/semantics.rs`); the interpreter decodes with them until the DEC and MEM
//! packages switch to [`EncodingSchema`](crate::aie2::isa::EncodingSchema).

/// Addressing mode extracted from the TableGen instruction name.
///
/// The AIE2 instruction naming convention encodes the addressing mode:
/// - `_idx_imm` -> IndexedImmediate:   *(ptr + imm), no pointer update
/// - `_idx` (no _imm) -> IndexedRegister: *(ptr + mN), no pointer update
/// - `_pstm_*_imm` -> PostModifyImmediate: t = *ptr, ptr += imm
/// - `_pstm_*` (no _imm) -> PostModifyRegister: t = *ptr, ptr += mN
///
/// The 2D / 3D descriptor modes ([`AddressingMode::TwoDim`], [`AddressingMode::ThreeDim`])
/// are not in this table: their record names carry no `_pstm_`, and the generated tables
/// take them from TableGen structure instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressingMode {
    #[default]
    Unknown,
    /// *(ptr + imm), pointer unchanged
    IndexedImmediate,
    /// *(ptr + mN), pointer unchanged
    IndexedRegister,
    /// t = *ptr; ptr += imm
    PostModifyImmediate,
    /// t = *ptr; ptr += mN
    PostModifyRegister,
    /// t = *ptr; then the 2D descriptor step: the pointer takes `dj` and the count advances,
    /// or takes `m` and the count restarts (`aie2::descriptors`, `execute::memory::agu`).
    /// Recognised from the descriptor register class of the operand and the TableGen
    /// `ag_pstm_2d` class, never from the instruction name.
    TwoDim,
    /// t = *ptr; then the 3D descriptor step over a pair of descriptors.
    ThreeDim,
}

/// Memory access width extracted from the instruction mnemonic.
///
/// The mnemonic suffix determines the width:
/// - `lda.s8` / `lda.u8` -> Byte (8-bit)
/// - `lda.s16` / `lda.u16` -> HalfWord (16-bit)
/// - `lda` (no suffix) -> Word (32-bit)
/// - `vlda.128` / `vldb.128` / `vst.128` -> QuadWord (128-bit)
/// - `vlda` / `vldb` / `vst` -> Vector256 (256-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InstrMemWidth {
    Byte,
    HalfWord,
    #[default]
    Word,
    /// 128-bit quadword access (vlda.128, vst.128, etc.)
    QuadWord,
    Vector256,
}

/// How a raw bit-field value should be decoded into an Operand.
///
/// Computed at build time from the operand's TableGen class name (`build_helpers/semantics.rs`).
/// This replaces all hand-written heuristics in the decoder with a single
/// data-driven dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandType {
    /// Simple register: raw value + base_offset IS the register index.
    /// base_offset is non-zero for register subclasses (e.g., eRS8 = r16-r23
    /// uses base_offset=16, so a 3-bit field value of 3 maps to r19).
    Register(RegisterKind),
    /// Register with a base offset for subclass encoding.
    RegisterWithOffset(RegisterKind, u8),
    /// Composite register: multiple register classes share one field,
    /// discriminated by low bits. Requires an inverse encoder function.
    CompositeRegister(CompositeEncoder),
    /// Immediate value with sign and scale.
    Immediate { signed: bool, scale: i32 },
    /// Lock ID.
    LockId,
    /// Unknown (safe fallback: treated as unsigned immediate).
    Unknown,
}

/// Which simple register file a field encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterKind {
    /// eR: scalar general-purpose registers (r0-r31)
    Scalar,
    /// eP: pointer registers (p0-p7)
    Pointer,
    /// eM: post-modify registers (m0-m7)
    ModifierM,
    /// eDN: dimension size registers (dn0-dn7)
    ModifierDN,
    /// eDJ: dimension stride/jump registers (dj0-dj7)
    ModifierDJ,
    /// eDC: dimension count registers (dc0-dc7)
    ModifierDC,
    /// eW*, mWm: 256-bit vector registers
    Vector256,
    /// mXm, eX*: 512-bit vector registers
    Vector512,
    /// eYs: 1024-bit vector registers (y2-y5)
    Vector1024,
    /// eAM, eBM, eCM, mAMm, mBMm: accumulator registers
    Accumulator,
    /// mCRm: control registers (crRnd, crSat, crSRSSign, crVaddSign, etc.)
    Control,
    /// mQX*: sparse composite registers qx0-qx3 (vector data + sparsity mask)
    SparseQx,
    /// eL: 64-bit scalar register pairs (l0..l7 = {r16:r17}..{r30:r31}).
    /// Encoding is pair index; decoded register = index * 2 + base_offset.
    /// The even register is emitted as ScalarReg; the execution handler
    /// reads reg+1 for the upper 32 bits.
    ScalarPair,
}

/// Which Peano composite encoder function was used.
///
/// Each variant corresponds to a `get<Name>OpValue` function in
/// `AIE2MCCodeEmitterRegOperandDef.h`. The decoder inverts these
/// to recover the original register from the encoded bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeEncoder {
    /// mLdaScl, mSclSt, mSclMS: load/store scalar composite
    LdaScl,
    /// mMvSclSrc, mMvSclDst, mMvSclDstCg: move scalar composite
    MvSclSrc,
    /// mLdaCg, mLdbCg: load destination composite group
    LdaCg,
    /// mAluCg: ALU destination composite group
    AluCg,
    /// mMvAMWQDst: accumulator/weight-queue destination
    MvAMWQDst,
    /// mMvAMWQSrc: accumulator/weight-queue source
    MvAMWQSrc,
    /// mMvBMXSrc: buffer/cross source
    MvBMXSrc,
    /// mMvBMXDst: buffer/cross destination
    MvBMXDst,
    /// eRS4: scalar register subset (r16-r31)
    ERS4,
    /// mShflDst: shuffle destination
    ShflDst,
    /// mWm_1: vector register with bit rearrangement
    Wm1,
    /// mQXHLb: cross high/low byte
    QXHLb,
}
