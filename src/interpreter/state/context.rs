//! Execution context for an AIE2 core.
//!
//! The `ExecutionContext` holds all processor state needed to execute
//! instructions: registers, program counter, flags, and execution statistics.
//!
//! This replaces the placeholder `ExecutionContext` in `traits.rs`.
//!
//! # Timing Support
//!
//! All execution is cycle-accurate. The `TimingContext` (always present) includes:
//! - `HazardDetector`: Tracks RAW/WAW/WAR register hazards
//! - `MemoryModel`: Tracks memory bank conflicts
//! - `LatencyTable`: Operation latency lookup
//!
//! ```ignore
//! let ctx = ExecutionContext::new();
//! assert!(ctx.has_timing()); // Always true
//! ```

use super::registers::{
    AccumulatorRegisterFile, Bypass, MaskRegisterFile, ModifierRegisterFile, PointerRegisterFile,
    ScalarRegisterFile, VectorRegisterFile,
};
use crate::interpreter::bundle::Operand;
use crate::interpreter::traits::{Flags, StateAccess};

// Re-export types that were previously defined in this file. Downstream code
// that imports via `super::context::EventType` etc. continues to compile.
pub use super::event_trace::{EventLog, EventType, TimestampedEvent};
pub use super::timing_context::{
    DeferredPcKind, PendingBranch, PendingDeferredEvent, SrsConfig, TimingContext, TRACE_PC_PIPELINE_DEPTH,
};

/// Snapshot of a ZOL boundary event for trace + diagnostic use.
///
/// Returned by [`ExecutionContext::check_hardware_loop`] when the just-fetched
/// PC matches LE and LC was non-zero (i.e. a hardware loop iteration just
/// completed). Carries the LC counter values around the decrement and the
/// LE PC at which the boundary was observed so callers can route the event
/// to the trace unit (mode-2 LC frames) or event log without re-reading
/// register state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopBoundaryInfo {
    pub lc_before: u32,
    pub lc_after: u32,
    pub le_pc: u32,
}

// ============================================================================
// Load Latency: Pending Write Queue
// ============================================================================

/// A deferred register write from a memory load operation.
///
/// AIE2 memory loads have a 7-cycle pipeline latency (AIE2Schedule.td
/// operandcycles[0] = 7 for all load itineraries). The compiler pipelines
/// multiple loads to the same register, relying on this latency to keep
/// earlier values alive until they are consumed. Without deferred writes,
/// later loads instantly overwrite earlier values, breaking the pipeline.
#[derive(Debug, Clone)]
pub struct PendingWrite {
    /// Destination register to write when ready.
    pub dest: Operand,
    /// Value to write (scalar: lower 32 bits; vector: use vec_value).
    pub scalar_value: u32,
    /// Vector value for vector loads (None for scalar loads).
    pub vec_value: Option<[u32; 8]>,
    /// Accumulator access width for AccumReg destinations.
    /// Determines which lanes to write (quarter/half/full).
    pub accum_width: Option<crate::interpreter::decode::register_map::AccumWidth>,
    /// Cycle at which this write becomes visible.
    pub ready_cycle: u64,
    /// Cycle at which this write was issued (for forwarding discrimination).
    ///
    /// Used to distinguish pending writes from previous bundles (forwardable)
    /// vs writes from the current bundle (not forwardable -- VLIW semantics
    /// require same-bundle reads to see pre-execution values).
    pub issued_cycle: u64,
    /// Full-width (up to 1024-bit) accumulator result for a deferred matmul
    /// write. When set, the AccumReg write uses this wide value instead of
    /// `vec_value`/`accum_width` (which only cover <=512-bit chunks). Boxed so
    /// the common scalar/vector/pointer pending writes stay small.
    pub wide_accum: Option<Box<DeferredAccum>>,
}

/// A deferred full-width matmul accumulator result.
///
/// The AIE2 matrix-multiply (VMUL/VMAC) writes a 1024-bit `cm` register (or a
/// 512-bit `bm` half) with the MAC result latency. This carries the computed
/// accumulator so `commit_pending_writes` can apply it at `ready_cycle`.
#[derive(Debug, Clone)]
pub struct DeferredAccum {
    /// The computed accumulator lanes ([u64; 16] = 1024 bits).
    pub value: [u64; 16],
    /// True for a 512-bit `bm` half write (low 8 lanes); false for the full
    /// 1024-bit `cm` wide-pair write.
    pub is_half: bool,
}

/// A deferred partial-word store awaiting its data register read.
///
/// AIE2 partial-word stores (st.s8/st.u8/st.s16/st.u16) use a read-modify-write
/// pipeline (II_STHB in AIE2Schedule.td). The address is computed at issue time,
/// but the data register is not read until 7 cycles later (operand latency = 7).
/// This allows the compiler to schedule computations that produce the store value
/// AFTER the store instruction itself (a common software-pipelining technique).
#[derive(Debug, Clone)]
pub struct PendingStore {
    /// Target byte address in core memory space (computed at issue time).
    pub address: u32,
    /// Source register operand to read when the store data is sampled.
    pub source: Operand,
    /// Memory access width (Byte or HalfWord).
    pub width: super::super::bundle::slot::MemWidth,
    /// Cycle at which the data register is read and the write occurs.
    pub ready_cycle: u64,
    /// PC of the bundle that issued this store.
    ///
    /// Used to model the AIE2 zero-overhead-loop back-edge: the fetch redirect
    /// at LE flushes a store issued *in the LE bundle* (the redirect shadow),
    /// while stores issued earlier in the body have already entered the
    /// decoupled store pipeline and still commit. See `check_hardware_loop`.
    pub issue_pc: u32,
}

/// Latency for partial-word store data register read (II_STHB operand 0).
///
/// AIE2Schedule.td specifies OperandCycles=[7,1,1] for II_STHB. The
/// operand cycle of 7 means a producer with latency 1 can be scheduled
/// up to 6 cycles AFTER the store. The actual hardware read occurs at
/// cycle 6 (before the cycle-7 write-back boundary), giving 1 cycle of
/// scheduling slack. Using 6 here ensures that:
/// - add_21_i8: add.nc at store+5 → result at store+6 → read at store+6 ✓
/// - add_12_i8: add.nc at store+6 → result at store+7 → read at store+6
///   reads the OLD r0 from the previous iteration (add.nc hasn't written yet) ✓
const PARTIAL_WORD_STORE_DATA_LATENCY: u64 =
    xdna_archspec::aie2::processor::PARTIAL_STORE_DATA_LATENCY as u64;

/// A deferred VUNPACK awaiting its late source-register read.
///
/// II_VUNPACK (AIE2Schedule.td) has OperandCycles=[7,7,7]: both the x-reg
/// destination write AND the w-reg source read happen at pipeline stage 7.
/// The compiler exploits the late read by scheduling the producer of the
/// source register AFTER the vunpack (Peano emits `vunpack x4, wl2` four
/// bundles before the `vsel.8 x2, ...` whose result it consumes; verified
/// against NPU1 silicon, vector fuzzer seed 7). The unpack must therefore
/// sample its source at issue+6 (one bundle before the cycle-7 write-back
/// boundary, same rationale as PARTIAL_WORD_STORE_DATA_LATENCY) and land
/// its destination at issue+7.
#[derive(Debug, Clone)]
pub struct PendingUnpack {
    /// Source w-register index (256-bit), sampled at `ready_bundle`.
    pub src_reg: u8,
    /// Destination x-register index (512-bit pair base).
    pub dest_reg: u8,
    /// Input lane width in bits.
    pub bits_i: u32,
    /// Output lane width in bits.
    pub bits_o: u32,
    /// Sign-extend (true) vs zero-extend.
    pub signed: bool,
    /// Issued-bundle count (`bundle_seq`) at which the source is read.
    pub ready_bundle: u64,
}

/// VUNPACK source-register read latency in issued bundles (II_VUNPACK
/// operand cycle 7, sampled one bundle before write-back; see PendingUnpack).
const VUNPACK_SRC_READ_LATENCY: u64 = 6;

/// A deferred accumulator add/sub awaiting its stage-3 source read.
///
/// II_VADDMAC / II_VADDMACf (AIE2Schedule.td) read their accumulator
/// sources at operand cycle 3, not at issue. The compiler relies on this:
/// Peano bundles `vconv.fp32.bf16 bmh2, wl2` (result latency 2) WITH the
/// `vadd.f bmh0, bmh0, bmh2` that consumes it, and the slot order executes
/// the vadd first. Sampling at issue read the stale accumulator (verified
/// against NPU1 silicon, vector fuzzer seed 4: low output half = a.lo +
/// b.HI because the prior bundle's conv was the freshest visible write).
/// All control bits are resolved at issue; only the accumulator source
/// read is deferred.
#[derive(Debug, Clone)]
pub struct PendingAccAdd {
    pub acc1_reg: u8,
    pub acc2_reg: u8,
    pub dst_reg: u8,
    pub negate_acc1: bool,
    pub negate_acc2: bool,
    pub zero_acc1: bool,
    pub shift16: bool,
    pub is_float: bool,
    pub is_wide: bool,
    /// Issued-bundle count at which the accumulator sources are read.
    pub ready_bundle: u64,
}

/// Accumulator add/sub source read latency in issued bundles (operand
/// cycle 3, sampled one bundle before the boundary; see PendingAccAdd).
const VACC_SRC_READ_LATENCY: u64 = 2;

/// A deferred fused `vlda.ups` awaiting its stage-7 shift-register read.
///
/// II_VLDA_UPS (AIE2Schedule.td) reads its shift S-register at operand
/// cycle 7, not at issue (OperandCycles `[9,7,1,1,...]`: acc dest at 9,
/// S-reg at 7). The conversion happens after the load data returns, so the
/// compiler schedules the `mov sN, rM` that sets the shift several bundles
/// AFTER the vlda.ups it controls. Sampling at issue read the stale shift
/// (verified against NPU1 silicon, vector fuzzer seeds 1142-1448: upshift
/// modes 1-7 produced the unshifted input). The load data, destination, and
/// element types are resolved at issue; only the shift amount is sampled at
/// issue+6, one bundle before the cycle-7 boundary (same rationale as
/// PendingUnpack). Plain register II_VUPS reads its shift at issue and is
/// not deferred.
#[derive(Debug, Clone)]
pub struct PendingUpsLoad {
    /// Vector data loaded from memory at issue.
    pub vec_data: [u32; 8],
    /// S-register holding the shift amount, sampled at `ready_bundle`.
    pub shift_reg: u8,
    /// Destination accumulator register.
    pub dest_reg: u8,
    /// Input element type.
    pub from: xdna_archspec::aie2::isa::ElementType,
    /// Output (accumulator lane) element type.
    pub to: xdna_archspec::aie2::isa::ElementType,
    /// Half (bml/bmh, 512-bit) vs full cm (1024-bit) destination.
    pub is_half: bool,
    /// Issued-bundle count at which the shift register is read.
    pub ready_bundle: u64,
}

/// VLDA.UPS shift-register read latency in issued bundles (operand cycle 7,
/// sampled one bundle before the boundary; see PendingUpsLoad).
const VLDA_UPS_SHIFT_READ_LATENCY: u64 = 6;

/// Maximum loop-body byte span for which the LE-bundle partial-word store is
/// flushed on the zero-overhead-loop back-edge.
///
/// Loop body = le-ls, always a multiple of the 16-byte AIE2 fetch packet.
/// Calibrated and validated against real silicon over ~3,500 differential-fuzz
/// kernels that park a partial-word store (st.s8/st.s16) in the LE bundle:
///
/// - body  > 96 bytes (>=7 fetch packets): HW COMMITS the store. 4/4 observed
///   (seed_1826, seed_1781, +2 in the widened HW sweep). The larger body has
///   already streamed past the front-end squash point when the LE bundle
///   issues.
/// - body == 96 bytes (6 fetch packets): HW FLUSHES the store in 107/109
///   observed cases -- the every-Nth-element drop. This is the dominant
///   first-order effect the threshold captures.
///
/// This is an OBSERVED first-order model, not the exact micro-architectural
/// rule. The flush is genuinely cycle-exact: ~1.8% of 96-byte cases COMMIT
/// instead (seed_1086, seed_1340 in the widened sweep), and they are
/// structurally indistinguishable from the flush cases (same body, same
/// store+induction LE bundle, same producer class -- cf. seed_1048 which
/// flushes). Three static discriminators were refuted: producer latency,
/// body size alone (this threshold), and LE-bundle composition. Resolving the
/// 96-byte split needs cycle-level pipeline visibility (aiesimulator) and more
/// commit samples than the hardware naturally produces (~0.04% of seeds).
/// Among simple models this threshold is the least-wrong: 2 mispredicts across
/// the corpus, vs 6 for unconditional flush and 109 for never-flush.
/// See docs/superpowers/findings/2026-05-31-bugb-zol-store-flush-investigation.md.
const ZOL_FLUSH_MAX_BODY_BYTES: u32 = 0x60;

/// Complete execution context for an AIE2 core.
///
/// Contains all register files and execution state needed for instruction
/// execution. Designed for efficient access patterns:
///
/// - Hot path: scalar regs, PC, flags (accessed every instruction)
/// - Warm path: pointer/modifier regs (accessed on memory ops)
/// - Cold path: vector/accumulator regs (accessed on vector ops)
#[derive(Clone)]
pub struct ExecutionContext {
    // === Hot path (accessed every cycle) ===
    /// Program counter.
    pc: u32,

    /// Condition flags.
    flags: Flags,

    /// Scalar general purpose registers (r0-r31).
    pub scalar: ScalarRegisterFile,

    // === Warm path (memory operations) ===
    /// Pointer registers (p0-p7).
    pub pointer: PointerRegisterFile,

    /// Modifier registers (m0-m7).
    pub modifier: ModifierRegisterFile,

    // === Cold path (vector operations) ===
    /// Vector registers (v0-v31).
    pub vector: VectorRegisterFile,

    /// Accumulator registers (acc0-acc7).
    pub accumulator: AccumulatorRegisterFile,

    /// The q register file (q0-q3): AIE2's 128-bit "mask registers". Named
    /// `mask` for their primary role (compare/sparsity masks), but the same
    /// registers also carry 128-bit vector data (`vmov q,w; st q`). See
    /// [`MaskRegisterFile`] -- `q` (the asm mnemonic) and `mask` are the same file.
    pub mask: MaskRegisterFile,

    // === Statistics ===
    /// Total cycles executed.
    pub cycles: u64,

    /// Total instructions executed.
    pub instructions: u64,

    /// Stall cycles (waiting on locks, DMA, etc.).
    pub stall_cycles: u64,

    // === Control ===
    /// Core is halted.
    pub halted: bool,

    /// Stack pointer register mapping.
    sp_reg: SpRegister,
    /// Dedicated SP register value (used when sp_reg == Dedicated).
    sp_value: u32,

    /// Link register (alias to a scalar reg).
    /// By convention, often r0 or r14.
    lr_reg: u8,

    // === Timing ===
    /// Timing context for cycle-accurate execution.
    /// Always present - all execution is cycle-accurate.
    pub timing: TimingContext,

    // === VLIW Bundle Support ===
    /// Snapshot of register files for VLIW parallel read semantics.
    /// When set, reads use the snapshot instead of live registers.
    /// This ensures all reads in a bundle see pre-execution values.
    scalar_snapshot: Option<ScalarRegisterFile>,
    pointer_snapshot: Option<PointerRegisterFile>,
    modifier_snapshot: Option<ModifierRegisterFile>,

    /// Same-bundle scalar immediate-MOV writes, for shift/control-operand
    /// forwarding only. AIE2 writes an S register at pipeline stage E1 but
    /// UPS/SRS read their shift operand at E7 (llvm-aie AIE2Schedule.td gives
    /// RAW latency -5), so a shift-setup `MOV sN,#imm` bundled with its consumer
    /// must forward the new value. This is deliberately NOT consulted by general
    /// scalar reads -- those keep pure read-old VLIW semantics (e.g. a same-bundle
    /// `st rX` must capture rX's pre-write value). (reg, value), last writer wins.
    bundle_shift_forward: Vec<(u8, u32)>,

    // === Branch Delay Slot Support ===
    /// Pending branch waiting for delay slots to complete.
    /// AIE2 has 5-cycle branch delay slots - after a branch is decided,
    /// the next 5 instructions still execute before the branch takes effect.
    pending_branch: Option<PendingBranch>,

    // === Load Latency Pipeline ===
    /// Deferred register writes from memory load operations.
    /// Loads have a 7-cycle pipeline latency (AIE2Schedule.td). The write to
    /// the destination register is deferred until `ready_cycle` is reached.
    pending_writes: Vec<PendingWrite>,

    /// Result latency (itinerary `operand_cycles[0]`) of the slot op currently
    /// being executed, set by the bundle executor before each slot dispatch.
    /// Vector-register writes pass this to the register file's bypass-network
    /// visibility model (`VectorRegisterFile::queue_write`). A value of 0 means
    /// "write immediately" -- direct test-path helpers that bypass the executor.
    pub result_latency: u8,

    /// Result bypass class of the slot op currently being executed (set
    /// alongside `result_latency`). Determines whether the result forwards to
    /// ALU consumers at issue+1 (`Mov`) or is visible only at full latency
    /// (`No`). See `VectorRegisterFile::resolve`.
    pub result_bypass: Bypass,

    /// Monotonic counter of issued bundles, incremented once per executed
    /// bundle by the cycle-accurate executor. This is the issue-slot-relative
    /// clock the compiler's result latencies are expressed in; it drives the
    /// vector register file's bypass visibility (`advance_bundle`) and, unlike
    /// `cycles`, does not advance on stalls (a stalled bundle re-issues).
    pub bundle_seq: u64,

    // === Partial-Word Store Pipeline ===
    /// Deferred partial-word stores awaiting data register read.
    /// Partial-word stores (st.s8, st.u8, etc.) use a RMW pipeline where
    /// the data register is sampled 7 cycles after issue. The address and
    /// post-modify are computed at issue time; only the data read is deferred.
    pending_stores: Vec<PendingStore>,

    // === Late-Read VUNPACK Pipeline ===
    /// Deferred VUNPACKs awaiting their stage-7 source read (see PendingUnpack).
    pending_unpacks: Vec<PendingUnpack>,

    /// Deferred accumulator add/subs awaiting their stage-3 source read
    /// (see PendingAccAdd).
    pending_acc_adds: Vec<PendingAccAdd>,

    /// Deferred MAC-family multiplies awaiting their stage-3 accumulator
    /// source read (see execute::vector_matmul::PendingMatmul).
    pending_matmuls: Vec<crate::interpreter::execute::vector_matmul::PendingMatmul>,

    /// Deferred fused vlda.ups awaiting their stage-7 shift read
    /// (see PendingUpsLoad).
    pending_ups_loads: Vec<PendingUpsLoad>,

    // === Vector Control Registers ===
    /// SRS/UPS control register state.
    ///
    /// In hardware, these are fields within the Core_CR MMIO register:
    ///   [1:0]  SATURATION_MODE  (crSat)
    ///   [5:2]  ROUND_MODE       (crRnd)
    ///   [17]   SRS_SIGN         (crSRSSign)
    ///
    /// Set by instructions like `set_satmode()` / `set_rnd()` (aie API) which
    /// compile to control register writes. Currently defaulted to the values
    /// most mlir-aie compiled code expects; will be wired to instruction-level
    /// writes when control register decoding is implemented.
    pub srs_config: SrsConfig,
}

/// Which register to use as stack pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpRegister {
    /// Use pointer register (p0-p7).
    Pointer(u8),
    /// Use scalar register (r0-r31).
    Scalar(u8),
    /// Dedicated SP register (AIE2: SPLReg<12, "sp">).
    ///
    /// AIE2's stack pointer is a separate special register, not aliased
    /// to any general-purpose pointer register. `PADDB_sp_imm` and
    /// `st/lda [sp, #offset]` operate on this dedicated register, while
    /// p0-p7 remain freely available for user code.
    Dedicated,
}

impl Default for SpRegister {
    fn default() -> Self {
        SpRegister::Dedicated
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Read the register initialization pattern from XDNA_EMU_REG_INIT.
///
/// Controls what value fills all registers before normal initialization.
/// Any register not explicitly initialized will retain this pattern,
/// making uninitialized reads immediately visible in output.
///
/// Values: "zero" -> 0, "<hex>" -> parsed, unset -> 0xDEADBEEF.
fn reg_init_pattern() -> u32 {
    match std::env::var("XDNA_EMU_REG_INIT").as_deref() {
        Ok("zero") => 0x0000_0000,
        Ok(hex) => u32::from_str_radix(hex, 16).unwrap_or(0xDEAD_BEEF),
        Err(_) => 0xDEAD_BEEF,
    }
}

impl ExecutionContext {
    /// Create a new execution context with all state zeroed.
    ///
    /// All execution is cycle-accurate with:
    /// - Hazard detection (RAW, WAW, WAR)
    /// - Memory bank conflict modeling
    /// - Event tracing (can be disabled via `timing.disable_tracing()`)
    ///
    /// CORE_ID is initialized to 0 (col=0, row=0). For production use,
    /// prefer `new_for_tile(col, row)` which sets the correct tile identity.
    pub fn new() -> Self {
        Self::new_for_tile(0, 0)
    }

    /// Create a new execution context for a specific tile position.
    ///
    /// All register files are first blasted with a sentinel pattern
    /// (default 0xDEADBEEF, configurable via XDNA_EMU_REG_INIT env var).
    /// Normal initialization then overwrites the fields it's responsible for.
    /// Any register NOT explicitly initialized retains the sentinel, making
    /// uninitialized reads immediately obvious in output.
    pub fn new_for_tile(col: u8, row: u8) -> Self {
        let pattern = reg_init_pattern();
        let mut timing = TimingContext::new();
        timing.enable_tracing();

        // Create register files and fill with sentinel pattern
        let mut scalar = ScalarRegisterFile::new();
        let mut pointer = PointerRegisterFile::new();
        let mut modifier = ModifierRegisterFile::new();
        let mut vector = VectorRegisterFile::new();
        let mut accumulator = AccumulatorRegisterFile::new();
        let mut mask = MaskRegisterFile::new();

        scalar.fill_pattern(pattern);
        pointer.fill_pattern(pattern);
        modifier.fill_pattern(pattern);
        vector.fill_pattern(pattern);
        accumulator.fill_pattern(pattern);
        mask.fill_pattern(pattern);

        // Normal initialization overwrites what it should.
        scalar.set_core_id(col, row);

        // Zero-initialize SRS shift registers (s0-s3 = ScalarReg 40-43).
        // Hardware resets these to 0. Compilers assume s0=0 at function
        // entry and may read it before writing (e.g., vlda.ups uses s0
        // for shift before the compiler emits `mov s0, r0`).
        for i in 40..=43 {
            scalar.write(i, 0);
        }

        Self {
            pc: 0,
            flags: Flags::default(),
            scalar,
            pointer,
            modifier,
            vector,
            accumulator,
            mask,
            cycles: 0,
            instructions: 0,
            stall_cycles: 0,
            halted: false,
            sp_reg: SpRegister::default(),
            sp_value: 0,
            lr_reg: super::registers::LR_REG_INDEX,
            timing,
            scalar_snapshot: None,
            pointer_snapshot: None,
            modifier_snapshot: None,
            bundle_shift_forward: Vec::new(),
            pending_branch: None,
            pending_writes: Vec::new(),
            result_latency: 0,
            result_bypass: Bypass::No,
            bundle_seq: 0,
            pending_stores: Vec::new(),
            pending_unpacks: Vec::new(),
            pending_acc_adds: Vec::new(),
            pending_matmuls: Vec::new(),
            pending_ups_loads: Vec::new(),
            srs_config: SrsConfig::default(),
        }
    }

    /// Create a new context with initial stack pointer.
    pub fn with_stack(stack_addr: u32) -> Self {
        let mut ctx = Self::new();
        ctx.set_sp(stack_addr);
        ctx
    }

    /// Check if cycle-accurate timing is enabled.
    ///
    /// Always returns `true` - all execution is cycle-accurate.
    #[inline]
    pub fn has_timing(&self) -> bool {
        true
    }

    /// Get the timing context.
    #[inline]
    pub fn timing_context(&self) -> &TimingContext {
        &self.timing
    }

    /// Get mutable timing context.
    #[inline]
    pub fn timing_context_mut(&mut self) -> &mut TimingContext {
        &mut self.timing
    }

    /// Get the program counter.
    #[inline]
    pub fn pc(&self) -> u32 {
        self.pc
    }

    /// Set the program counter.
    #[inline]
    pub fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    /// Advance PC by the given offset.
    #[inline]
    pub fn advance_pc(&mut self, offset: u32) {
        self.pc = self.pc.wrapping_add(offset);
    }

    /// Check the hardware zero-overhead loop (ZLS/ZLE/LC).
    ///
    /// AIE2 cores have a hardware loop mechanism: the LE register holds the
    /// address of the last instruction in the loop body. After that instruction
    /// executes, if LC > 0, the hardware decrements LC. If the new LC is
    /// still > 0, it redirects PC to LS; otherwise it falls through.
    ///
    /// This gives exactly LC iterations: the loop body executes, LC
    /// decrements, and the back-edge is taken while the decremented LC
    /// remains positive. When LC reaches 0, the current execution is
    /// the last and control falls through.
    ///
    /// Note: this differs from JNZD, which tests-before-decrement and
    /// needs trip_count - 1 loaded (see AIEBaseHardwareLoops.cpp). ZLS
    /// hardware loops load trip_count directly into LC.
    ///
    /// `fetch_pc` is the address of the instruction that just executed (before
    /// advance_pc). The check compares this against LE, because LE marks the
    /// last instruction to execute before looping, not the first instruction
    /// after the loop.
    ///
    /// If a branch resolved this cycle, the branch target takes priority
    /// and this check is skipped by the caller.
    #[inline]
    pub fn check_hardware_loop(&mut self, fetch_pc: u32) -> Option<LoopBoundaryInfo> {
        let le = self.scalar.read(super::registers::LE_REG_INDEX);
        if fetch_pc != le {
            return None;
        }
        let lc = self.scalar.read(super::registers::LC_REG_INDEX);
        if lc == 0 {
            return None;
        }
        let new_lc = lc - 1;
        self.scalar.write(super::registers::LC_REG_INDEX, new_lc);
        if new_lc > 0 {
            // Back-edge: the program-control-unit redirects fetch from LE to LS.
            // A partial-word store (st.s8/st.s16) issued *in the LE bundle* is in
            // that redirect shadow and is flushed before it commits to memory --
            // the every-4th-element drop seen on silicon for a Peano-unrolled i8
            // loop that parks a store at LE. Stores issued earlier in the body
            // have already entered the decoupled store pipeline (and commit at
            // E11 even across the back-edge -- software pipelining relies on
            // this), and loads/register writes retire into the register file, so
            // only stores with issue_pc == LE are dropped.
            //
            // The squash only reaches the LE store when the whole loop body fits
            // in the front-end fetch window (<= ZOL_FLUSH_MAX_BODY_BYTES, six
            // 16-byte fetch packets). A larger body has already streamed past
            // the squash point by the time the LE bundle issues, so its store
            // commits -- see ZOL_FLUSH_MAX_BODY_BYTES for the silicon basis.
            let ls = self.scalar.read(super::registers::LS_REG_INDEX);
            let body_bytes = le.wrapping_sub(ls);
            if body_bytes <= ZOL_FLUSH_MAX_BODY_BYTES {
                self.pending_stores.retain(|ps| ps.issue_pc != le);
            }

            self.pc = ls;
            log::debug!(
                "ZLS loop: executed instr at LE=0x{:X}, LC {} -> {}, jumping to LS=0x{:X}",
                le,
                lc,
                new_lc,
                ls
            );
        } else {
            log::debug!("ZLS loop: final iteration at LE=0x{:X}, LC {} -> 0, falling through", le, lc);
        }
        Some(LoopBoundaryInfo { lc_before: lc, lc_after: new_lc, le_pc: le })
    }

    /// Get the condition flags.
    #[inline]
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// Set the condition flags.
    #[inline]
    pub fn set_flags(&mut self, flags: Flags) {
        self.flags = flags;
    }

    /// Get the stack pointer value.
    #[inline]
    pub fn sp(&self) -> u32 {
        match self.sp_reg {
            SpRegister::Pointer(r) => self.pointer.read(r),
            SpRegister::Scalar(r) => self.scalar.read(r),
            SpRegister::Dedicated => self.sp_value,
        }
    }

    /// Set the stack pointer value.
    #[inline]
    pub fn set_sp(&mut self, value: u32) {
        match self.sp_reg {
            SpRegister::Pointer(r) => self.pointer.write(r, value),
            SpRegister::Scalar(r) => self.scalar.write(r, value),
            SpRegister::Dedicated => self.sp_value = value,
        }
    }

    /// Get the link register value.
    #[inline]
    pub fn lr(&self) -> u32 {
        self.scalar.read(self.lr_reg)
    }

    /// Set the link register value.
    #[inline]
    pub fn set_lr(&mut self, value: u32) {
        self.scalar.write(self.lr_reg, value);
    }

    /// Push a value onto the stack (decrement SP, write value).
    pub fn push(&mut self, _value: u32) {
        let new_sp = self.sp().wrapping_sub(4);
        self.set_sp(new_sp);
        // Note: actual memory write must be done by caller
    }

    /// Pop a value from the stack (read value, increment SP).
    /// Returns the stack address where the value should be read from.
    pub fn pop(&mut self) -> u32 {
        let sp = self.sp();
        self.set_sp(sp.wrapping_add(4));
        sp
    }

    /// Record one instruction executed.
    #[inline]
    pub fn record_instruction(&mut self, cycles: u64) {
        self.instructions += 1;
        self.cycles += cycles;
    }

    /// Record stall cycles.
    ///
    /// A stalled cycle FREEZES the core pipeline -- the cycle passes but no
    /// bundle issues, so no in-flight pipeline stage advances. The deferred
    /// queues keyed on `bundle_seq` (VUNPACK, acc-add, matmul, ups-load, and
    /// the vector register file's bypass network) are stall-immune by
    /// construction. The two queues keyed on the wall clock -- `pending_writes`
    /// (load/ALU write-back) and `pending_stores` (the partial-word store's late
    /// data read) -- are not, so their deadlines slip by the stall here. Without
    /// this, every stall cycle would land each deferred write one issued bundle
    /// EARLY, and a software-pipelined consumer would read the next iteration's
    /// value instead of the one the compiler scheduled it to read.
    #[inline]
    pub fn record_stall(&mut self, cycles: u64) {
        self.stall_cycles += cycles;
        self.cycles += cycles;
        for pw in &mut self.pending_writes {
            pw.ready_cycle += cycles;
        }
        for ps in &mut self.pending_stores {
            ps.ready_cycle += cycles;
        }
    }

    // === VLIW Bundle Support ===

    /// Begin a VLIW bundle execution.
    ///
    /// Takes a snapshot of scalar, pointer, and modifier registers so that
    /// all reads within the bundle see the pre-execution values, implementing
    /// VLIW parallel semantics where all operations execute "simultaneously".
    #[inline]
    pub fn begin_bundle(&mut self) {
        self.scalar_snapshot = Some(self.scalar.clone());
        self.pointer_snapshot = Some(self.pointer.clone());
        self.modifier_snapshot = Some(self.modifier.clone());
        self.bundle_shift_forward.clear();
        // Vector/accumulator/mask use an in-file shadow snapshot (the read
        // surface is too broad to route through context accessors). Same
        // pure-VLIW read-old/write-new semantics: in-bundle reads see
        // pre-execution values.
        self.vector.begin_bundle();
        self.accumulator.begin_bundle();
        self.mask.begin_bundle();
    }

    /// End a VLIW bundle execution.
    ///
    /// Clears register snapshots. Writes that occurred during bundle
    /// execution are already committed to the live registers.
    #[inline]
    pub fn end_bundle(&mut self) {
        self.scalar_snapshot = None;
        self.pointer_snapshot = None;
        self.modifier_snapshot = None;
        self.vector.end_bundle();
        self.accumulator.end_bundle();
        self.mask.end_bundle();
    }

    /// Record a same-bundle scalar immediate-MOV write (`MOV sN, #imm`) for
    /// shift/control-operand forwarding. Called by the bundle pre-pass before
    /// any slot executes, so a shift read sees the new value even though the
    /// MOV's own slot executes later in the bundle.
    #[inline]
    pub fn record_bundle_scalar_imm(&mut self, reg: u8, value: u32) {
        self.bundle_shift_forward.push((reg, value));
    }

    /// Forwarded value of a scalar register written by a same-bundle immediate
    /// MOV, for shift/control operand reads (E7) ONLY. `None` if no such write
    /// occurred this bundle. General scalar reads must NOT use this -- they keep
    /// pure read-old VLIW semantics. Last writer in the bundle wins.
    #[inline]
    pub fn shift_forward(&self, reg: u8) -> Option<u32> {
        self.bundle_shift_forward.iter().rev().find(|(r, _)| *r == reg).map(|(_, v)| *v)
    }

    // === Deferred Write Methods ===

    /// Queue a deferred register write with pipeline latency.
    ///
    /// The write will become visible after `latency` cycles. This models
    /// the AIE2 pipeline: a write with latency N issued at cycle C becomes
    /// visible at cycle C+N (committed by `commit_pending_writes()` at the
    /// start of that cycle).
    ///
    /// Used for:
    /// - Memory loads (latency 7): `queue_scalar_load()` / `queue_vector_load()`
    /// - Pointer register writes (latency 1): `queue_pointer_write()`
    ///
    /// At branch boundaries, `delay_pending_writes(1)` adds an extra cycle
    /// to model the loss of forwarding when the pipeline is flushed.
    pub fn queue_scalar_load(&mut self, dest: Operand, value: u32, latency: u64) {
        self.pending_writes.push(PendingWrite {
            dest,
            scalar_value: value,
            vec_value: None,
            accum_width: None,
            ready_cycle: self.cycles + latency,
            issued_cycle: self.cycles,
            wide_accum: None,
        });
    }

    /// Queue a vector load result for deferred register write.
    pub fn queue_vector_load(&mut self, dest: Operand, value: [u32; 8], latency: u64) {
        self.pending_writes.push(PendingWrite {
            dest,
            scalar_value: 0,
            vec_value: Some(value),
            accum_width: None,
            ready_cycle: self.cycles + latency,
            issued_cycle: self.cycles,
            wide_accum: None,
        });
    }

    /// Queue an accumulator load with width metadata for quarter/half/full writes.
    pub fn queue_accum_load(
        &mut self,
        dest: Operand,
        value: [u32; 8],
        width: crate::interpreter::decode::register_map::AccumWidth,
        latency: u64,
    ) {
        self.pending_writes.push(PendingWrite {
            dest,
            scalar_value: 0,
            vec_value: Some(value),
            accum_width: Some(width),
            ready_cycle: self.cycles + latency,
            issued_cycle: self.cycles,
            wide_accum: None,
        });
    }

    /// Queue a deferred full-width matmul accumulator write.
    ///
    /// The AIE2 matrix-multiply (VMUL/VMAC) result is not visible in the `cm`
    /// accumulator until the MAC result latency (5 cycles, II_VMAC
    /// operand_cycles[0]=5) elapses. Deferring the write models that pipeline:
    /// in a software-pipelined batch loop, the stores that drain a tile read
    /// the previous tile's accumulator while the next tile's VMUL is still in
    /// flight. `is_half` selects a 512-bit `bm` half write vs a 1024-bit `cm`
    /// wide-pair write.
    ///
    /// FIXME(bypass-model): the accumulator file is NOT yet part of the AIE2
    /// bypass/forwarding network that the vector register file now models
    /// (`VectorRegisterFile::resolve`). This path applies a single flat
    /// result-latency deferral and does not distinguish `VEC_Bypass` (MAC->MAC
    /// accumulator forwarding, effective latency L-1) from `NoBypass`
    /// consumers (e.g. accumulator stores). If a kernel surfaces a tight
    /// MAC->MAC vs MAC->store accumulator-visibility split, fold the
    /// accumulator file into the same `(l_def, def_bypass)` resolution rule:
    /// extract the `VEC_Bypass` id (see `Bypass::from_forwarding_id`, the live
    /// mapping, which today collapses any nonzero id to `Mov` because only
    /// vector-register results consume it) and give `AccumulatorRegisterFile`
    /// the same in-flight overlay.
    pub fn queue_matmul_accum_write(&mut self, dest: Operand, value: [u64; 16], is_half: bool, latency: u64) {
        self.pending_writes.push(PendingWrite {
            dest,
            scalar_value: 0,
            vec_value: None,
            accum_width: None,
            ready_cycle: self.cycles + latency,
            issued_cycle: self.cycles,
            wide_accum: Some(Box::new(DeferredAccum { value, is_half })),
        });
    }

    /// Land vector-register writes that have reached visibility and advance the
    /// register file's issued-bundle clock. Call once per issued bundle, at the
    /// start, before snapshotting. Vector writes themselves go through
    /// `self.vector.queue_write` (driven by the executor's `result_latency` /
    /// `result_bypass`), and the AIE2 forwarding network is modeled inside
    /// `VectorRegisterFile` (see its `resolve`).
    pub fn advance_vector_bundle(&mut self) {
        self.vector.advance_bundle(self.bundle_seq);
    }

    /// Queue a pointer register write with pipeline latency.
    ///
    /// Pointer register writes from MOV/MOVXM/PADDA/PADDB have latency 1
    /// in the scheduling model (II_MOV_P, II_PADDA). The write becomes
    /// visible in the next sequential bundle.
    ///
    /// At branch boundaries (delay slot -> branch target), forwarding is
    /// unavailable. `delay_pending_writes(1)` is called to push the write
    /// one cycle later, so the branch target's first instruction reads
    /// the pre-write value. This matches observed hardware behavior where
    /// Peano generates "prologue stores" that rely on pointer registers
    /// not being ready at function entry.
    pub fn queue_pointer_write(&mut self, reg: u8, value: u32, latency: u64) {
        if reg == super::registers::SP_PTR_INDEX {
            // Dedicated SP -- write immediately, no pipeline delay
            self.sp_value = value;
            return;
        }
        self.pending_writes.push(PendingWrite {
            dest: Operand::PointerReg(reg),
            scalar_value: value,
            vec_value: None,
            accum_width: None,
            ready_cycle: self.cycles + latency,
            issued_cycle: self.cycles,
            wide_accum: None,
        });
    }

    /// Delay all pending writes by the specified number of cycles.
    ///
    /// Called when a branch is taken (delay slots exhausted, PC redirected)
    /// to model the loss of forwarding at pipeline flush boundaries.
    /// On real hardware, the pipeline flush means that results from the
    /// last delay slot instruction(s) are not forwarded to the branch
    /// target's first instruction, adding effective latency.
    pub fn delay_pending_writes(&mut self, extra_cycles: u64) {
        for pw in &mut self.pending_writes {
            pw.ready_cycle += extra_cycles;
        }
    }

    /// Flush all pending writes immediately, ignoring cycle timing.
    ///
    /// Used in unit tests where memory operations are tested in isolation
    /// without the cycle-accurate executor. Production code should use
    /// `commit_pending_writes()` instead.
    #[cfg(test)]
    pub fn flush_pending_writes(&mut self) {
        self.force_commit_all_pending();
    }

    /// Force-commit ALL pending writes regardless of ready_cycle.
    ///
    /// Models hardware scoreboard stall: when an instruction reads a
    /// register that has a pending load, the hardware stalls until the
    /// load completes. We approximate this by committing all pending
    /// writes before the dependent instruction reads.
    pub fn force_commit_all_pending(&mut self) {
        // Deferred matmuls first: completing them queues their destination
        // writes, which the flush below then applies.
        crate::interpreter::execute::vector_matmul::drain_pending_matmuls(self);
        let writes: Vec<_> = self.pending_writes.drain(..).collect();
        for pw in &writes {
            self.apply_pending_write(pw);
        }
    }

    /// Commit all pending writes whose ready_cycle has been reached.
    ///
    /// Call this at the start of each cycle, BEFORE begin_bundle(), so that
    /// load results become visible at the correct time and are captured by
    /// the VLIW snapshot.
    pub fn commit_pending_writes(&mut self) {
        let current = self.cycles;
        // Commit writes whose ready_cycle has been reached.
        //
        // Using `<=` matches LLVM's LoadLatency semantics: a load with
        // latency L issued at cycle C has ready_cycle = C + L, and a
        // dependent instruction at cycle C + L can read the new value.
        //
        // When multiple pending writes target the SAME register, we must
        // apply them in ready_cycle order so the latest one wins. We sort
        // the ready writes by ready_cycle before applying.
        let mut ready_indices: Vec<usize> = Vec::new();
        for (i, pw) in self.pending_writes.iter().enumerate() {
            if pw.ready_cycle <= current {
                ready_indices.push(i);
            }
        }
        // Sort by ready_cycle so earlier writes are applied first
        // (later writes to the same register overwrite them correctly)
        ready_indices.sort_by_key(|&i| self.pending_writes[i].ready_cycle);
        // Drain in reverse index order to avoid invalidating indices
        // We collect the writes first, then remove them
        let mut ready_writes: Vec<PendingWrite> = Vec::with_capacity(ready_indices.len());
        // Remove from highest index to lowest to preserve indices
        ready_indices.sort_unstable_by(|a, b| b.cmp(a));
        for &idx in &ready_indices {
            ready_writes.push(self.pending_writes.swap_remove(idx));
        }
        // Sort by (ready_cycle, issued_cycle) ascending and apply.
        // When two writes have the same ready_cycle (e.g., a delayed
        // delay-slot write and a branch-target write), the one issued
        // later wins because it is applied last.
        ready_writes.sort_by_key(|pw| (pw.ready_cycle, pw.issued_cycle));
        for pw in &ready_writes {
            self.apply_pending_write(pw);
        }
    }

    /// Apply a single pending write to the register file.
    fn apply_pending_write(&mut self, pw: &PendingWrite) {
        match &pw.dest {
            Operand::ScalarReg(r) => self.scalar.write(*r, pw.scalar_value),
            Operand::PointerReg(r) if *r == super::registers::SP_PTR_INDEX => {
                self.sp_value = pw.scalar_value;
            }
            Operand::PointerReg(r) => {
                self.pointer.write(*r, pw.scalar_value);
            }
            Operand::ModifierReg(r) => self.modifier.write(*r, pw.scalar_value),
            Operand::VectorReg(r) => {
                if let Some(vec) = &pw.vec_value {
                    self.vector.write(*r, *vec);
                }
            }
            Operand::AccumReg(r) => {
                if let Some(wide) = &pw.wide_accum {
                    // Deferred matmul result: full 1024-bit cm pair, or 512-bit
                    // bm half.
                    if wide.is_half {
                        let mut half = [0u64; 8];
                        half.copy_from_slice(&wide.value[..8]);
                        self.accumulator.write(*r, half);
                    } else {
                        self.accumulator.write_wide(*r, wide.value);
                    }
                } else if let Some(vec) = &pw.vec_value {
                    self.apply_accum_write(*r, vec, pw.accum_width);
                }
            }
            Operand::ControlReg(id) => {
                if let Some(vec) = &pw.vec_value {
                    if *id >= 16 && *id <= 19 {
                        // q0-q3: full 128-bit mask write.
                        let q_idx = (*id - 16) as u8;
                        self.mask.write(q_idx, [vec[0], vec[1], vec[2], vec[3]]);
                    } else if *id >= 28 && *id <= 31 {
                        // ql0-ql3: write low 64 bits, preserve high 64 bits.
                        let q_idx = (*id - 28) as u8;
                        let mut cur = self.mask.read(q_idx);
                        cur[0] = vec[0];
                        cur[1] = vec[1];
                        self.mask.write(q_idx, cur);
                    } else if *id >= 32 && *id <= 35 {
                        // qh0-qh3: write high 64 bits, preserve low 64 bits.
                        let q_idx = (*id - 32) as u8;
                        let mut cur = self.mask.read(q_idx);
                        cur[2] = vec[0];
                        cur[3] = vec[1];
                        self.mask.write(q_idx, cur);
                    }
                }
            }
            _ => {}
        }
    }

    /// Write to an accumulator register with quarter/half/full width support.
    ///
    /// For quarter writes (AM registers), only 4 u64 lanes are updated while
    /// preserving the other 4 lanes. For half writes (bml/bmh), the full
    /// 512-bit register is overwritten.
    fn apply_accum_write(
        &mut self,
        reg: u8,
        vec: &[u32; 8],
        width: Option<crate::interpreter::decode::register_map::AccumWidth>,
    ) {
        use crate::interpreter::decode::register_map::AccumWidth;

        // Convert [u32; 8] (256 bits) to [u64; 4] by pairing adjacent words.
        let to_u64_lanes = |v: &[u32; 8]| -> [u64; 4] {
            [
                (v[0] as u64) | ((v[1] as u64) << 32),
                (v[2] as u64) | ((v[3] as u64) << 32),
                (v[4] as u64) | ((v[5] as u64) << 32),
                (v[6] as u64) | ((v[7] as u64) << 32),
            ]
        };

        match width {
            Some(AccumWidth::QuarterLow) => {
                // Write to lanes 0-3 of the 512-bit register, preserve 4-7.
                let mut current = self.accumulator.read(reg);
                let lanes = to_u64_lanes(vec);
                current[0] = lanes[0];
                current[1] = lanes[1];
                current[2] = lanes[2];
                current[3] = lanes[3];
                self.accumulator.write(reg, current);
            }
            Some(AccumWidth::QuarterHigh) => {
                // Write to lanes 4-7 of the 512-bit register, preserve 0-3.
                let mut current = self.accumulator.read(reg);
                let lanes = to_u64_lanes(vec);
                current[4] = lanes[0];
                current[5] = lanes[1];
                current[6] = lanes[2];
                current[7] = lanes[3];
                self.accumulator.write(reg, current);
            }
            Some(AccumWidth::Half) | None => {
                // Write the full 512-bit register (default for bml/bmh loads).
                let lanes = to_u64_lanes(vec);
                let mut full = [0u64; 8];
                full[0] = lanes[0];
                full[1] = lanes[1];
                full[2] = lanes[2];
                full[3] = lanes[3];
                // Upper 4 lanes come from the remaining vec data if available.
                // For 256-bit loads, lanes 4-7 are zero.
                self.accumulator.write(reg, full);
            }
            Some(AccumWidth::Full) => {
                // 1024-bit write: vec only has 256 bits, write as low half.
                // Full writes typically go through write_wide, not this path.
                let lanes = to_u64_lanes(vec);
                let mut full = [0u64; 8];
                full[0] = lanes[0];
                full[1] = lanes[1];
                full[2] = lanes[2];
                full[3] = lanes[3];
                self.accumulator.write(reg, full);
            }
        }
    }

    /// Queue a deferred partial-word store.
    ///
    /// The address is captured at issue time, but the data register will be
    /// read when `ready_cycle` is reached. This models the II_STHB RMW
    /// pipeline where the data operand has latency 7.
    pub fn queue_pending_store(
        &mut self,
        address: u32,
        source: Operand,
        width: super::super::bundle::slot::MemWidth,
    ) {
        self.pending_stores.push(PendingStore {
            address,
            source,
            width,
            ready_cycle: self.cycles + PARTIAL_WORD_STORE_DATA_LATENCY,
            issue_pc: self.pc,
        });
    }

    /// Check if there are pending stores waiting to commit.
    pub fn pending_stores_empty(&self) -> bool {
        self.pending_stores.is_empty()
    }

    /// Queue a VUNPACK for its stage-7 source read (see [`PendingUnpack`]).
    ///
    /// The source w-register is sampled `VUNPACK_SRC_READ_LATENCY` issued
    /// bundles after issue; the x-register result lands one bundle later
    /// (total dest latency 7, II_VUNPACK operand_cycles[0]).
    pub fn queue_pending_unpack(
        &mut self,
        src_reg: u8,
        dest_reg: u8,
        bits_i: u32,
        bits_o: u32,
        signed: bool,
    ) {
        self.pending_unpacks.push(PendingUnpack {
            src_reg,
            dest_reg,
            bits_i,
            bits_o,
            signed,
            ready_bundle: self.bundle_seq + VUNPACK_SRC_READ_LATENCY,
        });
    }

    /// Sample sources for pending VUNPACKs that reached their read bundle and
    /// queue their destination writes. Call once per issued bundle, after
    /// `advance_vector_bundle()` (so producer writes that land this bundle are
    /// architecturally visible) and before slot execution.
    pub fn process_pending_unpacks(&mut self) {
        if self.pending_unpacks.is_empty() {
            return;
        }
        let current = self.bundle_seq;
        let mut ready: Vec<PendingUnpack> = Vec::new();
        self.pending_unpacks.retain(|pu| {
            if pu.ready_bundle <= current {
                ready.push(pu.clone());
                false
            } else {
                true
            }
        });
        ready.sort_by_key(|pu| pu.ready_bundle);
        for pu in ready {
            use crate::interpreter::execute::vector_pack::unpack_half;
            let src = self.vector.read(pu.src_reg);
            let lanes_per_half = (256 / pu.bits_o) as usize;
            let lo = unpack_half(&src, 0, pu.bits_i, pu.bits_o, pu.signed);
            let hi = unpack_half(&src, lanes_per_half, pu.bits_i, pu.bits_o, pu.signed);
            let mut result = [0u32; 16];
            result[..8].copy_from_slice(&lo);
            result[8..].copy_from_slice(&hi);
            // Source read at issue+6, write-back at issue+7: one more bundle.
            self.vector.queue_write_wide(pu.dest_reg, result, 1, Bypass::No);
        }
    }

    /// Queue an accumulator add/sub for its stage-3 source read (see
    /// [`PendingAccAdd`]). Control bits are resolved by the caller at issue.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_pending_acc_add(
        &mut self,
        acc1_reg: u8,
        acc2_reg: u8,
        dst_reg: u8,
        negate_acc1: bool,
        negate_acc2: bool,
        zero_acc1: bool,
        shift16: bool,
        is_float: bool,
        is_wide: bool,
    ) {
        self.pending_acc_adds.push(PendingAccAdd {
            acc1_reg,
            acc2_reg,
            dst_reg,
            negate_acc1,
            negate_acc2,
            zero_acc1,
            shift16,
            is_float,
            is_wide,
            ready_bundle: self.bundle_seq + VACC_SRC_READ_LATENCY,
        });
    }

    /// Sample sources for pending accumulator add/subs that reached their
    /// read bundle and write their results. Call once per issued bundle,
    /// after `commit_pending_writes()` (so MAC results that land this cycle
    /// are visible) and before slot execution.
    pub fn process_pending_acc_adds(&mut self) {
        if self.pending_acc_adds.is_empty() {
            return;
        }
        let current = self.bundle_seq;
        let mut ready: Vec<PendingAccAdd> = Vec::new();
        self.pending_acc_adds.retain(|pa| {
            if pa.ready_bundle <= current {
                ready.push(pa.clone());
                false
            } else {
                true
            }
        });
        ready.sort_by_key(|pa| pa.ready_bundle);
        for pa in ready {
            if pa.is_wide {
                let a1 = self.accumulator.read_wide(pa.acc1_reg);
                let a2 = self.accumulator.read_wide(pa.acc2_reg);
                let mut result = [0u64; 16];
                for i in 0..16 {
                    result[i] = Self::acc_add_sub_lane_pair(a1[i], a2[i], &pa);
                }
                self.accumulator.write_wide(pa.dst_reg, result);
            } else {
                let a1 = self.accumulator.read(pa.acc1_reg);
                let a2 = self.accumulator.read(pa.acc2_reg);
                let mut result = [0u64; 8];
                for i in 0..8 {
                    result[i] = Self::acc_add_sub_lane_pair(a1[i], a2[i], &pa);
                }
                self.accumulator.write(pa.dst_reg, result);
            }
        }
    }

    /// Queue a MAC-family multiply for its stage-3 accumulator source read
    /// (see execute::vector_matmul::PendingMatmul). Vector sources and control
    /// flags are sampled by the caller at issue.
    pub fn queue_pending_matmul(&mut self, p: crate::interpreter::execute::vector_matmul::PendingMatmul) {
        self.pending_matmuls.push(p);
    }

    /// Take the deferred matmuls whose accumulator-read bundle arrived,
    /// in ready order. Completion lives in execute::vector_matmul.
    pub fn take_ready_pending_matmuls(
        &mut self,
    ) -> Vec<crate::interpreter::execute::vector_matmul::PendingMatmul> {
        if self.pending_matmuls.is_empty() {
            return Vec::new();
        }
        let current = self.bundle_seq;
        let mut ready = Vec::new();
        self.pending_matmuls.retain(|p| {
            if p.ready_bundle <= current {
                ready.push(p.clone());
                false
            } else {
                true
            }
        });
        ready.sort_by_key(|p| p.ready_bundle);
        ready
    }

    /// Take ALL deferred matmuls regardless of bundle (force-commit path).
    pub fn take_all_pending_matmuls(
        &mut self,
    ) -> Vec<crate::interpreter::execute::vector_matmul::PendingMatmul> {
        std::mem::take(&mut self.pending_matmuls)
    }

    /// Queue a fused `vlda.ups` for its stage-7 shift-register read (see
    /// [`PendingUpsLoad`]). Load data and destination are resolved at issue;
    /// only the shift amount is sampled at issue+6.
    pub fn queue_pending_ups_load(
        &mut self,
        vec_data: [u32; 8],
        shift_reg: u8,
        dest_reg: u8,
        from: xdna_archspec::aie2::isa::ElementType,
        to: xdna_archspec::aie2::isa::ElementType,
        is_half: bool,
    ) {
        self.pending_ups_loads.push(PendingUpsLoad {
            vec_data,
            shift_reg,
            dest_reg,
            from,
            to,
            is_half,
            ready_bundle: self.bundle_seq + VLDA_UPS_SHIFT_READ_LATENCY,
        });
    }

    /// Sample the shift register for pending vlda.ups conversions that
    /// reached their read bundle and write the converted accumulator. Call
    /// once per issued bundle, after `advance_vector_bundle()` and before
    /// slot execution.
    pub fn process_pending_ups_loads(&mut self) {
        if self.pending_ups_loads.is_empty() {
            return;
        }
        let current = self.bundle_seq;
        let mut ready: Vec<PendingUpsLoad> = Vec::new();
        self.pending_ups_loads.retain(|pu| {
            if pu.ready_bundle <= current {
                ready.push(pu.clone());
                false
            } else {
                true
            }
        });
        ready.sort_by_key(|pu| pu.ready_bundle);
        for pu in ready {
            use crate::interpreter::execute::vector_ups::{ups_vector_to_acc, ups_vector_to_acc_wide};
            let shift = self.scalar_read(pu.shift_reg);
            if pu.is_half {
                let acc = ups_vector_to_acc(&pu.vec_data, shift, pu.from, pu.to);
                self.accumulator.write(pu.dest_reg, acc);
            } else {
                let acc = ups_vector_to_acc_wide(&pu.vec_data, shift, pu.from, pu.to);
                self.accumulator.write_wide(pu.dest_reg, acc);
            }
        }
    }

    /// One 64-bit accumulator word of VADD/VSUB/VNEGADD/VNEGSUB: two
    /// independent 32-bit lanes (fp32 or int32 per `is_float`).
    fn acc_add_sub_lane_pair(a1: u64, a2: u64, pa: &PendingAccAdd) -> u64 {
        if pa.is_float {
            use crate::interpreter::execute::vector_float::aie2_acc_fp32_add;
            // Denormal inputs are NOT flushed: the acc add ALU computes them
            // at full precision and flushes only a denormal RESULT to signed
            // zero (silicon-verified, fuzzer seed 4; see aie2_acc_fp32_add).
            let mut a1_lo = if pa.zero_acc1 { 0 } else { a1 as u32 };
            let mut a1_hi = if pa.zero_acc1 { 0 } else { (a1 >> 32) as u32 };
            let mut a2_lo = a2 as u32;
            let mut a2_hi = (a2 >> 32) as u32;
            // Negate by flipping the sign bit -- but NOT for true zeros: the
            // hardware negation leaves zero positive, so vsub.f(-0, +0) = +0
            // (silicon-verified, fuzzer seed 4 sub stage; IEEE -0 + -0 would
            // give -0). Normals, denormals, inf, and NaN all flip.
            let neg = |bits: u32| {
                if bits & 0x7FFF_FFFF == 0 {
                    bits
                } else {
                    bits ^ 0x8000_0000
                }
            };
            if pa.negate_acc1 {
                a1_lo = neg(a1_lo);
                a1_hi = neg(a1_hi);
            }
            if pa.negate_acc2 {
                a2_lo = neg(a2_lo);
                a2_hi = neg(a2_hi);
            }
            let r_lo = aie2_acc_fp32_add(a1_lo, a2_lo);
            let r_hi = aie2_acc_fp32_add(a1_hi, a2_hi);
            (r_lo as u64) | ((r_hi as u64) << 32)
        } else {
            // Acc32 mode: no carry chain between halves, per-lane 32-bit math.
            let v1_lo = if pa.zero_acc1 { 0i32 } else { a1 as i32 };
            let v1_hi = if pa.zero_acc1 { 0i32 } else { (a1 >> 32) as i32 };
            let v2_lo = a2 as i32;
            let v2_hi = (a2 >> 32) as i32;
            let v1_lo = if pa.negate_acc1 {
                v1_lo.wrapping_neg()
            } else {
                v1_lo
            };
            let v1_hi = if pa.negate_acc1 {
                v1_hi.wrapping_neg()
            } else {
                v1_hi
            };
            let v2_lo = if pa.negate_acc2 {
                v2_lo.wrapping_neg()
            } else {
                v2_lo
            };
            let v2_hi = if pa.negate_acc2 {
                v2_hi.wrapping_neg()
            } else {
                v2_hi
            };
            let mut r_lo = v1_lo.wrapping_add(v2_lo);
            let mut r_hi = v1_hi.wrapping_add(v2_hi);
            if pa.shift16 {
                r_lo >>= 16;
                r_hi >>= 16;
            }
            (r_lo as u32 as u64) | ((r_hi as u32 as u64) << 32)
        }
    }

    /// Check if there are pending (deferred) register writes awaiting commit.
    pub fn pending_writes_is_empty(&self) -> bool {
        self.pending_writes.is_empty()
    }

    /// Drain pending stores whose data-read cycle has been reached.
    ///
    /// Returns the ready stores so the caller can read the registers and
    /// write to tile memory (which is not accessible from ExecutionContext).
    pub fn drain_ready_stores(&mut self) -> Vec<PendingStore> {
        let current = self.cycles;
        let mut ready = Vec::new();
        self.pending_stores.retain(|ps| {
            if ps.ready_cycle <= current {
                ready.push(ps.clone());
                false
            } else {
                true
            }
        });
        // Sort by ready_cycle so earlier stores commit first
        ready.sort_by_key(|ps| ps.ready_cycle);
        ready
    }

    /// Read a scalar register with VLIW semantics and load forwarding.
    ///
    /// If a pending load from a **previous** bundle targets this register
    /// AND has reached its ready_cycle, the pending value is forwarded.
    /// Pending writes from the **current** bundle or that haven't completed
    /// their latency are excluded.
    ///
    /// If no forwarding applies, returns the snapshot value (inside a bundle)
    /// or the current live value (outside a bundle).
    #[inline]
    pub fn scalar_read(&self, reg: u8) -> u32 {
        if let Some(val) = self.forward_scalar(reg) {
            return val;
        }
        if let Some(snapshot) = &self.scalar_snapshot {
            snapshot.read(reg)
        } else {
            self.scalar.read(reg)
        }
    }

    /// Read a pointer register with VLIW semantics and load forwarding.
    ///
    /// SP_PTR_INDEX (255) is intercepted and routed to the dedicated SP register.
    #[inline]
    pub fn pointer_read(&self, reg: u8) -> u32 {
        use crate::interpreter::state::SP_PTR_INDEX;
        if reg == SP_PTR_INDEX {
            return self.sp();
        }
        if let Some(val) = self.forward_pointer(reg) {
            return val;
        }
        if let Some(snapshot) = &self.pointer_snapshot {
            snapshot.read(reg)
        } else {
            self.pointer.read(reg)
        }
    }

    /// Read a modifier register with VLIW semantics and load forwarding.
    #[inline]
    pub fn modifier_read(&self, reg: u8) -> u32 {
        if let Some(val) = self.forward_modifier(reg) {
            return val;
        }
        if let Some(snapshot) = &self.modifier_snapshot {
            snapshot.read(reg)
        } else {
            self.modifier.read(reg)
        }
    }

    // === Load Forwarding ===
    //
    // AIE2 uses a write-back pipeline WITHOUT a hardware scoreboard.
    // Loads queue a pending write with ready_cycle = issued_cycle + latency.
    // The result is written back to the register file when commit_pending_writes()
    // runs at the start of a future cycle whose cycle >= ready_cycle.
    //
    // Forwarding covers the window between ready_cycle and the next commit:
    // if a load has completed its latency but commit hasn't run yet, the
    // forwarding functions return the loaded value.
    //
    // Two conditions gate forwarding:
    //   1. issued_cycle < current: excludes same-bundle writes (VLIW semantics)
    //   2. ready_cycle <= current: respects load latency
    //
    // If code reads a register BEFORE its load's ready_cycle (within the
    // latency window), forwarding returns None and the read falls through
    // to the register file, returning the OLD value. This is correct --
    // the compiler schedules around load latencies, and software-pipelined
    // code intentionally reads "stale" values from previous iterations.

    /// Forward a pending scalar load value if ready and from a previous bundle.
    fn forward_scalar(&self, reg: u8) -> Option<u32> {
        self.pending_writes.iter().rev().find_map(|pw| {
            if pw.issued_cycle < self.cycles && pw.ready_cycle <= self.cycles {
                if let Operand::ScalarReg(r) = &pw.dest {
                    if *r == reg {
                        return Some(pw.scalar_value);
                    }
                }
            }
            None
        })
    }

    /// Forward a pending pointer load value if ready and from a previous bundle.
    fn forward_pointer(&self, reg: u8) -> Option<u32> {
        self.pending_writes.iter().rev().find_map(|pw| {
            if pw.issued_cycle < self.cycles && pw.ready_cycle <= self.cycles {
                if let Operand::PointerReg(r) = &pw.dest {
                    if *r == reg {
                        return Some(pw.scalar_value);
                    }
                }
            }
            None
        })
    }

    /// Forward a pending modifier load value if ready and from a previous bundle.
    fn forward_modifier(&self, reg: u8) -> Option<u32> {
        self.pending_writes.iter().rev().find_map(|pw| {
            if pw.issued_cycle < self.cycles && pw.ready_cycle <= self.cycles {
                if let Operand::ModifierReg(r) = &pw.dest {
                    if *r == reg {
                        return Some(pw.scalar_value);
                    }
                }
            }
            None
        })
    }

    // === Branch Delay Slot Support ===

    /// Set a pending branch with 5 delay slots.
    ///
    /// Called when a branch instruction is executed. The branch won't
    /// actually change the PC until 5 more instructions have executed.
    #[inline]
    pub fn set_pending_branch(&mut self, target: u32) {
        // If there's already a pending branch, the new one replaces it
        // (this matches hardware behavior for back-to-back branches)
        self.pending_branch = Some(PendingBranch::new(target));
        log::debug!("Branch to 0x{:X} pending at PC=0x{:X}, 5 delay slots", target, self.pc);
    }

    /// Set a pending call (jl) with 5 delay slots.
    ///
    /// Like `set_pending_branch`, but also defers the LR update until
    /// delay slots are exhausted. At that point, LR is set to the current
    /// PC (the first instruction after all delay slots). This matches
    /// hardware behavior where delay slot instructions see the pre-call LR.
    #[inline]
    pub fn set_pending_call(&mut self, target: u32) {
        self.pending_branch = Some(PendingBranch::new_call(target));
        log::debug!("Call to 0x{:X} pending, 5 delay slots (LR deferred)", target);
    }

    /// Check if there's a pending branch.
    #[inline]
    pub fn has_pending_branch(&self) -> bool {
        self.pending_branch.is_some()
    }

    /// Get the pending branch target (if any).
    #[inline]
    pub fn pending_branch_target(&self) -> Option<u32> {
        self.pending_branch.map(|b| b.target)
    }

    /// Tick the delay slot counter after executing an instruction.
    ///
    /// Returns `Some(target)` if delay slots are exhausted and branch
    /// should now be taken, `None` otherwise.
    ///
    /// For call instructions (is_call=true), also updates LR to the
    /// current PC when delay slots are exhausted. This is the address
    /// of the first instruction after all delay slots, which is the
    /// correct return address.
    #[inline]
    pub fn tick_delay_slots(&mut self) -> Option<u32> {
        if let Some(ref mut pending) = self.pending_branch {
            if pending.tick() {
                let target = pending.target;
                let is_call = pending.is_call;
                self.pending_branch = None;
                if is_call {
                    // LR = current PC = first instruction after all delay slots.
                    // This is deferred from the jl execution to match hardware
                    // pipeline behavior (delay slots see the old LR).
                    let return_addr = self.pc;
                    self.set_lr(return_addr);
                    log::debug!(
                        "Call delay slots exhausted, LR=0x{:X}, branching to 0x{:X}",
                        return_addr,
                        target
                    );
                } else {
                    log::debug!("Delay slots exhausted, branching to 0x{:X}", target);
                }
                return Some(target);
            }
        }
        None
    }

    /// Clear any pending branch (used on halt or error).
    #[inline]
    pub fn clear_pending_branch(&mut self) {
        self.pending_branch = None;
    }

    /// Reset execution statistics.
    pub fn reset_stats(&mut self) {
        self.cycles = 0;
        self.instructions = 0;
        self.stall_cycles = 0;
    }

    /// Get instructions per cycle (IPC) ratio.
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 {
            0.0
        } else {
            self.instructions as f64 / self.cycles as f64
        }
    }

    /// Reset all state (registers, PC, flags, stats).
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Configure which register is used as stack pointer.
    pub fn set_sp_register(&mut self, reg: SpRegister) {
        self.sp_reg = reg;
    }

    /// Configure which scalar register is used as link register.
    ///
    /// Accepts indices 0-47 (including special register slots 32-47).
    pub fn set_lr_register(&mut self, reg: u8) {
        assert!((reg as usize) < super::registers::NUM_SCALAR_REGS, "lr register index {} out of range", reg);
        self.lr_reg = reg;
    }
}

impl StateAccess for ExecutionContext {
    fn read_scalar(&self, reg: u8) -> u32 {
        self.scalar.read(reg)
    }

    fn write_scalar(&mut self, reg: u8, value: u32) {
        self.scalar.write(reg, value);
    }

    fn read_vector(&self, reg: u8) -> [u32; 8] {
        self.vector.read(reg)
    }

    fn write_vector(&mut self, reg: u8, value: [u32; 8]) {
        self.vector.write(reg, value);
    }

    fn pc(&self) -> u32 {
        self.pc
    }

    fn set_pc(&mut self, pc: u32) {
        self.pc = pc;
    }

    fn flags(&self) -> Flags {
        self.flags
    }

    fn set_flags(&mut self, flags: Flags) {
        self.flags = flags;
    }
}

impl std::fmt::Debug for ExecutionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("pc", &format_args!("0x{:08X}", self.pc))
            .field("flags", &self.flags)
            .field("cycles", &self.cycles)
            .field("instructions", &self.instructions)
            .field("halted", &self.halted)
            .field("scalar", &self.scalar)
            .field("pointer", &self.pointer)
            .field("vector", &self.vector)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_hardware_loop_returns_some_on_boundary() {
        use crate::interpreter::state::{LC_REG_INDEX, LE_REG_INDEX, LS_REG_INDEX};
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LS_REG_INDEX, 0x100);
        ctx.scalar.write(LC_REG_INDEX, 3);
        let info = ctx.check_hardware_loop(0x200);
        assert_eq!(info, Some(LoopBoundaryInfo { lc_before: 3, lc_after: 2, le_pc: 0x200 }));
    }

    #[test]
    fn check_hardware_loop_returns_none_off_boundary() {
        use crate::interpreter::state::LE_REG_INDEX;
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        let info = ctx.check_hardware_loop(0x180);
        assert_eq!(info, None);
    }

    #[test]
    fn check_hardware_loop_returns_none_when_lc_zero() {
        use crate::interpreter::state::{LC_REG_INDEX, LE_REG_INDEX};
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LC_REG_INDEX, 0);
        let info = ctx.check_hardware_loop(0x200);
        assert_eq!(info, None);
    }

    #[test]
    fn test_context_creation() {
        let ctx = ExecutionContext::new();
        assert_eq!(ctx.pc(), 0);
        assert_eq!(ctx.cycles, 0);
        assert!(!ctx.halted);
    }

    #[test]
    fn test_context_with_stack() {
        let ctx = ExecutionContext::with_stack(0x7_0000);
        assert_eq!(ctx.sp(), 0x7_0000);
    }

    #[test]
    fn test_pc_operations() {
        let mut ctx = ExecutionContext::new();

        ctx.set_pc(0x1000);
        assert_eq!(ctx.pc(), 0x1000);

        ctx.advance_pc(4);
        assert_eq!(ctx.pc(), 0x1004);

        ctx.advance_pc(0xFFFF_FFFC); // -4 wrapping
        assert_eq!(ctx.pc(), 0x1000);
    }

    #[test]
    fn test_flags_operations() {
        let mut ctx = ExecutionContext::new();

        let flags = Flags { z: true, n: false, c: true, v: false };
        ctx.set_flags(flags);

        let read = ctx.flags();
        assert!(read.z);
        assert!(read.c);
        assert!(!read.n);
    }

    #[test]
    fn test_stack_operations() {
        let mut ctx = ExecutionContext::with_stack(0x1000);

        ctx.push(0xDEAD);
        assert_eq!(ctx.sp(), 0x0FFC);

        let addr = ctx.pop();
        assert_eq!(addr, 0x0FFC);
        assert_eq!(ctx.sp(), 0x1000);
    }

    #[test]
    fn test_link_register() {
        use crate::interpreter::state::LR_REG_INDEX;
        let mut ctx = ExecutionContext::new();

        // Explicitly zero r0 so we can verify set_lr doesn't clobber it.
        ctx.scalar.write(0, 0);

        ctx.set_lr(0x2000);
        assert_eq!(ctx.lr(), 0x2000);
        // lr is stored at dedicated index 32, NOT r0
        assert_eq!(ctx.scalar.read(LR_REG_INDEX), 0x2000);
        // r0 should be unaffected
        assert_eq!(ctx.scalar.read(0), 0);
    }

    #[test]
    fn test_statistics() {
        let mut ctx = ExecutionContext::new();

        ctx.record_instruction(1);
        ctx.record_instruction(2);
        ctx.record_stall(3);

        assert_eq!(ctx.instructions, 2);
        assert_eq!(ctx.cycles, 6); // 1 + 2 + 3
        assert_eq!(ctx.stall_cycles, 3);
        assert!((ctx.ipc() - 0.333).abs() < 0.01);

        ctx.reset_stats();
        assert_eq!(ctx.cycles, 0);
    }

    #[test]
    fn test_state_access_trait() {
        let mut ctx = ExecutionContext::new();

        // Test through trait methods
        <ExecutionContext as StateAccess>::write_scalar(&mut ctx, 10, 42);
        assert_eq!(<ExecutionContext as StateAccess>::read_scalar(&ctx, 10), 42);

        let vec_data = [1, 2, 3, 4, 5, 6, 7, 8];
        <ExecutionContext as StateAccess>::write_vector(&mut ctx, 5, vec_data);
        assert_eq!(<ExecutionContext as StateAccess>::read_vector(&ctx, 5), vec_data);
    }

    #[test]
    fn test_sp_register_config() {
        let mut ctx = ExecutionContext::new();

        // Explicitly zero p0 so we can verify set_sp doesn't clobber it.
        ctx.pointer.write(0, 0);

        // Default: dedicated SP register (not aliased to any pointer reg)
        ctx.set_sp(0x1000);
        assert_eq!(ctx.sp(), 0x1000);
        // Verify p0 is NOT affected (SP is separate)
        assert_eq!(ctx.pointer.read(0), 0);

        // Switch to scalar register r13 (for testing alternate configs)
        ctx.set_sp_register(SpRegister::Scalar(13));
        ctx.set_sp(0x2000);
        assert_eq!(ctx.scalar.read(13), 0x2000);
        assert_eq!(ctx.sp(), 0x2000);
    }

    #[test]
    fn test_reset() {
        let mut ctx = ExecutionContext::new();

        ctx.set_pc(0x1000);
        ctx.scalar.write(5, 42);
        ctx.cycles = 100;
        ctx.halted = true;

        ctx.reset();

        assert_eq!(ctx.pc(), 0);
        // After reset, r5 should NOT retain the pre-reset value (42).
        // It will have the register init pattern (tripwire), not zero.
        assert_ne!(ctx.scalar.read(5), 42, "r5 should not retain pre-reset value");
        assert_eq!(ctx.cycles, 0);
        assert!(!ctx.halted);
    }

    #[test]
    fn test_timing_context_access() {
        let mut ctx = ExecutionContext::new();
        assert!(ctx.has_timing());

        ctx.timing_context_mut().hazard_stalls = 5;
        assert_eq!(ctx.timing_context().hazard_stalls, 5);
    }

    #[test]
    fn test_hardware_loop_basic() {
        use crate::interpreter::state::{LS_REG_INDEX, LE_REG_INDEX, LC_REG_INDEX};
        let mut ctx = ExecutionContext::new();

        // Set up loop: LS=0x100, LE=0x200, LC=3
        // ZLS hardware loops load trip_count directly into LC.
        // LC=3 means 3 body executions: 2 back-edges + 1 fall-through.
        ctx.scalar.write(LS_REG_INDEX, 0x100);
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LC_REG_INDEX, 3);
        ctx.set_pc(0x204); // PC after advance_pc from LE

        // First check: LC 3->2, still > 0, loop back
        ctx.check_hardware_loop(0x200);
        assert_eq!(ctx.pc(), 0x100); // Redirected to LS
        assert_eq!(ctx.scalar.read(LC_REG_INDEX), 2);

        // Second check: LC 2->1, still > 0, loop back
        ctx.set_pc(0x204);
        ctx.check_hardware_loop(0x200);
        assert_eq!(ctx.pc(), 0x100);
        assert_eq!(ctx.scalar.read(LC_REG_INDEX), 1);

        // Third check: LC 1->0, fall through (this was the last iteration)
        ctx.set_pc(0x204);
        ctx.check_hardware_loop(0x200);
        assert_eq!(ctx.pc(), 0x204); // Falls through -- LC hit 0
        assert_eq!(ctx.scalar.read(LC_REG_INDEX), 0);

        // LC already 0: no action
        ctx.set_pc(0x208);
        ctx.check_hardware_loop(0x200);
        assert_eq!(ctx.pc(), 0x208); // Not at LE, unchanged
        assert_eq!(ctx.scalar.read(LC_REG_INDEX), 0);
    }

    #[test]
    fn test_hardware_loop_no_match() {
        use crate::interpreter::state::{LS_REG_INDEX, LE_REG_INDEX, LC_REG_INDEX};
        let mut ctx = ExecutionContext::new();

        ctx.scalar.write(LS_REG_INDEX, 0x100);
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LC_REG_INDEX, 5);
        ctx.set_pc(0x180);

        // fetch_pc != LE -> no loop
        ctx.check_hardware_loop(0x17C);
        assert_eq!(ctx.pc(), 0x180); // Unchanged
        assert_eq!(ctx.scalar.read(LC_REG_INDEX), 5); // Unchanged
    }

    // =====================================================================
    // Zero-overhead loop: in-flight store flush on the back-edge
    // =====================================================================
    //
    // AIE2 partial-word stores (st.s8/st.s16) are read-modify-write and commit
    // to data memory late (stage E11, ~11 cycles after issue; AM020 + AIE2
    // Schedule.td). When a ZOL back-edge is taken, a store that has not yet
    // reached its memory-commit stage is flushed by the program-control-unit
    // pipeline redirect -- so a store parked in the last bundle before LE is
    // lost on every back-edge and commits only on the final fall-through. This
    // is the every-4th-element drop observed on silicon for a Peano-unrolled
    // i8 loop. Loads/register writes are NOT flushed (they retire into the
    // register file), which is why a load in the LE bundle survives.
    // A still-pending store in the emulator is by definition not-yet-committed,
    // so flushing the pending-store queue on the back-edge models this exactly:
    // any correctly-scheduled store (committed >=11 cycles before LE) has
    // already drained and is unaffected.

    #[test]
    fn zol_backedge_flushes_pending_store() {
        use crate::interpreter::bundle::slot::MemWidth;
        use crate::interpreter::state::{LC_REG_INDEX, LE_REG_INDEX, LS_REG_INDEX};
        let mut ctx = ExecutionContext::new();
        // Body span le-ls = 0x60 (96 bytes, six fetch packets): inside the
        // flush window, so the LE-bundle store is squashed on the back-edge.
        ctx.scalar.write(LS_REG_INDEX, 0x1A0);
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LC_REG_INDEX, 3);

        // A store issued in the LE bundle is still pending at the back-edge.
        ctx.set_pc(0x200);
        ctx.queue_pending_store(0x400, Operand::ScalarReg(1), MemWidth::Byte);
        assert!(!ctx.pending_stores_empty());

        ctx.check_hardware_loop(0x200); // back-edge: LC 3 -> 2
        assert_eq!(ctx.pc(), 0x1A0);
        assert!(
            ctx.pending_stores_empty(),
            "a store issued in the LE bundle must be flushed by the ZOL back-edge"
        );
    }

    #[test]
    fn zol_backedge_large_body_keeps_le_store() {
        // A loop body that does NOT fit in the front-end fetch window
        // (le-ls > ZOL_FLUSH_MAX_BODY_BYTES) streams past the squash point
        // before the LE bundle issues, so even the LE-bundle store commits.
        // This is the seed_1826 / seed_1781 case (112-byte / seven-packet body)
        // that the unconditional issue_pc==LE flush wrongly dropped.
        use crate::interpreter::bundle::slot::MemWidth;
        use crate::interpreter::state::{LC_REG_INDEX, LE_REG_INDEX, LS_REG_INDEX};
        let mut ctx = ExecutionContext::new();
        // Body span le-ls = 0x70 (112 bytes, seven fetch packets): outside the
        // flush window.
        ctx.scalar.write(LS_REG_INDEX, 0x190);
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LC_REG_INDEX, 3);

        ctx.set_pc(0x200);
        ctx.queue_pending_store(0x400, Operand::ScalarReg(1), MemWidth::Byte);
        assert!(!ctx.pending_stores_empty());

        ctx.check_hardware_loop(0x200); // back-edge: LC 3 -> 2
        assert_eq!(ctx.pc(), 0x190);
        assert!(!ctx.pending_stores_empty(), "a large-body loop commits its LE-bundle store -- no flush");
    }

    #[test]
    fn zol_backedge_preserves_earlier_body_store() {
        // A store issued by a body bundle (PC < LE) has already entered the
        // store pipeline and commits across the back-edge -- only the LE-bundle
        // store is in the fetch-redirect shadow. This is the seed_13 case the
        // "flush all pending stores" model wrongly broke.
        use crate::interpreter::bundle::slot::MemWidth;
        use crate::interpreter::state::{LC_REG_INDEX, LE_REG_INDEX, LS_REG_INDEX};
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(LS_REG_INDEX, 0x1A0);
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LC_REG_INDEX, 3);

        // Store issued at a body bundle (0x1F0), still pending when LE is hit.
        ctx.set_pc(0x1F0);
        ctx.queue_pending_store(0x400, Operand::ScalarReg(1), MemWidth::Byte);
        ctx.set_pc(0x200);

        ctx.check_hardware_loop(0x200); // back-edge
        assert!(
            !ctx.pending_stores_empty(),
            "an earlier body store (issue_pc != LE) must survive the back-edge"
        );
    }

    #[test]
    fn zol_final_iteration_keeps_pending_store() {
        use crate::interpreter::bundle::slot::MemWidth;
        use crate::interpreter::state::{LC_REG_INDEX, LE_REG_INDEX, LS_REG_INDEX};
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(LS_REG_INDEX, 0x1A0);
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LC_REG_INDEX, 1); // last iteration

        ctx.set_pc(0x200);
        ctx.queue_pending_store(0x400, Operand::ScalarReg(1), MemWidth::Byte);
        ctx.set_pc(0x204); // models advance_pc past the LE bundle

        ctx.check_hardware_loop(0x200); // fall-through: LC 1 -> 0
        assert_eq!(ctx.pc(), 0x204);
        assert!(
            !ctx.pending_stores_empty(),
            "a store on the final (fall-through) iteration commits -- no flush"
        );
    }

    #[test]
    fn zol_backedge_preserves_pending_register_write() {
        use crate::interpreter::state::{LC_REG_INDEX, LE_REG_INDEX, LS_REG_INDEX};
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(LS_REG_INDEX, 0x1A0);
        ctx.scalar.write(LE_REG_INDEX, 0x200);
        ctx.scalar.write(LC_REG_INDEX, 3);

        // A load (deferred register write) in the LE bundle -- must survive,
        // unlike a memory store. This is the seed_13 case the crude squash broke.
        ctx.set_pc(0x200);
        ctx.queue_scalar_load(Operand::ScalarReg(5), 0xDEAD, 7);
        assert!(!ctx.pending_writes_is_empty());

        ctx.check_hardware_loop(0x200); // back-edge
        assert!(!ctx.pending_writes_is_empty(), "loads/register writes are NOT flushed by the back-edge");
    }

    // =====================================================================
    // Load Forwarding Tests
    // =====================================================================

    #[test]
    fn a_stall_cycle_does_not_age_the_deferred_write_pipeline() {
        // A core stall FREEZES the pipeline: the cycle passes but no bundle
        // issues, so no in-flight pipeline stage advances. A latency-7 load is
        // still seven ISSUED BUNDLES from landing, however many stall cycles
        // pass in between.
        //
        // Regression (bank-arbitration arc): `record_stall` advanced `cycles`
        // alone, so each stall cycle pulled every deferred write one issued
        // bundle EARLIER. In a software-pipelined vector loop the consumer then
        // read the load of the NEXT iteration and the output shifted by one
        // element. The bundle_seq-keyed queues (VUNPACK, acc-add, matmul,
        // ups-load, vector register file) are stall-immune by construction;
        // pending_writes and pending_stores are cycle-keyed and were not.
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 111);
        ctx.queue_scalar_load(Operand::ScalarReg(5), 222, 7); // issued at cycle 0

        ctx.record_stall(1); // e.g. this core lost a memory-bank arbitration

        // Six issued bundles. The seventh has not issued, so the load is not in.
        for _ in 0..6 {
            ctx.record_instruction(1);
        }
        assert_eq!(
            ctx.scalar_read(5),
            111,
            "six issued bundles after a latency-7 load -- a stall cycle is not an issued bundle"
        );

        ctx.record_instruction(1);
        assert_eq!(ctx.scalar_read(5), 222, "seven issued bundles after: the load has landed");
    }

    #[test]
    fn a_stall_cycle_does_not_age_the_partial_word_store_pipeline() {
        // Same contract for the partial-word store's late data read: its source
        // register is sampled PARTIAL_WORD_STORE_DATA_LATENCY *issued bundles*
        // after issue, not wall cycles.
        let mut ctx = ExecutionContext::new();
        ctx.queue_pending_store(0x400, Operand::ScalarReg(1), crate::interpreter::MemWidth::Byte);
        ctx.record_stall(1);

        for _ in 0..PARTIAL_WORD_STORE_DATA_LATENCY - 1 {
            ctx.record_instruction(1);
        }
        assert!(
            ctx.drain_ready_stores().is_empty(),
            "the store's data read is still one issued bundle away -- the stall cycle did not advance it"
        );

        ctx.record_instruction(1);
        assert_eq!(ctx.drain_ready_stores().len(), 1, "at the data-read bundle the store samples its source");
    }

    #[test]
    fn test_scalar_load_forwarding_basic() {
        // A scalar load queued at cycle 10 with latency 7 should NOT be
        // forwardable before ready_cycle (17). Reads see the old value
        // until the load completes.
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 111);

        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 222, 7); // ready=17

        // At cycle 11: load not ready -- read old value
        ctx.cycles = 11;
        assert_eq!(ctx.scalar_read(5), 111, "Before ready_cycle, should return old register value");
        assert_eq!(ctx.scalar.read(5), 111, "Live register should still have old value");

        // At cycle 16: still pending, still not ready
        ctx.cycles = 16;
        assert_eq!(ctx.scalar_read(5), 111, "Still before ready_cycle, should return old value");

        // At cycle 17: ready -- forward then commit
        ctx.cycles = 17;
        assert_eq!(ctx.scalar_read(5), 222, "At ready_cycle, forwarding should return load value");
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(5), 222, "After commit, live register should have new value");
        // No more pending writes, read should return live value
        assert_eq!(ctx.scalar_read(5), 222);
    }

    #[test]
    fn test_queue_matmul_accum_write_defers() {
        // A matmul (VMUL/VMAC) accumulator write has MAC result latency (5
        // cycles): the result is not visible in the accumulator until
        // ready_cycle, so in-flight stores in a software-pipelined loop read
        // the previous tile's value. Models AIE2 II_VMAC operand_cycles[0]=5.
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write_wide(0, [0xAAAA_AAAA_AAAA_AAAAu64; 16]);

        ctx.cycles = 10;
        let result = [0x1111_1111_1111_1111u64; 16];
        ctx.queue_matmul_accum_write(Operand::AccumReg(0), result, false, 5); // ready=15

        // Immediately and before ready_cycle: accumulator holds the OLD value.
        assert_eq!(ctx.accumulator.read_wide(0)[0], 0xAAAA_AAAA_AAAA_AAAA);
        ctx.cycles = 14;
        ctx.commit_pending_writes();
        assert_eq!(
            ctx.accumulator.read_wide(0)[0],
            0xAAAA_AAAA_AAAA_AAAA,
            "before ready_cycle the matmul result must not be visible",
        );

        // At ready_cycle (15): committed, new value visible.
        ctx.cycles = 15;
        ctx.commit_pending_writes();
        assert_eq!(
            ctx.accumulator.read_wide(0)[0],
            0x1111_1111_1111_1111,
            "at ready_cycle the matmul result becomes visible",
        );
    }

    // Vector-register result visibility (the AIE2 bypass/forwarding network) is
    // modeled inside `VectorRegisterFile` and unit-tested there
    // (`registers::tests::test_bypass_*`).

    #[test]
    fn test_pointer_load_forwarding_basic() {
        // Same as scalar but for pointer registers. This is the pattern
        // that matters for memcpy: lda pN, [sp, #-32] queues a pending
        // pointer write with latency 7.
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(7, 0xDEAD);

        ctx.cycles = 10;
        // Simulate: lda p7, [sp, #-32] loaded value 0x78000
        ctx.queue_scalar_load(Operand::PointerReg(7), 0x78000, 7);

        // Before ready_cycle: forwarding should NOT return the value
        ctx.cycles = 12;
        assert_eq!(ctx.pointer_read(7), 0xDEAD, "Before ready_cycle, pointer_read should return old value");
        assert_eq!(ctx.pointer.read(7), 0xDEAD, "Live pointer register should still have old value");

        // At ready_cycle: forwarding should work
        ctx.cycles = 17;
        assert_eq!(
            ctx.pointer_read(7),
            0x78000,
            "At ready_cycle, pointer forwarding should return pending load value"
        );

        // After ready: commit should apply
        ctx.cycles = 17;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000);
        assert_eq!(ctx.pointer_read(7), 0x78000);
    }

    #[test]
    fn test_modifier_load_forwarding_basic() {
        let mut ctx = ExecutionContext::new();
        ctx.modifier.write(3, 0x10);

        ctx.cycles = 5;
        ctx.queue_scalar_load(Operand::ModifierReg(3), 0x20, 7);

        // Before ready_cycle: should read old value
        ctx.cycles = 6;
        assert_eq!(ctx.modifier_read(3), 0x10, "Before ready_cycle, modifier_read should return old value");
        assert_eq!(ctx.modifier.read(3), 0x10, "Live modifier should still have old value");

        // At ready_cycle (5 + 7 = 12): forwarding should work
        ctx.cycles = 12;
        assert_eq!(
            ctx.modifier_read(3),
            0x20,
            "At ready_cycle, modifier forwarding should return pending value"
        );
    }

    #[test]
    fn test_forwarding_not_in_same_cycle() {
        // VLIW semantics: a write issued in the current cycle must NOT
        // be forwarded to reads in the same bundle.
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 111);

        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 222, 7);

        // Same cycle: forward_scalar should return None
        assert_eq!(ctx.forward_scalar(5), None, "Same-cycle pending write must not be forwarded");
        // scalar_read falls through to live value
        assert_eq!(ctx.scalar_read(5), 111);
    }

    #[test]
    fn test_forwarding_cleared_after_commit() {
        // After commit_pending_writes drains a pending write, forward
        // should return None (the value is now in the live register).
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 0);

        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 42, 3); // ready=13

        // Before ready_cycle: forward should NOT return the value
        // (load latency hasn't elapsed yet)
        ctx.cycles = 12;
        assert_eq!(
            ctx.forward_scalar(5),
            None,
            "Before ready_cycle, forward should not return pending value"
        );
        assert_eq!(ctx.scalar_read(5), 0, "Before ready_cycle, scalar_read should return old value");

        // At ready_cycle: forward should work
        ctx.cycles = 13;
        assert_eq!(ctx.forward_scalar(5), Some(42), "At ready_cycle, forward should return pending value");

        // Commit drains it
        ctx.commit_pending_writes();
        assert_eq!(ctx.forward_scalar(5), None);
        assert_eq!(ctx.scalar.read(5), 42);
    }

    #[test]
    fn test_multiple_pending_loads_most_recent_wins() {
        // Two loads to the same register: later load's value should be
        // forwarded only after its ready_cycle. Before that, reads see
        // either the earlier load's value (if ready) or the old value.
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 0);

        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 1001, 7); // ready=17

        ctx.cycles = 12;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 1002, 7); // ready=19

        // At cycle 13: neither load is ready yet -- read old value
        ctx.cycles = 13;
        assert_eq!(ctx.scalar_read(5), 0, "Neither load ready yet, should read old register value");

        // At cycle 17: first load is ready, second is not -- forward first
        ctx.cycles = 17;
        assert_eq!(
            ctx.forward_scalar(5),
            Some(1001),
            "First load ready, second not yet -- should forward first"
        );

        // Commit at 17 drains first load
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(5), 1001, "First load should have committed to live register");
        // Second still pending but not ready -- read committed value
        assert_eq!(ctx.scalar_read(5), 1001, "Second load not ready yet, should read committed first");

        // At cycle 19: second load ready -- forward it
        ctx.cycles = 19;
        assert_eq!(ctx.forward_scalar(5), Some(1002), "Second load now ready, should be forwarded");

        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(5), 1002, "Second load should overwrite first in live register");
    }

    #[test]
    fn test_forwarding_software_pipeline_pattern() {
        // Models the Chess software-pipelined byte store pattern from
        // add_21_i8: loads with 7-cycle latency interleaved with stores
        // of previous-iteration results. The pipeline relies on loads
        // NOT being visible until ready_cycle, so stores write the OLD
        // register value (from the previous iteration's computation).
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(24, 0); // r24 initially 0

        // Iteration 0: compute r24 = old_val + 21
        ctx.scalar.write(24, 42); // r24 = 21 + 21 = 42 (previous result)

        // Cycle 100: lda.u8 r24, [p7] -- loads new byte (say 5)
        ctx.cycles = 100;
        ctx.queue_scalar_load(Operand::ScalarReg(24), 5, 7); // ready=107

        // Cycles 101-106: the store of r24 should see 42 (old value),
        // NOT 5 (the pending load value)
        for c in 101..=106 {
            ctx.cycles = c;
            assert_eq!(ctx.scalar_read(24), 42, "At cycle {c}, load not ready, must read old value 42");
        }

        // Cycle 107: load completes, r24 becomes 5
        ctx.cycles = 107;
        assert_eq!(ctx.scalar_read(24), 5, "At ready_cycle 107, should forward new value 5");
    }

    #[test]
    fn test_forwarding_wrong_register_no_match() {
        // Forward should return None for a different register.
        let mut ctx = ExecutionContext::new();

        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 42, 7);

        ctx.cycles = 11;
        assert_eq!(ctx.forward_scalar(6), None, "Forward should not match different register");
        assert_eq!(ctx.forward_pointer(5), None, "Forward should not match different register class");
    }

    // =====================================================================
    // Scoreboard Stall Detection Tests
    // =====================================================================

    // =====================================================================
    // SP Register Isolation Tests
    // =====================================================================

    #[test]
    fn test_sp_does_not_alias_pointer_regs() {
        // Dedicated SP must not affect any pointer register p0-p7.
        let mut ctx = ExecutionContext::new();

        // Write distinctive values to all pointer regs
        for i in 0..8u8 {
            ctx.pointer.write(i, 0x1000 + i as u32);
        }

        // Write SP via set_sp
        ctx.set_sp(0xDEAD);
        assert_eq!(ctx.sp(), 0xDEAD);

        // All pointer regs should be unaffected
        for i in 0..8u8 {
            assert_eq!(ctx.pointer.read(i), 0x1000 + i as u32, "p{} should be unaffected by SP write", i);
        }
    }

    #[test]
    fn test_pointer_writes_dont_affect_sp() {
        let mut ctx = ExecutionContext::new();
        ctx.set_sp(0x70000);

        // Write all pointer regs
        for i in 0..8u8 {
            ctx.pointer.write(i, 0xBEEF);
        }

        assert_eq!(ctx.sp(), 0x70000, "SP should be unaffected by pointer register writes");
    }

    #[test]
    fn test_sp_via_pointer_read_intercept() {
        use crate::interpreter::state::SP_PTR_INDEX;
        let mut ctx = ExecutionContext::new();
        ctx.set_sp(0x70080);

        // pointer_read(SP_PTR_INDEX) should return SP, not a real pointer reg
        assert_eq!(ctx.pointer_read(SP_PTR_INDEX), 0x70080);
    }

    #[test]
    fn test_sp_via_queue_pointer_write_immediate() {
        use crate::interpreter::state::SP_PTR_INDEX;
        let mut ctx = ExecutionContext::new();

        // queue_pointer_write(SP_PTR_INDEX) should write immediately (no latency)
        ctx.cycles = 10;
        ctx.queue_pointer_write(SP_PTR_INDEX, 0x70080, 1);

        // SP should be updated immediately, NOT deferred
        assert_eq!(ctx.sp(), 0x70080);

        // No pending write should have been created
        assert!(ctx.pending_writes.is_empty(), "SP write should not create a pending write");
    }

    #[test]
    fn test_sp_not_affected_by_p0_clobber() {
        // This is the exact scenario that caused the memcpy bug:
        // SP = 0x70080, then mov p0, p1 clobbers p0.
        // With dedicated SP, SP must survive.
        let mut ctx = ExecutionContext::new();
        ctx.set_sp(0x70080);
        ctx.pointer.write(0, 0x74000); // p0 = output buffer
        ctx.pointer.write(1, 0x78000); // p1 = input buffer

        // Simulate: mov p0, p1 (clobbers p0)
        ctx.pointer.write(0, ctx.pointer.read(1));

        assert_eq!(ctx.pointer.read(0), 0x78000, "p0 should be clobbered");
        assert_eq!(ctx.sp(), 0x70080, "SP must survive p0 clobber");
    }

    // =====================================================================
    // VLIW Snapshot + Forwarding Interaction
    // =====================================================================

    #[test]
    fn test_forwarding_overrides_snapshot() {
        // When a pending write from a previous bundle has reached its
        // ready_cycle, forwarding should take priority over the snapshot.
        // Before ready_cycle, the snapshot (old) value is returned.
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 333);

        // Queue a load at cycle 10, ready at 17
        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 444, 7);

        // At cycle 11: begin a new bundle (takes snapshot of 333)
        // Load not ready yet -- snapshot value should be returned
        ctx.cycles = 11;
        ctx.begin_bundle();
        assert_eq!(ctx.scalar_read(5), 333, "Before ready_cycle, snapshot value should be returned");
        ctx.end_bundle();

        // At cycle 17: load is ready -- forwarding overrides snapshot
        ctx.cycles = 17;
        ctx.begin_bundle();
        assert_eq!(ctx.scalar_read(5), 444, "At ready_cycle, forwarding must override snapshot");
        ctx.end_bundle();
    }

    #[test]
    fn test_same_bundle_write_uses_snapshot_not_forward() {
        // A load issued in the current bundle must NOT be forwarded.
        // Instead, the snapshot (pre-execution) value is returned.
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 333);

        ctx.cycles = 10;
        ctx.begin_bundle();

        // Queue a load in this same bundle
        ctx.queue_scalar_load(Operand::ScalarReg(5), 222, 7);

        // Read in same bundle: should get snapshot value, not the pending load
        assert_eq!(ctx.scalar_read(5), 333, "Same-bundle load must not be forwarded");

        ctx.end_bundle();
    }

    // =====================================================================
    // commit_pending_writes Edge Cases
    // =====================================================================

    #[test]
    fn test_commit_multiple_writes_same_register_ordering() {
        // Two pending writes to the same register with different ready_cycles.
        // The later ready_cycle should win (applied last).
        let mut ctx = ExecutionContext::new();
        ctx.scalar.write(5, 0);

        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 2001, 2); // ready=12
        ctx.queue_scalar_load(Operand::ScalarReg(5), 2002, 5); // ready=15

        // Commit at cycle 15: both should be ready
        ctx.cycles = 15;
        ctx.commit_pending_writes();

        // Later ready_cycle writes second, overwriting the earlier one
        assert_eq!(ctx.scalar.read(5), 2002, "Later-ready write should overwrite earlier-ready write");
    }

    #[test]
    fn test_commit_partial_drain() {
        // Only writes whose ready_cycle <= current should be committed.
        // Others should remain in the queue.
        let mut ctx = ExecutionContext::new();

        // Explicitly zero the registers under test so we can detect
        // whether pending writes have committed or not.
        ctx.scalar.write(0, 0);
        ctx.scalar.write(1, 0);
        ctx.scalar.write(2, 0);

        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(0), 0xA, 2); // ready=12
        ctx.queue_scalar_load(Operand::ScalarReg(1), 0xB, 5); // ready=15
        ctx.queue_scalar_load(Operand::ScalarReg(2), 0xC, 3); // ready=13

        // Commit at cycle 13: r0 (ready=12) and r2 (ready=13) should commit
        ctx.cycles = 13;
        ctx.commit_pending_writes();

        assert_eq!(ctx.scalar.read(0), 0xA, "r0 should be committed");
        assert_eq!(ctx.scalar.read(2), 0xC, "r2 should be committed");
        assert_eq!(ctx.scalar.read(1), 0, "r1 should NOT be committed yet");

        // One pending write should remain
        assert_eq!(ctx.pending_writes.len(), 1);

        // Commit at cycle 15: r1 should now commit
        ctx.cycles = 15;
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(1), 0xB);
        assert!(ctx.pending_writes.is_empty());
    }

    #[test]
    fn test_commit_swap_remove_correctness() {
        // Regression test for the swap_remove drain pattern.
        // With many pending writes, partial commit must not corrupt indices.
        let mut ctx = ExecutionContext::new();

        // Zero all registers under test so we can distinguish committed
        // from not-yet-committed.
        for i in 0..6u8 {
            ctx.scalar.write(i, 0);
        }

        ctx.cycles = 10;
        // Queue 6 writes with staggered ready_cycles
        for i in 0..6u8 {
            ctx.queue_scalar_load(
                Operand::ScalarReg(i),
                (i as u32 + 1) * 100,
                (i as u64 + 1) * 2, // ready: 12, 14, 16, 18, 20, 22
            );
        }

        // Commit at cycle 16: r0 (12), r1 (14), r2 (16) should commit
        ctx.cycles = 16;
        ctx.commit_pending_writes();

        for i in 0..3u8 {
            assert_eq!(ctx.scalar.read(i), (i as u32 + 1) * 100, "r{} should be committed", i);
        }
        for i in 3..6u8 {
            assert_eq!(ctx.scalar.read(i), 0, "r{} should NOT be committed yet", i);
        }
        assert_eq!(ctx.pending_writes.len(), 3);

        // Commit the rest
        ctx.cycles = 22;
        ctx.commit_pending_writes();
        for i in 3..6u8 {
            assert_eq!(ctx.scalar.read(i), (i as u32 + 1) * 100, "r{} should now be committed", i);
        }
        assert!(ctx.pending_writes.is_empty());
    }

    #[test]
    fn test_delay_pending_writes_extends_forwarding_window() {
        // delay_pending_writes(1) at a branch boundary should push
        // the ready_cycle of all pending writes, extending the window
        // during which forwarding is active.
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(7, 0);

        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::PointerReg(7), 0x78000, 7); // ready=17

        // Simulate branch taken: delay by 1
        ctx.delay_pending_writes(1); // ready=18

        // At cycle 17: should NOT be committed yet (was delayed to 18)
        ctx.cycles = 17;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0, "Delayed write should not commit at original ready_cycle");
        // Forward also should not work (ready_cycle is 18)
        assert_eq!(ctx.pointer_read(7), 0, "Before delayed ready_cycle, forward should return old value");

        // At cycle 18: ready -- forward and commit
        ctx.cycles = 18;
        assert_eq!(ctx.pointer_read(7), 0x78000, "At delayed ready_cycle, forward should return value");
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000);
    }

    // =====================================================================
    // Save/Restore Pattern (memcpy scenario)
    // =====================================================================

    #[test]
    fn test_pointer_save_restore_roundtrip() {
        // Simulate the memcpy prologue/epilogue pattern:
        // 1. Save p7 to stack via store
        // 2. Clobber p7 (mov p7, sp)
        // 3. Restore p7 from stack via load
        //
        // This tests that the register pipeline correctly handles the
        // save -> clobber -> restore sequence.
        let mut ctx = ExecutionContext::new();
        ctx.set_sp(0x70080);
        ctx.pointer.write(7, 0x78000); // original p7

        // Step 1: Read p7 for store (at cycle 10)
        ctx.cycles = 10;
        let saved_value = ctx.pointer_read(7);
        assert_eq!(saved_value, 0x78000);

        // Step 2: Clobber p7 via queue_pointer_write (mov p7, sp)
        ctx.cycles = 11;
        ctx.queue_pointer_write(7, ctx.sp(), 1); // latency 1, ready=12

        // Step 3: Commit the clobber
        ctx.cycles = 12;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x70080, "p7 = sp after clobber");

        // ... memcpy body runs ...

        // Step 4: Restore p7 from stack (lda p7, [sp, #-32])
        // The "loaded value" would come from memory (0x78000 was stored there)
        ctx.cycles = 100;
        ctx.queue_scalar_load(Operand::PointerReg(7), 0x78000, 7); // ready=107

        // Step 5: Before ready_cycle, read returns the clobbered value
        ctx.cycles = 101;
        assert_eq!(ctx.pointer_read(7), 0x70080, "Before ready_cycle, should read clobbered sp value");

        // At ready_cycle (107), forward works and commit applies
        ctx.cycles = 107;
        assert_eq!(ctx.pointer_read(7), 0x78000, "At ready_cycle, forward should return restored value");
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000, "p7 should be restored to original value");
    }

    #[test]
    fn test_pointer_forwarding_survives_branch_delay() {
        // Full memcpy scenario with branch:
        // 1. lda p7, [sp, #-32] at cycle 50 (latency 7, ready=57)
        // 2. ret at cycle 54 (5 delay slots from cycle 50)
        // 3. delay_pending_writes(1) when branch takes effect: ready=58
        // 4. mov p2, p7 at cycle 55 (at branch target)
        //
        // At step 4, p7 should be forwarded from the pending load.
        let mut ctx = ExecutionContext::new();
        ctx.pointer.write(7, 0); // clobbered value (frame ptr)

        // Step 1: lda p7 restore
        ctx.cycles = 50;
        ctx.queue_scalar_load(Operand::PointerReg(7), 0x78000, 7); // ready=57

        // Step 2-3: Branch taken after delay slots
        ctx.delay_pending_writes(1); // ready=58

        // Step 4: At cycle 55 (before ready_cycle 58), reads old clobbered value
        ctx.cycles = 55;
        assert_eq!(ctx.pointer_read(7), 0, "Before ready_cycle, p7 should return old (clobbered) value");

        // At ready_cycle 58: forwarding works
        ctx.cycles = 58;
        assert_eq!(ctx.pointer_read(7), 0x78000, "At ready_cycle, p7 should be forwarded from pending load");
    }

    #[test]
    fn test_forwarding_does_not_return_stale_over_committed() {
        // Edge case: pending load to reg X, then a direct MOV to reg X.
        // Before the load's ready_cycle, the MOV value is visible.
        // After ready_cycle, the load value overwrites.
        let mut ctx = ExecutionContext::new();

        // Cycle 10: lda r5, [...] with latency 7 (ready=17)
        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(5), 3002, 7);

        // Cycle 12: mov r5, #3001 (immediate write, no latency)
        ctx.cycles = 12;
        ctx.scalar.write(5, 3001);

        // At cycle 13: load not ready yet -- read the MOV value
        ctx.cycles = 13;
        assert_eq!(ctx.scalar_read(5), 3001, "Before ready_cycle, MOV value should be visible");

        // At cycle 17: load ready -- forward overrides MOV
        ctx.cycles = 17;
        assert_eq!(ctx.scalar_read(5), 3002, "At ready_cycle, pending load should override MOV value");

        // Commit applies the load to the register file
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(5), 3002, "Load should overwrite the MOV value when committed");
    }

    // =====================================================================
    // Flush / Test Utility
    // =====================================================================

    #[test]
    fn test_flush_pending_writes() {
        // flush_pending_writes() ignores timing and applies all writes immediately.
        let mut ctx = ExecutionContext::new();
        ctx.cycles = 10;
        ctx.queue_scalar_load(Operand::ScalarReg(0), 42, 100); // ready=110
        ctx.queue_scalar_load(Operand::PointerReg(3), 0x1234, 100);

        ctx.flush_pending_writes();

        assert_eq!(ctx.scalar.read(0), 42);
        assert_eq!(ctx.pointer.read(3), 0x1234);
        assert!(ctx.pending_writes.is_empty());
    }

    // =====================================================================
    // Full Call/Return/Recall Sequence (init_values_repeat reproduction)
    // =====================================================================

    #[test]
    fn test_p7_survives_memcpy_return_and_loop_back() {
        // Simulates the EXACT sequence from init_values_repeat:
        //
        // 1. p7 = 0x78000 (set in main prologue via movxm)
        // 2. Call memcpy (jl): delay slot 4 does `mov p2, p7` -> p2 = 0x78000
        // 3. memcpy: save p7 to stack, clobber with sp, padda, loop body
        // 4. memcpy epilogue: `lda p7, [sp, #-32]` (deferred, latency 7)
        // 5. `ret lr`: 5 delay slots, then delay_pending_writes(1)
        // 6. Back in caller: arithmetic + jnz loop back + its delay_pending_writes(1)
        // 7. Second call to memcpy: delay slot 4 reads p7 -> must be 0x78000
        //
        // If this test fails, the pipeline mechanism is broken.
        // If it passes, the bug is in the executor or decoder layer.
        let mut ctx = ExecutionContext::new();
        ctx.set_sp(0x70040);

        // === Phase 1: main prologue - set p7 = 0x78000 ===
        ctx.cycles = 10;
        ctx.queue_pointer_write(7, 0x78000, 1); // movxm p7, #0x78000
        ctx.cycles = 11;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000, "p7 set in prologue");

        // === Phase 2: First jl call - delay slot reads p7 ===
        ctx.cycles = 15;
        assert_eq!(ctx.pointer_read(7), 0x78000, "First call delay slot: p7 correct");

        // === Phase 3: memcpy prologue ===
        // paddb sp, #0x20
        ctx.set_sp(0x70060);

        // st p7, [sp, #-32]: saves p7 to memory (simulated - memory write)
        ctx.cycles = 20;
        let saved = ctx.pointer_read(7);
        assert_eq!(saved, 0x78000, "memcpy prologue saves correct p7");

        // mov p7, sp: clobber p7 with frame pointer
        ctx.cycles = 21;
        ctx.queue_pointer_write(7, 0x70060, 1);
        ctx.cycles = 22;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x70060, "p7 clobbered to sp");

        // padda [p7], #-32: p7 = sp - 32
        ctx.cycles = 22;
        ctx.queue_pointer_write(7, 0x70060 - 32, 1);
        ctx.cycles = 23;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x70040, "p7 after padda");

        // === Phase 4: memcpy body runs (many cycles, p7 untouched) ===

        // === Phase 5: memcpy epilogue - lda p7, [sp, #-32] ===
        // Memory at sp-32 contains 0x78000 (saved in Phase 3)
        ctx.cycles = 100;
        ctx.queue_scalar_load(Operand::PointerReg(7), 0x78000, 7); // ready=107

        // ret lr at cycle 101
        ctx.cycles = 101;
        // (5 delay slots execute at cycles 102-106)

        // Last delay slot at cycle 106
        ctx.cycles = 106;
        // After branch takes effect: delay_pending_writes(1)
        ctx.delay_pending_writes(1); // p7 ready: 107 -> 108

        // === Phase 6: Back at caller (return point) ===
        ctx.cycles = 107;
        ctx.commit_pending_writes(); // p7 ready=108 > 107, NOT committed
                                     // Before ready_cycle: reads old clobbered value
        assert_eq!(
            ctx.pointer_read(7),
            0x70040,
            "Before ready_cycle, p7 should read clobbered value (from padda)"
        );

        // At ready_cycle 108: forward works and commit applies
        ctx.cycles = 108;
        assert_eq!(ctx.pointer_read(7), 0x78000, "At ready_cycle, p7 should be forwarded");
        ctx.commit_pending_writes(); // p7 ready=108 <= 108, COMMITTED!
        assert_eq!(ctx.pointer.read(7), 0x78000, "p7 committed in register file after return");
        assert!(
            ctx.pending_writes
                .iter()
                .all(|pw| { !matches!(&pw.dest, Operand::PointerReg(7)) }),
            "p7 pending write drained from queue"
        );

        // === Phase 7: Several instructions in caller (no p7 writes) ===
        // rel, add, ltu, add, xor, xor, or at cycles 107-114
        for c in 109..=115 {
            ctx.cycles = c;
            ctx.commit_pending_writes(); // no-op for p7
        }

        // === Phase 8: jnz loop back (second branch) ===
        ctx.cycles = 116; // jnz
                          // 5 delay slots at 117-121
        ctx.cycles = 121;
        // Branch takes effect - delay_pending_writes(1) on empty queue
        ctx.delay_pending_writes(1); // no-op (p7 already committed)

        // === Phase 9: Back at loop start ===
        ctx.cycles = 122;
        ctx.commit_pending_writes(); // nothing pending
        assert_eq!(ctx.pointer.read(7), 0x78000, "p7 intact after jnz loop back");

        // === Phase 10: acq (stalls, cycles advance) ===
        for c in 123..=130 {
            ctx.cycles = c;
            ctx.commit_pending_writes();
        }

        // === Phase 11: Second jl call - delay slot reads p7 ===
        ctx.cycles = 135;
        let p7_second_call = ctx.pointer_read(7);
        assert_eq!(p7_second_call, 0x78000, "CRITICAL: p7 must be 0x78000 at second memcpy call");
    }

    #[test]
    fn test_multiple_delay_pending_writes_dont_corrupt() {
        // Regression test: two branches in sequence, only the first
        // has a pending p7 write. The second delay_pending_writes(1)
        // must not corrupt or duplicate anything.
        let mut ctx = ExecutionContext::new();

        // Queue a p7 load at cycle 50
        ctx.cycles = 50;
        ctx.queue_scalar_load(Operand::PointerReg(7), 0xDEAD, 7); // ready=57

        // First branch: delay_pending_writes(1) → ready=58
        ctx.delay_pending_writes(1);

        // p7 commits at cycle 58
        ctx.cycles = 58;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0xDEAD);
        assert!(ctx.pending_writes.is_empty(), "Queue should be empty");

        // Second branch: delay_pending_writes(1) on empty queue
        ctx.cycles = 70;
        ctx.delay_pending_writes(1); // no-op
        assert_eq!(ctx.pointer.read(7), 0xDEAD, "p7 unchanged by second delay");

        // Third branch: still OK
        ctx.cycles = 80;
        ctx.delay_pending_writes(1);
        assert_eq!(ctx.pointer.read(7), 0xDEAD, "p7 still correct");
    }

    /// VLIW: a vector-register write during a bundle must NOT be visible to
    /// reads in the same bundle. Regression for vec_srs_i32: a Store-slot VSRS
    /// writes a w-register that a Scalar1 vmov reads in the same bundle (Store
    /// executes before Scalar1 in slot order), so the vmov must see the
    /// pre-bundle value -- pure read-old/write-new.
    #[test]
    fn test_bundle_snapshot_hides_in_bundle_vector_write() {
        let mut ctx = ExecutionContext::new();
        ctx.vector.write(1, [10, 20, 30, 40, 50, 60, 70, 80]);
        ctx.begin_bundle();
        ctx.vector.write(1, [1, 2, 3, 4, 5, 6, 7, 8]); // in-bundle writer (the VSRS)
        assert_eq!(
            ctx.vector.read(1),
            [10, 20, 30, 40, 50, 60, 70, 80],
            "in-bundle vector read must see the pre-bundle value"
        );
        ctx.end_bundle();
        assert_eq!(ctx.vector.read(1), [1, 2, 3, 4, 5, 6, 7, 8], "after the bundle, the write is live");
    }

    /// VLIW read-old/write-new for accumulator (cm) registers, including the
    /// wide (1024-bit cm) read path that the SRS uses.
    #[test]
    fn test_bundle_snapshot_hides_in_bundle_accum_write() {
        let mut ctx = ExecutionContext::new();
        ctx.accumulator.write(0, [10, 20, 30, 40, 50, 60, 70, 80]);
        ctx.accumulator.write(1, [11, 21, 31, 41, 51, 61, 71, 81]);
        ctx.begin_bundle();
        ctx.accumulator.write(0, [1; 8]);
        assert_eq!(ctx.accumulator.read(0), [10, 20, 30, 40, 50, 60, 70, 80], "narrow read sees pre-bundle");
        let wide = ctx.accumulator.read_wide(0);
        assert_eq!(&wide[..8], &[10, 20, 30, 40, 50, 60, 70, 80], "wide read low half sees pre-bundle");
        assert_eq!(&wide[8..], &[11, 21, 31, 41, 51, 61, 71, 81], "wide read high half sees pre-bundle");
        ctx.end_bundle();
        assert_eq!(ctx.accumulator.read(0), [1; 8], "after the bundle, the write is live");
    }

    /// VLIW read-old/write-new for q (mask) registers.
    #[test]
    fn test_bundle_snapshot_hides_in_bundle_mask_write() {
        let mut ctx = ExecutionContext::new();
        ctx.mask.write(0, [10, 20, 30, 40]);
        ctx.begin_bundle();
        ctx.mask.write(0, [1, 2, 3, 4]);
        assert_eq!(ctx.mask.read(0), [10, 20, 30, 40], "in-bundle mask read must see pre-bundle value");
        ctx.end_bundle();
        assert_eq!(ctx.mask.read(0), [1, 2, 3, 4], "after the bundle, the write is live");
    }

    #[test]
    fn test_register_init_tripwire() {
        // Default pattern is 0xDEADBEEF (unless XDNA_EMU_REG_INIT overrides).
        // Registers NOT touched by init should retain the sentinel.
        let pattern = reg_init_pattern();
        let ctx = ExecutionContext::new_for_tile(1, 2);

        // Scalar r0 is not initialized -- should contain sentinel
        assert_eq!(ctx.scalar.read(0), pattern, "r0 should contain sentinel");
        assert_eq!(ctx.scalar.read(15), pattern, "r15 should contain sentinel");

        // CORE_ID IS initialized -- should NOT contain sentinel
        assert_ne!(
            ctx.scalar.read(crate::interpreter::state::registers::CORE_ID_REG_INDEX),
            pattern,
            "CORE_ID should be overwritten by init"
        );
        // Verify CORE_ID has the right value
        assert_eq!(
            ctx.scalar.read(crate::interpreter::state::registers::CORE_ID_REG_INDEX),
            (1u32 << 16) | 2u32,
            "CORE_ID should encode col=1, row=2"
        );

        // PC IS initialized to 0
        assert_eq!(ctx.pc(), 0, "PC should be initialized to 0");

        // Vector register should contain sentinel
        let v0 = ctx.vector.read(0);
        assert_eq!(v0, [pattern; 8], "v0 should contain sentinel");

        // Accumulator should contain doubled sentinel
        let wide = (pattern as u64) << 32 | (pattern as u64);
        let acc0 = ctx.accumulator.read(0);
        assert_eq!(acc0, [wide; 8], "acc0 should contain sentinel");

        // Pointer registers should contain sentinel
        assert_eq!(ctx.pointer.read(0), pattern, "p0 should contain sentinel");

        // Modifier registers should contain sentinel
        assert_eq!(ctx.modifier.read(0), pattern, "m0 should contain sentinel");

        // Mask registers should contain sentinel
        assert_eq!(ctx.mask.read(0), [pattern; 4], "q0 should contain sentinel");

        // SRS shift registers (s0-s3 = ScalarReg 40-43) should be zero,
        // not sentinel, because hardware resets them to 0.
        assert_eq!(ctx.scalar.read(40), 0, "s0 should be zero (hardware reset)");
        assert_eq!(ctx.scalar.read(41), 0, "s1 should be zero (hardware reset)");
        assert_eq!(ctx.scalar.read(42), 0, "s2 should be zero (hardware reset)");
        assert_eq!(ctx.scalar.read(43), 0, "s3 should be zero (hardware reset)");
    }
}
