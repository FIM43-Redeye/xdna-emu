//! Behavioral units: the explicit override registry (spec Section 3) and the
//! single hand-curated CapabilityDomain spine (spec Section 6). Seeded coarse;
//! Phase 2 refines. Co-located on purpose (one location, spec Section 6).

use crate::aie2::isa::{ElementType, SemanticOp};
use crate::coverage::variant::{variants_of, OpVariant};
use crate::coverage::verdict::{Verdict, Verification};
use crate::coverage::CoverageNode;
use crate::types::Architecture;

/// Nodes for every variant of each of `ops`: for evidence that genuinely
/// covers the whole op (the op has one variant, or the evidence spans all).
fn every_variant(arch: Architecture, ops: &[SemanticOp]) -> Vec<CoverageNode> {
    ops.iter()
        .flat_map(|&op| {
            variants_of(arch, op)
                .into_iter()
                .map(move |variant| CoverageNode::Semantic { arch, op, variant })
        })
        .collect()
}

/// Nodes for exactly the listed (op, variant) pairs.
fn listed_variants(arch: Architecture, pairs: &[(SemanticOp, OpVariant)]) -> Vec<CoverageNode> {
    pairs
        .iter()
        .map(|&(op, variant)| CoverageNode::Semantic { arch, op, variant })
        .collect()
}

/// What fine nodes a behavioral unit claims (spec Section 3). A unit either
/// claims explicit nodes (override) or is the derived bucket for a category.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Claims {
    /// Explicit override: this exact set of nodes (spec Section 3).
    Nodes(Vec<CoverageNode>),
}

/// A behavioral unit: a cluster of fine nodes with one verdict (spec Section 3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BehavioralUnit {
    pub id: String,
    pub arch: Architecture,
    pub claims: Claims,
    pub verdict: Verdict,
    /// Free-text human narrative for the Section-3 no-silent-shadow rule:
    /// when an override pulls a node off the toolchain-derived path, this
    /// records why for a human reader. NEVER substring-matched for
    /// enforcement -- see `shared_from` for the typed soundness gate.
    pub shadows_derived: Option<String>,
    /// Typed cross-arch provenance (spec Section 7). Some(other_arch) means
    /// this verdict was shared from other_arch. enforce_coverage panics on
    /// Verified + shared_from.is_some() -- verification never transfers
    /// across silicon. Typed, so phrasing can neither bypass nor trip it.
    pub shared_from: Option<Architecture>,
}

/// A top-level hardware capability the manual names (spec Section 6), now
/// carrying the retired index's per-subsystem detail as seeded data (spec
/// Section 2). Each must be claimed by >= 1 behavioral unit per applicable
/// arch, or the build panics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityDomain {
    pub id: String,
    pub arches: Vec<Architecture>,
    /// Authoritative source (aie-rt path / AM025 / device model). Non-empty,
    /// enforced.
    pub source_ref: String,
    /// Our emulator src/ path(s). Non-empty unless an OOS/MISSING narrative is
    /// present (enforced).
    pub src_locations: Vec<String>,
    /// Human-readable notes and known gaps for this domain.
    pub narrative: String,
    /// The domain's own asserted coverage verdict (spec Section 1/2).
    pub verdict: Verdict,
    /// Documents a deliberate own-vs-rollup divergence (spec Section 3); a
    /// silent material drift is a hard failure, an annotated one is allowed.
    pub drift_rationale: Option<String>,
}

impl CapabilityDomain {
    pub fn applies_to(&self, arch: Architecture) -> bool {
        self.arches.contains(&arch)
    }
}

/// The explicit override registry, per arch. Phase 2 adds entries here; the
/// sole current entry Accepts the `SemanticOp::Intrinsic(_)` catch-all (#104).
pub fn override_registry(arch: Architecture) -> Vec<BehavioralUnit> {
    match arch {
        Architecture::Aie2 => {
            vec![
                intrinsic_catchall_accepted(),
                dma_ops_verified(),
                vector_ops_verified(),
                lane_goldens_verified(),
                gated_not_modeled(),
            ]
        }
        _ => Vec::new(),
    }
}

/// The (op, variant) nodes the vector differential fuzzer's silicon-matched
/// corpus executed, re-derived per variant (2026-10-29): the corpus's replayable
/// seeds are replayed on the emulator with the variant-keyed fuzz recorder
/// (`xdna-emu fuzz-vector --replay`), and a variant is listed only if it
/// executed in a seed whose replay currently matches the banked NPU1 silicon
/// output. Under-claim is safe, over-claim is a correctness bug.
///
/// Union of the executed keys of the 3 matching seeds (5026, 5038, 70000),
/// Vector-category ops only, in `category()` order with each op's variants in
/// `OpVariant` order (`vector_verified_variants_are_sorted` checks it). Left
/// out on purpose: `Shuffle/untyped` (executed incidentally by lowering, never
/// a fuzzed stage, and shuffle mode is not a variant axis), and every executed
/// key whose op is not Vector-category (`Add`, `Sub`, `And`, `Or`, `Copy`,
/// `SetEq`, `SetLt`), which the derived defaults own.
pub(crate) const VECTOR_VERIFIED_VARIANTS: &[(SemanticOp, OpVariant)] = &[
    (SemanticOp::Mac, mac(0, 1, 0, true, true)),
    (SemanticOp::Mac, mac(1, 3, 1, true, true)),
    (SemanticOp::MatMul, mac(0, 1, 0, true, true)),
    (SemanticOp::MatMul, mac(1, 3, 1, true, true)),
    (SemanticOp::Srs, converted(ElementType::Int8, ElementType::Int32)),
    (SemanticOp::Srs, converted(ElementType::UInt8, ElementType::Int32)),
    (SemanticOp::Srs, converted(ElementType::Int16, ElementType::Int32)),
    (SemanticOp::Srs, converted(ElementType::Int16, ElementType::Int64)),
    (SemanticOp::Srs, converted(ElementType::UInt16, ElementType::Int32)),
    (SemanticOp::Srs, converted(ElementType::Int32, ElementType::Int64)),
    (SemanticOp::Ups, converted(ElementType::Int32, ElementType::Int8)),
    (SemanticOp::Ups, converted(ElementType::Int32, ElementType::UInt8)),
    (SemanticOp::Ups, converted(ElementType::Int32, ElementType::Int16)),
    (SemanticOp::Ups, converted(ElementType::Int32, ElementType::UInt16)),
    (SemanticOp::Ups, converted(ElementType::Int64, ElementType::Int32)),
    (SemanticOp::Pack, converted(ElementType::Int8, ElementType::Int16)),
    (SemanticOp::Unpack, converted(ElementType::Int16, ElementType::Int8)),
    (SemanticOp::VectorBroadcast, typed(ElementType::Int8)),
    (SemanticOp::VectorBroadcast, typed(ElementType::Int16)),
    (SemanticOp::VectorBroadcast, typed(ElementType::Int32)),
    (SemanticOp::VectorSelect, typed(ElementType::Int8)),
    (SemanticOp::VectorSelect, typed(ElementType::Int16)),
    (SemanticOp::VectorSelect, typed(ElementType::Int32)),
    (SemanticOp::VectorClear, OpVariant::Untyped),
    (SemanticOp::VectorExtract, typed(ElementType::Int16)),
    (SemanticOp::VectorInsert, typed(ElementType::Int32)),
    (SemanticOp::MaxDiffLt, typed(ElementType::Int16)),
    (SemanticOp::MaxDiffLt, typed(ElementType::Int32)),
    (SemanticOp::MaxLt, typed(ElementType::Int8)),
    (SemanticOp::MaxLt, typed(ElementType::Int16)),
    (SemanticOp::MaxLt, typed(ElementType::Int32)),
    (SemanticOp::MinGe, typed(ElementType::Int8)),
    (SemanticOp::MinGe, typed(ElementType::Int16)),
    (SemanticOp::MinGe, typed(ElementType::Int32)),
    (SemanticOp::AbsGtz, typed(ElementType::Int32)),
    (SemanticOp::NegGtz, typed(ElementType::Int16)),
    (SemanticOp::NegGtz, typed(ElementType::Int32)),
    (SemanticOp::NegLtz, typed(ElementType::Int16)),
    (SemanticOp::NegLtz, typed(ElementType::Int32)),
];

const fn typed(out: ElementType) -> OpVariant {
    OpVariant::Typed { out, from: None }
}

const fn converted(out: ElementType, from: ElementType) -> OpVariant {
    OpVariant::Typed { out, from: Some(from) }
}

const fn mac(amode: u8, bmode: u8, variant: u8, sgn_x: bool, sgn_y: bool) -> OpVariant {
    OpVariant::MacMode { amode, bmode, variant, sgn_x, sgn_y }
}

const fn acc(amode: u8, bmode: u8, variant: u8) -> OpVariant {
    OpVariant::AccMode { amode, bmode, variant }
}

/// The (op, variant) nodes the lane-level silicon goldens pin, by what the
/// capture kernels execute (Chess disassembly of the kernels, forms kernel
/// in `build/experiments/d6-ops/forms.dis`):
///
/// - `vec_srs_i32*`: `vups.s64.s32` then `vst.srs.s16.s64` -> `Srs`
///   Int16<-Int64 (and `Ups` Int64<-Int32, which no lib test checks: the
///   `vec_ups_i32*` captures are not wired to one, so UPS is not claimed here).
///   `matches_every_silicon_srs_capture` checks the SRS lanes.
/// - `vec_conv_bf16*`: `vst.conv.bf16.fp32` -> `Convert` BFloat16<-Float32.
/// - `vec_mac_bf16`, `_ovf`, `_forms`: `vmul.f`, `vnegmul.f`, `vmac.f`,
///   `vmsc.f`, `vaddmac.f`, `vaddmsc.f` with config words amode 2, bmode 3,
///   sign bits clear; variant 0 (dense 4x8x4) for all six forms, variant 1
///   (elementwise 16x2) for mul/negmul/mac/msc only.
///   Semantics by decoder: vmul.f MatMul, vnegmul.f NegMul, vmac.f Mac,
///   vmsc.f MatMulSub, vaddmac.f and vaddmsc.f AddMac.
/// - `vec_acc_addsub_f32`: `vadd.f`, `vsub.f`, `vnegsub.f` with the fp32
///   accumulator config (amode 2, bmode 3) -> `Accumulate`/`AccumSub`/
///   `AccumNegSub` acc(2,3,0); `vnegadd.f` is not config-typed (the header has
///   no `negadd` accfloat builtin of its own) -> `AccumNegAdd` Float32. Its
///   `vneg.f` forms key as `Neg`, a category with no verification axis.
/// - `vec_vshift_wide`: `vshift` and `vshift.align` -> `Align` (untyped).
///
/// Only the variants no other unit claims are claimed by the unit (a node has
/// one owner); the SRS and UPS variants the fuzzer also executed stay with the
/// vector unit.
const LANE_GOLDEN_VARIANTS: &[(SemanticOp, OpVariant)] = &[
    (SemanticOp::MatMul, mac(2, 3, 0, false, false)),
    (SemanticOp::MatMul, mac(2, 3, 1, false, false)),
    (SemanticOp::NegMul, mac(2, 3, 0, false, false)),
    (SemanticOp::NegMul, mac(2, 3, 1, false, false)),
    (SemanticOp::Mac, mac(2, 3, 0, false, false)),
    (SemanticOp::Mac, mac(2, 3, 1, false, false)),
    (SemanticOp::MatMulSub, mac(2, 3, 0, false, false)),
    (SemanticOp::MatMulSub, mac(2, 3, 1, false, false)),
    (SemanticOp::AddMac, mac(2, 3, 0, false, false)),
    (SemanticOp::Srs, converted(ElementType::Int16, ElementType::Int64)),
    (SemanticOp::Convert, converted(ElementType::BFloat16, ElementType::Float32)),
    (SemanticOp::Accumulate, acc(2, 3, 0)),
    (SemanticOp::AccumSub, acc(2, 3, 0)),
    (SemanticOp::AccumNegSub, acc(2, 3, 0)),
    (SemanticOp::AccumNegAdd, typed(ElementType::Float32)),
    (SemanticOp::Align, OpVariant::Untyped),
];

/// Variants the emulator deliberately declines ("[NOT MODELED] ... awaiting
/// clean-room derivation"): the implementation was removed in the re-root and
/// the executor reports the op unhandled. The toolchain describes them and we
/// assert no model, so they are comprehension gaps, not perishable models.
///
/// - Sparse MAC, every MAC-family variant in a mode whose geometry is sparse:
///   gated on the sparse-operand source in `vector_matmul/mod.rs`.
///
/// Wide VSHIFT and the fp32 accumulator add/sub were gated here too until they
/// were derived from silicon (2026-10-29, `lane_goldens_verified`).
fn gated_not_modeled() -> BehavioralUnit {
    use crate::aie2::matmul::CONFIG_GEOMETRY_TABLE;
    use crate::coverage::all_semantic_ops;
    use crate::coverage::verdict::Provenance;
    let arch = Architecture::Aie2;
    let sparse = |k: (u8, u8, u8)| {
        CONFIG_GEOMETRY_TABLE
            .iter()
            .any(|(a, b, v, g)| g.sparse && (*a as u8, *b as u8, *v as u8) == k)
    };
    let gated = |_op: SemanticOp, v: &OpVariant| match *v {
        OpVariant::MacMode { .. } => v.mode_key().is_some_and(sparse),
        _ => false,
    };
    let nodes = all_semantic_ops()
        .into_iter()
        .flat_map(|op| variants_of(arch, op).into_iter().map(move |variant| (op, variant)))
        .filter(|(op, v)| gated(*op, v))
        .map(|(op, variant)| CoverageNode::Semantic { arch, op, variant })
        .collect();
    BehavioralUnit {
        id: "aie2.gated.not_modeled".into(),
        arch,
        claims: Claims::Nodes(nodes),
        verdict: Verdict { provenance: Provenance::Unspecified, verification: Verification::Unverified },
        shadows_derived: Some(
            "These variants otherwise derive to the Vector default (ModelInformed/Unverified -> \
             perishable), but the emulator declines them: sparse MAC modes are gated \
             \"[NOT MODELED] awaiting clean-room derivation\" since the re-root."
                .into(),
        ),
        shared_from: None,
    }
}

/// Flip the silicon-verified Vector-category variants to Verified (#126, #127),
/// re-derived per variant.
///
/// The vector differential fuzzer (#112/#114) silicon-matched its 218-key
/// universe against real NPU1. An op-level claim from that was too coarse (`Mac`
/// was "verified" by integer fuzzing while every bf16 `Mac` was wrong), so the
/// claim is now per (op, variant): the corpus's replayable seeds are replayed on
/// the emulator with the variant-keyed `fuzz_recorder` armed, and only variants
/// that executed in a currently-matching seed are listed
/// (`VECTOR_VERIFIED_VARIANTS`). Method and the historical op-level audit:
/// `xdna-emu-priv: research/findings/2026-06-12-vector-semanticop-empirical-audit.md`.
///
/// 2026-10-29 replay: 28 live seeds (all `seed_*` directories of the corpus;
/// the 45 under `archive/` are pre-pool banks that replay cannot run and are
/// not evidence). Before the wide-vshift and fp32 accumulator derivation 3
/// matched and 25 could not run; after it all 28 match silicon, 0 diverge, 0
/// error. A seed that errors proves nothing, so only keys of matching seeds
/// are listed. The fuzzer's `Align` and fp32 `Accumulate`/`AccumSub` keys are
/// owned by the lane goldens, which pin them lane by lane.
///
/// The integer `aie::min`/`max`/`abs`/`neg`/`maxdiff` lower to the hardware's
/// fused compare-select instructions, so the verified ops are `MaxLt`/`MinGe`/
/// `AbsGtz`/`NegLtz`/`MaxDiffLt`, not `Min`/`Max`. `pack16` (I32->I16) and
/// `unpack32` (I16->I32) lower to `Srs`/`Ups`.
///
/// The fuzzer's bf16 `.f` MAC keys (`Mac`/`MatMul` `Float32`) are NOT claimed
/// from the fuzzer: its bf16 pool is 97.6% NaN/Inf/overflow lanes and its
/// comparison rounds the fp32 result to bf16 and ignores NaN payloads, so those
/// stages do not discriminate the MAC datapath (see
/// `build/experiments/per-variant/bf16_discrimination.py`); the lane goldens
/// own them (`lane_goldens_verified`).
///
/// SRS and f32->bf16 rounding is additionally pinned lane by lane across every
/// crRnd and crSat mode by the lane goldens; `Srs` Int16<-Int64 sits in both
/// evidence sets and is claimed here (a node has one owner).
///
/// Provenance stays `ModelInformed`; only verification moves to `Verified`.
fn vector_ops_verified() -> BehavioralUnit {
    use crate::coverage::verdict::{Provenance, Verdict};
    BehavioralUnit {
        id: "aie2.vector_ops.verified".into(),
        arch: Architecture::Aie2,
        claims: Claims::Nodes(listed_variants(Architecture::Aie2, VECTOR_VERIFIED_VARIANTS)),
        verdict: Verdict {
            provenance: Provenance::ModelInformed,
            verification: Verification::Verified {
                evidence: "Vector differential fuzzer (#112/#114) vs real NPU1, re-derived per \
                    variant on 2026-10-29: the corpus's 28 live seeds (seed_* directories; the 45 in \
                    archive/ are pre-pool banks and not evidence) were replayed on the emulator with \
                    the variant-keyed fuzz recorder: all 28 match the banked silicon output, 0 \
                    diverge, 0 error (3/28 before wide vshift and the fp32 accumulator adder were \
                    derived from silicon). The 39 listed (op, variant) nodes are the integer \
                    Vector-category keys the matching seeds executed (67 distinct keys in all; the \
                    rest are non-Vector ops, Shuffle, keys the lane goldens own, or bf16 keys the \
                    fuzzer's bf16-rounded NaN-equal compare cannot discriminate). Fused \
                    compare-select lowering: min/max/abs/neg/maxdiff execute as MaxLt/MinGe/AbsGtz/\
                    NegLtz/MaxDiffLt (Min/Max are dead); pack16 and unpack32 lower to Srs/Ups. \
                    Earlier op-level history: #126 (20 ops from the executed-op audit, xdna-emu-priv: \
                    research/findings/2026-06-12-vector-semanticop-empirical-audit.md) and #127 \
                    (Pack/Unpack re-fuzzed on real NPU1, 60/60 HW match), re-confirmed per variant by \
                    the current replay. SRS \
                    rounding/saturation is additionally pinned lane by lane (Srs Int16<-Int64 is in \
                    both evidence sets: tools/golden/silicon/vec_srs_i32*.json, 77 kernels, every \
                    crRnd 0-15, crSat none/saturate/symmetric, shifts 0-31, checked by \
                    matches_every_silicon_srs_capture in vector_srs.rs)."
                    .into(),
            },
        },
        shadows_derived: Some(
            "These Vector-category variants otherwise derive to the Vector default \
             (ModelInformed/Unverified -> perishable). Claimed ONLY where the variant executed in a \
             currently-replay-confirmed silicon match; everything else (Min/Max/MatMulSub/\
             VectorPush/..., the bf16 fuzzer keys, other element types) is NOT claimed and stays \
             perishable."
                .into(),
        ),
        shared_from: None,
    }
}

/// Verified by lane-level comparison with NPU1 silicon captures: the bf16
/// multiply-accumulate forms (D6), the fp32 accumulator add/sub, wide vshift,
/// and the SRS / f32->bf16 conversion rounding.
///
/// The `matches_every_silicon_*` tests run the emulator's lane functions over
/// the captures in `tools/golden/silicon/` and demand bit-exact equality:
/// `matches_every_silicon_srs_capture` (`vector_srs.rs`),
/// `matches_every_silicon_bf16_capture` (`vector_float.rs`),
/// `matches_silicon_on_every_form_lane` and `matches_older_vmul_captures_exactly`
/// (`vector_matmul/bf16.rs`), and `test_bf16_forms_match_silicon_through_execute_matmul`
/// and `test_vmul_f_matches_silicon_exactly_on_older_captures` (`vector_matmul/mod.rs`,
/// end to end through the instruction handler), `acc_add_matches_silicon_on_every_form_lane`
/// (`vector_matmul/bf16.rs`) and `wide_shift_matches_silicon_on_every_entry` (`vector_misc.rs`). See `LANE_GOLDEN_VARIANTS` for which
/// (op, variant)s the capture kernels execute. A node may be claimed by one unit
/// only, so variants the fuzzer unit already claims are left to it.
fn lane_goldens_verified() -> BehavioralUnit {
    use crate::coverage::verdict::{Provenance, Verdict};
    let nodes = listed_variants(Architecture::Aie2, LANE_GOLDEN_VARIANTS)
        .into_iter()
        .filter(|n| !listed_variants(Architecture::Aie2, VECTOR_VERIFIED_VARIANTS).contains(n))
        .collect();
    BehavioralUnit {
        id: "aie2.lane_goldens.verified".into(),
        arch: Architecture::Aie2,
        claims: Claims::Nodes(nodes),
        verdict: Verdict {
            provenance: Provenance::ModelInformed,
            verification: Verification::Verified {
                evidence: "Lane-by-lane bit-exact against NPU1 silicon captures (2026-10): \
                    tools/golden/silicon/vec_mac_bf16_forms.json (23 forms x 336 tiles = 123,648 \
                    lanes: vmul.f, vnegmul.f, vmac.f, vmsc.f, vaddmac.f, vaddmsc.f, dense 4x8x4 and \
                    elementwise, config sub/zero controls; bit-identical across two hardware runs), \
                    vec_mac_bf16.json and vec_mac_bf16_ovf.json (vmul.f incl. NaN/Inf tiles), and \
                    vec_conv_bf16*.json (27 kernels, vst.conv.bf16.fp32, every crRnd incl. \
                    denormal/Inf/NaN edge sets). Tests: matches_silicon_on_every_form_lane and \
                    matches_older_vmul_captures_exactly (vector_matmul/bf16.rs), \
                    test_bf16_forms_match_silicon_through_execute_matmul and \
                    test_vmul_f_matches_silicon_exactly_on_older_captures (vector_matmul/mod.rs), \
                    matches_every_silicon_bf16_capture (vector_float.rs), \
                    matches_every_silicon_srs_capture (vector_srs.rs; its Srs Int16<-Int64 node is \
                    claimed by aie2.vector_ops.verified). Variants identified from the Chess \
                    disassembly of the capture kernels (build/experiments/d6-ops/forms.dis for the \
                    forms kernel). fp32 accumulator add/sub (2026-10-29): \
                    tools/golden/silicon/vec_acc_addsub_f32.json (20 forms x 704 tiles x 16 lanes: \
                    vadd.f, vsub.f, vnegadd.f, vnegsub.f, vneg.f with zero_acc1/sub_acc1/sub_acc2 \
                    config variants; 202,752 non-identity lanes, all explained by the D6 lane adder \
                    with zero product terms), checked by acc_add_matches_silicon_on_every_form_lane \
                    (vector_matmul/bf16.rs). Wide vshift/vshift.align: \
                    tools/golden/silicon/vec_vshift_wide.json (2,432 (step, shift) entries, shifts \
                    0-127 plus out-of-range), checked by wide_shift_matches_silicon_on_every_entry \
                    (vector_misc.rs). Both kernels bit-identical across three hardware runs, \
                    disassembly in build/experiments/gated-forms/. Not covered: vnegmac.f, \
                    vsubmac.f, vsubmsc.f (no capture, gated), sparse bf16, elementwise \
                    vaddmac.f/vaddmsc.f (not in the forms kernel)."
                    .into(),
            },
        },
        shadows_derived: Some(
            "These variants otherwise derive to the Vector default (ModelInformed/Unverified -> \
             perishable); the lane goldens are silicon evidence for them. Only the forms the \
             capture kernels execute are claimed."
                .into(),
        ),
        shared_from: None,
    }
}

/// Flip `DmaStart` + `DmaWait` to silicon-Verified (#113 axis-2).
///
/// The diff-fuzzing framework's DMA/data-movement tenant drove the DMA
/// access-pattern engine against real NPU1 silicon to ledger completion: the
/// full 81-key universe `{feature}/{engine}/{dir}/{dtype}` (33 shim + 48
/// memtile; linear/strided-2d/3d/4d/transpose/overlap/packet/pad-{before,
/// after,both}) was silicon-matched at depth ~10 seeds/key -- 810 HW cases, 0
/// emulator-vs-silicon divergences, 0 crashes, 0 wedges. Every n-dimensional BD
/// program the generator can express lands byte-for-byte identically on the
/// emulator and the hardware. In SemanticOp terms that is exactly `DmaStart`
/// (issue + walk the BD) and `DmaWait` (await channel completion).
///
/// Scope is deliberately TWO ops, not the whole `SideEffect` category. The
/// remaining five SideEffect ops -- `CascadeRead`/`CascadeWrite` (cascade) and
/// `StreamRead`/`StreamWrite`/`StreamWritePacketHeader` (core-side stream) --
/// are the contention/cascade tenants (4/5) and are NOT exercised by the DMA
/// tenant, so they keep their honest `DocSpecified`/`Unverified` default.
///
/// Provenance stays `DocSpecified` (the access-pattern semantics are
/// AM025/aie-rt described); only the verification axis moves to `Verified`,
/// mirroring the modeled-then-silicon-checked transition. This Verified flip
/// does NOT green `semantic_provenance_clean(Aie2)`: the perishable queue still
/// holds the vector ops and the unclaimed SideEffect ops, so the gate stays
/// honestly red until the vector and tenant-4/5 work lands.
fn dma_ops_verified() -> BehavioralUnit {
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verdict};
    BehavioralUnit {
        id: "aie2.dma_ops.verified".into(),
        arch: Architecture::Aie2,
        claims: Claims::Nodes(every_variant(
            Architecture::Aie2,
            &[SemanticOp::DmaStart, SemanticOp::DmaWait],
        )),
        verdict: Verdict {
            provenance: Provenance::DocSpecified,
            verification: Verification::Verified {
                evidence: "Diff-fuzzing framework DMA/data-movement tenant vs real NPU1: the full \
                    81-key access-pattern universe ({feature}/{engine}/{dir}/{dtype}, 33 shim + 48 \
                    memtile, linear/strided-2d/3d/4d/transpose/overlap/packet/pad-*) silicon-matched \
                    at depth ~10 seeds/key -- 810 HW cases, 0 divergent, 0 crash, 0 wedge. The \
                    emulator DMA model is indistinguishable from NPU1 silicon across the whole \
                    n-dimensional BD access-pattern space. See \
                    xdna-emu-priv: research/plans/2026-06-11-framework-step3a-dma-tenant.md (Outcome) and \
                    xdna-emu-priv: research/specs/2026-06-11-dma-data-movement-domain.md."
                    .into(),
            },
        },
        shadows_derived: Some(
            "DmaStart/DmaWait otherwise derive to the SideEffect category default \
             (DocSpecified/Unverified -> perishable). This override claims ONLY these two ops -- \
             the silicon evidence is the DMA-engine access-pattern campaign, which does not touch \
             the cascade (CascadeRead/Write) or core-side stream (StreamRead/Write/ \
             WritePacketHeader) ops; those stay on their derived default for tenants 4/5."
                .into(),
        ),
        shared_from: None,
    }
}

/// Accept the `SemanticOp::Intrinsic(_)` catch-all node (#104).
///
/// The node `SemanticOp::Intrinsic(0)` represents "an intrinsic we could not
/// classify to a concrete SemanticOp." It derives to the NeedsTriage category
/// default (`Unspecified` / `Unverified`) and so registers as a comprehension
/// gap. It is closed here by explicit Accept rather than left open, on these
/// confirmed facts:
///
///   * It is **never constructed** by the live pipeline. Every intrinsic-backed
///     instruction is resolved to a concrete SemanticOp at build time
///     (`build_helpers/extract.rs:semantic_from_intrinsic`), and the ISA build
///     asserts 100% semantic coverage (`aie2/isa/mod.rs`), so no decoded
///     instruction can carry `Intrinsic`. The runtime classifier
///     `SemanticOp::from_intrinsic` that would mint `Intrinsic(idx)` is dead
///     code (never called outside its own unit tests).
///   * It is **fail-loud, not silent-wrong**: the surface probe classes
///     `Intrinsic(_)` as `Absent`, so any value reaching execution hits a hard
///     `ExecuteResult::Error`, never a wrong result.
///   * The concrete vector SemanticOps that intrinsics actually resolve to
///     (`Srs`/`Ups`/`Pack`/`Convert`/`Mac`/`MatMul`) are
///     model-differential checked bit-exact (#103 Half A; silicon verification pending;
///     `xdna-emu-priv: research/findings/2026-06-08-vector-compute-audit-half-a-rollup.md`).
///
/// Provenance stays `Unspecified` -- the catch-all genuinely has no model; it is
/// Accepted because it is unconstructible and fail-loud, not because it is
/// understood. Silicon verification of the *resolved* ops is Half B (HW-gated)
/// and is tracked separately in the perishable queue, so this Accept does not
/// green the release gate.
fn intrinsic_catchall_accepted() -> BehavioralUnit {
    use crate::aie2::isa::SemanticOp;
    use crate::coverage::verdict::{Provenance, Verdict};
    BehavioralUnit {
        id: "aie2.intrinsic_catchall.accepted".into(),
        arch: Architecture::Aie2,
        claims: Claims::Nodes(every_variant(Architecture::Aie2, &[SemanticOp::Intrinsic(0)])),
        verdict: Verdict {
            provenance: Provenance::Unspecified,
            verification: Verification::Accepted {
                rationale: "SemanticOp::Intrinsic(_) catch-all: never constructed by the live \
                    pipeline (every intrinsic-backed instruction resolves to a concrete SemanticOp \
                    at build time via semantic_from_intrinsic; the ISA build asserts 100% semantic \
                    coverage; the runtime from_intrinsic classifier is dead code). Fail-loud: the \
                    surface probe classes Intrinsic(_) as Absent -> hard ExecuteResult::Error, never \
                    a wrong value. The concrete ops it resolves to (Srs/Ups/Pack/Convert/Mac/MatMul) \
                    are model-differential checked bit-exact (#103 Half \
                    A). Accepted as unconstructible + fail-loud, not understood; silicon \
                    verification of the resolved ops is Half B (perishable queue)."
                    .into(),
            },
        },
        shadows_derived: Some(
            "Intrinsic(0) otherwise derives to the NeedsTriage default (Unspecified/Unverified -> \
             comprehension gap). This Accept shadows no real unmodeled behavior: the catch-all is \
             never constructed (build-time 100% concrete-SemanticOp coverage) and is fail-loud."
                .into(),
        ),
        shared_from: None,
    }
}

/// The single hand-curated capability spine (spec Section 6). Seeded once from
/// the AM020 ToC + aie-rt module tree; maintained ONLY here. Deliberately
/// COARSER than `SubsystemKind` (spec Section 6: architectural domains, not a
/// 1:1 register-taxonomy mirror). Documented folds so a Phase-2 author does
/// not recreate a domain or think one is missing:
///   - `core` covers `SubsystemKind::Processor` (the VLIW compute core)
///   - `program_counter` covers `SubsystemKind::ProgramCounter` (PC sampling)
///   - `events_trace` covers `SubsystemKind::Trace` AND `::Event`
///   - `locks` covers `SubsystemKind::Lock` AND `::LockRequest`
///   - `debug_halt` covers `SubsystemKind::Debug`
///   - `control_packets` covers on-chip control-packet reassembly, packet
///     handler status, and the NPU host instruction stream (spec Appendix)
///   - `clock_control` covers module/column/tile clock + reset control
///   - `tile_isolation` covers Tile_Control isolation bits / N-S-E-W gates
///   - `binary_load` covers CDO/ELF/XCLBIN ingest, SS-routing reconstruction,
///     and array-topology construction (spec Appendix N1 resolutions)
/// `SubsystemKind::Unknown` is a classifier placeholder, NOT a hardware
/// capability -- intentionally excluded. Coarse, arch-invariant; AIE2 is the
/// only wired arch today (Plan 1). Phase 1 auto-claims every domain via the
/// derived shim (Task 6); the SubsystemKind<->spine partition is Phase 2.
pub fn capability_spine() -> Vec<CapabilityDomain> {
    use crate::coverage::verdict::{Completeness::*, Provenance::*, Verification::*};
    let aie2 = || vec![Architecture::Aie2];
    let d = |id: &str,
             source_ref: &str,
             locs: &[&str],
             narrative: &str,
             v: Verification,
             drift_rationale: Option<&str>| CapabilityDomain {
        id: id.into(),
        arches: aie2(),
        source_ref: source_ref.into(),
        src_locations: locs.iter().map(|s| s.to_string()).collect(),
        narrative: narrative.into(),
        verdict: Verdict { provenance: ToolchainDerived, verification: v },
        drift_rationale: drift_rationale.map(|s| s.to_string()),
    };
    vec![
        // Order must match SPINE_DOMAIN_IDS exactly (spec Section 6: one location).
        d("core", "aie-rt core/, llvm-aie TableGen; AM025 Core_Control/Core_Status/Error_Halt_*",
          &["src/interpreter/", "src/device/core_debug/"],
          "VLIW core, control (enable/done/reset), and error-halt path. 100% ISA decode; SemanticOp coverage ~33%, rest in legacy handlers. Generic error_halt fires INSTR_ERROR (event 69) on every CoreStatus::Error; ECC fires ECC_ERROR_STALL. Saturation/watchdog error sources not yet detected.",
          Modeled { completeness: Full },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (NeedsTriage=Unspecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("program_memory", "AM025",
          &["src/parser/elf.rs"],
          "16KB program memory; ELF load -> run.", Modeled { completeness: Full }, None),
        d("program_counter", "AM025 PC_Event0..3 (0x32020/4/8/C); aie-rt xaiemlgbl_params.h",
          &["src/interpreter/", "src/device/core_debug/"],
          "PC sampling + PC_Event0..3 / Core_PC_Range matching; drives event-halt selector.",
          Modeled { completeness: Full }, None),
        d("data_memory", "aie-rt memory/, AM025",
          &["src/device/banking.rs", "src/device/state/memtile.rs", "src/interpreter/timing/memory.rs"],
          "64KB compute (8 banks x 128-bit) / 512KB memtile; conflict detection done; per-bank MEM_CONFLICT_DM_BANK_N fired. ECC: status bit readable, no scrubber/fault-injection -- accepted out of scope unless workloads require.",
          Modeled { completeness: Full }, None),
        d("dma", "aie-rt dma/, AM025 (112/433/144 reg)",
          &["src/device/dma/"],
          "BDs per tile-type 16/48/16 (memtile 48 not 64 -- aie-rt xaiemlgbl_reginit.c), channels 2/6/2, address dims 3/4/3; n-d addressing, padding, lock coupling, packet header, compression. Repeat / out-of-order BD execution: verify. Stream-side back-pressure is modeled functionally and as observable status: S2MM stalls in Transferring on empty input FIFO (transfer_s2mm returns Stalled, status emits Stalled_Stream_Starvation at bit 4 of DMA_S2MM_Status_0); MM2S stalls in Transferring when stream_out reaches the local-slave FIFO depth (output_fifo_capacity() = STREAM_LOCAL_SLAVE_FIFO_DEPTH per AM020 ch2, status emits Stalled_Stream_Backpressure at bit 4 of DMA_MM2S_Status_0). The S2MM and MM2S stall bits share bit position but use distinct field names in the regdb and are loaded separately so a future register split would surface as a build failure (audit landed 2026-05-24, see xdna-emu-priv: research/findings/2026-05-24-bridge-sweep-objectfifo-wedges.md).",
          Modeled { completeness: Full },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (SideEffect=DocSpecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("locks", "aie-rt locks/, AM025",
          &["src/device/tile/locks.rs"],
          "Counts per tile-type 16/64/16 (aie-rt xaiemlgbl_reginit.c; the 192 in AieMlMemTileDmaMod.NumLocks is the cross-tile reference range, not slot count). acquire/release/get/set, semaphore semantics, round-robin arbiter.",
          Modeled { completeness: Full }, None),
        d("stream_switch", "aie-rt stream_switch/, AM025 (160/119/149 reg)",
          &["src/device/stream_switch/"],
          "Circuit + packet, FIFOs, port events, packet-header matching. Parse-time routing reconstruction is a binary_load concern (spec Appendix N1). Per-port data-available / space-available signals propagate to DMA consumers/producers: S2MM stalls on empty per-channel input FIFO; MM2S stalls on local-slave-FIFO-full (paired with dma side).",
          Modeled { completeness: Full },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (SideEffect=DocSpecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("events_trace", "aie-rt events/ + trace/, AM025 (128/161/51 events)",
          &["src/device/events/", "src/device/trace_unit/"],
          "Events (broadcast 16ch, combo, group, port), cross-tile broadcast network, trace unit modes 0/1/2, pipelined start/stop + multi-tile timer sync. Combo/edge generator boundary cases need targeted tests; L2 broadcast propagation verify.",
          Modeled { completeness: Full }, None),
        d("performance_counters", "aie-rt perfcnt/, AM025 (compute/memtile/shim 4/11/6 reg)",
          &["src/device/perf_counters/"],
          "4 counters, threshold events. DMA/stream FIFO-size events not emitted (cycle-accuracy gap, tracked in xdna-emu-priv: research/coverage/cycle-accuracy-mission.md).",
          Modeled { completeness: Full }, None),
        d("timer", "aie-rt timer/, AM025 (5 reg)",
          &["src/device/timer.rs"],
          "Free-running 64-bit per-module; Reset_Event consumed via pending_reset latch; multi-tile timer-sync modeled. Trig_Event_Low/High_Value write effect: verification follow-up.",
          Modeled { completeness: Full }, None),
        d("watchpoint", "AM025 Compute WatchPoint0/1 (2), MemTile WatchPoint0..3 (4)",
          &["src/interpreter/execute/cycle_accurate.rs"],
          "Compute 2 / memtile 4 slots; WriteStrobes==0xF gate, direction + address comparator, AXI/DMA/quadrant origin filters, scalar+vector+DMA-engine paths, modifier-register effective address. Locked by 17+ unit tests.",
          Modeled { completeness: Full }, None),
        d("debug_halt", "AM025 Debug_*; aie-rt core/",
          &["src/device/core_debug/", "src/interpreter/engine/coordinator.rs"],
          "Halt + status bits, synchronous PC-event breakpoints (before-commit pre-execute seam, G1 silicon-derived 2026-05-18), async halt paths, and debug-register read/write routing all modeled. Count-step (Debug_Control0[5:2] Single_Step_Count) is a live N-committed-bundle budget that halts before the (N+1)th bundle commits (G2 silicon-derived 2026-05-19). Event-driven single-step boundary is the principled split: PC-wired (SSTEP_EVENT==Core_PC_0..3) halts before-commit via the same seam; watchpoint/mem/lock/range-wired stays after-commit (documented modeling decision -- no coherent before-commit point). Section 8 close-out (2026-05-19): Core_Status RESET-bit divergence and OUTBUF_ADDR/TRAP_PC probe fragility RESOLVED; G1 bounded-escalation tracker retired (contingency never fired). Open (tracked, spec section 8, HW-budget-gated): count-step finer silicon characterization (decrement cadence / larger-N / 0x11-on-silicon -- only N=4 observed) and resume hardware-verification.",
          Modeled { completeness: Full }, None),
        d("cascade", "aie-rt events",
          &["src/interpreter/execute/cascade.rs"],
          "Tile<->tile cascade read/write. Deadlock detection is a placeholder (deadlock.rs) -- promote to real detection or remove (verification follow-up).",
          Modeled { completeness: Full },
          Some("Own verdict reflects the actually-modeled subsystem; the rolled-up verdict is the pessimistic Phase-1 coarse category default (SideEffect=DocSpecified/Unverified), a bootstrap floor, not a ceiling. Per-op refinement that would raise the rollup is Phase-2 override work (spec Section 3).")),
        d("interrupt", "aie-rt interrupt/, AM025 (shim interrupt 23 reg: 18 L1 + 5 L2)",
          &["src/device/interrupts/l1.rs", "src/device/interrupts/l2.rs",
            "src/device/interrupts/mod.rs", "src/device/tile/registers.rs",
            "src/device/tile/mod.rs", "src/device/state/effects.rs",
            "src/device/state/dispatch.rs", "src/interpreter/core/interpreter.rs"],
          "Full AIE2 shim interrupt path MODELED. All 23 registers (18 L1 over 2 switches 0x35000-0x35050, 5 L2 0x15000-0x15010) read- and write-routed with exact semantics (write-1-to-clear status, enable/disable->mask, read-only mask). Stimulus path wired: event/error -> EventModule -> L1 (both switches, per-event-slot independent match + enable gating) -> broadcast network (block-mask honored) -> L2 sink -> NPI line, driven to a fixed point so an L1 output reaches L2 within one dispatch. Hardware errors enter via raise_instr_error -> EventModule. Privilege gating (only the L2 interrupt-routing register) is a driver-side concern, scoped out unrestricted per the noc/shim_mux precedent. Host-visible delivery is firmware-mediated (the AIE interrupt never reaches the x86 host directly) -- Tier B Spec 1 (firmware async-event mailbox plumbing + INSTR_ERROR producer) is shipped: cache + per-column rings + push callback + driver-mirror categorization tables + DRM_AMDXDNA_HW_LAST_ASYNC_ERR ioctl wired through the plugin. Per-class detection follow-ups (DMA / parity / ECC / lock / stream) remain tracked. Tier C (wedge-recovery / context-restart) is shipped: per-context state model (Connected/Stopped/Failed) with completed_counter; device-side TdrDetector classifying engine run state into Progressing/NaturalCompletion/MaskPollUnsatisfied/Wedged per cycle; xdna_emu_run consumes the classifier and returns the new XdnaEmuHaltReason::WedgeRecovered halt code on wedge, which the plugin translates to an EIO-shaped XRT command state. Plugin xdna_emu_reset_context resolution upgraded to fail-loud (required). Plumbed for multi-context throughout (single ContextId(0) today, all APIs take ContextId). Multi-context engine scheduling and lifecycle ioctls are a separate spec. See xdna-emu-priv: research/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md and xdna-emu-priv: research/specs/2026-05-19-interrupt-tier-b-firmware-mailbox-design.md and xdna-emu-priv: research/findings/2026-05-19-interrupt-tier-c-tdr.md and xdna-emu-priv: research/specs/2026-05-19-interrupt-tier-c-tdr-design.md.",
          Modeled { completeness: Full }, None),
        d("noc", "AM025 (NoC_Interface_AIE_to_NoC 4 reg, AIE_AXIMM_Config); aie-rt npi/; hardware spec",
          &[],
          "SCOPE: this domain is the AIE-array <-> SoC-data-fabric (Infinity Fabric) edge ONLY -- the off-chip handoff where shim DMA reaches DDR. It is NOT the on-chip interconnect: tile-to-tile data movement (stream_switch), the broadcast/event network (interrupt), cascade forwarding (cascade), and the memory-mapped control bus (control_packets) are all separately MODELED/Full. Three pieces: (1) Functional data path -- shim DMA <-> DDR is emulated (data arrives correctly via host_memory); no separate NoC model is needed for functional correctness. (2) Fabric timing (AXI latency / NoC arbitration / DDR egress rate) -- DELIBERATELY ABSTRACTED: the data fabric is a shared SoC IP, not the NPU; its latency is nondeterministic from the NPU's view (depends on CPU/GPU/DDR-controller traffic) and has no AM025-style register spec. First-order egress is folded into the calibrated shim-DMA constants (shim_ddr_cold_start_*, words_per_cycle); cycle-level re-calibration is gated on a trace-independent oracle (Perf_Counter readback, currently blocked) -- xdna-emu-priv: research/coverage/cycle-accuracy-mission.md item #5 (DEFERRED). (3) Config surface -- AIE_AXIMM_Config (incl. SLVERR_Block) and the NoC_Interface registers are AM025-derivable on demand but unused by every observed binary: control-packet SLVERR is modeled for the reset-default (SLVERR-enabled) config every binary uses; honoring SLVERR_Block would only matter for a binary that programs it (none do). NPI privileged register access is driver-side privilege -- emulator gives unrestricted access, out of scope.",
          Accepted { rationale: "AIE <-> SoC-data-fabric (Infinity Fabric) edge: a deliberate abstraction boundary, not unbuilt NPU work. Functional path is covered by shim DMA + host_memory; the on-chip interconnect (stream_switch / interrupt / cascade / control_packets) is separately Full. Fabric timing is a shared non-NPU SoC IP with no register-level spec, folded into the calibrated shim-DMA constants (cycle-calibration gated on the perf-counter oracle -- xdna-emu-priv: research/coverage/cycle-accuracy-mission.md #5). The AXIMM config surface (SLVERR_Block) is AM025-derivable on demand but unused by all observed binaries.".into() }, None),
        d("shim_mux", "aie-rt, AM025 (shim Mux/Demux 2 reg); aie-rt pl/ (PL Interface)",
          &["src/device/stream_switch/"],
          "Shim master/slave NoC-facing mux/demux. PL Interface (Upsizer/Downsizer) is Versal-FPGA stream-width adaptation -- NPU1 exposes no programmable PL: accepted out of scope.",
          Modeled { completeness: Full }, None),
        d("control_packets", "AM025 Control_Packet_Handler_Status (0x3FF30/0xB0F30); aie-rt xaiegbl_params.h:7761; XRT host protocol",
          &["src/device/control_packets/", "src/device/tile/mod.rs", "src/device/state/ctrl_access.rs", "src/interpreter/engine/coordinator.rs", "src/npu/"],
          "Control-packet headers, reassembly, register read/write effects, response packets MODELED. NPU host instruction stream (WRITE32/BLOCKWRITE/BLOCKSET/MASKWRITE/MASKPOLL/CONFIG_SHIMDMA_*/DDR_PATCH) MODELED. All four Tile_Control_Packet_Handler_Status sticky bits have faithful detecting paths: First/Second-header parity and Tlast via the reassembler, SLVERR_On_Access via undecoded-address decode at the dispatch boundary, all with poll-only sticky-continue semantics (aie-rt/AM025: latch + continue, no interrupt, no abort). The AIE_AXIMM_Config.SLVERR_Block config-suppression refinement is a tracked NoC-gated goal -- see the noc domain. Keystone subsystem.",
          Modeled { completeness: Full }, None),
        d("clock_control", "AM025 Module_Clock_Control / Column_Clock_Control / Reset_Control_1 / AIE_Tile_Column_Reset; aie-rt pm/xaie_clock.c",
          &["src/device/clock_control/", "src/device/state/dispatch.rs", "src/device/array/mod.rs", "src/device/array/routing.rs", "src/device/array/dma_ops.rs", "src/interpreter/engine/coordinator.rs"],
          "Three-tier clock gating MODELED: Column_Clock_Control, per-module Module_Clock_Control (compute/memtile/shim_0/shim_1 layouts decoded from regdb), and adaptive clock gating (idle counters + abort_period + wake-on-event). Step loop consults all three tiers before DMA / StreamSwitch / Core stepping. Adaptive counters tick once per data-movement cycle, gated by per-module clocks (silicon-accurate); column or module re-ungate transitions reset the relevant counter; Phase-5 tick consults DmaEngine::any_channel_has_pending_work so a queued-but-FSM-Idle channel is treated as not-yet-idle. Wake-on-event paths (xdna-emu-priv: research/coverage/cycle-accuracy-mission.md item #8): (Wake 1) register-bus access to the Dma/Lock/DataMemory subsystem wakes the DMA counter, StreamSwitch subsystem wakes the SS counter, via DeviceState::wake_adaptive_for_subsystem invoked from write_register/mask_write_register/dma_write; (Wake 2) stream beat arriving at a slave port sets cycle_active, which the existing Phase-5 SS branch converts to tick_adaptive_ss(active=true); (Wake 3) lock-value changes reduce to Wake 1 cross-tile (control-packet decode into Lock register write) plus the Phase-5 has_pending_work check for same-tile (AcquiringLock implies is_active(), so the gate cannot engage). Silicon-accurate boot default: all tiles gated, ungate-via-CDO is the documented path; ungate_all() test helper exercises the real register-write path. Gated-tile non-clock-control writes serve-and-warn (dedup per (col,row,offset); XDNA_EMU_WARN_GATED_ACCESS=0 silences). AIE_Tile_Column_Reset (shim 0xFFF28) partition teardown MODELED: asserting bit 0 tears down every non-shim tile in the column (cores, DMAs, locks, stream switches, adaptive counters) via Array::reset_column, preserving tile memory and exempting the shim row, on the assert edge (aie-rt pm/xaie_reset.c); Reset_Control_1/0xFFF14 (shim-NoC reset) is intentionally a no-op pending the NoC stub. Clock-gating counter-freeze MODELED (\"no clock, no tick\"): a clock-gated module advances neither its free-running timer (core/mem module timers) nor its performance counters. Phase 3e consults the Core module clock (compute core bank) or Memory module clock (compute/memtile mem bank), falling back to the column clock for the shim-PL bank (no separate shim-PL module gate in the model), before ticking -- so gating a module or column freezes Performance_Counter0..3 and the module timers, and they resume from the held value on re-ungate (clock-gating preserves state; the separate AIE_Tile_Column_Reset path is what clears them). FULL: the AIE2 Core module has no adaptive clock gating to model -- Module_Clock_Control (core, 0x60000) exposes only DMA_Adaptive_Clock_Gate (bit 5), Stream_Switch_Adaptive_Clock_Gate (bit 4), and the plain Core_Module_Clock_Enable (bit 2); the AM025 register DB and aie-rt define no Core idle counter, no Core abort_period register, and no cascade wake event. A core waiting on cascade input stalls at the FSM level (Cascade_Stall_MCD/SCD in Core_Status, already modeled via the cascade queues + core stall) -- it does not clock-gate-and-wake -- so the former \"Wake 4\" cascade-arrival item is a non-feature (hardware has no such mechanism), not deferred work. Privilege enforcement is out of scope (no privilege model in the emulator yet) -- a candidate for later bug-probing, not a modeling gap. See xdna-emu-priv: research/specs/2026-05-24-clock-control-design.md.",
          Modeled { completeness: Full }, None),
        d("tile_isolation", "aie-rt pm/xaie_tilectrl.c, AM025 (Tile_Control compute 0x36030 / memtile 0x96030)",
          &["src/device/tile/mod.rs", "src/device/state/effects.rs", "src/device/array/routing.rs", "src/interpreter/execute/memory/neighbor.rs", "src/interpreter/engine/coordinator.rs"],
          "Tile_Control low 4 bits (S/W/N/E) snapshotted on register write. Inter-tile stream transfers, cross-tile NeighborMemory snapshots/reads/buffered writes, and NeighborLocks slices all consult the destination/own isolation byte. Shim isolation snapshotted; only memtile->shim south-bound routing gate consults it today. Clock-gating bits of Tile_Control pass through unmodeled (see clock_control).",
          Modeled { completeness: Full }, None),
        d("binary_load", "XRT container / CDO / ELF formats; mlir-aie device model (tools/aie-device-models.json)",
          &["src/parser/xclbin.rs", "src/parser/cdo/", "src/parser/elf.rs", "src/parser/stream_switch_topology.rs", "src/device/array/"],
          "XCLBIN container, CDO framing/syntax/semantics -> DeviceOps, per-core ELF load, all MODELED. Stream-switch routing reconstruction from CDO writes (parse-side, distinct from the runtime stream_switch subsystem -- spec Appendix N1). NPU1 physical topology is a 5x6 envelope with 29 present tiles: the mlir-aie logical tile map is placed at the xdna-driver-derived column origin, preserving the DPU prefix and `(0,0)` hole. Folded here as the array-constructed-from-binary concern (spec Appendix N1 rationale), not a reachability-forced tag.",
          Modeled { completeness: Full }, None),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Architecture;

    #[test]
    fn override_registry_has_intrinsic_accept() {
        // #104: one Phase-2 override is the Accept of the SemanticOp::Intrinsic(_)
        // catch-all (never constructed, fail-loud; the concrete ops it resolves
        // to are differentially verified -- #103). Located by id, not index, so
        // it survives the registry growing (the DMA-ops Verified override, #113,
        // is the sibling entry).
        use crate::coverage::verdict::Verification;
        let regs = override_registry(Architecture::Aie2);
        let intr = regs
            .iter()
            .find(|u| u.id == "aie2.intrinsic_catchall.accepted")
            .expect("intrinsic-catchall Accept present");
        assert!(matches!(intr.verdict.verification, Verification::Accepted { .. }));
        assert!(intr.shadows_derived.is_some(), "Accept pulls Intrinsic off its derived NeedsTriage default");
    }

    #[test]
    fn override_registry_has_dma_ops_verified() {
        // #113 axis-2: the DMA/data-movement tenant of the diff-fuzzing
        // framework silicon-verified the DMA access-pattern engine against
        // real NPU1 (81/81 keys, 810 HW cases at depth ~10, 0 divergent),
        // which exercises exactly the DmaStart + DmaWait SemanticOps. The
        // other five SideEffect ops (CascadeRead/Write, StreamRead/Write/
        // WritePacketHeader) are the core-side stream/cascade tenants 4/5 --
        // deliberately NOT claimed here. Under-claim is safe; over-claim is a
        // correctness bug.
        use crate::aie2::isa::SemanticOp;
        use crate::coverage::verdict::Verification;
        let regs = override_registry(Architecture::Aie2);
        let dma = regs
            .iter()
            .find(|u| u.id == "aie2.dma_ops.verified")
            .expect("DMA-ops Verified override present");
        assert!(matches!(dma.verdict.verification, Verification::Verified { .. }));
        let Claims::Nodes(nodes) = &dma.claims;
        let ops: std::collections::HashSet<_> = nodes
            .iter()
            .map(|n| match n {
                CoverageNode::Semantic { op, .. } => *op,
                other => panic!("DMA override claims a non-semantic node: {other:?}"),
            })
            .collect();
        assert_eq!(
            ops,
            [SemanticOp::DmaStart, SemanticOp::DmaWait].into_iter().collect(),
            "claims DmaStart + DmaWait ONLY -- not the whole SideEffect category (cascade/stream are tenants 4/5)"
        );
        let all = |op| variants_of(Architecture::Aie2, op).len();
        assert_eq!(
            nodes.len(),
            all(SemanticOp::DmaStart) + all(SemanticOp::DmaWait),
            "every variant of both"
        );
        assert!(
            dma.shadows_derived.is_some(),
            "Verified pulls DMA ops off their DocSpecified/Unverified default"
        );
        assert!(dma.shared_from.is_none(), "earned on AIE2 silicon directly, not shared");
    }

    #[test]
    fn override_registry_has_vector_ops_verified() {
        // #126/#127 axis-1, re-derived per variant: the vector differential
        // fuzzer's silicon-matched corpus, replayed on the emulator with the
        // variant-keyed recorder. Only Vector-category variants a matched run
        // executed are claimed; the never-executed ops stay perishable.
        use crate::aie2::isa::SemanticOp::*;
        use crate::coverage::derive::{category, Category};
        use crate::coverage::verdict::Verification;
        let regs = override_registry(Architecture::Aie2);
        let vec_unit = regs
            .iter()
            .find(|u| u.id == "aie2.vector_ops.verified")
            .expect("vector-ops Verified override present");
        assert!(matches!(vec_unit.verdict.verification, Verification::Verified { .. }));
        let Claims::Nodes(nodes) = &vec_unit.claims;
        for n in nodes {
            let CoverageNode::Semantic { op, .. } = n else {
                panic!("vector override claims a non-semantic node: {n:?}")
            };
            assert_eq!(category(op), Category::Vector, "{n:?} is not a Vector op");
        }
        let claimed_op = |want| {
            nodes
                .iter()
                .any(|n| matches!(n, CoverageNode::Semantic { op, .. } if *op == want))
        };
        for not_claimed in [Min, Max, MatMulSub, VectorPush] {
            assert!(!claimed_op(not_claimed), "{not_claimed:?} must NOT be claimed (perishable)");
        }
        assert!(vec_unit.shadows_derived.is_some());
        assert!(vec_unit.shared_from.is_none());
    }

    fn is_verified(v: &Verdict) -> bool {
        matches!(v.verification, Verification::Verified { .. })
    }

    #[test]
    fn vector_verified_variants_are_sorted() {
        // Per op, variants in `OpVariant` order (the order `variants_of` sorts
        // by), and no pair listed twice.
        let mut seen = std::collections::HashSet::new();
        for w in VECTOR_VERIFIED_VARIANTS.windows(2) {
            if w[0].0 == w[1].0 {
                assert!(w[0].1 < w[1].1, "{:?}: {} before {}", w[0].0, w[0].1, w[1].1);
            }
        }
        for pair in VECTOR_VERIFIED_VARIANTS {
            assert!(seen.insert(*pair), "{pair:?} listed twice");
        }
    }

    #[test]
    fn replay_evidence_is_per_variant() {
        // Concrete expectations from the 2026-10-29 replay (all 28 live seeds
        // match silicon). Verified: a variant a matching seed executed.
        // Perishable: a variant of the same op that no matching seed executed,
        // so verifying Mac in one mode never vouches for another.
        use crate::aie2::isa::SemanticOp::*;
        let m = crate::coverage::CoverageModel::build(Architecture::Aie2);
        let verified = |op, v: &OpVariant| is_verified(&m.variant_verdict(&op, v));
        let perishable = |op, v: &OpVariant| m.variant_verdict(&op, v).is_perishable();

        assert!(verified(Mac, &mac(0, 1, 0, true, true)));
        assert!(verified(MatMul, &mac(1, 3, 1, true, true)));
        assert!(verified(MaxLt, &typed(ElementType::Int32)));
        assert!(verified(Srs, &converted(ElementType::Int16, ElementType::Int64)));

        // Same op, other config mode / signedness / element type: not executed.
        assert!(perishable(Mac, &mac(0, 1, 0, false, false)));
        assert!(perishable(Mac, &mac(0, 0, 0, true, true)));
        assert!(perishable(MaxLt, &typed(ElementType::UInt8)));
        assert!(perishable(VectorSelect, &typed(ElementType::UInt16)));
        assert!(verified(Pack, &converted(ElementType::Int8, ElementType::Int16)));
        // Never executed in a matching seed.
        assert!(perishable(Min, &typed(ElementType::Int32)));
        // bf16 fuzzer keys do not discriminate and are not claimed.
        assert!(perishable(MaxLt, &typed(ElementType::BFloat16)));
        // An op with one verified variant is not verified as a whole.
        assert!(!is_verified(&m.semantic_verdict(&Mac)));
    }

    #[test]
    fn override_registry_has_lane_goldens_verified() {
        // The bf16 MAC forms (D6) and SRS / f32->bf16 rounding are pinned lane
        // by lane against NPU1 captures. Claims exclude whatever the fuzzer
        // unit owns (a node has one owner) and reach exactly the bf16 modes.
        use crate::aie2::isa::SemanticOp::*;
        use crate::coverage::verdict::Verification;
        let regs = override_registry(Architecture::Aie2);
        let lane = regs
            .iter()
            .find(|u| u.id == "aie2.lane_goldens.verified")
            .expect("lane-golden Verified override present");
        let Verification::Verified { evidence } = &lane.verdict.verification else {
            panic!("lane goldens must be Verified")
        };
        for name in [
            "vec_mac_bf16_forms",
            "vec_conv_bf16",
            "matches_every_silicon_srs_capture",
            "vec_acc_addsub_f32",
            "vec_vshift_wide",
        ] {
            assert!(evidence.contains(name), "evidence must name {name}");
        }
        let Claims::Nodes(nodes) = &lane.claims;
        let has = |op, v: OpVariant| {
            nodes.contains(&CoverageNode::Semantic { arch: Architecture::Aie2, op, variant: v })
        };
        assert!(has(Mac, mac(2, 3, 0, false, false)));
        assert!(has(MatMul, mac(2, 3, 1, false, false)));
        assert!(has(Convert, converted(ElementType::BFloat16, ElementType::Float32)));
        assert!(has(Accumulate, acc(2, 3, 0)));
        assert!(has(AccumNegAdd, typed(ElementType::Float32)));
        assert!(has(Align, OpVariant::Untyped));
        // Elementwise addmac.f has no capture; integer modes are not bf16.
        assert!(!has(AddMac, mac(2, 3, 1, false, false)));
        assert!(!has(Mac, mac(0, 1, 0, true, true)));
        // The SRS variant the fuzzer unit already claims stays with it.
        assert!(!has(Srs, converted(ElementType::Int16, ElementType::Int64)));
        // bf16 MAC variants are Verified through the lane goldens, not the fuzzer.
        let m = crate::coverage::CoverageModel::build(Architecture::Aie2);
        assert!(is_verified(&m.variant_verdict(&Mac, &mac(2, 3, 0, false, false))));
        // Not captured: vnegmac.f / vsubmac.f forms.
        assert!(m.variant_verdict(&NegMatMul, &mac(2, 3, 0, false, false)).is_perishable());
        assert!(m.variant_verdict(&SubMac, &mac(2, 3, 0, false, false)).is_perishable());
    }

    #[test]
    fn gated_variants_are_comprehension_gaps() {
        use crate::aie2::isa::SemanticOp::*;
        let m = crate::coverage::CoverageModel::build(Architecture::Aie2);
        let gap = |op, v: OpVariant| {
            let r = m.semantic_ledger().into_iter().find(|r| r.op == op && r.variant == v).unwrap();
            assert!(r.verdict.is_comprehension_gap(), "{op:?} {v}");
            assert_eq!(r.claimed_by.as_deref(), Some("aie2.gated.not_modeled"), "{op:?} {v}");
        };
        // int8 x int4 sparse 4x32x8, and bf16 sparse 4x16x4.
        gap(Mac, OpVariant::MacMode { amode: 0, bmode: 0, variant: 1, sgn_x: true, sgn_y: true });
        gap(MatMul, OpVariant::MacMode { amode: 2, bmode: 3, variant: 2, sgn_x: false, sgn_y: false });
        // Derived from silicon since: no longer gated.
        let owner = |op, v: OpVariant| {
            m.semantic_ledger()
                .into_iter()
                .find(|r| r.op == op && r.variant == v)
                .unwrap()
                .claimed_by
        };
        assert_eq!(owner(Align, OpVariant::Untyped).as_deref(), Some("aie2.lane_goldens.verified"));
        assert_eq!(
            owner(Accumulate, OpVariant::AccMode { amode: 2, bmode: 3, variant: 0 }).as_deref(),
            Some("aie2.lane_goldens.verified")
        );
        // The integer accumulator adder is modelled: not claimed.
        let int_acc = m
            .semantic_ledger()
            .into_iter()
            .find(|r| {
                r.op == Accumulate && r.variant == OpVariant::AccMode { amode: 1, bmode: 0, variant: 0 }
            })
            .unwrap();
        assert!(int_acc.claimed_by.is_none());
    }

    #[test]
    fn override_registry_partitions_and_claims_only_real_variants() {
        // The real registry never passes through enforce_coverage, so check its
        // partition here, and that every semantic claim names a variant the
        // derived universe actually has (a typo'd variant would otherwise
        // silently verify nothing).
        let regs = override_registry(Architecture::Aie2);
        let mut seen = std::collections::HashMap::new();
        for u in &regs {
            let Claims::Nodes(ns) = &u.claims;
            for n in ns {
                if let Some(first) = seen.insert(n.clone(), u.id.clone()) {
                    panic!("{n:?} claimed by both {first} and {}", u.id);
                }
                if let CoverageNode::Semantic { arch, op, variant } = n {
                    assert!(
                        variants_of(*arch, *op).contains(variant),
                        "{}: {op:?} has no variant {variant}",
                        u.id
                    );
                }
            }
        }
    }

    #[test]
    fn capability_spine_seeded_for_aie2() {
        let spine = capability_spine();
        assert!(spine.iter().any(|d| d.id == "dma"));
        assert!(spine.iter().any(|d| d.id == "locks"));
        assert!(spine.iter().any(|d| d.id == "stream_switch"));
        // Every seeded domain applies to AIE2 (the only wired arch today).
        assert!(spine.iter().all(|d| d.applies_to(Architecture::Aie2)));
        // Spine extended to 20 (spec Section 2 hybrid decision).
        assert_eq!(spine.len(), 20);
        for id in ["control_packets", "clock_control", "tile_isolation", "binary_load"] {
            assert!(spine.iter().any(|d| d.id == id), "missing new domain {id}");
        }
        // Every domain carries non-empty source_ref.
        assert!(spine.iter().all(|d| !d.source_ref.is_empty()));
        // src_locations may be empty only for OOS/MISSING states (Stub/Partial/Accepted).
        // Domains with Modeled{Full} + Verified must name where they live in src/.
        // This mirrors what enforce_coverage block-4 gates on (spec Section 2 N2).
        use crate::coverage::verdict::{Completeness, Verification};
        for d in &spine {
            let oos_or_missing = match &d.verdict.verification {
                Verification::Accepted { .. } => true,
                Verification::Modeled { completeness } => {
                    matches!(completeness, Completeness::Stub | Completeness::Partial { .. })
                }
                Verification::NotApplicable
                | Verification::Verified { .. }
                | Verification::Unverified
                | Verification::Contradicted { .. } => false,
            };
            if !oos_or_missing {
                assert!(
                    !d.src_locations.is_empty(),
                    "domain '{}' is not OOS/MISSING but has empty src_locations",
                    d.id
                );
            }
        }
    }

    #[test]
    fn capability_domain_arch_applicability_is_explicit() {
        let dma = capability_spine().into_iter().find(|d| d.id == "dma").unwrap();
        assert!(dma.applies_to(Architecture::Aie2));
    }

    #[test]
    fn capability_spine_matches_the_leaf_id_list() {
        // The leaf list (build.rs-visible) and the rich spine MUST agree --
        // they are the same source of truth, just two views (spec Section 6
        // one location; Plan 2 cycle note).
        use crate::coverage::spine_ids::SPINE_DOMAIN_IDS;
        let rich: Vec<String> = capability_spine().into_iter().map(|d| d.id).collect();
        let leaf: Vec<String> = SPINE_DOMAIN_IDS.iter().map(|s| s.to_string()).collect();
        assert_eq!(rich, leaf);
    }
}
