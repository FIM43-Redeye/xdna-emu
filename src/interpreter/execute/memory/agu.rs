//! Pointer update of the multi-dimensional addressing modes (`[p], d` with a 2D or 3D
//! descriptor), shared by every load and store that has one.
//!
//! A descriptor `d<k>` bundles the registers `m<k>` (modifier), `dn<k>` (size), `dj<k>`
//! (stride) and `dc<k>` (count); which registers a descriptor names comes from TableGen
//! (`xdna_archspec::aie2::descriptors`). The access is made at the pointer, then the pointer
//! and the counts are updated. Observed on NPU1 silicon, first through the fused
//! `vldb.*unpack*` loads (`vec_unpack_addr`, 1888 accesses) and then through every other
//! family (`vec_agu_*`: scalar `lda` / `st` of every width, `vlda` / `vldb` / `vst` in 256-
//! and 128-bit and accumulator-half forms, the q-register forms, the fused `vlda.ups` and
//! `vlda.conv` loads, the fused `vst.srs`, `vst.pack` and `vst.conv` stores; 9600 records,
//! 2D and 3D, descriptors `d0`..`d3`), with counts that start below, at and above their
//! sizes and with negative strides. Every pointer and count matched the one rule:
//!
//! 2D, one dimension `(size, stride, count)` and the modifier `m`:
//! ```text
//! if count < size { ptr += stride; count += 1 } else { ptr += m; count = 0 }
//! ```
//!
//! 3D, dimension 1 from `d<k>` and dimension 2 from `d<k+4>`; the modifier is `m<k>`:
//! ```text
//! if count1 < size1      { ptr += stride1; count1 += 1 }
//! else if count2 < size2 { ptr += stride2; count1 = 0; count2 += 1 }
//! else                   { ptr += m;       count1 = 0; count2 = 0 }
//! ```
//!
//! A count at or above its size takes the wrap branch, so a descriptor whose count starts
//! past its size behaves as if the count had just reached it. Comparisons are unsigned;
//! negative strides are two's complement and the pointer update wraps.

use crate::interpreter::state::{
    is_descriptor_operand, ExecutionContext, MOD_BASE_D2D, MOD_BASE_D3D, MOD_BASE_DC, MOD_BASE_DJ,
    MOD_BASE_DN, MOD_BASE_M,
};
use xdna_archspec::aie2::descriptors::{descriptor_2d, descriptor_3d, Descriptor2D, ModifierKind, SubRegister};

/// One dimension of a descriptor: the `dn`, `dj` and `dc` registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimension {
    pub size: u32,
    pub stride: u32,
    pub count: u32,
}

/// The result of one access: what to add to the pointer and the new counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PointerStep {
    pub delta: u32,
    pub count1: u32,
    /// Unchanged by a 2D step.
    pub count2: u32,
}

/// 2D update with modifier `m` and dimension `d`.
pub fn step_2d(m: u32, d: Dimension) -> PointerStep {
    if d.count < d.size {
        PointerStep { delta: d.stride, count1: d.count.wrapping_add(1), count2: 0 }
    } else {
        PointerStep { delta: m, count1: 0, count2: 0 }
    }
}

/// 3D update with modifier `m`, inner dimension `d1` and outer dimension `d2`.
pub fn step_3d(m: u32, d1: Dimension, d2: Dimension) -> PointerStep {
    if d1.count < d1.size {
        PointerStep { delta: d1.stride, count1: d1.count.wrapping_add(1), count2: d2.count }
    } else if d2.count < d2.size {
        PointerStep { delta: d2.stride, count1: 0, count2: d2.count.wrapping_add(1) }
    } else {
        PointerStep { delta: m, count1: 0, count2: 0 }
    }
}

/// The modifier-file index of a descriptor component (`m`, `dn`, `dj` or `dc` of some index).
fn component(sub: SubRegister) -> u8 {
    let base = match sub.kind {
        ModifierKind::Modifier => MOD_BASE_M,
        ModifierKind::Size => MOD_BASE_DN,
        ModifierKind::Stride => MOD_BASE_DJ,
        ModifierKind::Count => MOD_BASE_DC,
    };
    base + sub.index
}

/// One dimension of a 2D descriptor, read from the context.
fn read_dimension(ctx: &ExecutionContext, d: &Descriptor2D) -> Dimension {
    Dimension {
        size: ctx.modifier_read(component(d.size)),
        stride: ctx.modifier_read(component(d.stride)),
        count: ctx.modifier_read(component(d.count)),
    }
}

/// Run one descriptor update for the descriptor operand `desc` (a `ModifierReg` index:
/// `d<k>` from `MOD_BASE_D2D`, `d<k>_3d` from `MOD_BASE_D3D`): read the registers the
/// descriptor names (`xdna_archspec::aie2::descriptors`, from TableGen), write the new counts
/// back, and return the step so the caller can apply the pointer delta with the timing of its
/// instruction (post-modify of a load or store, or pointer arithmetic).
///
/// # Panics
///
/// If `desc` is not a descriptor operand: that is a decode bug, and reading a modifier
/// register instead would be silently wrong.
pub fn descriptor_step(ctx: &mut ExecutionContext, desc: u8) -> PointerStep {
    assert!(is_descriptor(desc), "operand {desc} is not a 2D/3D descriptor");
    if desc >= MOD_BASE_D2D {
        let d = descriptor_2d(desc - MOD_BASE_D2D)
            .unwrap_or_else(|| panic!("no 2D descriptor d{}", desc - MOD_BASE_D2D));
        let step = step_2d(ctx.modifier_read(component(d.modifier)), read_dimension(ctx, d));
        ctx.modifier.write(component(d.count), step.count1);
        step
    } else {
        let d = descriptor_3d(desc - MOD_BASE_D3D)
            .unwrap_or_else(|| panic!("no 3D descriptor d{}_3d", desc - MOD_BASE_D3D));
        let (inner, outer) = (
            descriptor_2d(d.inner).expect("3D descriptor names a 2D descriptor"),
            descriptor_2d(d.outer).expect("3D descriptor names a 2D descriptor"),
        );
        // The modifier is the inner descriptor's.
        let step = step_3d(
            ctx.modifier_read(component(inner.modifier)),
            read_dimension(ctx, inner),
            read_dimension(ctx, outer),
        );
        ctx.modifier.write(component(inner.count), step.count1);
        ctx.modifier.write(component(outer.count), step.count2);
        step
    }
}

/// Whether a `ModifierReg` index is a 2D / 3D descriptor operand rather than an m / dn /
/// dj / dc register.
pub fn is_descriptor(reg: u8) -> bool {
    is_descriptor_operand(reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim(size: u32, stride: u32, count: u32) -> Dimension {
        Dimension { size, stride, count }
    }

    #[test]
    fn two_d_walks_a_row_then_jumps_by_the_modifier() {
        // size 2, stride 32, modifier 64 (silicon set (64, 2, 32, 0)): 0, 32, 64, then +64.
        let mut d = dim(2, 32, 0);
        let mut ptr = 0u32;
        let mut seen = vec![];
        for _ in 0..6 {
            seen.push(ptr);
            let step = step_2d(64, d);
            ptr = ptr.wrapping_add(step.delta);
            d.count = step.count1;
        }
        assert_eq!(seen, [0, 32, 64, 128, 160, 192]);
    }

    #[test]
    fn two_d_count_past_its_size_takes_the_wrap_branch() {
        let step = step_2d(32, dim(2, 32, 5));
        assert_eq!(step, PointerStep { delta: 32, count1: 0, count2: 0 });
    }

    #[test]
    fn three_d_carries_into_the_outer_dimension_then_wraps() {
        // Silicon set (32, 3, 32, 2, 160, 0, 0): inner size 3 stride 32, outer size 2 stride 160.
        let (mut d1, mut d2) = (dim(3, 32, 0), dim(2, 160, 0));
        let mut ptr = 0u32;
        let mut seen = vec![];
        for _ in 0..13 {
            seen.push(ptr);
            let step = step_3d(32, d1, d2);
            ptr = ptr.wrapping_add(step.delta);
            d1.count = step.count1;
            d2.count = step.count2;
        }
        assert_eq!(seen, [0, 32, 64, 96, 256, 288, 320, 352, 512, 544, 576, 608, 640]);
    }

    #[test]
    fn negative_strides_wrap_the_pointer() {
        let step = step_2d(0, dim(4, (-32i32) as u32, 0));
        assert_eq!(0x100u32.wrapping_add(step.delta), 0xE0);
    }

    fn ctx_with(regs: &[(u8, u32)]) -> ExecutionContext {
        let mut ctx = ExecutionContext::new();
        for (reg, value) in regs {
            ctx.modifier.write(*reg, *value);
        }
        ctx
    }

    #[test]
    fn a_2d_descriptor_reads_the_registers_of_its_own_index() {
        // d5 = m5, dn5, dj5, dc5; every other index holds a decoy that would change the step.
        let mut regs = vec![];
        for k in 0..8u8 {
            let own = k == 5;
            regs.push((MOD_BASE_M + k, if own { 640 } else { 1 }));
            regs.push((MOD_BASE_DN + k, if own { 2 } else { 100 }));
            regs.push((MOD_BASE_DJ + k, if own { 32 } else { 2 }));
            regs.push((MOD_BASE_DC + k, if own { 2 } else { 0 }));
        }
        let mut ctx = ctx_with(&regs);
        // count 2 has reached size 2: wrap, take the modifier, reset the count.
        let step = descriptor_step(&mut ctx, MOD_BASE_D2D + 5);
        assert_eq!(step, PointerStep { delta: 640, count1: 0, count2: 0 });
        assert_eq!(ctx.modifier_read(MOD_BASE_DC + 5), 0);
        for k in [0u8, 1, 4, 6, 7] {
            assert_eq!(ctx.modifier_read(MOD_BASE_DC + k), 0, "dc{k} untouched");
        }
    }

    #[test]
    fn a_3d_descriptor_pairs_index_k_with_k_plus_4_and_takes_the_modifier_of_k() {
        // d2_3d = d2 (inner) + d6 (outer); the modifier is m2.
        let mut ctx = ctx_with(&[
            (MOD_BASE_M + 2, 7000),
            (MOD_BASE_M + 6, 9999),
            (MOD_BASE_DN + 2, 1),
            (MOD_BASE_DJ + 2, 10),
            (MOD_BASE_DC + 2, 1),
            (MOD_BASE_DN + 6, 1),
            (MOD_BASE_DJ + 6, 100),
            (MOD_BASE_DC + 6, 0),
        ]);
        // inner full, outer not: carry into the outer dimension.
        assert_eq!(
            descriptor_step(&mut ctx, MOD_BASE_D3D + 2),
            PointerStep { delta: 100, count1: 0, count2: 1 }
        );
        assert_eq!((ctx.modifier_read(MOD_BASE_DC + 2), ctx.modifier_read(MOD_BASE_DC + 6)), (0, 1));
        // inner restarts: one more inner step.
        assert_eq!(
            descriptor_step(&mut ctx, MOD_BASE_D3D + 2),
            PointerStep { delta: 10, count1: 1, count2: 1 }
        );
        // both full: the modifier of the inner descriptor, both counts reset.
        assert_eq!(
            descriptor_step(&mut ctx, MOD_BASE_D3D + 2),
            PointerStep { delta: 7000, count1: 0, count2: 0 }
        );
    }

    #[test]
    #[should_panic(expected = "not a 2D/3D descriptor")]
    fn a_modifier_register_is_not_a_descriptor() {
        descriptor_step(&mut ExecutionContext::new(), MOD_BASE_DC);
    }
}
