//! Pointer update of the multi-dimensional addressing modes (`[p], d` with a 2D or 3D
//! descriptor).
//!
//! A descriptor `d<k>` bundles the registers `m<k>` (modifier), `dn<k>` (size), `dj<k>`
//! (stride) and `dc<k>` (count). The access is made at the pointer, then the pointer and
//! the counts are updated. Observed on NPU1 silicon through the fused `vldb.*unpack*`
//! loads, with counts that start below, at and above their sizes (`vec_unpack_addr`, 8
//! 2D and 6 3D descriptor sets plus 4 sets that start past their sizes, 1888 accesses,
//! every pointer and count matched):
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
    ExecutionContext, MOD_BASE_D2D, MOD_BASE_D3D, MOD_BASE_DC, MOD_BASE_DJ, MOD_BASE_DN, MOD_BASE_M,
};

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

/// Run one descriptor update for the descriptor operand `desc` (a `ModifierReg` index:
/// `d<k>` from `MOD_BASE_D2D`, `d<k>_3d` from `MOD_BASE_D3D`): read `m`, `dn`, `dj`, `dc`,
/// write the new counts back, and return the step so the caller can apply the pointer
/// delta with the timing of its instruction (post-modify of a load, or pointer arithmetic).
pub fn descriptor_step(ctx: &mut ExecutionContext, desc: u8) -> PointerStep {
    let read_dim = |ctx: &ExecutionContext, k: u8| Dimension {
        size: ctx.modifier_read(MOD_BASE_DN + k),
        stride: ctx.modifier_read(MOD_BASE_DJ + k),
        count: ctx.modifier_read(MOD_BASE_DC + k),
    };
    if desc >= MOD_BASE_D2D {
        let k = desc - MOD_BASE_D2D;
        let step = step_2d(ctx.modifier_read(MOD_BASE_M + k), read_dim(ctx, k));
        ctx.modifier.write(MOD_BASE_DC + k, step.count1);
        step
    } else {
        let k = desc - MOD_BASE_D3D;
        let step = step_3d(ctx.modifier_read(MOD_BASE_M + k), read_dim(ctx, k), read_dim(ctx, k + 4));
        ctx.modifier.write(MOD_BASE_DC + k, step.count1);
        ctx.modifier.write(MOD_BASE_DC + k + 4, step.count2);
        step
    }
}

/// Whether a `ModifierReg` index is a 2D / 3D descriptor operand rather than an m / dn /
/// dj / dc register.
pub fn is_descriptor(reg: u8) -> bool {
    reg >= MOD_BASE_D3D
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
}
