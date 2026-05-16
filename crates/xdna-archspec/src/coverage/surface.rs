//! Axis 1: surface presence. The TYPE and CONTRACT live here (archspec is the
//! single point of definition for both axes, spec Section 2). The concrete
//! per-arch `impl SurfaceProbe` lives in the interpreter crate (Plan 2),
//! because "does a handler exist" is implementation state, not architecture.

use crate::coverage::NodeId;

/// Whether a generated node is wired into execution. Spec Section 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceClass {
    /// A real execution handler / register consumer exists.
    Wired,
    /// Reaches a `_ =>` default; no dedicated handling.
    Fallthrough,
    /// Decoded but nothing consumes it.
    Absent,
}

/// Axis-1 contract. archspec declares it; the interpreter implements it
/// per-arch (spec Section 2, Section 7). A node simply does not exist in an
/// arch's node set if the toolchain does not emit it for that arch, so no
/// `Inapplicable` value is needed (spec Section 7).
pub trait SurfaceProbe {
    fn surface_class(&self, node: &NodeId) -> SurfaceClass;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage::NodeId;
    use crate::types::Architecture;

    /// A fake probe proves the trait is object-safe and usable before the
    /// real interpreter impl exists (Plan 2).
    struct FakeProbe;
    impl SurfaceProbe for FakeProbe {
        fn surface_class(&self, _node: &NodeId) -> SurfaceClass {
            SurfaceClass::Wired
        }
    }

    #[test]
    fn trait_is_object_safe_and_usable() {
        let p: &dyn SurfaceProbe = &FakeProbe;
        let node = NodeId::Capability { arch: Architecture::Aie2, domain: "dma".into() };
        assert_eq!(p.surface_class(&node), SurfaceClass::Wired);
    }
}
