//! DMA coverage universe (3a) and key parsing. Keys are
//! `{feature}/{engine}/{dir}/{dtype}`, capability-gated per the spike matrix.
use super::chain::{Direction, Dtype, Engine, Feature};

/// Is this (engine, feature, dir) combination part of the coverage universe?
/// Gating: strided4d + pad* are memtile-only; packet + pad* are MM2S-only
/// (all spike-verified, 3a). 3b adds chain (memtile multi-BD next_bd, both dirs).
///
/// `iter` (shim BD iteration) is DEFERRED: it executes on the emulator but fails
/// on real NPU1 silicon -- the overlapping iteration patterns are order-dependent
/// on S2MM (writes) and the MM2S read variant times out the kernel. Making iter
/// HW-deterministic needs non-overlapping interleaved patterns + a timeout
/// root-cause. Evidence + redesign plan: docs/superpowers/findings/
/// 2026-06-12-dma-iter-hw-deferral.md. Feature::Iter stays in the enum so the
/// follow-up is a gating flip, not a refactor.
pub fn supported(engine: Engine, feature: Feature, dir: Direction) -> bool {
    use Feature::*;
    match feature {
        Linear | Strided2d | Strided3d | Transpose | Overlap => true,
        Strided4d => engine == Engine::Memtile,
        Packet => dir == Direction::Mm2s,
        PadBefore | PadAfter | PadBoth => engine == Engine::Memtile && dir == Direction::Mm2s,
        // 3b: memtile-only multi-BD next_bd chain (silicon-verified).
        Chain => engine == Engine::Memtile,
        // Deferred (see doc-comment above): never in the universe yet.
        Iter => false,
    }
}

/// The full sorted coverage-key universe (87 keys: 81 3a + 6 chain).
pub fn universe_keys() -> Vec<String> {
    let mut keys = Vec::new();
    for engine in Engine::all() {
        for feature in Feature::all() {
            for dir in Direction::all() {
                if !supported(engine, feature, dir) {
                    continue;
                }
                for dtype in Dtype::all() {
                    keys.push(format!(
                        "{}/{}/{}/{}",
                        feature.key_str(),
                        engine.key_str(),
                        dir.key_str(),
                        dtype.key_str()
                    ));
                }
            }
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

/// A parsed coverage key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    pub feature: Feature,
    pub engine: Engine,
    pub dir: Direction,
    pub dtype: Dtype,
}

pub fn parse_key(key: &str) -> Result<Target, String> {
    let parts: Vec<&str> = key.split('/').collect();
    if parts.len() != 4 {
        return Err(format!("bad key (need feature/engine/dir/dtype): {key}"));
    }
    let feature = Feature::all()
        .into_iter()
        .find(|f| f.key_str() == parts[0])
        .ok_or_else(|| format!("unknown feature: {}", parts[0]))?;
    let engine = Engine::all()
        .into_iter()
        .find(|e| e.key_str() == parts[1])
        .ok_or_else(|| format!("unknown engine: {}", parts[1]))?;
    let dir = Direction::all()
        .into_iter()
        .find(|d| d.key_str() == parts[2])
        .ok_or_else(|| format!("unknown dir: {}", parts[2]))?;
    let dtype = Dtype::all()
        .into_iter()
        .find(|t| t.key_str() == parts[3])
        .ok_or_else(|| format!("unknown dtype: {}", parts[3]))?;
    Ok(Target { feature, engine, dir, dtype })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn universe_is_87_keys() {
        // 81 (3a) + 6 chain (memtile x 2dir x 3dtype). iter is deferred.
        let u = universe_keys();
        assert_eq!(u.len(), 87, "expected 87 keys, got {}", u.len());
        let mut s = u.clone();
        s.sort();
        s.dedup();
        assert_eq!(s, u);
        assert_eq!(u.iter().filter(|k| k.starts_with("chain/")).count(), 6);
        assert_eq!(u.iter().filter(|k| k.starts_with("iter/")).count(), 0, "iter is deferred");
    }

    #[test]
    fn gating_invariants() {
        for k in universe_keys() {
            if k.starts_with("strided4d/") {
                assert!(k.contains("/memtile/"), "strided4d must be memtile: {k}");
            }
            if k.starts_with("pad") {
                assert!(k.contains("/memtile/") && k.contains("/mm2s/"), "pad must be memtile mm2s: {k}");
            }
            if k.starts_with("packet/") {
                assert!(k.contains("/mm2s/"), "packet must be mm2s: {k}");
            }
            // chain is memtile-only, both directions.
            if k.starts_with("chain/") {
                assert!(k.contains("/memtile/"), "chain must be memtile: {k}");
            }
        }
    }

    #[test]
    fn parse_round_trips_every_key() {
        for k in universe_keys() {
            let t = parse_key(&k).unwrap();
            let back = format!(
                "{}/{}/{}/{}",
                t.feature.key_str(),
                t.engine.key_str(),
                t.dir.key_str(),
                t.dtype.key_str()
            );
            assert_eq!(back, k);
        }
    }
}
