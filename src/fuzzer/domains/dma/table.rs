//! DMA coverage universe (3a) and key parsing. Keys are
//! `{feature}/{engine}/{dir}/{dtype}`, capability-gated per the spike matrix.
use super::chain::{Direction, Dtype, Engine, Feature};

/// Is this (engine, feature, dir) combination part of the 3a universe?
/// Gating (spike-verified): strided4d + pad* are memtile-only; packet + pad*
/// are MM2S-only; iter/chain are deferred to 3b.
pub fn supported(engine: Engine, feature: Feature, dir: Direction) -> bool {
    use Feature::*;
    match feature {
        Linear | Strided2d | Strided3d | Transpose | Overlap => true,
        Strided4d => engine == Engine::Memtile,
        Packet => dir == Direction::Mm2s,
        PadBefore | PadAfter | PadBoth => engine == Engine::Memtile && dir == Direction::Mm2s,
        // 3b -- never in the 3a universe.
        Iter | Chain => false,
    }
}

/// The full sorted 3a key universe (81 keys).
pub fn universe_keys() -> Vec<String> {
    let mut keys = Vec::new();
    for engine in Engine::all() {
        for feature in Feature::all_3a() {
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
    let feature = Feature::all_3a()
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
    fn universe_is_81_keys() {
        let u = universe_keys();
        assert_eq!(u.len(), 81, "expected 81 3a keys, got {}", u.len());
        let mut s = u.clone();
        s.sort();
        s.dedup();
        assert_eq!(s, u);
    }

    #[test]
    fn no_iter_or_chain_keys() {
        for k in universe_keys() {
            assert!(!k.starts_with("iter/"), "iter is 3b: {k}");
            assert!(!k.starts_with("chain/"), "chain is 3b: {k}");
        }
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
