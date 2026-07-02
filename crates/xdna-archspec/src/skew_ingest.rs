//! Build-time ingestion of the SP-5c `skew_constants.json` handoff into a
//! `BroadcastTiming`. Fail-loud: a calibrated file with any null constant is an
//! error (never a silent fallback to zeros). Design Sec.7.

use crate::types::BroadcastTiming;

pub fn uncalibrated() -> BroadcastTiming {
    BroadcastTiming {
        per_hop_horizontal: 0,
        per_hop_vertical: 0,
        intra_tile_core_offset: 0,
        intra_tile_mem_offset: 0,
        calibrated: false,
    }
}

fn field_u8(v: &serde_json::Value, path: &[&str]) -> Option<u8> {
    let mut cur = v;
    for k in path {
        cur = cur.get(k)?;
    }
    cur.as_u64().map(|n| n as u8)
}

pub fn broadcast_timing_from_json(v: &serde_json::Value) -> Result<BroadcastTiming, String> {
    let calibrated = v.get("calibrated").and_then(|c| c.as_bool()).unwrap_or(false);
    if !calibrated {
        return Ok(uncalibrated());
    }
    let d_h = field_u8(v, &["d_h"]);
    let d_v = field_u8(v, &["d_v"]);
    let core = field_u8(v, &["intra", "core"]);
    let mem = field_u8(v, &["intra", "mem"]);
    match (d_h, d_v, core, mem) {
        (Some(h), Some(vv), Some(c), Some(m)) => Ok(BroadcastTiming {
            per_hop_horizontal: h,
            per_hop_vertical: vv,
            intra_tile_core_offset: c,
            intra_tile_mem_offset: m,
            calibrated: true,
        }),
        _ => Err(format!(
            "calibrated skew_constants.json has a null/missing constant \
             (d_h={d_h:?} d_v={d_v:?} core={core:?} mem={mem:?}); refusing to \
             build a model that claims calibrated with incomplete data"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncalibrated_is_all_zero() {
        let t = uncalibrated();
        assert_eq!(t.per_hop_horizontal, 0);
        assert_eq!(t.per_hop_vertical, 0);
        assert_eq!(t.intra_tile_core_offset, 0);
        assert_eq!(t.intra_tile_mem_offset, 0);
        assert!(!t.calibrated);
    }

    #[test]
    fn uncalibrated_json_maps_to_zeros() {
        let v = serde_json::json!({
            "calibrated": false, "d_h": null, "d_v": null,
            "intra": {"core": null, "mem": null}
        });
        let t = broadcast_timing_from_json(&v).unwrap();
        assert!(!t.calibrated);
        assert_eq!(t.per_hop_horizontal, 0);
    }

    #[test]
    fn calibrated_json_maps_all_fields() {
        // Distinct values for every field so a future field-path swap
        // (e.g. transposing intra.core <-> intra.mem, or d_h <-> d_v) fails
        // this test instead of passing silently.
        let v = serde_json::json!({
            "calibrated": true, "d_h": 4, "d_v": 2,
            "intra": {"core": 1, "mem": 3}
        });
        let t = broadcast_timing_from_json(&v).unwrap();
        assert!(t.calibrated);
        assert_eq!(t.per_hop_horizontal, 4);
        assert_eq!(t.per_hop_vertical, 2);
        assert_eq!(t.intra_tile_core_offset, 1);
        assert_eq!(t.intra_tile_mem_offset, 3);
    }

    #[test]
    fn calibrated_with_null_d_h_is_error() {
        let v = serde_json::json!({
            "calibrated": true, "d_h": null, "d_v": 2,
            "intra": {"core": 1, "mem": 3}
        });
        let err = broadcast_timing_from_json(&v).unwrap_err();
        assert!(err.contains("calibrated"), "{err}");
    }

    #[test]
    fn calibrated_with_null_d_v_is_error() {
        let v = serde_json::json!({
            "calibrated": true, "d_h": 4, "d_v": null,
            "intra": {"core": 1, "mem": 3}
        });
        let err = broadcast_timing_from_json(&v).unwrap_err();
        assert!(err.contains("calibrated"), "{err}");
    }

    #[test]
    fn calibrated_with_null_intra_core_is_error() {
        let v = serde_json::json!({
            "calibrated": true, "d_h": 4, "d_v": 2,
            "intra": {"core": null, "mem": 3}
        });
        let err = broadcast_timing_from_json(&v).unwrap_err();
        assert!(err.contains("calibrated"), "{err}");
    }

    #[test]
    fn calibrated_with_null_intra_mem_is_error() {
        let v = serde_json::json!({
            "calibrated": true, "d_h": 4, "d_v": 2,
            "intra": {"core": 1, "mem": null}
        });
        let err = broadcast_timing_from_json(&v).unwrap_err();
        assert!(err.contains("calibrated"), "{err}");
    }
}
