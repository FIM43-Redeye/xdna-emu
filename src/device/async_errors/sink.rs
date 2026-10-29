//! Tier B sink: cache + per-column rings + drain queue.
//!
//! Single owner of all async-error state. `record_error` is the only
//! mutation entry point; reads (cache, ring bytes, drain) are read-only
//! aside from `drain_newly_recorded` (which empties the queue).

use std::collections::VecDeque;

use xdna_archspec::aie2::async_errors::{self, AieErrorOrigin};

use super::types::{AieError, AmdxdnaAsyncError, AsyncRing, RET_CODE_OVERFLOW};

/// Async-error subsystem state. Lives on `DeviceState`.
pub struct AsyncErrorSink {
    /// Last-error cache. `None` until any error fires; mirrors the driver's
    /// `amdxdna_async_err_cache` last-write-wins behavior.
    cache: Option<AmdxdnaAsyncError>,
    /// Per-column ring buffers. Indexed by physical column. Sized to match
    /// the driver's per-column `async_event` slot allocation (5 cols on NPU1,
    /// but we size dynamically to support any width).
    rings: Vec<AsyncRing>,
    /// Records added since last drain. Drained by the FFI layer between
    /// engine steps to fire the registered push callback.
    newly_recorded: VecDeque<AmdxdnaAsyncError>,
}

impl AsyncErrorSink {
    /// Create a sink with `num_cols` independent ring buffers.
    pub fn new(num_cols: usize) -> Self {
        Self {
            cache: None,
            rings: (0..num_cols).map(|_| AsyncRing::new()).collect(),
            newly_recorded: VecDeque::new(),
        }
    }

    /// Record an error. Updates the cache (last-write-wins), appends to
    /// the column's ring (or sets RET_CODE_OVERFLOW if full), and queues
    /// the cache record for FFI drain.
    pub fn record_error(&mut self, col: u8, row: u8, origin: AieErrorOrigin, event_id: u8, cycle: u64) {
        // 1. Append to the column's ring.
        let col_idx = col as usize;
        if let Some(ring) = self.rings.get_mut(col_idx) {
            let record = AieError {
                row,
                col,
                reserved_0: 0,
                mod_type: origin.wire_mod_type(),
                event_id,
                reserved_1: 0,
                reserved_2: 0,
            };
            if ring.push(record).is_err() {
                ring.set_ret_code(RET_CODE_OVERFLOW);
            }
        }

        // 2. Categorize and update cache. `is_error_event` gates the call
        // site, so `event_to_category` returning None here would be a bug;
        // we expect Some and unwrap with a clear message.
        let category = async_errors::event_to_category(event_id, origin)
            .expect("record_error: event_to_category returned None; caller must gate via is_error_event");
        let num = async_errors::category_to_error_num(category);
        let module = async_errors::mod_type_to_amdxdna_module(origin);
        // Convenience helper hardcodes driver=Aie, severity=Critical, class=Aie,
        // matching driver's AMDXDNA_CRITICAL_ERROR_CODE_BUILD (aie2_error.c:239).
        let err_code = async_errors::build_critical_aie_error_code(num, module);
        let ex_err_code = async_errors::build_ex_err_code(row, col);
        // Cycle-as-nanoseconds at ~1 GHz silicon -> microseconds = cycle / 1000.
        // Deterministic; spec section 2 ts_us decision.
        let ts_us = cycle / 1000;
        let record = AmdxdnaAsyncError { err_code, ts_us, ex_err_code };
        self.cache = Some(record);
        self.newly_recorded.push_back(record);
    }

    /// Read the last-error cache. `None` until any error fires.
    pub fn last_cache(&self) -> Option<&AmdxdnaAsyncError> {
        self.cache.as_ref()
    }

    /// Read-only view of column `col`'s ring. `None` if `col` is out of range.
    pub fn ring(&self, col: u8) -> Option<&AsyncRing> {
        self.rings.get(col as usize)
    }

    /// Drain and return records queued since the last drain. FIFO order.
    /// FFI layer calls this between engine steps to fire push callbacks.
    pub fn drain_newly_recorded(&mut self) -> Vec<AmdxdnaAsyncError> {
        self.newly_recorded.drain(..).collect()
    }

    /// Zero the cache, all rings, and the drain queue. Called from
    /// `xdna_emu_reset_context` and `xdna_emu_clear_async_errors`.
    pub fn clear(&mut self) {
        self.cache = None;
        for r in self.rings.iter_mut() {
            r.clear();
        }
        self.newly_recorded.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xdna_archspec::aie2::async_errors::{AmdxdnaErrorModule, AmdxdnaErrorNum};

    #[test]
    fn new_sink_has_empty_cache_and_rings() {
        let sink = AsyncErrorSink::new(5);
        assert!(sink.last_cache().is_none());
        for c in 0u8..5 {
            assert_eq!(sink.ring(c).unwrap().header().err_cnt, 0);
        }
    }

    #[test]
    fn record_error_populates_cache_and_ring() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(1, 2, AieErrorOrigin::Core, 69, 50_000);
        let cache = sink.last_cache().expect("cache must be populated");
        assert_eq!(
            cache.err_code,
            async_errors::build_critical_aie_error_code(
                AmdxdnaErrorNum::AieInstruction,
                AmdxdnaErrorModule::AieCore,
            )
        );
        assert_eq!(cache.ts_us, 50); // 50_000 cycles / 1000
        assert_eq!(cache.ex_err_code, (2u64 << 8) | 1u64);
        let ring = sink.ring(1).unwrap();
        assert_eq!(ring.header().err_cnt, 1);
        let rec = &ring.records()[0];
        assert_eq!(rec.event_id, 69);
        assert_eq!(rec.row, 2);
        assert_eq!(rec.col, 1);
        assert_eq!(rec.mod_type, AieErrorOrigin::Core.wire_mod_type());
    }

    #[test]
    fn second_record_overwrites_cache_appends_ring() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(1, 2, AieErrorOrigin::Core, 69, 1_000);
        sink.record_error(1, 3, AieErrorOrigin::Core, 70, 2_000);
        let cache = sink.last_cache().unwrap();
        assert_eq!(cache.ts_us, 2);
        assert_eq!(cache.ex_err_code, (3u64 << 8) | 1u64);
        assert_eq!(sink.ring(1).unwrap().header().err_cnt, 2);
    }

    #[test]
    fn per_column_rings_independent() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(1, 2, AieErrorOrigin::Core, 69, 1_000);
        sink.record_error(3, 2, AieErrorOrigin::Core, 69, 2_000);
        assert_eq!(sink.ring(1).unwrap().header().err_cnt, 1);
        assert_eq!(sink.ring(3).unwrap().header().err_cnt, 1);
        // Column 2 untouched.
        assert_eq!(sink.ring(2).unwrap().header().err_cnt, 0);
    }

    #[test]
    fn out_of_range_col_is_silent_noop_on_ring_but_still_updates_cache() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(99, 2, AieErrorOrigin::Core, 69, 1_000);
        // Cache populated even though no ring matched (categorization still ran).
        assert!(sink.last_cache().is_some());
        // No rings updated.
        for c in 0u8..5 {
            assert_eq!(sink.ring(c).unwrap().header().err_cnt, 0);
        }
    }

    #[test]
    fn overflow_at_capacity_sets_ret_code() {
        use super::super::types::MAX_ERRORS_PER_RING;
        let mut sink = AsyncErrorSink::new(5);
        for _ in 0..MAX_ERRORS_PER_RING {
            sink.record_error(0, 2, AieErrorOrigin::Core, 69, 1_000);
        }
        assert_eq!(sink.ring(0).unwrap().header().err_cnt as usize, MAX_ERRORS_PER_RING);
        // One more triggers overflow.
        sink.record_error(0, 2, AieErrorOrigin::Core, 69, 1_000);
        assert_eq!(sink.ring(0).unwrap().header().ret_code, RET_CODE_OVERFLOW);
        // err_cnt unchanged.
        assert_eq!(sink.ring(0).unwrap().header().err_cnt as usize, MAX_ERRORS_PER_RING);
    }

    #[test]
    fn drain_returns_records_in_fifo_order_and_empties_queue() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(1, 2, AieErrorOrigin::Core, 69, 1_000);
        sink.record_error(2, 3, AieErrorOrigin::Core, 70, 2_000);
        let drained = sink.drain_newly_recorded();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].ts_us, 1);
        assert_eq!(drained[1].ts_us, 2);
        // Second drain returns empty.
        assert!(sink.drain_newly_recorded().is_empty());
    }

    #[test]
    fn clear_resets_cache_rings_and_drain_queue() {
        let mut sink = AsyncErrorSink::new(5);
        sink.record_error(0, 2, AieErrorOrigin::Core, 69, 1_000);
        sink.clear();
        assert!(sink.last_cache().is_none());
        assert_eq!(sink.ring(0).unwrap().header().err_cnt, 0);
        assert!(sink.drain_newly_recorded().is_empty());
    }
}
