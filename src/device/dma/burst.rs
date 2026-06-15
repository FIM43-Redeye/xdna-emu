//! DDR burst-delivery gate.
//!
//! Real shim host-DDR delivery is *bursty*: the AXI master / DDR controller
//! delivers a chunk of words, then idles while the next row/burst is fetched.
//! The emulator's default delivery is uniform (a fixed words-per-cycle stream),
//! which is "too smooth" -- downstream S2MM channels rarely starve, so
//! `DMA_S2MM_*_STREAM_STARVATION` and `PORT_RUNNING` level events under-emit
//! versus hardware (known-fidelity-gaps row 50/117).
//!
//! `BurstGate` models that burstiness as a per-channel state machine: deliver
//! `burst_words` words, then idle for `inter_burst_cycles`, after an initial
//! `first_access_latency` before the first word. During an idle gap the gate
//! delivers nothing, so the downstream consumer FIFO drains and the existing
//! S2MM stall-edge logic fires a starvation event naturally -- the same way
//! silicon produces it. We never emit starvation directly.
//!
//! The model is a *framework*: every parameter is configurable (Phoenix-shaped
//! defaults, env-overridable). `burst_words == 0` disables it entirely, giving
//! the historical uniform delivery -- the default, so enabling bursting is opt-in
//! and never silently perturbs existing cycle-accuracy baselines.

/// Tunable DDR burst-delivery parameters. A copy lives in `DmaTimingConfig`;
/// the gate reads it each cycle so a single config drives every channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurstParams {
    /// Words delivered per burst before an inter-burst idle gap. `0` disables
    /// the burst model (uniform per-cycle delivery, the historical default).
    pub burst_words: u16,
    /// Idle cycles between bursts (the DDR row/refresh/refill gap). Ignored
    /// when `burst_words == 0`.
    pub inter_burst_cycles: u16,
    /// Cycles before the first word of a channel session is delivered (DDR
    /// read-pipeline fill). Paid once per gate session; `reset()` re-arms it.
    pub first_access_latency: u16,
}

impl BurstParams {
    /// The disabled model: uniform delivery, no bursting.
    pub const DISABLED: BurstParams =
        BurstParams { burst_words: 0, inter_burst_cycles: 0, first_access_latency: 0 };

    /// AXI4-faithful DDR burst cadence for AIE2 shim host-memory reads -- the
    /// default delivery model.
    ///
    /// `burst_words = 16` is AXI4's maximum read-burst length (ARLEN 0..15 ->
    /// 1..16 beats); `inter_burst_cycles = 1` is the inter-burst address re-arm
    /// bubble. HW-confirmed: NPU1 Phoenix memtile `PORT_RUNNING_0` (S2MM ch0
    /// input, fed by the shim DDR read) traces a clean `on16 off1` cadence, and
    /// the data-dependent burst size (ch1 carries 8-beat bursts) is the
    /// fingerprint of AXI burst-splitting differently-sized transfers, not a
    /// fixed per-tile internal cadence. `first_access_latency = 0` because the
    /// DDR read-pipeline cold-start is already modeled by
    /// `shim_ddr_cold_start_mm2s_cycles`; charging it here would double-count.
    pub const AIE2_DDR_DEFAULT: BurstParams =
        BurstParams { burst_words: 16, inter_burst_cycles: 1, first_access_latency: 0 };

    /// Whether the burst model is active. `burst_words == 0` means uniform
    /// delivery -- the gate is a no-op pass-through.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.burst_words != 0
    }
}

/// Per-channel burst-delivery state. Deterministic and resume-safe: the only
/// state is the words remaining in the current burst and the cycles remaining
/// in the current gap/first-access latency.
#[derive(Debug, Clone, Default)]
pub struct BurstGate {
    /// Cycles left in the current idle gap (or first-access latency). While
    /// `> 0` the gate delivers nothing.
    gap_left: u16,
    /// Words left in the current burst before the next gap opens. Reloaded to
    /// `burst_words` after a gap closes.
    burst_left: u16,
    /// Whether the first-access latency has been armed this session.
    armed: bool,
}

impl BurstGate {
    /// Re-arm for a fresh channel session (channel reset / stop). The next
    /// `words_allowed` re-charges the first-access latency.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// How many words the gate permits this cycle, given the params. The caller
    /// delivers `min(this, its own rate, remaining)` words and reports the
    /// actual count via [`consume`](Self::consume).
    ///
    /// Returns `u16::MAX` (unbounded) when the model is disabled, so a disabled
    /// gate never constrains delivery. Returns `0` during a gap or first-access
    /// latency. Otherwise returns the words left in the current burst.
    ///
    /// This advances the gap countdown: each call while in a gap consumes one
    /// gap cycle. A cycle that returns `0` is an idle DDR cycle.
    pub fn words_allowed(&mut self, p: BurstParams) -> u16 {
        if !p.enabled() {
            return u16::MAX;
        }
        if !self.armed {
            self.armed = true;
            self.gap_left = p.first_access_latency;
            self.burst_left = p.burst_words;
        }
        if self.gap_left > 0 {
            self.gap_left -= 1;
            return 0;
        }
        if self.burst_left == 0 {
            // Burst exhausted: open an inter-burst gap, consuming one of its
            // cycles now (this cycle is the first idle cycle).
            self.burst_left = p.burst_words;
            if p.inter_burst_cycles > 0 {
                self.gap_left = p.inter_burst_cycles - 1;
                return 0;
            }
        }
        self.burst_left
    }

    /// Account for `n` words actually delivered this cycle. The caller may
    /// deliver fewer than `words_allowed` returned (rate cap, end of transfer,
    /// downstream backpressure); only the delivered count draws down the burst.
    pub fn consume(&mut self, n: u16) {
        self.burst_left = self.burst_left.saturating_sub(n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aie2_ddr_default_is_axi_faithful() {
        // AXI4 max read burst (16 beats) + 1-cycle re-arm; cold-start is
        // modeled elsewhere so first_access_latency stays 0 (no double-count).
        let p = BurstParams::AIE2_DDR_DEFAULT;
        assert!(p.enabled());
        assert_eq!(p.burst_words, 16);
        assert_eq!(p.inter_burst_cycles, 1);
        assert_eq!(p.first_access_latency, 0);
    }

    #[test]
    fn disabled_gate_never_constrains() {
        let mut g = BurstGate::default();
        for _ in 0..10 {
            assert_eq!(g.words_allowed(BurstParams::DISABLED), u16::MAX);
            g.consume(4);
        }
    }

    #[test]
    fn first_access_latency_delays_first_word() {
        let p = BurstParams { burst_words: 4, inter_burst_cycles: 3, first_access_latency: 2 };
        let mut g = BurstGate::default();
        // Two latency cycles deliver nothing.
        assert_eq!(g.words_allowed(p), 0);
        assert_eq!(g.words_allowed(p), 0);
        // Then the first burst opens.
        assert_eq!(g.words_allowed(p), 4);
    }

    #[test]
    fn burst_then_gap_cadence_at_one_word_per_cycle() {
        // burst of 3 words, gap of 2 cycles, no first-access latency.
        // Delivering 1 word/cycle (shim rate), expect: 1,1,1, gap,gap, 1,1,1, ...
        let p = BurstParams { burst_words: 3, inter_burst_cycles: 2, first_access_latency: 0 };
        let mut g = BurstGate::default();
        let mut timeline = Vec::new();
        for _ in 0..10 {
            let allowed = g.words_allowed(p);
            let delivered = allowed.min(1); // shim rate = 1 word/cyc
            timeline.push(delivered);
            g.consume(delivered);
        }
        // burst(3) gap(2) burst(3) gap(2): 1,1,1,0,0,1,1,1,0,0
        assert_eq!(timeline, vec![1, 1, 1, 0, 0, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn gap_count_matches_inter_burst_cycles() {
        // Over a long run, count how many cycles deliver 0 (gaps). With burst=4
        // gap=10, each 4-word burst (delivered 1/cyc => 4 cycles) is followed by
        // exactly 10 idle cycles.
        let p = BurstParams { burst_words: 4, inter_burst_cycles: 10, first_access_latency: 0 };
        let mut g = BurstGate::default();
        let (mut gaps, mut bursts) = (0, 0);
        for _ in 0..(4 + 10) * 3 {
            let allowed = g.words_allowed(p);
            let delivered = allowed.min(1);
            if delivered == 0 {
                gaps += 1;
            } else {
                bursts += 1;
            }
            g.consume(delivered);
        }
        // 3 full cycles of (4 burst + 10 gap) = 42 cycles: 12 burst, 30 gap.
        assert_eq!((bursts, gaps), (12, 30));
    }

    #[test]
    fn reset_re_arms_first_access_latency() {
        let p = BurstParams { burst_words: 4, inter_burst_cycles: 1, first_access_latency: 3 };
        let mut g = BurstGate::default();
        assert_eq!(g.words_allowed(p), 0); // latency
        g.reset();
        // After reset the latency is re-charged: three more idle cycles.
        assert_eq!(g.words_allowed(p), 0);
        assert_eq!(g.words_allowed(p), 0);
        assert_eq!(g.words_allowed(p), 0);
        assert_eq!(g.words_allowed(p), 4);
    }
}
