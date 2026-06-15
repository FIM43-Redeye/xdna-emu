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
///
/// Burst size and inter-burst gap are **ranges**: each burst draws a fresh value
/// uniformly from `[min, max]`. Real DDR delivery is stochastic (AXI
/// burst-splitting + DDR row/refresh timing under shared SoC traffic), so HW
/// PORT_RUNNING cadence is a *distribution* (NPU1: slot0 5.73 +- 0.44 sub-bursts,
/// slot4 7.07 +- 0.25), not a fixed count -- a single fixed cadence cannot match
/// it. `min == max` collapses to the historical fixed-param behavior. See
/// `docs/superpowers/specs/2026-06-15-ddr-stochastic-delivery-jitter.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BurstParams {
    /// Lower bound (inclusive) of the per-burst word count. `max == 0` disables
    /// the burst model (uniform per-cycle delivery, the historical default).
    pub burst_words_min: u16,
    /// Upper bound (inclusive) of the per-burst word count.
    pub burst_words_max: u16,
    /// Lower bound (inclusive) of the inter-burst idle gap (the DDR
    /// row/refresh/refill gap). Ignored when the model is disabled.
    pub inter_burst_cycles_min: u16,
    /// Upper bound (inclusive) of the inter-burst idle gap.
    pub inter_burst_cycles_max: u16,
    /// Cycles before the first word of a channel session is delivered (DDR
    /// read-pipeline fill). Deterministic (not drawn): paid once per gate
    /// session; `reset()` re-arms it.
    pub first_access_latency: u16,
}

impl BurstParams {
    /// The disabled model: uniform delivery, no bursting.
    pub const DISABLED: BurstParams = BurstParams {
        burst_words_min: 0,
        burst_words_max: 0,
        inter_burst_cycles_min: 0,
        inter_burst_cycles_max: 0,
        first_access_latency: 0,
    };

    /// AXI4-faithful *fixed* DDR burst cadence for AIE2 shim host-memory reads.
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
    ///
    /// This is the *degenerate* (min == max) point of the range model; the
    /// stochastic delivery jitter that band-matches HW uses widened bounds,
    /// calibrated against a fresh HW capture (the calibration surface, not a
    /// model constant).
    pub const AIE2_DDR_DEFAULT: BurstParams = BurstParams {
        burst_words_min: 16,
        burst_words_max: 16,
        inter_burst_cycles_min: 1,
        inter_burst_cycles_max: 1,
        first_access_latency: 0,
    };

    /// Whether the burst model is active. `burst_words_max == 0` means uniform
    /// delivery -- the gate is a no-op pass-through.
    #[inline]
    pub fn enabled(&self) -> bool {
        self.burst_words_max != 0
    }
}

/// SplitMix64 -- a tiny, dependency-free PRNG. One `u64` of state, advanced in
/// place. Used to draw per-burst sizes/gaps so DDR delivery varies run-to-run
/// the way silicon does. Resume-safe: the state is plain data carried in
/// `BurstGate`, so a paused-and-resumed run replays the identical sequence.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Per-channel burst-delivery state. Resume-safe: the only state is the words
/// remaining in the current burst, the cycles remaining in the current
/// gap/first-access latency, and the PRNG state. Stochastic *across* fresh
/// sessions (the seed varies), deterministic *within* one (the PRNG replays).
#[derive(Debug, Clone, Default)]
pub struct BurstGate {
    /// Cycles left in the current idle gap (or first-access latency). While
    /// `> 0` the gate delivers nothing.
    gap_left: u16,
    /// Words left in the current burst before the next gap opens. Redrawn from
    /// `[burst_words_min, burst_words_max]` after a gap closes.
    burst_left: u16,
    /// Whether the first-access latency has been armed this session.
    armed: bool,
    /// PRNG state. Seeded per-channel (see [`set_seed`](Self::set_seed)); `0`
    /// (the default) is a valid fixed seed, used until a seed is wired in.
    rng_state: u64,
}

impl BurstGate {
    /// Construct a gate with an explicit PRNG seed.
    pub fn seeded(seed: u64) -> Self {
        Self { rng_state: seed, ..Self::default() }
    }

    /// Set the PRNG seed (per-channel, mixed from the master seed and
    /// `(col,row,channel)` by the caller so channels decorrelate). Does not
    /// disturb in-flight burst/gap counters.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = seed;
    }

    /// Re-arm for a fresh channel session (channel reset / stop). The next
    /// `words_allowed` re-charges the first-access latency. The PRNG sequence
    /// **continues** (the seed is preserved) -- a channel restart does not
    /// re-roll DDR randomness, and preserving it keeps resume reproducible.
    pub fn reset(&mut self) {
        let seed = self.rng_state;
        *self = Self::default();
        self.rng_state = seed;
    }

    /// Draw an inclusive `[min, max]` value. `max <= min` returns `min` (the
    /// degenerate fixed case -- no PRNG advance, so fixed params stay
    /// bit-for-bit deterministic regardless of seed).
    fn draw(&mut self, min: u16, max: u16) -> u16 {
        if max <= min {
            return min;
        }
        let span = (max - min) as u64 + 1;
        min + (splitmix64(&mut self.rng_state) % span) as u16
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
            self.burst_left = self.draw(p.burst_words_min, p.burst_words_max);
        }
        if self.gap_left > 0 {
            self.gap_left -= 1;
            return 0;
        }
        if self.burst_left == 0 {
            // Burst exhausted: redraw the next burst size, then open an
            // inter-burst gap (also drawn), consuming one of its cycles now
            // (this cycle is the first idle cycle).
            self.burst_left = self.draw(p.burst_words_min, p.burst_words_max);
            let gap = self.draw(p.inter_burst_cycles_min, p.inter_burst_cycles_max);
            if gap > 0 {
                self.gap_left = gap - 1;
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

    /// Build fixed (degenerate, min == max) params -- the historical
    /// fixed-cadence behavior, used by the back-compat tests below.
    fn fixed(burst_words: u16, inter_burst_cycles: u16, first_access_latency: u16) -> BurstParams {
        BurstParams {
            burst_words_min: burst_words,
            burst_words_max: burst_words,
            inter_burst_cycles_min: inter_burst_cycles,
            inter_burst_cycles_max: inter_burst_cycles,
            first_access_latency,
        }
    }

    #[test]
    fn aie2_ddr_default_is_axi_faithful() {
        // AXI4 max read burst (16 beats) + 1-cycle re-arm; cold-start is
        // modeled elsewhere so first_access_latency stays 0 (no double-count).
        // The default is the degenerate (min == max) point of the range model.
        let p = BurstParams::AIE2_DDR_DEFAULT;
        assert!(p.enabled());
        assert_eq!((p.burst_words_min, p.burst_words_max), (16, 16));
        assert_eq!((p.inter_burst_cycles_min, p.inter_burst_cycles_max), (1, 1));
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
        let p = fixed(4, 3, 2);
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
        let p = fixed(3, 2, 0);
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
        let p = fixed(4, 10, 0);
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
        let p = fixed(4, 1, 3);
        let mut g = BurstGate::default();
        assert_eq!(g.words_allowed(p), 0); // latency
        g.reset();
        // After reset the latency is re-charged: three more idle cycles.
        assert_eq!(g.words_allowed(p), 0);
        assert_eq!(g.words_allowed(p), 0);
        assert_eq!(g.words_allowed(p), 0);
        assert_eq!(g.words_allowed(p), 4);
    }

    // --- stochastic range model ---

    /// Drive a gate to completion and record the per-cycle delivery timeline
    /// (1 word/cyc shim rate), for `n` cycles.
    fn timeline(g: &mut BurstGate, p: BurstParams, n: usize) -> Vec<u16> {
        let mut t = Vec::with_capacity(n);
        for _ in 0..n {
            let d = g.words_allowed(p).min(1);
            t.push(d);
            g.consume(d);
        }
        t
    }

    #[test]
    fn same_seed_replays_identically() {
        // Reproducibility / resume-safety: a given seed yields a bit-for-bit
        // identical delivery sequence.
        let p = BurstParams {
            burst_words_min: 4,
            burst_words_max: 16,
            inter_burst_cycles_min: 1,
            inter_burst_cycles_max: 12,
            first_access_latency: 0,
        };
        let a = timeline(&mut BurstGate::seeded(0xABCD), p, 200);
        let b = timeline(&mut BurstGate::seeded(0xABCD), p, 200);
        assert_eq!(a, b, "same seed must replay identically");
    }

    #[test]
    fn distinct_seeds_decorrelate() {
        let p = BurstParams {
            burst_words_min: 4,
            burst_words_max: 16,
            inter_burst_cycles_min: 1,
            inter_burst_cycles_max: 12,
            first_access_latency: 0,
        };
        let a = timeline(&mut BurstGate::seeded(1), p, 200);
        let b = timeline(&mut BurstGate::seeded(2), p, 200);
        assert_ne!(a, b, "distinct seeds should produce distinct cadences");
    }

    #[test]
    fn draws_stay_within_bounds() {
        // Every burst length lands in [min, max] and every gap in [min, max].
        let p = BurstParams {
            burst_words_min: 5,
            burst_words_max: 9,
            inter_burst_cycles_min: 3,
            inter_burst_cycles_max: 7,
            first_access_latency: 0,
        };
        let mut g = BurstGate::seeded(0x1234_5678);
        // Walk many bursts, measuring contiguous run/gap lengths.
        let t = timeline(&mut g, p, 5000);
        let mut i = 0;
        while i < t.len() {
            // measure a run of 1s (burst), then a run of 0s (gap)
            let start = i;
            while i < t.len() && t[i] == 1 {
                i += 1;
            }
            let burst = i - start;
            // The first/last partial run at the window edges may be clipped; skip
            // zero-length and only assert on interior complete runs.
            if burst > 0 && start > 0 && i < t.len() {
                assert!((5..=9).contains(&burst), "burst len {burst} out of [5,9]");
            }
            let gstart = i;
            while i < t.len() && t[i] == 0 {
                i += 1;
            }
            let gap = i - gstart;
            if gap > 0 && gstart > 0 && i < t.len() {
                assert!((3..=7).contains(&gap), "gap len {gap} out of [3,7]");
            }
        }
    }

    #[test]
    fn min_eq_max_is_seed_independent() {
        // Degenerate (fixed) params must be bit-for-bit identical regardless of
        // seed -- the `max <= min` short-circuit must not advance the PRNG.
        let p = fixed(8, 4, 0);
        let a = timeline(&mut BurstGate::seeded(111), p, 100);
        let b = timeline(&mut BurstGate::seeded(999), p, 100);
        assert_eq!(a, b, "fixed params must not depend on the seed");
    }

    #[test]
    fn reset_preserves_seed() {
        // A channel restart re-arms the latency but continues the PRNG sequence
        // (does not re-roll), keeping a resumed run reproducible.
        let p = BurstParams {
            burst_words_min: 2,
            burst_words_max: 16,
            inter_burst_cycles_min: 1,
            inter_burst_cycles_max: 9,
            first_access_latency: 0,
        };
        let mut g = BurstGate::seeded(0x5151);
        let _ = timeline(&mut g, p, 50);
        let post_reset_seed_before = g.rng_state;
        g.reset();
        assert_eq!(g.rng_state, post_reset_seed_before, "reset must preserve PRNG state");
    }
}
