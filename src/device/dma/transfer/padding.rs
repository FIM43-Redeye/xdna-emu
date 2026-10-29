//! Zero-padding state machine for MemTile MM2S DMA transfers.
//!
//! Tracks where to insert zero-padded words at dimension boundaries.
//! Per AM025, the padding dimensions have different units:
//! - D0 before/after: individual 32-bit zero words
//! - D1 before/after: "wraps of dim0" (complete D0 output rows of zeros)
//! - D2 before/after: "wraps of dim0dim1" (complete D1 output blocks of zeros)

use super::super::addressing::{AddressGenerator, ZeroPadConfig};

/// What a padded transfer should output on the next cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadAction {
    /// Read a data word from memory at the given address.
    Data(u64),
    /// Emit a zero word (padding, no memory read).
    Zero,
}

/// Phase within the zero-padding state machine.
///
/// The output pattern for a padded transfer follows nested dimension loops:
///
/// ```text
/// for each D2 iteration:
///   [d2_before zeros]
///   for each D1 iteration:
///     [d1_before zeros]
///     [d0_before zeros] [d0_size data words] [d0_after zeros]
///     [d1_after zeros]
///   [d2_after zeros]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PadPhase {
    D2Before,
    D1Before,
    D0Before,
    D0Data,
    D0After,
    D1After,
    D2After,
    Done,
}

/// Tracks zero-padding state for a MemTile MM2S transfer.
///
/// Wraps around the address generator: on each `next()` call, returns either
/// `PadAction::Data(addr)` (read from memory) or `PadAction::Zero` (emit zero).
/// The address generator only advances for data words.
///
/// Per AM025, the padding dimensions have different units:
/// - D0 before/after: individual 32-bit zero words
/// - D1 before/after: "wraps of dim0" (complete D0 output rows of zeros)
/// - D2 before/after: "wraps of dim0dim1" (complete D1 output blocks of zeros)
#[derive(Debug, Clone)]
pub struct ZeroPadState {
    config: ZeroPadConfig,
    phase: PadPhase,
    /// Remaining words in the current phase
    phase_remaining: u32,
    /// D0 data size (from dimension config)
    d0_size: u32,
    /// Size of one complete D0 output row (d0_before + d0_size + d0_after)
    d0_wrap_size: u32,
    /// D1 data iteration count
    d1_size: u32,
    /// Total D1 iterations including padding (d1_before + d1_size + d1_after)
    d1_total: u32,
    /// D2 iteration count
    d2_size: u32,
    /// Current D1 counter (0..d1_size, data iterations only)
    d1_counter: u32,
    /// Current D2 counter (0..d2_size)
    d2_counter: u32,
    /// Total output words (data + padding) for the whole transfer
    total_output_words: u64,
    /// Output words emitted so far
    words_emitted: u64,
}

impl ZeroPadState {
    /// Create a new padding state from BD dimensions and padding config.
    pub fn new(config: ZeroPadConfig, d0_size: u32, d1_size: u32, d2_size: u32) -> Self {
        let d0_eff = if d0_size == 0 { 1 } else { d0_size };
        let d1_eff = if d1_size == 0 { 1 } else { d1_size };
        let d2_eff = if d2_size == 0 { 1 } else { d2_size };

        // D0 wrap = one complete D0 output row including D0 padding
        let d0_wrap = config.d0_before as u32 + d0_eff + config.d0_after as u32;
        // D1 total = data iterations + D1 padding iterations
        let d1_total = config.d1_before as u32 + d1_eff + config.d1_after as u32;

        let data_words = d0_eff as u64 * d1_eff as u64 * d2_eff as u64;
        let pad_words = config.total_pad_words(d0_eff, d1_eff, d2_eff);
        let total = data_words + pad_words;

        // Start at D2Before if there are D2-level before zeros, otherwise
        // cascade down through the phase hierarchy
        let (phase, phase_remaining) = if config.d2_before > 0 {
            // D2 before: emit d2_before complete D1 blocks of zeros
            (PadPhase::D2Before, config.d2_before as u32 * d1_total * d0_wrap)
        } else {
            Self::enter_d1_iteration(&config, d0_eff, d0_wrap)
        };

        Self {
            config,
            phase,
            phase_remaining,
            d0_size: d0_eff,
            d0_wrap_size: d0_wrap,
            d1_size: d1_eff,
            d1_total,
            d2_size: d2_eff,
            d1_counter: 0,
            d2_counter: 0,
            total_output_words: total,
            words_emitted: 0,
        }
    }

    /// Total output words (data + padding).
    pub fn total_output_words(&self) -> u64 {
        self.total_output_words
    }

    /// Whether all output words have been emitted.
    pub fn is_finished(&self) -> bool {
        self.phase == PadPhase::Done || self.words_emitted >= self.total_output_words
    }

    /// Remaining output words.
    pub fn remaining(&self) -> u64 {
        self.total_output_words.saturating_sub(self.words_emitted)
    }

    /// Get the next output action without advancing state.
    pub fn current_action(&self, addr_gen: &AddressGenerator) -> PadAction {
        match self.phase {
            PadPhase::D0Data => PadAction::Data(addr_gen.current()),
            PadPhase::Done => PadAction::Zero, // shouldn't be called, but safe
            _ => PadAction::Zero,
        }
    }

    /// Advance the state machine by one output word.
    ///
    /// Returns true if the address generator should also be advanced
    /// (i.e., the word was data, not padding).
    pub fn advance(&mut self) -> bool {
        if self.phase == PadPhase::Done {
            return false;
        }

        self.words_emitted += 1;
        let is_data = self.phase == PadPhase::D0Data;

        self.phase_remaining = self.phase_remaining.saturating_sub(1);
        if self.phase_remaining == 0 {
            self.transition();
        }

        is_data
    }

    /// Transition to the next phase when the current phase is exhausted.
    fn transition(&mut self) {
        match self.phase {
            PadPhase::D2Before => {
                // D2 before padding done, start first D1 iteration
                let (phase, remaining) =
                    Self::enter_d1_iteration(&self.config, self.d0_size, self.d0_wrap_size);
                self.phase = phase;
                self.phase_remaining = remaining;
            }
            PadPhase::D1Before => {
                // D1 before padding done, start D0 data pattern
                self.phase = if self.config.d0_before > 0 {
                    PadPhase::D0Before
                } else {
                    PadPhase::D0Data
                };
                self.phase_remaining = if self.config.d0_before > 0 {
                    self.config.d0_before as u32
                } else {
                    self.d0_size
                };
            }
            PadPhase::D0Before => {
                self.phase = PadPhase::D0Data;
                self.phase_remaining = self.d0_size;
            }
            PadPhase::D0Data => {
                if self.config.d0_after > 0 {
                    self.phase = PadPhase::D0After;
                    self.phase_remaining = self.config.d0_after as u32;
                } else {
                    self.advance_d1_data_iter();
                }
            }
            PadPhase::D0After => {
                // D0 complete for this D1 data iteration.
                // Advance to next D1 data iteration, or emit D1 after.
                self.advance_d1_data_iter();
            }
            PadPhase::D1After => {
                // D1 after padding done (all D1 data iterations complete).
                // Advance to the next D2 iteration.
                self.finish_d2_iteration();
            }
            PadPhase::D2After => {
                self.d2_counter += 1;
                if self.d2_counter < self.d2_size {
                    // Start next D2 iteration
                    if self.config.d2_before > 0 {
                        self.phase = PadPhase::D2Before;
                        self.phase_remaining =
                            self.config.d2_before as u32 * self.d1_total * self.d0_wrap_size;
                    } else {
                        let (phase, remaining) =
                            Self::enter_d1_iteration(&self.config, self.d0_size, self.d0_wrap_size);
                        self.phase = phase;
                        self.phase_remaining = remaining;
                    }
                    self.d1_counter = 0;
                } else {
                    self.phase = PadPhase::Done;
                    self.phase_remaining = 0;
                }
            }
            PadPhase::Done => {}
        }
    }

    /// Advance after completing one D1 data iteration's D0 pattern.
    ///
    /// If more D1 data iterations remain, start the next D0 pattern.
    /// If all D1 data iterations are done, emit D1 after padding (or
    /// advance to the next D2 iteration if no D1 after padding).
    fn advance_d1_data_iter(&mut self) {
        self.d1_counter += 1;
        if self.d1_counter < self.d1_size {
            // More D1 data iterations -- start next D0 pattern directly
            // (D1 before was already emitted once at the start of this D1 block)
            self.phase = if self.config.d0_before > 0 {
                PadPhase::D0Before
            } else {
                PadPhase::D0Data
            };
            self.phase_remaining = if self.config.d0_before > 0 {
                self.config.d0_before as u32
            } else {
                self.d0_size
            };
        } else {
            // All D1 data iterations done -- emit D1 after padding
            if self.config.d1_after > 0 {
                self.phase = PadPhase::D1After;
                // Each D1 after unit = one complete D0 output row of zeros
                self.phase_remaining = self.config.d1_after as u32 * self.d0_wrap_size;
            } else {
                self.finish_d2_iteration();
            }
        }
    }

    /// Called when all data D1 iterations within a D2 iteration are done.
    fn finish_d2_iteration(&mut self) {
        if self.config.d2_after > 0 {
            self.phase = PadPhase::D2After;
            // Each D2 after unit = one complete D1 block of zeros
            self.phase_remaining = self.config.d2_after as u32 * self.d1_total * self.d0_wrap_size;
        } else {
            self.d2_counter += 1;
            if self.d2_counter < self.d2_size {
                self.d1_counter = 0;
                if self.config.d2_before > 0 {
                    self.phase = PadPhase::D2Before;
                    self.phase_remaining = self.config.d2_before as u32 * self.d1_total * self.d0_wrap_size;
                } else {
                    let (phase, remaining) =
                        Self::enter_d1_iteration(&self.config, self.d0_size, self.d0_wrap_size);
                    self.phase = phase;
                    self.phase_remaining = remaining;
                }
            } else {
                self.phase = PadPhase::Done;
                self.phase_remaining = 0;
            }
        }
    }

    /// Determine initial phase when entering a new D1 iteration.
    ///
    /// D1 before padding = d1_before complete D0 wraps of zeros (per AM025:
    /// "wraps of dim0 before dim1").
    fn enter_d1_iteration(config: &ZeroPadConfig, d0_size: u32, d0_wrap_size: u32) -> (PadPhase, u32) {
        if config.d1_before > 0 {
            // Each D1 before unit = one complete D0 output row of zeros
            (PadPhase::D1Before, config.d1_before as u32 * d0_wrap_size)
        } else if config.d0_before > 0 {
            (PadPhase::D0Before, config.d0_before as u32)
        } else {
            (PadPhase::D0Data, d0_size)
        }
    }
}
