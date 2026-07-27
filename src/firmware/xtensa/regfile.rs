//! Xtensa windowed register file: 64 physical AR registers, WINDOWBASE/
//! WINDOWSTART, and the logical->physical rotation the windowed call ABI uses.

/// PS.CALLINC field shift (bits 17:16): the call increment a `callN` records
/// so the matching `entry` knows how far to rotate the window.
const PS_CALLINC_SHIFT: u32 = 16;
/// PS.EXCM bit (bit 4): exception mode. While set, window overflow/underflow
/// detection is masked (the handler runs with EXCM=1 so it can't re-fault).
pub const PS_EXCM: u32 = 1 << 4;
/// PS.UM bit (bit 5): user mode selects UserExceptionVector for a non-double
/// exception; clear selects KernelExceptionVector.
pub const PS_UM: u32 = 1 << 5;
/// PS.WOE bit (bit 18): window-overflow-detection enable. Overflow/underflow
/// exceptions are only raised when WOE=1 (boot sets it once the ABI is live).
pub const PS_WOE: u32 = 1 << 18;
/// PS.OWB field shift (bits 11:8): Old WindowBase. A window overflow/underflow
/// exception saves the pre-rotation WINDOWBASE here; `rfwo`/`rfwu` restore
/// WINDOWBASE from it (QEMU `win_helper.c`, `PS_OWB_SHIFT`).
const PS_OWB_SHIFT: u32 = 8;
/// PS.INTLEVEL field mask (bits 3:0): the current interrupt level, below
/// which interrupts are masked. `waiti imm4` writes `imm4` here and idles
/// until an interrupt at a higher level arrives.
const PS_INTLEVEL_MASK: u32 = 0xF;

/// Number of window frames (quads) the WINDOWBASE/WINDOWSTART bookkeeping wraps
/// through: 64 physical AR registers / 4 registers per quad.
pub const NUM_FRAMES: u32 = 16;

/// Xtensa windowed register file: 64 physical AR registers with windowed-ABI state.
///
/// `Clone` supports the FLIX `xt_format1` parallel-bundle executor, which
/// snapshots the file so each slot's op reads pre-bundle state (all fields are
/// `Copy`, so the derive is a cheap memcpy).
#[derive(Clone)]
pub struct RegFile {
    // Physical AR registers; accessed via logical indices through the window mechanism.
    ar: [u32; 64],
    /// Window base register: selects which 16-register block is the logical AR0-AR15.
    pub windowbase: u32,
    /// Window start register: bitmask indicating which window frames contain live state.
    pub windowstart: u32,
    /// Shift amount register: used for bit shifts and field rotation operations.
    pub sar: u32,
    /// Processor state register: current execution state and mode flags.
    pub ps: u32,
    /// Loop begin (LBEG): the address of the first instruction inside the
    /// active zero-overhead loop body. Set by `loop`/`loopnez` to `pc + 3`
    /// (the setup instruction's own length) -- see
    /// `interp::control::exec`'s `Loop`/`Loopnez` handling and the LBEG/LEND
    /// asymmetry documented there.
    pub lbeg: u32,
    /// Loop end (LEND): one past the last instruction of the active
    /// zero-overhead loop body. `interp::Cpu::step`'s per-retire loop-back
    /// fires when an instruction retires by advancing `pc` SEQUENTIALLY
    /// (not via a taken branch/jump/call/ret) to exactly `lend` while
    /// `lcount != 0` and `PS.EXCM` is clear.
    pub lend: u32,
    /// Loop count (LCOUNT): remaining zero-overhead-loop iterations after
    /// the one currently in flight. Xtensa's trip count is `AR[s] - 1`
    /// (`loop`/`loopnez` always run the iteration in progress before the
    /// count is ever checked), so `lcount == 0` means "this is the last
    /// iteration."
    pub lcount: u32,
    /// Exception cause register (EXCCAUSE, SR number 0xE8 -- see
    /// `interp::SR_EXCCAUSE`): the cause code of the most recently raised
    /// general exception (M2a Task 9's `raise_general_exception`), e.g.
    /// `EXCCAUSE_SYSCALL` (1) or `EXCCAUSE_INTEGER_DIVIDE_BY_ZERO` (6). A
    /// plain independent register, like `lbeg`/`lend`/`lcount` -- no
    /// bit-packing, so direct field access is the right shape (see
    /// `loop_registers_default_zero_and_round_trip`'s comment for why that
    /// convention was chosen over a getter/setter pair for this kind of SR).
    pub exccause: u32,
}

impl RegFile {
    /// Create a new windowed register file with all registers zeroed and default ABI state.
    pub fn new() -> Self {
        Self {
            ar: [0; 64],
            windowbase: 0,
            windowstart: 1,
            sar: 0,
            ps: 0,
            lbeg: 0,
            lend: 0,
            lcount: 0,
            exccause: 0,
        }
    }

    /// Compute the physical AR index for a logical register number (0-15).
    /// Maps logical to physical via: `(windowbase * 4 + logical) mod 64`.
    pub fn phys(&self, logical: u8) -> usize {
        (((self.windowbase as usize) * 4) + logical as usize) % 64
    }

    /// Read a logical AR register.
    pub fn read_ar(&self, logical: u8) -> u32 {
        self.ar[self.phys(logical)]
    }

    /// Write a logical AR register.
    pub fn write_ar(&mut self, logical: u8, v: u32) {
        let p = self.phys(logical);
        self.ar[p] = v;
    }

    /// Rotate the window by `delta` register-quads (each quad is 4 registers).
    /// `delta` can be positive (forward; used by `entry`) or negative
    /// (backward; used by `retw`).
    pub fn rotate(&mut self, delta: i32) {
        self.windowbase = (self.windowbase as i32 + delta).rem_euclid(NUM_FRAMES as i32) as u32;
    }

    /// The current PS.CALLINC (0-3): how many quads the pending `entry` rotates.
    pub fn callinc(&self) -> u32 {
        (self.ps >> PS_CALLINC_SHIFT) & 0x3
    }

    /// Set PS.CALLINC (low 2 bits of `k`), leaving the rest of PS untouched.
    /// A `callN` records the call size here for the callee's `entry` to read.
    pub fn set_callinc(&mut self, k: u32) {
        self.ps = (self.ps & !(0x3 << PS_CALLINC_SHIFT)) | ((k & 0x3) << PS_CALLINC_SHIFT);
    }

    /// The current PS.INTLEVEL (0-15): the interrupt level `waiti` last set.
    pub fn intlevel(&self) -> u32 {
        self.ps & PS_INTLEVEL_MASK
    }

    /// Set PS.INTLEVEL (low 4 bits of `level`), leaving the rest of PS
    /// untouched. `waiti imm4` records the wait level here.
    pub fn set_intlevel(&mut self, level: u32) {
        self.ps = (self.ps & !PS_INTLEVEL_MASK) | (level & PS_INTLEVEL_MASK);
    }

    /// True when window overflow/underflow detection is enabled (PS.WOE=1 and
    /// PS.EXCM=0). Both gates must hold before a window exception can be raised.
    pub fn window_exceptions_enabled(&self) -> bool {
        (self.ps & PS_WOE != 0) && (self.ps & PS_EXCM == 0)
    }

    /// True when PS.EXCM is set (the CPU is in exception mode).
    pub fn excm(&self) -> bool {
        self.ps & PS_EXCM != 0
    }

    /// True when PS.UM selects user mode.
    pub fn um(&self) -> bool {
        self.ps & PS_UM != 0
    }

    /// Enter exception mode (set PS.EXCM), masking further window exceptions.
    pub fn set_excm(&mut self) {
        self.ps |= PS_EXCM;
    }

    /// Leave exception mode (clear PS.EXCM). Used by `rfwo`/`rfwu` on return
    /// from a window-spill/fill handler.
    pub fn clear_excm(&mut self) {
        self.ps &= !PS_EXCM;
    }

    /// Read PS.OWB (the saved Old WindowBase, 0-15).
    pub fn owb(&self) -> u32 {
        (self.ps >> PS_OWB_SHIFT) & 0xF
    }

    /// Save `wb` into PS.OWB, leaving the rest of PS untouched. A window
    /// exception stashes the pre-rotation WINDOWBASE here for `rfwo`/`rfwu`.
    pub fn set_owb(&mut self, wb: u32) {
        self.ps = (self.ps & !(0xF << PS_OWB_SHIFT)) | ((wb & 0xF) << PS_OWB_SHIFT);
    }

    /// Detect a window overflow for an `entry`/access needing `w` quads of new
    /// window. Returns `None` if no live frame lies within `w` quads ahead of
    /// WINDOWBASE; else `Some((n, spill_quads))` where `n` is the forward
    /// rotation to reach the frame that must be spilled and `spill_quads` (1/2/3)
    /// is that frame's size, selecting the Overflow4/8/12 vector. Mirrors QEMU
    /// `HELPER(window_check)`: `n = ctz(replicate(WS) >> (WB+1)) + 1`, size =
    /// `ctz(that >> n) + 1` (capped at 3). The replicate (`WS | WS<<NUM_FRAMES`)
    /// handles the wrap past frame 15.
    pub fn overflow_check(&self, w: u32) -> Option<(u32, u32)> {
        let replicated = self.windowstart | (self.windowstart << NUM_FRAMES);
        let shifted = replicated >> (self.windowbase + 1);
        let n = shifted.trailing_zeros() + 1;
        if n > w {
            return None;
        }
        let spill_quads = ((shifted >> n).trailing_zeros() + 1).min(3);
        Some((n, spill_quads))
    }

    /// True if the window frame quad `q` (taken mod [`NUM_FRAMES`]) is marked
    /// live in WINDOWSTART -- i.e. it holds a not-yet-returned frame's registers.
    pub fn frame_live(&self, q: u32) -> bool {
        self.windowstart & (1 << (q % NUM_FRAMES)) != 0
    }

    /// Mark the frame quad `q` (mod [`NUM_FRAMES`]) live in WINDOWSTART.
    pub fn mark_frame_live(&mut self, q: u32) {
        self.windowstart |= 1 << (q % NUM_FRAMES);
    }

    /// Clear the live bit for frame quad `q` (mod [`NUM_FRAMES`]) in WINDOWSTART.
    pub fn clear_frame_live(&mut self, q: u32) {
        self.windowstart &= !(1 << (q % NUM_FRAMES));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_maps_through_windowbase() {
        let mut rf = RegFile::new();
        rf.windowbase = 0; // physical base 0
        rf.write_ar(3, 0xaaaa);
        assert_eq!(rf.phys(3), 3);
        assert_eq!(rf.read_ar(3), 0xaaaa);

        rf.windowbase = 2; // physical base = 2*4 = 8
        rf.write_ar(0, 0xbbbb);
        assert_eq!(rf.phys(0), 8);
        assert_eq!(rf.read_ar(0), 0xbbbb);
    }

    #[test]
    fn rotate_advances_windowbase_mod_16() {
        let mut rf = RegFile::new();
        rf.windowbase = 15;
        rf.rotate(2); // 15 + 2 = 17 -> 1 (mod 16)
        assert_eq!(rf.windowbase, 1);
    }

    #[test]
    fn owb_round_trips_in_ps_bits_11_8() {
        let mut rf = RegFile::new();
        rf.ps = 0xFFFF_F0FF; // all PS bits set except the OWB field
        rf.set_owb(0xB);
        assert_eq!(rf.owb(), 0xB);
        assert_eq!(rf.ps, 0xFFFF_FBFF, "set_owb must touch only bits 11:8");
        rf.set_owb(0x2);
        assert_eq!(rf.owb(), 0x2);
    }

    #[test]
    fn overflow_check_detects_fully_nested_call8_wrap() {
        let mut rf = RegFile::new();
        // Every quad holds a call8 frame's base bit (0,2,4,..14 -> 0x5555); the
        // window is full. The newest frame is at WB=14; a further call8 (w=2)
        // would rotate onto the live frame at quad 0 -> Overflow8 (spill 2 quads).
        rf.windowstart = 0x5555;
        rf.windowbase = 14;
        assert_eq!(rf.overflow_check(2), Some((2, 2)));
    }

    #[test]
    fn overflow_check_none_when_room() {
        let mut rf = RegFile::new();
        // Only the current frame is live; a call8 has free quads ahead.
        rf.windowstart = 0x1;
        rf.windowbase = 0;
        assert_eq!(rf.overflow_check(2), None);
    }

    #[test]
    fn physical_wraps_mod_64() {
        let mut rf = RegFile::new();
        rf.windowbase = 15; // base = 60
        assert_eq!(rf.phys(7), (60 + 7) % 64); // 3
    }

    #[test]
    fn rotate_negative_delta_retw_decrement() {
        let mut rf = RegFile::new();
        rf.windowbase = 1;
        rf.rotate(-3); // 1 - 3 = -2 -> 14 (mod 16)
        assert_eq!(rf.windowbase, 14);
    }

    #[test]
    fn callinc_round_trips_through_ps() {
        let mut rf = RegFile::new();
        rf.set_callinc(2);
        assert_eq!(rf.callinc(), 2);
        // Only the 2-bit field is written; the rest of PS is untouched.
        rf.ps |= PS_WOE;
        rf.set_callinc(3);
        assert_eq!(rf.callinc(), 3);
        assert_ne!(rf.ps & PS_WOE, 0);
    }

    #[test]
    fn intlevel_round_trips_through_ps() {
        let mut rf = RegFile::new();
        assert_eq!(rf.intlevel(), 0);
        rf.set_intlevel(5);
        assert_eq!(rf.intlevel(), 5);
        // Only the low 4 bits are written; the rest of PS (e.g. CALLINC) is
        // untouched, same convention as set_callinc.
        rf.set_callinc(2);
        rf.set_intlevel(15);
        assert_eq!(rf.intlevel(), 15);
        assert_eq!(rf.callinc(), 2, "set_intlevel must not disturb CALLINC");
    }

    #[test]
    fn window_exceptions_gated_on_woe_and_not_excm() {
        let mut rf = RegFile::new();
        assert!(!rf.window_exceptions_enabled(), "WOE clear by default");
        rf.ps |= PS_WOE;
        assert!(rf.window_exceptions_enabled());
        rf.set_excm();
        assert!(!rf.window_exceptions_enabled(), "EXCM masks detection");
        assert!(rf.excm());
    }

    #[test]
    fn loop_registers_default_zero_and_round_trip() {
        // lbeg/lend/lcount (M2a Task 7): plain independent architectural
        // registers, no bit-packing -- unlike callinc/excm (which pack into
        // PS), a direct field round-trip is the correct "getter/setter"
        // shape here, matching windowbase/windowstart/sar/ps's existing
        // convention in this file.
        let mut rf = RegFile::new();
        assert_eq!(rf.lbeg, 0);
        assert_eq!(rf.lend, 0);
        assert_eq!(rf.lcount, 0);
        rf.lbeg = 0x1000;
        rf.lend = 0x1020;
        rf.lcount = 7;
        assert_eq!(rf.lbeg, 0x1000);
        assert_eq!(rf.lend, 0x1020);
        assert_eq!(rf.lcount, 7);
    }

    #[test]
    fn exccause_round_trips() {
        // EXCCAUSE (M2a Task 9): a plain register, same round-trip shape as
        // lbeg/lend/lcount above -- defaults to 0, holds whatever cause code
        // `raise_general_exception` last wrote.
        let mut rf = RegFile::new();
        assert_eq!(rf.exccause, 0);
        rf.exccause = 6; // EXCCAUSE_INTEGER_DIVIDE_BY_ZERO
        assert_eq!(rf.exccause, 6);
    }

    #[test]
    fn frame_live_marks_and_clears_mod_frames() {
        let mut rf = RegFile::new();
        rf.windowstart = 0;
        rf.mark_frame_live(2);
        assert!(rf.frame_live(2));
        assert!(!rf.frame_live(3));
        // Indices wrap mod NUM_FRAMES: quad 18 aliases quad 2.
        assert!(rf.frame_live(2 + NUM_FRAMES));
        rf.clear_frame_live(2);
        assert!(!rf.frame_live(2));
    }
}
