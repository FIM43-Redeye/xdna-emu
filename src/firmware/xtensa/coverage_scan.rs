//! M2a exit-gate coverage scan (Task 10, firmware-gated): proves the base ISA
//! is complete over the firmware's real code, not just "every op we happened
//! to write a unit test for." Asserts zero `Op::Unknown` across two of the
//! three parts of the firmware's executed code that the M2a spec's
//! "Completeness bar" names:
//!
//! - the Ghidra-identified body (`0x2730+`), and
//! - the MMU-prologue executed path (`0x320..jx`).
//!
//! The third part -- the `~0x200..0x320` reset head -- is executed code but is
//! **covered-by-implementation, not by this scan**: its instructions are all
//! already-implemented pre-M2a opcodes (per the M2a spec's inspection), but it
//! is deferred from the scan for lack of an instruction-boundary oracle (see
//! the "reset region below" note further down for the full argument). So this
//! gate proves zero-Unknown over the Ghidra body + the MMU prologue, which is
//! the whole M2a opcode surface; it does not independently re-audit the reset
//! head. The two regions it does cover:
//!
//! 1. **The Ghidra-identified body** (`0x2730..0x3ca0e`): `listing.txt` is
//!    the authoritative instruction-boundary oracle (produced by Ghidra's
//!    disassembler, which -- unlike a blind linear byte scan -- already
//!    resolved code vs. literal-pool data). For every listed line, re-derive
//!    its bytes from the loaded image at the listed offset and decode them.
//! 2. **The boot/reset prologue** (`0x320..0x399`, entry -> `jx`): Ghidra did
//!    NOT disassemble this range (it starts before the identified body), and
//!    a blind linear scan of it is NOT reliable -- confirmed by hand while
//!    building this scan: `xtensa-lx106-elf-objdump` run linearly from
//!    ~`0x1e0` desyncs into repeated `ill`/`excw` noise because the region
//!    interleaves literal-pool data with code (e.g. `d2 73 03` linearly
//!    "decodes" as an RRI8 cache op with `t=0xD`, which matches none of
//!    `dhwbi`/`dhi`/`dii`/`ihi`'s `t` values in either objdump's table or
//!    this decoder -- consistent with it being data, not a real instruction).
//!    So this region is audited by **following the real executed path**:
//!    driving the actual interpreter (`FirmwareProcessor`'s `Cpu`/`Bus`) from
//!    the pinned boot entry (`firmware::BOOT_ENTRY`, `0x320`) forward,
//!    asserting no `Step::Unknown` fires, until the `jx` that leaves ROM for
//!    (unmapped, pre-MMU) virtual space is executed -- the M2b MMU wall,
//!    genuinely out of scope for M2a (see the M2a spec's Scope Boundaries).
//!
//! The reset region below that (the M2a spec's "reset head at ~`0x200`") was
//! investigated too: `objdump -D -b binary -m xtensa` from `0x1e0` mostly
//! desyncs into the same data-interleaving noise, but happens to resync onto
//! a real `j 0x320` at file offset `0x28e` preceded by what look like TLB/
//! cache pre-fill loops. Without a control-flow-driven RE pass (Ghidra
//! didn't cover this range either), the true reset-vector address and its
//! instruction boundaries are not established well enough to assert
//! "zero-Unknown" against by any means safer than eyeballing objdump noise --
//! which is exactly the failure mode this scan exists to avoid. `BOOT_ENTRY`
//! (`0x320`) is the M1.7-pinned, coherence-verified start of the MMU-setup
//! routine and is what this scan drives from, per the task brief's own
//! "simplest faithful approach" allowance. If the true `~0x200` reset head is
//! ever reconstructed with real instruction boundaries (a `listing.txt`-style
//! oracle), it belongs in a follow-up to this scan, not a guess here.

use std::path::Path;

use crate::firmware::xtensa::decode::{self, Op};
use crate::firmware::xtensa::interp::Step;
use crate::firmware::{firmware_path, FirmwareImage, FirmwareProcessor, BOOT_ENTRY};

/// One parsed `listing.txt` line: `OFFSET: HEXBYTES  MNEMONIC operands`.
/// `text` (the mnemonic + operands) is kept only for diagnostics -- it plays
/// no role in the decode itself, which re-derives everything from `offset`
/// and the loaded image.
struct ListingLine {
    offset: u32,
    num_bytes: usize,
    text: String,
}

/// Parse one `listing.txt` line. Returns `None` for anything that doesn't
/// match `OFFSET: HEXBYTES ...` (a defensive skip per the task brief -- in
/// practice every one of the 33768 lines in the captured listing matches,
/// but a future re-capture is not guaranteed to).
fn parse_listing_line(line: &str) -> Option<ListingLine> {
    let (offset_str, rest) = line.split_once(':')?;
    let offset_str = offset_str.trim();
    if offset_str.is_empty() || !offset_str.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let offset = u32::from_str_radix(offset_str, 16).ok()?;

    let rest = rest.trim_start();
    let (hex_bytes, text) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    if hex_bytes.is_empty() || hex_bytes.len() % 2 != 0 || !hex_bytes.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    Some(ListingLine { offset, num_bytes: hex_bytes.len() / 2, text: text.trim().to_string() })
}

/// **Region 1**: every instruction boundary Ghidra identified in the firmware
/// body must decode to something other than `Op::Unknown`. Also cross-checks
/// that this decoder's own notion of instruction length agrees with Ghidra's
/// byte count at that offset -- a length disagreement would mean a
/// format-selection bug (narrow vs. wide) distinct from an outright missing
/// opcode, so it's collected and reported separately.
#[test]
fn zero_unknown_in_ghidra_identified_body() {
    let Some(fw_path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };
    let listing_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("build/experiments/firmware-re/listing.txt");
    if !listing_path.exists() {
        eprintln!("skip: Ghidra listing.txt not present at {}", listing_path.display());
        return;
    }

    let raw = std::fs::read(&fw_path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse firmware image");
    let bytes = img.bytes();

    let listing = std::fs::read_to_string(&listing_path).expect("read listing.txt");

    let mut scanned = 0u64;
    let mut skipped_lines = 0u64;
    let mut unknowns: Vec<(u32, u32, String)> = Vec::new();
    let mut length_mismatches: Vec<(u32, u8, usize, String)> = Vec::new();

    for line in listing.lines() {
        let Some(parsed) = parse_listing_line(line) else {
            skipped_lines += 1;
            continue;
        };
        scanned += 1;

        let start = parsed.offset as usize;
        let end = start + parsed.num_bytes;
        let slice: &[u8] = if end <= bytes.len() {
            &bytes[start..end]
        } else {
            &[]
        };

        let decoded = decode::decode(slice, parsed.offset);
        if let Op::Unknown { word } = decoded.op {
            unknowns.push((parsed.offset, word, parsed.text.clone()));
            continue;
        }
        if decoded.len as usize != parsed.num_bytes {
            length_mismatches.push((parsed.offset, decoded.len, parsed.num_bytes, parsed.text.clone()));
        }
    }

    eprintln!("=== M2a coverage scan: Ghidra-identified body ===");
    eprintln!("lines scanned     = {scanned}");
    eprintln!("lines skipped     = {skipped_lines}");
    eprintln!("unknown count     = {}", unknowns.len());
    eprintln!("length mismatches = {}", length_mismatches.len());

    // Non-vacuousness floor: an empty/reformatted listing.txt (present but
    // yielding no parseable lines) would otherwise let both emptiness
    // assertions below PASS with scanned==0 -- a silently vacuous gate. The
    // real capture has 33768 instruction lines; 5000 is a safe structural
    // floor that no legitimate shrink could cross while still being real
    // coverage. Paired with a skipped-lines ceiling so a format drift that
    // makes most lines parse-skip (shrinking coverage while a few lines
    // still parse and stay above the floor) can't slip through either.
    assert!(
        scanned > 5000,
        "coverage scan parsed only {scanned} instruction lines -- listing.txt missing/emptied/reformatted?",
    );
    assert!(
        skipped_lines < 16,
        "coverage scan skipped {skipped_lines} unparseable listing.txt lines -- a format drift may be \
         silently shrinking coverage (the captured listing parses cleanly with 0 skips)",
    );

    assert!(
        length_mismatches.is_empty(),
        "decoded length disagreed with Ghidra's byte count at these offsets (decoder format-selection \
         bug, or a listing mis-alignment -- distinct from a missing opcode): {:#?}",
        length_mismatches,
    );
    assert!(
        unknowns.is_empty(),
        "real instructions in the Ghidra-identified body decoded to Op::Unknown -- each is either a \
         genuine missing opcode (a bug in M2a Tasks 2-9) or a listing mis-identification, listed here as \
         (offset, raw word, what Ghidra calls it): {:#?}",
        unknowns,
    );
}

/// **Region 2**: the boot/reset MMU-setup prologue Ghidra never disassembled
/// (it precedes the identified body). Drives the real interpreter from
/// `BOOT_ENTRY` and follows the actual executed path -- not a blind linear
/// sweep, which the module doc above shows is unreliable here -- asserting
/// no `Step::Unknown` fires. Stops the instant the `jx` into (unmapped,
/// pre-MMU) virtual space is executed: that's the M2b MMU wall, and
/// M2a's scope ends at the ROM boundary.
#[test]
fn zero_unknown_in_boot_prologue() {
    let Some(fw_path) = firmware_path() else {
        eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
        return;
    };

    let raw = std::fs::read(&fw_path).expect("read firmware");
    let img = FirmwareImage::parse(&raw).expect("parse firmware image");
    let mut proc = FirmwareProcessor::load(img, BOOT_ENTRY);

    // Generous ceiling: the real prologue is 42 instructions (independently
    // confirmed against `xtensa-lx106-elf-objdump` while building this scan,
    // and matching the M2a-carried-forward M1.7 observation). A bug that
    // stalls or loops the interpreter before reaching `jx` fails loudly here
    // rather than hanging the test suite.
    const MAX_STEPS: u32 = 200;

    let mut n = 0u32;
    loop {
        assert!(
            n < MAX_STEPS,
            "boot prologue did not reach jx within {MAX_STEPS} steps -- stuck at pc={:#x}",
            proc.cpu.pc,
        );

        let pc = proc.cpu.pc;
        // Peek (no side effects) so we know BEFORE stepping whether this is
        // the jx that leaves ROM -- the loop's stop condition is derived
        // from the decode, not a hardcoded exit address.
        let peek =
            [proc.bus.peek8(pc), proc.bus.peek8(pc.wrapping_add(1)), proc.bus.peek8(pc.wrapping_add(2))];
        let is_jx = matches!(decode::decode(&peek, pc).op, Op::Jx { .. });

        let step = proc.cpu.step(&mut proc.bus);
        n += 1;

        if let Step::Unknown { pc, word } = step {
            panic!(
                "boot-region real instruction at {pc:#x} (word {word:#08x}) decoded to Op::Unknown -- the \
                 entry->jx prologue should be entirely M1-era pre-existing opcodes (movi.n/wsr/isync/l32r/ \
                 dsync/witlb/wdtlb/or/iitlb/idtlb/jx); this is a refactor regression, not a missing M2a op",
            );
        }
        if is_jx {
            break;
        }
    }

    eprintln!("=== M2a coverage scan: boot prologue (0x{BOOT_ENTRY:x}..jx) ===");
    eprintln!("instructions executed = {n}");
    eprintln!("pc after jx            = {:#x}", proc.cpu.pc);

    // Secondary regression guard beyond "no Unknown": the exact instruction
    // count is a derived fact (objdump-confirmed), not a guess, so a drift
    // here means the interpreter's control flow desynced even though every
    // individual op still decoded -- worth failing loudly on too.
    assert_eq!(n, 42, "boot prologue instruction count drifted from the confirmed entry->jx length of 42");
}
