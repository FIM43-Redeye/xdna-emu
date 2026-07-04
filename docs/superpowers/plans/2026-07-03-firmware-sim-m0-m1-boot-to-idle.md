# Firmware-sim M0+M1 (boot-to-idle) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load the real Phoenix management firmware (`npu.dev.sbin`) and run it on a new in-tree base-Xtensa interpreter from its entry vector until it reaches a stable command-loop idle -- proving the interpreter + memory model + windowed-ABI + boot path before any device-timing work.

**Architecture:** A new `src/firmware/` module: a `$PS1`-container image loader, a base-ISA Xtensa decoder (validated against two golden oracles), a windowed register file, a routed memory/MMIO bus, a fetch/execute core with windowed-call + window-exception handling, and off-array system-aperture stubs with spin-detection. A `FirmwareProcessor` ties them together and runs boot-to-idle. Device (`0x04000000`) and mailbox (`0x27000000`) semantics are deliberately stubbed as plain memory in this phase; real routing into `DeviceState` is M2 (out of scope here).

**Tech Stack:** Rust, `zerocopy` (fixed header parse), `thiserror` (`FirmwareError`), `log` (diagnostics). Ground-truth derivation via `xtensa-lx106-elf-objdump` (base ISA) and the captured Ghidra `listing.txt` (windowed ABI).

**Scope boundary:** This plan is **M0 + M1 only** (spec section 9). M2-M5 (MMIO-into-DeviceState bridge, mailbox seam, EXEC_DPU dispatch, clock reconciliation, stub deletion) are deliberately excluded -- they depend on observations M1 produces. Design spec: `docs/superpowers/specs/2026-07-03-firmware-sim-subsystem-design.md`. Recon: `build/experiments/firmware-re/INFODUMP.md`.

## Global Constraints

- **DERIVE FROM THE TOOLCHAIN.** No hand-guessed Xtensa bit layouts. Every decode is a test vector derived from an oracle: `xtensa-lx106-elf-objdump -D -b binary -m xtensa` for base ISA; the captured `build/experiments/firmware-re/listing.txt` (Ghidra Xtensa:LE) for windowed-ABI ops that lx106 cannot decode.
- **Base ISA only, instructions on demand.** Implement an opcode when the firmware executes it; do not front-load the full ISA. No FPU/vector TIE ops (recon: none on the control plane).
- **Phoenix / NPU1 (`1502_00`) only.** Firmware at `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`, 248592 bytes.
- **The firmware binary is NOT in the repo** (downloaded by the driver build). Unit tests MUST be hermetic (in-tree byte-blob builders + captured golden vectors). The one firmware-dependent test (M1.7 boot-to-idle) is env-gated and skips cleanly when the binary is absent, mirroring the HW-gated test pattern.
- **Conventions:** dir + `mod.rs` for multi-file modules; `//!` file headers + `///` on public items; `thiserror` error enum named `FirmwareError` carrying `offset`/`context` (mirror `ParseError`, `src/parser/error.rs`); `log::warn!`/`log::error!` on bad external input, never panic; snake_case sentence-style test names with inline-captured format args.
- **Tile-addressing constants** (needed only when M2 arrives) come from `xdna_archspec::aie2::{TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK}` -- do not redefine locally.
- **After every task:** `cargo test --lib` passes. Commit at the end of each task. No emoji. Commit messages end with:
  ```
  Generated using Claude Code.
  Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
  ```

## File Structure

```
src/firmware/
  mod.rs        FirmwareProcessor; boot-to-idle run loop; module wiring + pub exports
  error.rs      FirmwareError (thiserror; offset/context fields)
  image.rs      $PS1 container parse -> FirmwareImage (base-0 addressable payload)
  mmio.rs       Bus: region routing (Rom/Ram/Mailbox/ArrayLog/System), load/store
  sysstub.rs    SysStub: off-array aperture reads + access log + spin-detection
  xtensa/
    mod.rs      xtensa submodule wiring + re-exports
    decode.rs   Op enum + decode(bytes) -> Decoded{op,len}; two-oracle test vectors
    regfile.rs  RegFile: AR[64], WINDOWBASE/WINDOWSTART/SAR/PS, windowed mapping+rotation
    interp.rs   Cpu{pc,regs}; step(bus)->Step; base ISA exec; windowed calls + window exc
```

Files that change together live together; each has one responsibility. `decode.rs` (pure) and `regfile.rs` (state) are independently testable; `interp.rs` wires them; `mmio.rs`/`sysstub.rs` are the memory seam; `mod.rs` integrates.

---

### Task 1: M0 -- Firmware image loader

**Files:**
- Create: `src/firmware/mod.rs` (initial: module wiring only)
- Create: `src/firmware/error.rs`
- Create: `src/firmware/image.rs`
- Modify: `src/lib.rs` (add `pub mod firmware;` in the always-available group ~line 28, and a `- [`firmware`]: ...` doc bullet)
- Test: inline `#[cfg(test)] mod tests` in `src/firmware/image.rs`

**Interfaces:**
- Produces:
  - `FirmwareError` (enum): variants `BadMagic { offset: usize, found: [u8;4] }`, `Truncated { offset: usize, needed: usize, got: usize }`, `SizeMismatch { header: u32, file: usize }`.
  - `pub struct FirmwareImage { payload: Vec<u8>, payload_size: u32 }`
  - `impl FirmwareImage { pub fn parse(raw: &[u8]) -> Result<Self, FirmwareError>; pub fn bytes(&self) -> &[u8]; pub fn payload_size(&self) -> u32; }`
  - `bytes()` returns the full base-0 addressable image: file offset == link address for the base-0 `.text`/`.rodata` segment (recon `README.md`). The trailing ~512-byte signature sits past the code range (`0x3ca0e`) and is inert data.

**Facts (derived, do not re-guess):** `$PS1` magic (`24 50 53 31`) at file offset `0x10`; payload-size `u32` LE at offset `0x14` = `0x0003c910` (248080 = file_len - 0x200); file length 248592 (`0x3cb10`).

- [ ] **Step 1: Add the module to the crate**

In `src/lib.rs`, in the always-available module group (after `pub mod device;`), add:
```rust
pub mod firmware;
```
And in the crate `//!` doc bullet list, add:
```rust
//! - [`firmware`]: in-tree Xtensa interpreter running the real NPU management firmware
```

- [ ] **Step 2: Write `error.rs`**

```rust
//! Error type for the firmware loader/interpreter.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FirmwareError {
    #[error("bad firmware magic at offset {offset:#x}: found {found:02x?}")]
    BadMagic { offset: usize, found: [u8; 4] },

    #[error("firmware truncated at offset {offset:#x}: need {needed} bytes, have {got}")]
    Truncated { offset: usize, needed: usize, got: usize },

    #[error("firmware size mismatch: header says {header:#x}, file is {file:#x}")]
    SizeMismatch { header: u32, file: usize },
}
```

- [ ] **Step 3: Write the initial `mod.rs` wiring**

```rust
//! In-tree base-Xtensa interpreter that runs the real NPU management firmware.
//!
//! Phase M0+M1 scope: load the `$PS1` image and boot it to a command-loop idle.
//! Device/mailbox MMIO routing into `DeviceState` is later (M2).

mod error;
mod image;

pub use error::FirmwareError;
pub use image::FirmwareImage;
```

- [ ] **Step 4: Write the failing test in `image.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal valid $PS1 container: 0x10 bytes of hash, "$PS1",
    // a size field, then payload, then a 0x200-byte signature trailer.
    fn build_image(payload: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; 0x18];
        v[0x10..0x14].copy_from_slice(b"$PS1");
        let size = (0x18 + payload.len()) as u32; // header+payload, excl. sig
        v[0x14..0x18].copy_from_slice(&size.to_le_bytes());
        v.extend_from_slice(payload);
        v.extend_from_slice(&[0u8; 0x200]); // signature trailer
        v
    }

    #[test]
    fn parses_valid_container_and_exposes_base0_bytes() {
        let raw = build_image(&[0xde, 0xad, 0xbe, 0xef]);
        let img = FirmwareImage::parse(&raw).expect("valid image");
        // base-0 addressable: the "$PS1" magic is still visible at 0x10
        assert_eq!(&img.bytes()[0x10..0x14], b"$PS1");
        assert_eq!(img.payload_size(), (0x18 + 4) as u32);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut raw = build_image(&[0x00]);
        raw[0x10] = b'X';
        let err = FirmwareImage::parse(&raw).unwrap_err();
        assert!(matches!(err, FirmwareError::BadMagic { offset: 0x10, .. }), "got {err}");
    }

    #[test]
    fn rejects_truncated_before_header() {
        let err = FirmwareImage::parse(&[0u8; 0x12]).unwrap_err();
        assert!(matches!(err, FirmwareError::Truncated { .. }), "got {err}");
    }
}
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `cargo test --lib firmware::image`
Expected: FAIL (`FirmwareImage` / `parse` not defined).

- [ ] **Step 6: Implement `image.rs`**

```rust
//! `$PS1` PSP-signed firmware container loader.
//!
//! The image is signed but NOT encrypted or compressed (recon
//! `build/experiments/firmware-re/INFODUMP.md`). We bypass signing and expose
//! the plaintext payload as a base-0 addressable byte image: for the base-0
//! `.text`/`.rodata` segment, file offset == link address.

use crate::firmware::error::FirmwareError;

const MAGIC_OFFSET: usize = 0x10;
const MAGIC: &[u8; 4] = b"$PS1";
const SIZE_OFFSET: usize = 0x14;
const HEADER_END: usize = 0x18;

pub struct FirmwareImage {
    payload: Vec<u8>,
    payload_size: u32,
}

impl FirmwareImage {
    pub fn parse(raw: &[u8]) -> Result<Self, FirmwareError> {
        if raw.len() < HEADER_END {
            return Err(FirmwareError::Truncated {
                offset: 0,
                needed: HEADER_END,
                got: raw.len(),
            });
        }
        let found: [u8; 4] = raw[MAGIC_OFFSET..MAGIC_OFFSET + 4].try_into().unwrap();
        if &found != MAGIC {
            return Err(FirmwareError::BadMagic { offset: MAGIC_OFFSET, found });
        }
        let payload_size =
            u32::from_le_bytes(raw[SIZE_OFFSET..SIZE_OFFSET + 4].try_into().unwrap());
        // The whole file (minus the inert signature trailer) is the base-0 image.
        Ok(Self { payload: raw.to_vec(), payload_size })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.payload
    }

    pub fn payload_size(&self) -> u32 {
        self.payload_size
    }
}
```

Add `mod tests;`-equivalent: the `#[cfg(test)] mod tests` block from Step 4 already lives in this file.

- [ ] **Step 7: Run tests to verify they pass**

Run: `cargo test --lib firmware::image`
Expected: PASS (3 tests).

- [ ] **Step 8: Commit**

```bash
git add src/lib.rs src/firmware/
git commit -m "$(cat <<'EOF'
feat(#140): firmware M0 -- $PS1 image loader

Parse the PSP-signed npu.dev.sbin container (magic @0x10, size @0x14) into a
base-0 addressable FirmwareImage. Hermetic tests via an in-tree builder; no
dependency on the (downloaded, non-tracked) firmware binary.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 2: M1.1 -- Xtensa base-ISA decoder

**Files:**
- Create: `src/firmware/xtensa/mod.rs`
- Create: `src/firmware/xtensa/decode.rs`
- Modify: `src/firmware/mod.rs` (add `pub mod xtensa;`)
- Test: inline `#[cfg(test)] mod tests` in `decode.rs`

**Interfaces:**
- Produces:
  - `pub enum Op` -- one variant per implemented instruction, carrying decoded operands (register indices as `u8`, immediates as `i32`/`u32`, branch targets as absolute `u32` where the encoding is PC-relative and the decoder is given the PC). Includes `Unknown { word: u32 }` for not-yet-implemented opcodes.
  - `pub struct Decoded { pub op: Op, pub len: u8 }` -- `len` is 2 (narrow `.n`) or 3 (standard).
  - `pub fn decode(bytes: &[u8], pc: u32) -> Decoded` -- decodes the instruction at `bytes[0..]`; `pc` resolves PC-relative targets (`l32r`, `call8`, `beqz`). Reads at most 3 bytes.
- Consumed by: `interp.rs` (Task M1.4/M1.5).

**Derivation rule:** every test vector below is a real instruction from the firmware, with the oracle noted. Add new opcodes the same way: find the bytes in the firmware, disassemble with the appropriate oracle, add a vector, then implement. **Do not invent encodings.**

**Ground-truth vectors (derived this session):**

| Bytes (LE) | Oracle | Disassembly |
|------------|--------|-------------|
| `36 41 00` | Ghidra | `entry a1, 0x20` (frame 32) |
| `e5 20 f9` | Ghidra | `call8 0x33244` (from pc 0x3a034) |
| `48 45` | both | `l32i.n a4, a5, 0x10` |
| `bd 03` | both | `mov.n a11, a3` |
| `0c 52` | both | `movi.n a2, 5` |
| `21 bd e7` | lx106 | `l32r a2, <lit>` (from pc 0x33262) |
| `d2 a0 ac` | lx106 | `movi a13, 172` |
| `52 22 0a` | lx106 | `l32i a5, a2, 40` |
| `30 30 f4` | lx106 | `extui a3, a3, 0, 16` |
| `20 a2 20` | lx106 | `or a10, a2, a2` |
| `70 64 50` | lx106 | `witlb a7, a4` |
| `00 20 00` | lx106 | `isync` |

- [ ] **Step 1: Write the failing decoder test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Each case: (bytes, pc, expected Op, expected len). Vectors derived from
    // the real firmware via objdump (base ISA) / Ghidra listing.txt (windowed).
    #[test]
    fn decodes_entry() {
        let d = decode(&[0x36, 0x41, 0x00], 0x33244);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Entry { s: 1, imm: 32 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l32i_n() {
        let d = decode(&[0x48, 0x45], 0x33259);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::L32iN { t: 4, s: 5, imm: 16 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_movi_n() {
        let d = decode(&[0x0c, 0x52], 0x33278);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::MoviN { t: 2, imm: 5 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_movi() {
        let d = decode(&[0xd2, 0xa0, 0xac], 0x33270);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Movi { t: 13, imm: 172 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_or() {
        let d = decode(&[0x20, 0xa2, 0x20], 0x33256);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Or { r: 10, s: 2, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn unknown_opcode_is_reported_not_panicked() {
        // 0xff byte region (padding) must not panic.
        let d = decode(&[0xff, 0xff, 0xff], 0);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --lib firmware::xtensa::decode`
Expected: FAIL (`decode` / `Op` undefined).

- [ ] **Step 3: Implement `xtensa/mod.rs` wiring and the decoder**

In `src/firmware/mod.rs` add `pub mod xtensa;`. Create `src/firmware/xtensa/mod.rs`:
```rust
//! In-tree base-Xtensa interpreter (decoder, register file, execution core).

pub mod decode;
```

Create `src/firmware/xtensa/decode.rs`. Implement `Op`, `Decoded`, and `decode` by extracting Xtensa instruction fields. **Field layout is derived from the oracle disassembly**, cross-checked against the Xtensa ISA reference (op0 = bits[3:0] of byte0 selects format; narrow `.n` ops have op0 = 0x8..0xD). Implement exactly the opcodes covered by the vectors above (`entry`, `call8`, `l32i.n`, `mov.n`, `movi.n`, `movi`, `l32i`, `l32r`, `or`, `extui`, `witlb`, `isync`), each with a decode arm justified by its vector. Structure:

```rust
//! Xtensa instruction decoder. Base ISA + the windowed-call ops the firmware
//! uses. DERIVED FROM THE TOOLCHAIN: every opcode here has a test vector taken
//! from the real firmware, disassembled by xtensa-lx106-elf-objdump (base) or
//! the captured Ghidra listing.txt (windowed ops lx106 cannot decode).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Entry { s: u8, imm: u32 },
    Call8 { target: u32 },
    L32iN { t: u8, s: u8, imm: u32 },
    MovN { t: u8, s: u8 },
    MoviN { t: u8, imm: i32 },
    Movi { t: u8, imm: i32 },
    L32i { t: u8, s: u8, imm: u32 },
    L32r { t: u8, target: u32 },
    Or { r: u8, s: u8, t: u8 },
    Extui { r: u8, t: u8, shiftimm: u8, maskimm: u8 },
    Witlb { t: u8, s: u8 },
    Isync,
    Unknown { word: u32 },
}

pub struct Decoded {
    pub op: Op,
    pub len: u8,
}

pub fn decode(bytes: &[u8], pc: u32) -> Decoded {
    // op0 = low nibble of byte0 selects the instruction format/length.
    // Narrow (2-byte) ops: op0 in 0x8..=0xd. Standard (3-byte) otherwise.
    // Extract fields per the Xtensa ISA formats (RRR/RRI8/RI16/CALLn/BRI8...),
    // one decode arm per vectored opcode. Unknown -> Op::Unknown, never panic.
    // (Full field math implemented here, each arm justified by its M1.1 vector.)
    todo!("implement per the vectors; see derivation rule")
}
```

The implementer fills the field math opcode-by-opcode, running the test after each arm. This is the derive loop, not a placeholder: the vectors are the spec, the oracle is the check.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib firmware::xtensa::decode`
Expected: PASS (6 tests). If any fails, re-derive that instruction's fields from the oracle -- do not adjust the expected value to match a buggy decode.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/
git commit -m "$(cat <<'EOF'
feat(#140): firmware M1.1 -- Xtensa base-ISA decoder

decode(bytes,pc) -> Decoded{op,len} for the starter opcode set, every arm
derived from a real firmware instruction (objdump base ISA / Ghidra listing.txt
windowed). Unknown opcodes reported, never panicked.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 3: M1.2 -- Windowed register file

**Files:**
- Create: `src/firmware/xtensa/regfile.rs`
- Modify: `src/firmware/xtensa/mod.rs` (add `pub mod regfile;`)
- Test: inline `#[cfg(test)] mod tests` in `regfile.rs`

**Interfaces:**
- Produces:
  - `pub struct RegFile { ar: [u32; 64], pub windowbase: u32, pub windowstart: u32, pub sar: u32, pub ps: u32 }`
  - `impl RegFile { pub fn new() -> Self; pub fn phys(&self, logical: u8) -> usize; pub fn read_ar(&self, logical: u8) -> u32; pub fn write_ar(&mut self, logical: u8, v: u32); pub fn rotate(&mut self, delta: i32); }`
  - Logical->physical: `phys = ((windowbase * 4) + logical) mod 64`. `rotate(delta)` adds `delta` (in register-quads) to `windowbase` mod 16. These are the windowed-ABI mechanics `call8`/`entry`/`retw` drive (Task M1.5).
- Consumed by: `interp.rs`.

- [ ] **Step 1: Write the failing test**

```rust
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
    fn physical_wraps_mod_64() {
        let mut rf = RegFile::new();
        rf.windowbase = 15; // base = 60
        assert_eq!(rf.phys(7), (60 + 7) % 64); // 3
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib firmware::xtensa::regfile`
Expected: FAIL (undefined).

- [ ] **Step 3: Implement `regfile.rs`**

```rust
//! Xtensa windowed register file: 64 physical AR registers, WINDOWBASE/
//! WINDOWSTART, and the logical->physical rotation the windowed call ABI uses.

pub struct RegFile {
    ar: [u32; 64],
    pub windowbase: u32,
    pub windowstart: u32,
    pub sar: u32,
    pub ps: u32,
}

impl RegFile {
    pub fn new() -> Self {
        Self { ar: [0; 64], windowbase: 0, windowstart: 1, sar: 0, ps: 0 }
    }

    pub fn phys(&self, logical: u8) -> usize {
        (((self.windowbase as usize) * 4) + logical as usize) % 64
    }

    pub fn read_ar(&self, logical: u8) -> u32 {
        self.ar[self.phys(logical)]
    }

    pub fn write_ar(&mut self, logical: u8, v: u32) {
        let p = self.phys(logical);
        self.ar[p] = v;
    }

    pub fn rotate(&mut self, delta: i32) {
        self.windowbase = (self.windowbase as i32 + delta).rem_euclid(16) as u32;
    }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib firmware::xtensa::regfile`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/firmware/
git commit -m "$(cat <<'EOF'
feat(#140): firmware M1.2 -- Xtensa windowed register file

64 physical ARs, WINDOWBASE/WINDOWSTART, logical->physical mapping and mod-16
window rotation -- the mechanics the windowed call ABI drives.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 4: M1.3 -- Routed memory/MMIO bus

**Files:**
- Create: `src/firmware/mmio.rs`
- Modify: `src/firmware/mod.rs` (add `mod mmio;` and `pub use mmio::Bus;`)
- Test: inline `#[cfg(test)] mod tests` in `mmio.rs`

**Interfaces:**
- Produces:
  - `pub enum Region { Rom, Ram, Mailbox, Array, System }`
  - `pub struct Bus { rom: Vec<u8>, ram: Vec<u8>, mailbox: Vec<u8>, /* logs */ }`
  - `impl Bus { pub fn new(rom: Vec<u8>) -> Self; pub fn region(addr: u32) -> Region; pub fn load32(&mut self, addr: u32) -> u32; pub fn store32(&mut self, addr: u32, v: u32); pub fn load8(&mut self, addr: u32) -> u8; pub fn store8(&mut self, addr: u32, v: u32); }`
  - Routing (spec section 5): `0x00000000..0x04000000` -> `Rom` (base-0 image, RW-store logged as violation); `0x08b00000..` window -> `Ram`; `0x27000000..0x28000000` -> `Mailbox` (plain RAM this phase); `0x04000000..0x08000000` array windows -> `Array` (store logged, load returns 0 this phase); everything else -> `System` (Task M1.6 SysStub; returns 0 + logs until then).
- Consumed by: `interp.rs`, `mod.rs`.

**Note:** RAM/mailbox are lazily-sized backing `Vec`s keyed by offset from their region base. Little-endian 32-bit access.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_addresses_to_regions() {
        assert_eq!(Bus::region(0x00002730), Region::Rom);
        assert_eq!(Bus::region(0x08b00010), Region::Ram);
        assert_eq!(Bus::region(0x27010d00), Region::Mailbox);
        assert_eq!(Bus::region(0x04000000), Region::Array);
        assert_eq!(Bus::region(0xf7000000), Region::System);
    }

    #[test]
    fn rom_reads_little_endian_from_image() {
        let mut bus = Bus::new(vec![0x78, 0x56, 0x34, 0x12]); // @0
        assert_eq!(bus.load32(0), 0x12345678);
    }

    #[test]
    fn ram_round_trips() {
        let mut bus = Bus::new(vec![]);
        bus.store32(0x08b00100, 0xcafebabe);
        assert_eq!(bus.load32(0x08b00100), 0xcafebabe);
    }

    #[test]
    fn mailbox_round_trips_as_ram_this_phase() {
        let mut bus = Bus::new(vec![]);
        bus.store32(0x27010d00, 0x11223344);
        assert_eq!(bus.load32(0x27010d00), 0x11223344);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib firmware::mmio`
Expected: FAIL (undefined).

- [ ] **Step 3: Implement `mmio.rs`**

Implement `Region`, `Bus`, routing by the address ranges above, LE 32-bit and 8-bit access, RAM/mailbox as offset-keyed lazily-grown `Vec<u8>`, ROM store -> `log::warn!` (read-only violation), Array store -> `log::debug!` + record (load returns 0), System -> 0 + `log::debug!`. Full code written here by the implementer following the interface; each access path covered by a Step-1 test.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib firmware::mmio`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/firmware/
git commit -m "$(cat <<'EOF'
feat(#140): firmware M1.3 -- routed memory/MMIO bus

Bus routes firmware loads/stores by aperture (Rom/Ram/Mailbox/Array/System)
per spec section 5. This phase: mailbox+array are plain-memory/logged stubs;
DeviceState routing is M2.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 5: M1.4 -- Interpreter core (base-ISA execution)

**Files:**
- Create: `src/firmware/xtensa/interp.rs`
- Modify: `src/firmware/xtensa/mod.rs` (add `pub mod interp;`)
- Test: inline `#[cfg(test)] mod tests` in `interp.rs`

**Interfaces:**
- Produces:
  - `pub struct Cpu { pub pc: u32, pub regs: RegFile }`
  - `pub enum WaitReason { MailboxEmpty, Waiti, PollSpin { addr: u32 } }`
  - `pub enum Step { Ran, Wait(WaitReason), Exception { cause: u32, pc: u32 }, Unknown { pc: u32, word: u32 } }`
  - `impl Cpu { pub fn new(entry: u32) -> Self; pub fn step(&mut self, bus: &mut Bus) -> Step; }`
  - `step` fetches at `pc` via `bus`, decodes (`decode::decode`), executes one instruction, advances `pc` by `Decoded::len` (or the branch target), returns a `Step`. This task implements the **non-windowed** subset (`movi`, `movi.n`, `mov.n`, `l32i`, `l32i.n`, `l32r`, `or`, `extui`, `isync`, `witlb` as no-op-with-log); `entry`/`call8`/`retw` return `Step::Unknown` for now (implemented in M1.5).
- Consumes: `RegFile` (M1.2), `Bus` (M1.3), `decode` (M1.1).

- [ ] **Step 1: Write the failing test (execute a tiny hand-built program from ROM)**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware::mmio::Bus;

    // movi a2, 5 (0c 52 as movi.n) then or a3,a2,a2 (20 32 20) -> a3 == 5.
    #[test]
    fn executes_movi_n_then_or() {
        let rom = vec![0x0c, 0x52, /* movi.n a2,5 */ 0x30, 0x32, 0x20 /* or a3,a2,a2 */];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 5);
    }

    #[test]
    fn executes_wide_movi() {
        // The 3-byte movi form (distinct from movi.n above). Vector from
        // M1.1: d2 a0 ac = movi a13, 172.
        let rom = vec![0xd2, 0xa0, 0xac];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(13), 172);
    }

    #[test]
    fn windowed_op_returns_unknown_until_m1_5() {
        let rom = vec![0x36, 0x41, 0x00]; // entry a1, 32
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Unknown { .. }));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib firmware::xtensa::interp`
Expected: FAIL (undefined).

- [ ] **Step 3: Implement `interp.rs` (non-windowed subset)**

Implement `Cpu::step`: read 3 bytes at `pc` from `bus` (`load8` x3, tolerating ROM end), `decode`, match each implemented `Op` and mutate `regs`/`bus`, advance `pc`. `entry`/`call8`/other windowed -> `Step::Unknown`. `Op::Unknown` -> `Step::Unknown`. Full arm code written by the implementer, each op covered by a test derived as needed.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib firmware::xtensa::interp`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/
git commit -m "$(cat <<'EOF'
feat(#140): firmware M1.4 -- interpreter core (non-windowed base ISA)

Cpu::step fetch/decode/execute over the Bus for the base-ISA subset; windowed
call ops deferred to M1.5 (return Step::Unknown). Tested on hand-built programs.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 6: M1.5 -- Windowed calls and window exceptions (the crux)

> **CORRECTION (2026-07-03, firmware-derived, human-approved).** Deriving from the
> real firmware during implementation refuted two premises in the original text
> below; the corrected model governs:
> - **WINDOWBASE rotates at `entry`, NOT at the call.** `call8`/`callx8` only set
>   `PS.CALLINC=2` and stash the return address; **`entry`** does `WINDOWBASE +=
>   CALLINC` + frame-alloc + WINDOWSTART + overflow-check; `retw`/`retw.n` rotate
>   back per `a0[31:30]` + underflow-check + return. (Proof: firmware uses `a1`(sp)
>   as a valid frame ptr immediately after `entry`, e.g. `0xc58c`.) The Step-1
>   `call8` test is rewritten: windowbase unchanged after `call8`; +2 only after the
>   subsequent `entry`.
> - **This firmware has ZERO `s32e`/`l32e`/`rfwo`/`rfwu`** (verified across 33k
>   listing lines); it software-spills via `rotw`+`s32i.n` (`_xtos_spill_windows`
>   @0x0e098). So **`s32e`/`l32e` are DROPPED** (no oracle, never executed). The
>   architectural overflow(at `entry`)/underflow(at `retw`) **raise+vector mechanism
>   is still built** (faithful CPU behavior: cause+EPC, `pc=vecbase+offset`,
>   `Step::Exception`), unit-proven with synthetic WINDOWSTART + a stub handler.
>   Whether THIS firmware ever fires it vs. proactively software-spilling is the
>   hypothesis **M1.7's observation settles** (instrument-first).
> - `Cpu.vecbase` is a directly-settable field for now; full `wsr`/`rsr`/`wur`
>   special-register decode+exec is deferred to the **M1.7 derive-loop** (add when
>   the real boot executes them), not half-built in the crux.

**Files:**
- Modify: `src/firmware/xtensa/interp.rs` (extend `step`; add window-exception logic)
- Modify: `src/firmware/xtensa/decode.rs` (add `retw`/`retw.n`/`callx8`/`s32e`/`l32e` vectors + arms as encountered)
- Modify: `src/firmware/xtensa/regfile.rs` if the exception path needs `WINDOWSTART` helpers
- Test: inline tests in `interp.rs`

**Interfaces:**
- Produces (extends M1.4): `Cpu::step` now executes `entry` (allocate frame, rotate window per the call size), `call8`/`callx8` (rotate +2 quads, set return PC in `a0` per Xtensa convention), `retw`/`retw.n` (restore window). On a window overflow/underflow condition, `step` raises: sets exception cause + `EPC`, sets `pc` to the appropriate window-exception vector, returns `Step::Exception { cause, pc }` (the firmware's own handler, which uses `s32e`/`l32e`, then runs as ordinary instructions).
- `pub struct Cpu` gains `pub vecbase: u32` (set by `wsr VECBASE` during boot, or a boot-derived default).
- Consumed by: `mod.rs` (M1.7).

**Derivation:** windowed-op encodings come from the Ghidra `listing.txt` oracle (lx106 cannot decode them). The window-exception *mechanism* (overflow on `call` when `WINDOWSTART` marks the target window live; underflow on `retw`) follows the Xtensa ISA windowed-register option; vector offsets are read from the firmware's `wsr VECBASE` + the standard window-vector layout, confirmed by M1.7 execution.

- [ ] **Step 1: Write failing tests for the window mechanics**

```rust
#[cfg(test)]
mod window_tests {
    use super::*;
    use crate::firmware::mmio::Bus;

    #[test]
    fn entry_allocates_frame_and_sets_stack() {
        // entry a1, 32: rotates nothing by itself (rotation is on call), but
        // records the frame. After entry, a1 (stack) is decremented by 32.
        let rom = vec![0x36, 0x41, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(1, 0x0000_1000); // sp
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(1), 0x0000_1000 - 32);
    }

    #[test]
    fn call8_rotates_window_and_saves_return() {
        // call8 target: windowbase += 2, and the caller's next-pc is recorded
        // in the callee a0 top bits (call size in bits 30:31).
        let rom = vec![0xe5, 0x20, 0xf9]; // call8 (target resolved from pc)
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0x3a034);
        let wb0 = cpu.regs.windowbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.windowbase, (wb0 + 2) % 16);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib firmware::xtensa::interp`
Expected: FAIL.

- [ ] **Step 3: Implement `entry`/`call8`/`callx8`/`retw` and the window-exception raise/vector path**

Extend `step` with the windowed arms and overflow/underflow detection, per the interface. Add `s32e`/`l32e` decode+exec (used only inside the firmware's window handlers). Implement `wsr VECBASE`/`wsr PS`/`rsr` for the special registers the handlers touch. Each arm derived from the oracle + ISA windowed-option reference; each covered by a test.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib firmware::xtensa::interp`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/firmware/
git commit -m "$(cat <<'EOF'
feat(#140): firmware M1.5 -- windowed calls + window exceptions

entry/call8/callx8/retw with WINDOWBASE rotation, and the window
overflow/underflow raise-and-vector path (firmware's own s32e/l32e handlers run
as ordinary instructions). The load-bearing windowed-ABI crux of M1.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 7: M1.6 -- System-aperture stubs + spin-detection

**Files:**
- Create: `src/firmware/sysstub.rs`
- Modify: `src/firmware/mmio.rs` (route `Region::System` through `SysStub`)
- Modify: `src/firmware/mod.rs` (add `mod sysstub;`)
- Test: inline tests in `sysstub.rs`

**Interfaces:**
- Produces:
  - `pub struct SysStub { log: Vec<(u32, u32)>, read_counts: HashMap<u32, u32>, spin_threshold: u32 }`
  - `impl SysStub { pub fn new() -> Self; pub fn read(&mut self, addr: u32) -> u32; pub fn write(&mut self, addr: u32, v: u32); pub fn spinning(&self) -> Option<u32>; pub fn accesses(&self) -> &[(u32, u32)]; }`
  - `read` returns a benign value (0) and increments a per-address counter; `spinning()` returns `Some(addr)` if any address was read more than `spin_threshold` times consecutively without an intervening different access -- the "waiting on unmodeled state" signal (spec section 8).
- Consumed by: `Bus` (M1.3, `Region::System` path), `FirmwareProcessor` (M1.7 idle/hang diagnosis).

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_reads_return_zero_and_log() {
        let mut s = SysStub::new();
        assert_eq!(s.read(0xf7000000), 0);
        assert_eq!(s.accesses().len(), 1);
    }

    #[test]
    fn detects_a_tight_read_spin() {
        let mut s = SysStub::new();
        for _ in 0..(s_threshold() + 1) {
            s.read(0x03001000);
        }
        assert_eq!(s.spinning(), Some(0x03001000));
    }

    #[test]
    fn interleaved_reads_are_not_a_spin() {
        let mut s = SysStub::new();
        for _ in 0..100 {
            s.read(0x03001000);
            s.read(0x03002000);
        }
        assert_eq!(s.spinning(), None);
    }

    fn s_threshold() -> u32 { SysStub::new_threshold() }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib firmware::sysstub`
Expected: FAIL (undefined).

- [ ] **Step 3: Implement `sysstub.rs` and wire it into `Bus`**

Implement `SysStub` with a "consecutive-read" spin counter (reset on a different address), a `pub fn new_threshold() -> u32` exposing the default (e.g. 1024), the access log, and benign zero reads. In `mmio.rs`, give `Bus` a `SysStub` field and route `Region::System` loads/stores through it. Add a `Bus::sysstub(&self) -> &SysStub` accessor for M1.7.

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test --lib firmware::sysstub`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/firmware/
git commit -m "$(cat <<'EOF'
feat(#140): firmware M1.6 -- system-aperture stubs + spin-detection

SysStub answers off-array SMN/NoC reads with benign zeros, logs every access,
and flags a tight consecutive-read spin as the "waiting on unmodeled state"
signal (spec section 8). Wired into Bus's System region.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

### Task 8: M1.7 -- FirmwareProcessor boot-to-idle (integration + entry pinning)

**Files:**
- Modify: `src/firmware/mod.rs` (add `FirmwareProcessor`, `IdleReport`, the run loop, re-exports)
- Test: a **firmware-gated** integration test in `src/firmware/mod.rs` (`#[cfg(test)] mod boot_tests`)

**Interfaces:**
- Produces:
  - `pub struct FirmwareProcessor { pub cpu: Cpu, pub bus: Bus, pub entry: u32 }`
  - `pub struct IdleReport { pub reached_idle: bool, pub instrs_executed: u64, pub wait_reason: Option<WaitReason>, pub funcs_entered: Vec<(u32, String)>, pub unresolved_spin: Option<u32>, pub last_pc: u32 }`
  - `impl FirmwareProcessor { pub fn load(image: FirmwareImage, entry: u32) -> Self; pub fn boot_to_idle(&mut self, max_instrs: u64) -> IdleReport; }`
  - `boot_to_idle` steps the CPU until: (a) `Step::Wait(_)` at a stable PC (idle -- success), (b) `SysStub::spinning()` fires (`unresolved_spin`), (c) `Step::Unknown` (unimplemented op -- report `last_pc` + word), or (d) `max_instrs` exceeded. `funcs_entered` records `call8`/`callx8` targets that match the recovered symbol map (loaded from `build/experiments/firmware-re/symbols.txt` if present).
- Consumes: everything above.

**Entry-point pinning (derive, do not hardcode):** the reset/boot entry is unknown a priori. `load(image, entry)` takes the candidate; the test tries the boot-vector candidate and asserts the interpreter decodes a coherent stream (reaches the `witlb`/`wdtlb` TLB-setup sequence) rather than immediately hitting `Unknown`. The correct entry is the one that runs the TLB setup then proceeds; a wrong entry desyncs into `Unknown` within a few instructions.

**MMU decision (resolve here):** instrument `witlb`/`wdtlb` (Task M1.4 logs them). If the firmware's subsequent accesses stay within the identity-consistent ranges we already back (base-0 / `0x08b00000` / `0x27` / `0x04`), a flat physical model suffices and `mmu.rs` is unnecessary; if it depends on the virtual `~0x40000000` remap, add the minimal translation. Record the finding in the test output / a follow-up note.

- [ ] **Step 1: Write the firmware-gated integration test**

```rust
#[cfg(test)]
mod boot_tests {
    use super::*;
    use std::path::Path;

    fn firmware_path() -> Option<std::path::PathBuf> {
        // Env override first, then the known repo-relative download location.
        if let Ok(p) = std::env::var("XDNA_FIRMWARE") {
            let p = std::path::PathBuf::from(p);
            return p.exists().then_some(p);
        }
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin");
        p.exists().then_some(p)
    }

    #[test]
    fn boots_real_firmware_to_idle() {
        let Some(path) = firmware_path() else {
            eprintln!("skip: firmware binary not present (set XDNA_FIRMWARE)");
            return;
        };
        let raw = std::fs::read(&path).expect("read firmware");
        let img = FirmwareImage::parse(&raw).expect("parse");

        // Candidate entry: the boot/TLB region (derive; see task notes).
        let entry = 0x334; // candidate -- adjust per coherence check
        let mut proc = FirmwareProcessor::load(img, entry);
        let report = proc.boot_to_idle(5_000_000);

        // Primary success: reached a stable idle wait, no unresolved spin,
        // no unimplemented opcode.
        assert!(report.unresolved_spin.is_none(),
            "unresolved spin at {:#x}", report.unresolved_spin.unwrap());
        assert!(report.reached_idle,
            "did not reach idle; last_pc={:#x} instrs={}", report.last_pc, report.instrs_executed);
    }
}
```

- [ ] **Step 2: Run to verify it fails (or skips)**

Run: `cargo test --lib firmware::boot_tests -- --nocapture`
Expected: FAIL (`FirmwareProcessor` undefined), or a clean skip line if the binary is absent on this machine (it is present on the dev box).

- [ ] **Step 3: Implement `FirmwareProcessor` and the run loop**

Implement `load` (build `Bus` from `image.bytes()`, `Cpu::new(entry)`), and `boot_to_idle` (the step loop with the four termination conditions, `funcs_entered` symbol lookup, spin check via `bus.sysstub().spinning()`). Optionally load `symbols.txt` for names; absent file -> empty names, not an error.

- [ ] **Step 4: Iterate entry-point + unimplemented opcodes to reach idle**

Run: `cargo test --lib firmware::boot_tests -- --nocapture`
This is the **observation milestone**. Expected iteration loop:
- If `Step::Unknown` at some pc: disassemble that pc with the oracle, add the decode vector + arm (Task M1.1/M1.4/M1.5 pattern), re-run.
- If a spin fires: inspect the flagged address -- it is an off-array status bit the firmware waits on; decide (benign constant vs. a modeled transition) and adjust `SysStub`.
- If entry desyncs immediately: adjust the `entry` candidate.
Repeat until `reached_idle == true`. Record: the pinned entry, the MMU finding, the `funcs_entered` sequence (cross-checked against the symbol map), and the idle `wait_reason` (this is the H1 input for M2).

- [ ] **Step 5: Commit**

```bash
git add src/firmware/
git commit -m "$(cat <<'EOF'
feat(#140): firmware M1.7 -- boot the real firmware to command-loop idle

FirmwareProcessor::boot_to_idle runs npu.dev.sbin from its pinned entry to a
stable idle wait, reporting instrs, funcs entered (vs symbol map), the idle
wait reason (H1 input for M2), and any unresolved system-aperture spin.
Firmware-gated test skips cleanly without the binary. Resolves the MMU
flat-vs-remap question and the entry point empirically.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh
EOF
)"
```

---

## Self-Review

**Spec coverage (M0+M1 rows of spec section 9, plus the modules of section 3):**
- M0 image parse -> Task M0. [covered]
- `src/firmware/` layout (`mod`, `error`, `image`, `mmio`, `sysstub`, `xtensa/{decode,regfile,interp}`) -> Tasks M0-M1.7. `xtensa/exc.rs` from the spec is folded into `interp.rs` (M1.5) -- the window-exception logic is small and inseparable from the step loop; noted deviation, keeps related code together per the file-structure rule. `xtensa/mmu.rs` is created only if M1.7 shows the remap is needed (deferred by design, not omitted).
- Interpreter (decode + windowed ABI) -> M1.1, M1.2, M1.4, M1.5. [covered]
- MMIO bridge memory model + array/mailbox stubbed this phase -> M1.3. Real `DeviceState` routing is M2 (out of scope, correctly deferred). [covered as scoped]
- Boot + stub layer + spin-detection -> M1.6, M1.7. [covered]
- Execution/timing model -> intentionally NOT implemented here; M1.7 only *observes* the idle wait reason (the H1 input). The timing accounting is M2+ per the spec's instrument-first decision. [correctly deferred]
- Validation: symbol-map landmark cross-check -> M1.7 `funcs_entered`. Differential-vs-HW timing + 112-cy reconciliation -> M2+ (out of scope). [covered as scoped]

**Placeholder scan:** The `todo!()` in Task M1.1 Step 3 and the prose "implementer fills..." in M1.3/M1.4/M1.5 are **derive-loop scaffolding, not placeholders**: each is backed by concrete derived test vectors (the real spec of the behavior) and a named oracle to check against. Writing hand-transcribed full ISA semantics inline would violate DERIVE-FROM-THE-TOOLCHAIN and risk shipping wrong bit-math as authoritative. The tests are complete and concrete; the implementation is a mechanical derive-and-verify loop. This is the honest structure for an ISA interpreter.

**Type consistency:** `Bus::new(rom: Vec<u8>)`, `Cpu::new(entry: u32)`, `Cpu::step(&mut self, bus: &mut Bus) -> Step`, `RegFile::{read_ar,write_ar,rotate,phys}`, `decode(bytes,pc) -> Decoded{op,len}`, `SysStub::{read,write,spinning,accesses,new_threshold}`, `FirmwareProcessor::{load,boot_to_idle}` / `IdleReport` -- names consistent across producing and consuming tasks.

**Known iteration point:** M1.7 Step 4 is an explicit derive-loop (new opcodes surface as `Step::Unknown`, get vectored + implemented). This is expected and bounded by the firmware's actual boot-path instruction set, not open-ended.
