# Memory Watch Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an environment-variable-driven memory watch mechanism that logs every access to specified address ranges, plus compile-out trace logging in release.

**Architecture:** A `debug::watch` module with global `OnceLock` storage, parsed from `XDNA_EMU_WATCH` at startup. Inline `is_watched()` check at each memory access point (zero cost when unset). Log at INFO level for reliable visibility.

**Tech Stack:** Rust std `OnceLock`, `log` crate with `release_max_level_info` feature.

---

### Task 1: Clean Up Temporary Debug Instrumentation

**Files:**
- Modify: `src/interpreter/execute/memory/mod.rs`

Remove all temporary watch-point code added during the 2026-04-11
debugging session. This includes the static COUNTER, the WATCH-LD /
WATCH-ST / WATCH-VLDA / WATCH-ENTRY / MEM-EXEC log lines, and the
hardcoded address-range checks.

- [ ] **Step 1: Remove temporary instrumentation from memory/mod.rs**

Revert the `execute()` function's catch-all counter:

```rust
// REMOVE the entire block starting with:
//   static COUNTER: std::sync::atomic::AtomicU64 ...
//   let count = COUNTER.fetch_add(...);
//   if count < 5 || count % 500 == 0 { log::info!("[MEM-EXEC] ..."); }
```

Revert `execute_load()` -- remove the `[WATCH-LD]` block (the one using
`log::info!` with hardcoded address ranges 0x0400, 0xC000, 0x8000).
Restore the original `log::trace!` watch on out_buff_0 if desired, or
remove it too (the new watch mechanism replaces it).

Revert `execute_store()` -- remove the `[WATCH-ST]` block.

Revert `execute_vector_load_a()` -- remove the `[WATCH-VLDA]` block.

- [ ] **Step 2: Build and verify no regressions**

Run: `cargo build && TMPDIR=/tmp/claude-1000 cargo test --lib -q`
Expected: Clean build, all tests pass.

- [ ] **Step 3: Commit**

```
git add src/interpreter/execute/memory/mod.rs
git commit -m "chore: remove temporary debug instrumentation from memory unit"
```

---

### Task 2: Create debug::watch Module

**Files:**
- Create: `src/debug/mod.rs`
- Create: `src/debug/watch.rs`
- Modify: `src/lib.rs` (add `pub mod debug;`)

- [ ] **Step 1: Write tests for WatchRange and parse logic**

Create `src/debug/watch.rs` with tests at the bottom:

```rust
//! Memory watch mechanism for debugging.
//!
//! Set `XDNA_EMU_WATCH=0xC000:40,0x428:40` to log every memory access
//! (core loads/stores, DMA reads/writes) to the specified address ranges.
//! Zero overhead when not configured.

use std::sync::OnceLock;

/// A watched address range.
#[derive(Debug, Clone)]
pub struct WatchRange {
    pub start: u64,
    pub len: usize,
}

impl WatchRange {
    /// Check if an access at `addr` of `size` bytes overlaps this range.
    #[inline]
    pub fn overlaps(&self, addr: u64, size: usize) -> bool {
        let end = self.start + self.len as u64;
        let access_end = addr + size as u64;
        self.start < access_end && addr < end
    }
}

/// Parse an XDNA_EMU_WATCH value into watch ranges.
///
/// Format: comma-separated `address:bytes` pairs.
/// Address is hex (0x prefix optional). Bytes defaults to 4 if omitted.
///
/// Examples:
/// - `0xC000:40` -- 40 bytes starting at 0xC000
/// - `C000` -- 4 bytes (one word) at 0xC000
/// - `0xC000:40,0x428:40` -- two ranges
pub fn parse_watch_env(value: &str) -> Vec<WatchRange> {
    let mut ranges = Vec::new();
    for part in value.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (addr_str, len) = if let Some((a, l)) = part.split_once(':') {
            let len: usize = l.parse().unwrap_or(4);
            (a, len)
        } else {
            (part, 4)
        };
        let addr_str = addr_str.trim_start_matches("0x").trim_start_matches("0X");
        if let Ok(addr) = u64::from_str_radix(addr_str, 16) {
            ranges.push(WatchRange { start: addr, len });
        } else {
            log::warn!("XDNA_EMU_WATCH: invalid address '{}', skipping", part);
        }
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlap_exact() {
        let r = WatchRange { start: 0xC000, len: 40 };
        assert!(r.overlaps(0xC000, 4));   // start
        assert!(r.overlaps(0xC024, 4));   // last word
        assert!(!r.overlaps(0xC028, 4));  // just past end
        assert!(!r.overlaps(0xBFFC, 4));  // just before start
    }

    #[test]
    fn test_overlap_partial() {
        let r = WatchRange { start: 0xC000, len: 40 };
        // Access that straddles the boundary
        assert!(r.overlaps(0xBFFE, 4));   // starts before, overlaps
        assert!(r.overlaps(0xC026, 4));   // starts inside, extends past
    }

    #[test]
    fn test_parse_basic() {
        let ranges = parse_watch_env("0xC000:40");
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0xC000);
        assert_eq!(ranges[0].len, 40);
    }

    #[test]
    fn test_parse_no_prefix() {
        let ranges = parse_watch_env("C000:40");
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0xC000);
    }

    #[test]
    fn test_parse_default_len() {
        let ranges = parse_watch_env("0xC000");
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start, 0xC000);
        assert_eq!(ranges[0].len, 4);
    }

    #[test]
    fn test_parse_multiple() {
        let ranges = parse_watch_env("0xC000:40,0x428:40,0x400:40");
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].start, 0xC000);
        assert_eq!(ranges[1].start, 0x428);
        assert_eq!(ranges[2].start, 0x400);
    }

    #[test]
    fn test_parse_empty_and_whitespace() {
        let ranges = parse_watch_env(" 0xC000:40 , , 0x428:40 ");
        assert_eq!(ranges.len(), 2);
    }

    #[test]
    fn test_parse_invalid_skipped() {
        let ranges = parse_watch_env("0xC000:40,notahex:10,0x428:40");
        assert_eq!(ranges.len(), 2);
    }
}
```

- [ ] **Step 2: Run tests to verify parsing logic**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib debug::watch -- -v`
Expected: All 7 tests pass.

- [ ] **Step 3: Add global state and init/query functions**

Add above the `#[cfg(test)]` block in `src/debug/watch.rs`:

```rust
static WATCHES: OnceLock<Vec<WatchRange>> = OnceLock::new();

/// Initialize the watch mechanism from the XDNA_EMU_WATCH environment variable.
///
/// Call once at emulator startup. Safe to call multiple times (subsequent
/// calls are no-ops due to OnceLock).
pub fn init() {
    WATCHES.get_or_init(|| {
        match std::env::var("XDNA_EMU_WATCH") {
            Ok(val) if !val.is_empty() => {
                let ranges = parse_watch_env(&val);
                if !ranges.is_empty() {
                    log::info!(
                        "[WATCH] Initialized {} watch range(s) from XDNA_EMU_WATCH",
                        ranges.len()
                    );
                    for r in &ranges {
                        log::info!("[WATCH]   0x{:X}..0x{:X} ({} bytes)",
                            r.start, r.start + r.len as u64, r.len);
                    }
                }
                ranges
            }
            _ => Vec::new(),
        }
    });
}

/// Check if an address range overlaps any watch range.
///
/// Returns false immediately when no watches are configured (fast path).
#[inline]
pub fn is_watched(addr: u64, len: usize) -> bool {
    match WATCHES.get() {
        Some(ranges) if !ranges.is_empty() => {
            ranges.iter().any(|r| r.overlaps(addr, len))
        }
        _ => false,
    }
}

/// Log a core load that hit a watch range.
pub fn log_core_load(cycle: u64, pc: u32, addr: u64, value: u32, dest: &str) {
    log::info!(
        "[WATCH] cycle={} CORE-LD  pc=0x{:03X} addr=0x{:05X} value=0x{:08X} -> {}",
        cycle, pc, addr, value, dest
    );
}

/// Log a core store that hit a watch range.
pub fn log_core_store(cycle: u64, pc: u32, addr: u64, value: u32) {
    log::info!(
        "[WATCH] cycle={} CORE-ST  pc=0x{:03X} addr=0x{:05X} value=0x{:08X}",
        cycle, pc, addr, value
    );
}

/// Log a DMA write (S2MM) that hit a watch range.
pub fn log_dma_write(cycle: u64, col: u8, row: u8, addr: u64, value: u32, channel: &str) {
    log::info!(
        "[WATCH] cycle={} DMA-WR   tile=({},{}) addr=0x{:05X} value=0x{:08X} ch={}",
        cycle, col, row, addr, value, channel
    );
}

/// Log a DMA read (MM2S) that hit a watch range.
pub fn log_dma_read(cycle: u64, col: u8, row: u8, addr: u64, value: u32, channel: &str) {
    log::info!(
        "[WATCH] cycle={} DMA-RD   tile=({},{}) addr=0x{:05X} value=0x{:08X} ch={}",
        cycle, col, row, addr, value, channel
    );
}
```

- [ ] **Step 4: Create module files**

Create `src/debug/mod.rs`:
```rust
//! Debug instrumentation for the emulator.
//!
//! Provides runtime-configurable tools for investigating emulator behavior:
//! - `watch`: Memory address watch mechanism (XDNA_EMU_WATCH)

pub mod watch;
```

Add to `src/lib.rs` after the existing module declarations:
```rust
pub mod debug;
```

- [ ] **Step 5: Build and test**

Run: `cargo build && TMPDIR=/tmp/claude-1000 cargo test --lib debug:: -- -v`
Expected: Clean build, all watch tests pass.

- [ ] **Step 6: Commit**

```
git add src/debug/ src/lib.rs
git commit -m "feat: add debug::watch module with XDNA_EMU_WATCH support"
```

---

### Task 3: Wire Up Initialization

**Files:**
- Modify: `crates/xdna-emu-ffi/src/lib.rs:126` (after env_logger::try_init)
- Modify: `src/main.rs:17` (after env_logger::init)

- [ ] **Step 1: Add init call in FFI crate**

In `crates/xdna-emu-ffi/src/lib.rs`, after the `env_logger::try_init()`
call (line 126), add:

```rust
xdna_emu::debug::watch::init();
```

- [ ] **Step 2: Add init call in main binary**

In `src/main.rs`, after the `env_logger::init()` call (line 17), add:

```rust
xdna_emu::debug::watch::init();
```

- [ ] **Step 3: Build all workspace members and test**

Run: `cargo build && TMPDIR=/tmp/claude-1000 cargo test --lib -q`
Expected: Clean build, all tests pass.

- [ ] **Step 4: Commit**

```
git add crates/xdna-emu-ffi/src/lib.rs src/main.rs
git commit -m "feat: initialize watch mechanism at emulator startup"
```

---

### Task 4: Instrument Core Load Paths

**Files:**
- Modify: `src/interpreter/execute/memory/mod.rs`

Instrument the three core load functions: `execute_load`,
`execute_vector_load_a`, `execute_vector_load_b`. Each needs a watch
check after the memory read, before the deferred register write.

- [ ] **Step 1: Instrument execute_load**

In `execute_load()`, after the `read_memory` call and value extension
logic (around the existing `Self::write_dest_with_latency` call), add
the watch check:

```rust
if crate::debug::watch::is_watched(addr as u64, width.bytes()) {
    let dest_str = op.dest.as_ref()
        .map(|d| format!("{:?}", d))
        .unwrap_or_default();
    crate::debug::watch::log_core_load(
        ctx.cycles, ctx.pc(), addr as u64, value as u32, &dest_str,
    );
}
```

Place this just before `Self::write_dest_with_latency(...)`.

- [ ] **Step 2: Instrument execute_vector_load_a**

In `execute_vector_load_a()`, after `read_vector_from_memory`, add:

```rust
if crate::debug::watch::is_watched(addr as u64, if op.mem_width == MemWidth::QuadWord { 16 } else { 32 }) {
    let dest_str = op.dest.as_ref()
        .map(|d| format!("{:?}", d))
        .unwrap_or_default();
    // Log first word as representative value
    crate::debug::watch::log_core_load(
        ctx.cycles, ctx.pc(), addr as u64, vec_data[0], &dest_str,
    );
}
```

Place this after `vec_data` is computed but before the queue call.

- [ ] **Step 3: Instrument execute_vector_load_b**

Same pattern as execute_vector_load_a. Find the `read_vector_from_memory`
call in `execute_vector_load_b()` and add the same watch check.

- [ ] **Step 4: Build and test**

Run: `cargo build && TMPDIR=/tmp/claude-1000 cargo test --lib -q`
Expected: Clean build, all tests pass (watches not configured = no output).

- [ ] **Step 5: Commit**

```
git add src/interpreter/execute/memory/mod.rs
git commit -m "feat: instrument core load paths with watch mechanism"
```

---

### Task 5: Instrument Core Store Paths

**Files:**
- Modify: `src/interpreter/execute/memory/mod.rs`

Instrument `execute_store` and `execute_vector_store`.

- [ ] **Step 1: Instrument execute_store (scalar)**

In `execute_store()`, in the non-partial-word branch, after
`get_store_value` and before `write_memory`, add:

```rust
if crate::debug::watch::is_watched(addr as u64, width.bytes()) {
    crate::debug::watch::log_core_store(
        ctx.cycles, ctx.pc(), addr as u64, value as u32,
    );
}
```

- [ ] **Step 2: Instrument execute_vector_store**

In `execute_vector_store()`, after the data and address are resolved
but before `write_vector_to_memory`, add:

```rust
if crate::debug::watch::is_watched(addr as u64, if op.mem_width == MemWidth::QuadWord { 16 } else { 32 }) {
    // Log first word as representative value
    if let Some(ref data) = vec_data {
        crate::debug::watch::log_core_store(
            ctx.cycles, ctx.pc(), addr as u64, data[0],
        );
    }
}
```

Find the correct variable names by reading the function.

- [ ] **Step 3: Build and test**

Run: `cargo build && TMPDIR=/tmp/claude-1000 cargo test --lib -q`
Expected: Clean build, all tests pass.

- [ ] **Step 4: Commit**

```
git add src/interpreter/execute/memory/mod.rs
git commit -m "feat: instrument core store paths with watch mechanism"
```

---

### Task 6: Instrument DMA Paths

**Files:**
- Modify: `src/device/dma/engine/stepping.rs`

Instrument the four DMA transfer functions: `transfer_s2mm`,
`transfer_mm2s`, `transfer_stream_to_host`, `transfer_host_to_stream`.

- [ ] **Step 1: Instrument transfer_s2mm (DMA write to tile memory)**

In the word loop inside `transfer_s2mm()`, after writing each word to
tile memory, add:

```rust
if crate::debug::watch::is_watched(
    (offset + bytes_written) as u64, 4,
) {
    crate::debug::watch::log_dma_write(
        self.current_cycle, self.col, self.row,
        (offset + bytes_written) as u64, word,
        &format!("S2MM{}", channel),
    );
}
```

Place inside the `for word_idx` loop, after the `data[...] = word_bytes[j]`
writes.

- [ ] **Step 2: Instrument transfer_mm2s (DMA read from tile memory)**

In the word loop inside `transfer_mm2s()`, after reading each word from
tile memory, add:

```rust
if crate::debug::watch::is_watched(word_offset as u64, 4) {
    crate::debug::watch::log_dma_read(
        self.current_cycle, self.col, self.row,
        word_offset as u64, word,
        &format!("MM2S{}", channel),
    );
}
```

Place inside the `for i in 0..word_count` loop, after `word` is computed.

- [ ] **Step 3: Instrument transfer_stream_to_host (shim DMA write)**

In `transfer_stream_to_host()`, after `host_memory.write_u32(word_addr, word)`:

```rust
if crate::debug::watch::is_watched(word_addr, 4) {
    crate::debug::watch::log_dma_write(
        self.current_cycle, self.col, self.row,
        word_addr, word,
        &format!("S2MM{}", channel),
    );
}
```

- [ ] **Step 4: Instrument transfer_host_to_stream (shim DMA read)**

In `transfer_host_to_stream()`, after `host_memory.read_u32(word_addr)`:

```rust
if crate::debug::watch::is_watched(word_addr, 4) {
    crate::debug::watch::log_dma_read(
        self.current_cycle, self.col, self.row,
        word_addr, word,
        &format!("MM2S{}", channel),
    );
}
```

- [ ] **Step 5: Build and test**

Run: `cargo build && TMPDIR=/tmp/claude-1000 cargo test --lib -q`
Expected: Clean build, all tests pass.

- [ ] **Step 6: Commit**

```
git add src/device/dma/engine/stepping.rs
git commit -m "feat: instrument DMA transfer paths with watch mechanism"
```

---

### Task 7: Add Release Log Level Gating

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add release_max_level_info feature to log dependency**

In the root `Cargo.toml`, change:
```toml
log = "0.4"
```
to:
```toml
log = { version = "0.4", features = ["release_max_level_info"] }
```

This compiles out `trace!()` and `debug!()` in release builds. Debug
builds retain full logging, toggled at runtime via RUST_LOG.

- [ ] **Step 2: Build both profiles**

Run: `cargo build && cargo build --release`
Expected: Both build cleanly. Release build may show fewer warnings
(trace-level log statements become no-ops).

- [ ] **Step 3: Commit**

```
git add Cargo.toml
git commit -m "perf: compile out trace/debug logging in release builds"
```

---

### Task 8: Smoke Test with Real Bridge Test

**Files:** None (validation only)

- [ ] **Step 1: Rebuild the FFI plugin**

Run: `nice -n 19 cargo build -p xdna-emu-ffi`

This ensures the debug .so has the watch mechanism.

- [ ] **Step 2: Run sliding_window chess test WITH watches**

```bash
cd /home/triple/npu-work/mlir-aie/build/test/npu-xrt/dynamic_object_fifo/sliding_window/chess
XDNA_EMU=debug XDNA_EMU_WATCH=0xC000:4,0x428:4,0x400:4 RUST_LOG=info \
  timeout 60 ./test.exe 2>/tmp/claude-1000/watch_smoke.log
grep "\[WATCH\]" /tmp/claude-1000/watch_smoke.log | head -20
```

Expected: `[WATCH]` lines showing cycle-correlated CORE-LD, CORE-ST,
DMA-WR, and DMA-RD operations at the watched addresses.

- [ ] **Step 3: Run sliding_window chess test WITHOUT watches (performance baseline)**

```bash
XDNA_EMU=debug RUST_LOG=info timeout 60 ./test.exe 2>/dev/null
```

Expected: Completes without extra output. No performance regression.

- [ ] **Step 4: Run full library test suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -q`
Expected: All tests pass.

- [ ] **Step 5: Update CLAUDE.md Build Commands section**

Add a note clarifying the build/plugin relationship:

```markdown
# Building for bridge tests (plugin path)
cargo build -p xdna-emu-ffi     # Builds the debug .so loaded by XRT plugin
./scripts/rebuild-plugin.sh     # Builds release .so + C++ plugin + installs
```

- [ ] **Step 6: Commit**

```
git add CLAUDE.md
git commit -m "docs: clarify FFI build commands for bridge test workflow"
```
