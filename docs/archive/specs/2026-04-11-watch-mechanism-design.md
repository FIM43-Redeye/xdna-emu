# Memory Watch Mechanism Design

**Date**: 2026-04-11
**Status**: Approved

## Problem

Debugging emulator memory issues requires correlating core loads/stores
with DMA writes across cycle boundaries. Current approach (editing source
to add ad-hoc log statements, rebuilding, filtering RUST_LOG) is slow,
error-prone, and produces either too much or too little output.

## Solution

An environment-variable-driven watch mechanism that logs every memory
access to specified address ranges at INFO level. Zero overhead when
not configured.

### Environment Variable

`XDNA_EMU_WATCH` -- comma-separated `address:bytes` pairs.

```
XDNA_EMU_WATCH=0xC000:40,0x428:40,0x400:40
XDNA_EMU_WATCH=0xC000          # single word (4 bytes default)
XDNA_EMU_WATCH=C000:40,428:40  # 0x prefix optional
```

### Log Format

```
[WATCH] cycle=283 CORE-LD  pc=0x1A0 addr=0x0C000 value=0x00000001 -> ScalarReg(24)
[WATCH] cycle=283 CORE-ST  pc=0x1B4 addr=0x00400 value=0x00000005
[WATCH] cycle=285 DMA-WR   tile=(0,2) addr=0x0C000 value=0x00000004 ch=S2MM0
[WATCH] cycle=285 DMA-RD   tile=(0,2) addr=0x00400 value=0x00000005 ch=MM2S2
```

### Architecture

**New module**: `src/debug/watch.rs` + `src/debug/mod.rs`

- `WatchRange { start: u64, len: usize }` -- a watched address range
- `init()` -- parses `XDNA_EMU_WATCH`, stores in global `OnceLock`
- `is_watched(addr: u64, len: usize) -> bool` -- inline; returns false
  immediately when no watches configured (single pointer check)
- `log_core_load(cycle, pc, addr, value, dest)` -- format + emit
- `log_core_store(cycle, pc, addr, value)` -- format + emit
- `log_dma_write(cycle, col, row, addr, value, channel)` -- format + emit
- `log_dma_read(cycle, col, row, addr, value, channel)` -- format + emit

**Global state**: `static WATCHES: OnceLock<Vec<WatchRange>>`

### Intercept Points

| Point | File | Catches |
|-------|------|---------|
| Core scalar load | `execute_load()` in memory/mod.rs | LDA to scalar regs |
| Core vector load A | `execute_vector_load_a()` in memory/mod.rs | LDA/VLDA via LoadA slot |
| Core vector load B | `execute_vector_load_b()` in memory/mod.rs | LDB/VLDB via LoadB slot |
| Core scalar store | `execute_store()` in memory/mod.rs | ST from scalar regs |
| Core vector store | `execute_vector_store()` in memory/mod.rs | VST from vector regs |
| DMA S2MM tile write | `transfer_s2mm()` in stepping.rs | DMA writing tile memory |
| DMA MM2S tile read | `transfer_mm2s()` in stepping.rs | DMA reading tile memory |
| DMA S2MM host write | `transfer_stream_to_host()` in stepping.rs | DMA writing host DDR |
| DMA MM2S host read | `transfer_host_to_stream()` in stepping.rs | DMA reading host DDR |

Each point checks against the address it naturally works with (tile-local
offsets for tile DMA, host addresses for shim DMA). No address translation.

### Address Matching

Watch addresses are stored as u64. The `is_watched` check tests overlap:
`watch.start < addr + len && addr < watch.start + watch.len`. This handles
both 16-bit tile-local and 64-bit host addresses transparently.

## Companion Changes

### Release Log Level Gating

Add `release_max_level_info` feature to the `log` dependency:

```toml
log = { version = "0.4", features = ["release_max_level_info"] }
```

This compiles out `trace!()` and `debug!()` in release builds. Debug
builds retain full logging, toggled at runtime via `RUST_LOG`.

### Cleanup

Remove all temporary watch-point instrumentation added during the
2026-04-11 debugging session (static counters, hardcoded address checks
in memory/mod.rs).

### Build Documentation

Add note to CLAUDE.md Build Commands clarifying that `cargo build` from
the xdna-emu root builds all workspace members including the FFI crate.
The `scripts/rebuild-plugin.sh` script remains the canonical way to
build + install the plugin for bridge tests.

## Future Work

- **Register watches** (`XDNA_EMU_WATCH_REG=r24,p0`): log every
  read/write of specified registers. Separate env var, same module.
- **Expensive debug feature**: `deep-debug` Cargo feature for debug
  paths that do expensive computation beyond what `trace!()` covers.
  Default off, opt-in when needed.
