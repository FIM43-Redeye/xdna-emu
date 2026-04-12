# Adaptive Timeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed cycle limit with monotonic progress detection so heavy-scalar tests complete while true deadlocks/livelocks are caught quickly.

**Architecture:** Track two monotonic indicators (total DMA bytes transferred, cumulative lock release count) in the coordinator. If neither advances for `stall_threshold` cycles, declare stall. The existing `max_cycles` becomes a high safety cap. The test runner's `StallDetector` is updated to use lock releases instead of instruction count.

**Tech Stack:** Rust, no new dependencies

**Spec:** `docs/superpowers/specs/2026-04-11-adaptive-timeout-design.md`

---

### Task 1: Add lock release counter to tile state

**Files:**
- Modify: `src/device/tile/mod.rs`
- Test: `src/device/tile/mod.rs` (existing test module)

The tile already has `resolve_lock_requests()` (line 699) which returns granted operations. We need a cumulative counter that increments on each granted release.

- [ ] **Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` in `src/device/tile/mod.rs`:

```rust
#[test]
fn test_lock_release_counter() {
    use crate::device::tile::LockRequestor;

    let mut tile = Tile::new_compute(0, 2, 64 * 1024);
    assert_eq!(tile.lock_release_count(), 0);

    // Set lock 0 to value 1 so a release (decrement) can succeed.
    tile.locks_mut()[0].set_value(1);

    // Submit a release request and resolve.
    tile.defer_core_lock_release(0, 1);
    tile.resolve_lock_requests(0);

    assert_eq!(tile.lock_release_count(), 1);

    // Second release: set lock back to 1 and release again.
    tile.locks_mut()[0].set_value(1);
    tile.defer_core_lock_release(0, 1);
    tile.resolve_lock_requests(0);

    assert_eq!(tile.lock_release_count(), 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib tile::tests::test_lock_release_counter`
Expected: FAIL -- `lock_release_count` method does not exist.

- [ ] **Step 3: Implement the counter**

In `src/device/tile/mod.rs`, add a field to the `Tile` struct (near line 50, alongside other counters):

```rust
    /// Cumulative count of granted lock releases (core + DMA).
    /// Monotonically increasing -- used by stall detection.
    lock_release_count: u64,
```

Initialize it to 0 in all constructors (`new_compute`, `new_memtile`, `new_shim`).

Add the accessor method:

```rust
    /// Cumulative count of granted lock releases on this tile.
    pub fn lock_release_count(&self) -> u64 {
        self.lock_release_count
    }
```

In `resolve_lock_requests()` (line 699), after the arbiter resolves, count granted releases:

```rust
    pub fn resolve_lock_requests(&mut self, cycle: u64) -> Vec<(LockRequestor, usize, bool, bool)> {
        let results = self.lock_arbiter.resolve(&mut self.locks);
        // Emit trace events for granted lock operations.
        for &(_, lock_id, granted, is_acquire) in results {
            if granted {
                if !is_acquire {
                    self.lock_release_count += 1;
                }
                let event = if is_acquire {
                    EventType::LockAcquire { lock_id: lock_id as u8 }
                } else {
                    EventType::LockRelease { lock_id: lock_id as u8 }
                };
                self.mem_trace_pending.push((cycle, event));
            }
        }
        results.to_vec()
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib tile::tests::test_lock_release_counter`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/device/tile/mod.rs
git commit -m "feat: add cumulative lock release counter to tile state

Tracks granted lock releases (both core and DMA) for monotonic
progress detection. Used by stall detection to distinguish
'slow but working' from 'stuck in infinite loop'.

Generated using Claude Code."
```

---

### Task 2: Add array-level lock release aggregation

**Files:**
- Modify: `src/device/array/dma_ops.rs`
- Test: inline verification in Task 4

The array already has `total_dma_bytes_transferred()` (line 258). Add a matching `total_lock_releases()` that sums across all tiles.

- [ ] **Step 1: Implement `total_lock_releases()`**

In `src/device/array/dma_ops.rs`, near the existing `total_dma_bytes_transferred()` (line 258), add:

```rust
    /// Total lock releases granted across all tiles.
    ///
    /// Monotonically increasing counter used by stall detection.
    /// Counts both core and DMA lock releases.
    pub fn total_lock_releases(&self) -> u64 {
        self.tiles.iter().map(|t| t.lock_release_count()).sum()
    }
```

- [ ] **Step 2: Verify it compiles**

Run: `TMPDIR=/tmp/claude-1000 cargo build --lib 2>&1`
Expected: compiles without errors.

- [ ] **Step 3: Commit**

```bash
git add src/device/array/dma_ops.rs
git commit -m "feat: add total_lock_releases() aggregation to array

Sums lock_release_count across all tiles for stall detection.

Generated using Claude Code."
```

---

### Task 3: Update coordinator stall detection

**Files:**
- Modify: `src/interpreter/engine/coordinator.rs`

Replace the current 50-cycle `no_progress_cycles` deadlock detector with monotonic progress tracking using DMA bytes and lock releases. Add `EngineStatus::Stalled` variant.

- [ ] **Step 1: Add `Stalled` variant to `EngineStatus`**

In `src/interpreter/engine/coordinator.rs` (line 28), add the variant:

```rust
pub enum EngineStatus {
    #[default]
    Ready,
    Running,
    Paused,
    Halted,
    /// No monotonic progress for stall_threshold cycles.
    Stalled,
    Error,
}
```

- [ ] **Step 2: Replace progress tracking fields**

Replace the three fields (lines 108-113):

```rust
    /// Counter for cycles with no progress while all cores halted.
    /// Used to detect deadlock where DMAs are stalled waiting for resources.
    no_progress_cycles: u32,
    /// Last cycle's words routed (to detect progress).
    last_words_routed: usize,
    /// Last cycle's total DMA bytes transferred (to detect DMA-level progress
    /// even when no stream words are routed -- e.g., during lock operations).
    last_dma_bytes: u64,
```

with:

```rust
    /// Stall detection: last observed total DMA bytes transferred.
    last_dma_bytes: u64,
    /// Stall detection: last observed total lock releases.
    last_lock_releases: u64,
    /// Consecutive cycles with no monotonic progress.
    stall_cycles: u64,
    /// Cycles of no progress before declaring stall. 0 = disabled.
    stall_threshold: u64,
```

- [ ] **Step 3: Update constructor**

In `new()` (line 140), replace the old initializers (lines 151-153):

```rust
            no_progress_cycles: 0,
            last_words_routed: 0,
            last_dma_bytes: 0,
```

with:

```rust
            last_dma_bytes: 0,
            last_lock_releases: 0,
            stall_cycles: 0,
            stall_threshold: 0,  // disabled by default; callers set via set_stall_threshold()
```

Do the same for `new_npu1()` and any other constructors.

- [ ] **Step 4: Add setter method**

Add a public method to configure the threshold:

```rust
    /// Set the stall detection threshold. 0 disables stall detection.
    pub fn set_stall_threshold(&mut self, threshold: u64) {
        self.stall_threshold = threshold;
    }
```

- [ ] **Step 5: Replace the progress detection section in `step()`**

Replace the entire block from line 980 to line 1029 (the `if cores_done { ... } else { ... }` section) with:

```rust
        if cores_done && !dma_active {
            // No DMA activity at all -- clean halt.
            self.status = EngineStatus::Halted;
            return;
        }

        // -- Monotonic progress detection --
        // Two indicators that can only advance in a correct program:
        // 1. Total DMA bytes transferred (all channels, all tiles)
        // 2. Total lock releases granted (core + DMA, all tiles)
        //
        // If EITHER advances, the system is making forward progress.
        // If NEITHER advances for stall_threshold cycles, declare stall.
        if self.stall_threshold > 0 {
            let dma_bytes = self.device.array.total_dma_bytes_transferred();
            let lock_releases = self.device.array.total_lock_releases();

            if dma_bytes > self.last_dma_bytes || lock_releases > self.last_lock_releases {
                self.stall_cycles = 0;
                self.last_dma_bytes = dma_bytes;
                self.last_lock_releases = lock_releases;
            } else {
                self.stall_cycles += 1;
                if self.stall_cycles >= self.stall_threshold {
                    log::warn!(
                        "Stall detected: no progress for {} cycles (dma_bytes={}, lock_releases={})",
                        self.stall_cycles, dma_bytes, lock_releases,
                    );
                    self.status = EngineStatus::Stalled;
                }
            }
        } else if cores_done && dma_active {
            // Stall detection disabled -- fall back to the old 50-cycle
            // deadlock check for post-core DMA drainage.
            let dma_bytes = self.device.array.total_dma_bytes_transferred();
            if dma_bytes > self.last_dma_bytes {
                self.stall_cycles = 0;
                self.last_dma_bytes = dma_bytes;
            } else {
                self.stall_cycles += 1;
                if self.stall_cycles >= 50 {
                    log::info!(
                        "Engine halting: all cores done, DMA stalled for {} cycles",
                        self.stall_cycles,
                    );
                    self.status = EngineStatus::Halted;
                }
            }
        }
```

- [ ] **Step 6: Verify it compiles**

Run: `TMPDIR=/tmp/claude-1000 cargo build --lib 2>&1`
Expected: compiles. There may be warnings about the new `Stalled` variant not being matched everywhere -- those are fixed in later tasks.

- [ ] **Step 7: Commit**

```bash
git add src/interpreter/engine/coordinator.rs
git commit -m "feat: monotonic progress detection in coordinator

Replace fixed 50-cycle deadlock check with stall detection based
on two monotonic indicators: total DMA bytes transferred and total
lock releases. Stall fires only when neither advances for
stall_threshold cycles. Catches deadlocks and livelocks while
allowing slow-but-working tests to complete.

Generated using Claude Code."
```

---

### Task 4: Add config support for stall_threshold

**Files:**
- Modify: `src/config.rs`
- Test: existing config tests in `src/config.rs`

- [ ] **Step 1: Write the failing test**

Add to the existing test module in `src/config.rs`:

```rust
#[test]
fn test_stall_threshold_defaults() {
    let config = Config::default();
    assert_eq!(config.stall_threshold(), 100_000);
    assert_eq!(config.max_cycles(), 10_000_000);
}

#[test]
fn test_stall_threshold_env_override() {
    let mut config = Config::default();
    // Simulate what apply_env_overrides does for XDNA_EMU_STALL_THRESHOLD.
    config.stall_threshold = Some(50_000);
    assert_eq!(config.stall_threshold(), 50_000);
}

#[test]
fn test_stall_threshold_zero_disables() {
    let mut config = Config::default();
    config.stall_threshold = Some(0);
    assert_eq!(config.stall_threshold(), 0);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib config::tests::test_stall_threshold`
Expected: FAIL -- `stall_threshold` field and method don't exist.

- [ ] **Step 3: Add the field and methods**

In `src/config.rs`, add to the `Config` struct (near `max_cycles`, line 53):

```rust
    pub stall_threshold: Option<u64>,
```

Add the accessor (near `max_cycles()`, line 139):

```rust
    /// Stall detection threshold in cycles. Default: 100,000.
    ///
    /// The emulator declares a stall when neither DMA bytes transferred
    /// nor lock release count advances for this many consecutive cycles.
    /// Set to 0 to disable stall detection (fall back to max_cycles only).
    pub fn stall_threshold(&self) -> u64 {
        self.stall_threshold.unwrap_or(100_000)
    }
```

Change the `max_cycles` default from 500,000 to 10,000,000:

```rust
    pub fn max_cycles(&self) -> u64 {
        self.max_cycles.unwrap_or(10_000_000)
    }
```

In `apply_env_overrides()` (after the `XDNA_EMU_MAX_CYCLES` block, line 302), add:

```rust
        if let Ok(val) = std::env::var("XDNA_EMU_STALL_THRESHOLD") {
            if let Ok(threshold) = val.parse::<u64>() {
                log::info!("Using XDNA_EMU_STALL_THRESHOLD from environment: {}", threshold);
                self.stall_threshold = Some(threshold);
            }
        }
```

In the `merge()` method (near line 269 where `max_cycles` is merged):

```rust
        if other.stall_threshold.is_some() {
            self.stall_threshold = other.stall_threshold;
        }
```

Initialize to `None` in `Default` impl or default constructor.

- [ ] **Step 4: Run tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib config::tests::test_stall_threshold`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/config.rs
git commit -m "feat: add stall_threshold config, raise max_cycles default to 10M

stall_threshold (default 100K) controls monotonic progress detection.
max_cycles (now 10M) becomes a safety cap. Both configurable via
config file, env vars (XDNA_EMU_STALL_THRESHOLD, XDNA_EMU_MAX_CYCLES),
and FFI.

Generated using Claude Code."
```

---

### Task 5: Wire stall_threshold into FFI execution path

**Files:**
- Modify: `crates/xdna-emu-ffi/src/lib.rs`
- Modify: `crates/xdna-emu-ffi/src/execution.rs`

- [ ] **Step 1: Pass stall_threshold from config to engine in FFI handle creation**

In `crates/xdna-emu-ffi/src/lib.rs`, wherever the `XdnaEmuHandle` is created and `handle.max_cycles = config.max_cycles()` is set, also set the stall threshold on the engine:

```rust
handle.engine.set_stall_threshold(config.stall_threshold());
```

- [ ] **Step 2: Handle `EngineStatus::Stalled` in the run loop**

In `crates/xdna-emu-ffi/src/execution.rs`, in the `xdna_emu_run()` function, add a check for the Stalled status. After the existing `EngineStatus::Halted` check (line 133), add:

```rust
        if handle.engine.status() == EngineStatus::Stalled {
            log::warn!("Stall detected after {} cycles: no monotonic progress", cycles);
            break;
        }
```

Also update the `match` import to include `Stalled` if needed.

- [ ] **Step 3: Verify it compiles**

Run: `TMPDIR=/tmp/claude-1000 cargo build -p xdna-emu-ffi 2>&1`
Expected: compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add crates/xdna-emu-ffi/src/lib.rs crates/xdna-emu-ffi/src/execution.rs
git commit -m "feat: wire stall detection into FFI execution path

The FFI run loop now checks EngineStatus::Stalled and breaks
with a diagnostic log message. Stall threshold is configured
from the global config at handle creation time.

Generated using Claude Code."
```

---

### Task 6: Update test runner StallDetector

**Files:**
- Modify: `src/testing/quiescence.rs`
- Modify: `src/testing/xclbin_suite.rs`

The test runner has its own `StallDetector` that currently tracks DMA bytes and instruction count. Update it to track DMA bytes and lock releases instead.

- [ ] **Step 1: Update StallDetector to use lock releases**

In `src/testing/quiescence.rs`, replace the `StallDetector` struct fields (line 268):

```rust
pub struct StallDetector {
    /// Last observed total DMA bytes transferred.
    last_dma_bytes: u64,
    /// Last observed total lock releases.
    last_lock_releases: u64,
    /// Consecutive cycles with no progress (DMA or lock releases).
    cycles_since_progress: u64,
    /// Threshold before declaring a stall.
    threshold: u64,
}
```

Update `new()`:

```rust
    pub fn new(threshold: u64) -> Self {
        Self {
            last_dma_bytes: 0,
            last_lock_releases: 0,
            cycles_since_progress: 0,
            threshold,
        }
    }
```

Update `check()` -- replace the progress comparison (lines 318-319):

```rust
        let current_bytes = engine.device().array.total_dma_bytes_transferred();
        let current_lock_releases = engine.device().array.total_lock_releases();

        if current_bytes != self.last_dma_bytes
            || current_lock_releases != self.last_lock_releases
        {
            self.last_dma_bytes = current_bytes;
            self.last_lock_releases = current_lock_releases;
            self.cycles_since_progress = 0;
            StallStatus::Progressing
        } else {
            // ... rest stays the same ...
        }
```

- [ ] **Step 2: Update test runner constants**

In `src/testing/xclbin_suite.rs`, update the stall threshold constant (line 1132):

```rust
/// No monotonic progress (DMA bytes or lock releases) for this many cycles
/// with pending syncs = stall.
const STALL_CYCLES: u64 = 100_000;
```

- [ ] **Step 3: Handle `EngineStatus::Stalled` in the match**

In `src/testing/xclbin_suite.rs`, in the `run_engine()` match on `engine.status()` (line 1183), add:

```rust
                EngineStatus::Stalled => {
                    let dma_bytes = engine.device().array.total_dma_bytes_transferred();
                    log::warn!(
                        "Stall detected after {} cycles in test {} ({} bytes transferred)",
                        cycles, test.name, dma_bytes,
                    );
                    return TestOutcome::Timeout { cycles };
                }
```

- [ ] **Step 4: Verify it compiles**

Run: `TMPDIR=/tmp/claude-1000 cargo build --lib 2>&1`
Expected: compiles without errors or warnings.

- [ ] **Step 5: Run existing tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib quiescence 2>&1`
Expected: existing quiescence tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/testing/quiescence.rs src/testing/xclbin_suite.rs
git commit -m "feat: update test runner stall detection to use lock releases

Replace instruction count with lock release count as a progress
indicator. A core in an infinite loop stops releasing locks, so
the stall detector now catches livelocks. Threshold raised from
50K to 100K to match the coordinator.

Generated using Claude Code."
```

---

### Task 7: Wire stall_threshold into test runner

**Files:**
- Modify: `src/testing/xclbin_suite.rs`

The test runner creates its own engine. It needs to set the stall threshold on it from config.

- [ ] **Step 1: Set stall_threshold on engine**

In `src/testing/xclbin_suite.rs`, find where the engine is created (likely in a setup method that calls `InterpreterEngine::new()`). After creation, add:

```rust
engine.set_stall_threshold(config.stall_threshold());
```

If the test runner uses a builder pattern with `max_cycles`, also check whether the coordinator-level stall detection and the test runner's `StallDetector` overlap. Both should use the same threshold from config. The `STALL_CYCLES` constant from Task 6 should be replaced with `config.stall_threshold()`:

```rust
let mut stall = StallDetector::new(config.stall_threshold());
```

(The config is available via `self.config` or similar in the test runner.)

- [ ] **Step 2: Verify it compiles and tests pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib xclbin_suite 2>&1`
Expected: compiles and existing tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/testing/xclbin_suite.rs
git commit -m "feat: wire config stall_threshold into test runner

Both the coordinator-level and test-runner-level stall detectors
now use the same configurable threshold from config.

Generated using Claude Code."
```

---

### Task 8: Integration test -- verify timeout tests still work

**Files:**
- No new files; run existing test suite

This is a verification task to ensure we haven't broken deadlock detection.

- [ ] **Step 1: Run the full library test suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib 2>&1`
Expected: all existing tests pass. No regressions.

- [ ] **Step 2: Run a quick bridge test with a known-passing test**

Run: `nice -n 19 ./scripts/emu-bridge-test.sh --no-hw add_one_using_dma 2>&1`
Expected: PASS for both Chess and Peano EMU.

- [ ] **Step 3: Run the two previously-timing-out tests**

Run: `nice -n 19 ./scripts/emu-bridge-test.sh --no-hw dma_task_large_linear 2>&1`
Expected: Peano EMU should now PASS (or at least run much longer before any timeout).

Run: `nice -n 19 ./scripts/emu-bridge-test.sh --no-hw objectfifo_repeat/init_values_repeat 2>&1`
Expected: Peano EMU should now PASS (or at least run much longer).

Note: these may take several minutes each due to the heavy scalar work.

- [ ] **Step 4: Commit any fixes needed**

If tests revealed issues, fix them and commit.

---

### Task 9: Rebuild FFI plugin and run bridge tests

**Files:**
- No new files; rebuild and test

The bridge tests use the FFI `.so` loaded by the XRT plugin. We need to rebuild it.

- [ ] **Step 1: Rebuild FFI crate**

Run: `nice -n 19 cargo build -p xdna-emu-ffi 2>&1`
Expected: builds successfully.

- [ ] **Step 2: Rebuild and install plugin**

Run: `nice -n 19 ./scripts/rebuild-plugin.sh 2>&1`
Expected: builds and installs successfully.

- [ ] **Step 3: Run the two target tests via bridge**

Run: `nice -n 19 ./scripts/emu-bridge-test.sh --no-hw dma_task_large_linear 2>&1`
Expected: Peano EMU PASS.

Run: `nice -n 19 ./scripts/emu-bridge-test.sh --no-hw objectfifo_repeat/init_values_repeat 2>&1`
Expected: Peano EMU PASS.

- [ ] **Step 4: Run the full bridge suite**

Run: `nice -n 19 ./scripts/emu-bridge-test.sh 2>&1`
Expected: at least the same pass rate as before (59 Chess, 44 Peano), with the two timeouts now passing.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "test: verify adaptive timeout with bridge test suite

dma_task_large_linear and objectfifo_repeat/init_values_repeat
now complete successfully with monotonic progress detection.

Generated using Claude Code."
```
