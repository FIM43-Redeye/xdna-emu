# C.4 Findings: QUERY_HW_CONTEXTS per-batch sample (2026-04-25)

## Original framing

`docs/observability-leads.md` lead #5:
> Per-batch sanity check independent of the trace buffer. Useful for
> distinguishing "trace was empty because nothing ran" from "trace was
> empty because event config was wrong."

Action priority: "cheap meta-anchor for batch validity. Append to
runner JSON status output."

## Status: design uncovered, implementation deferred

The data is reachable -- driver ioctl shape is well-documented:

```c
struct amdxdna_drm_query_hwctx {  // 64 bytes
    __u32 context_id;
    __u32 start_col, num_col, pad;
    __s64 pid;
    __u64 command_submissions, command_completions;
    __u64 migrations, preemptions, errors;
};
// DRM_IOCTL_AMDXDNA_GET_INFO = DRM_IOWR(0x40 + 7, amdxdna_drm_get_info)
// param = DRM_AMDXDNA_QUERY_HW_CONTEXTS (= 5)
```

Public XRT API does NOT expose this -- `xrt::info::device` enum has no
`aie_partition_info` member (only the internal `xrt_core::query` does,
which lives behind private headers). xrt-smi *does* expose it via
`xrt-smi examine -r aie-partitions -f JSON`, but only between runs --
once we shell out, the partition is gone.

## The "cheap" framing was wrong

What the lead actually requires:

1. **Sampling-during-run.** Hardware contexts only exist while their
   owning process is alive. Bridge-trace-runner spawns, attaches, runs
   the kernel, then exits. By the time we shell out to xrt-smi or run
   a separate sampler, the partition has been torn down.

   Three approaches:
   - **In-process sampler thread**: bridge-trace-runner spawns a
     sampler thread that calls the ioctl every N ms, recording deltas
     in shared memory. Adds C++ thread complexity to the runner.
   - **Pre/post sampling around the runner**: parent shell records
     `command_submissions` count for our pid before fork, and after
     join. Doesn't work for our pid because the runner reports
     pre-fork=0 (no context yet) / post-join=0 (context destroyed).
     Could work if we proxied via a long-lived holder process.
   - **Driver-side counter**: counter that survives across context
     teardown. Currently no such counter is exposed.

2. **Per-context filtering.** The query returns all contexts, not just
   ours. We need to filter by pid (we know our pid) -- straightforward
   once we get a sample.

3. **Counter semantics.** `command_submissions` / `command_completions`
   are cumulative for the context's lifetime. To get a per-batch
   delta we'd need start/end snapshots within one context's lifespan,
   or we'd need each batch to use a fresh context (matches current
   bridge-runner behaviour: hw_context recreated per launch).

## Recommendation

This is its own ~1 day implementation task: in-process sampler thread
in bridge-trace-runner, plus a JSON-emit-on-shutdown step that summarises
the final counter values and includes them in the batch status line.

Not blocking the cycle-budget validation work today. The
"distinguishing empty trace because nothing ran" use case is partially
served by the existing classifier (`EMPTY` vs `NO_CORE` already
distinguishes the two structural cases via traced-MLIR inspection).
What we'd gain is a runtime sanity check ("did anything actually
submit?") -- useful but not urgent.

## Data shapes (recorded for the implementation pass)

When implemented, the output should look like:

```json
{
  "run_idx": 0,
  "ok": true,
  "trace_out": "...",
  "elapsed_ms": 123,
  "hwctx": {
    "context_id": 17,
    "start_col": 1,
    "num_col": 2,
    "submissions": 5,
    "completions": 5,
    "migrations": 0,
    "preemptions": 0,
    "errors": 0
  }
}
```

Diffing successive batches' hwctx blocks gives per-batch work
counters without needing a separate sampler tool.

## What this means for thread C

C.4 closed as designed-not-implemented. The data path is fully mapped;
the implementation is deferred until the bridge-trace-runner gets
another iteration cycle (good neighbour change to land alongside
the next runner-side improvement).
