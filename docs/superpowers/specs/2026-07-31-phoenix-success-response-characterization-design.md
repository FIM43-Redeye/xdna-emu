# Phoenix Successful-Response Characterization -- Design

**Date:** 2026-07-31

**Status:** Completed -- causal rule confirmed on physical NPU1

## Purpose

Close three firmware-behavior unknowns left by the NPU1 Phase 3A vertical
pair:

- the successful `CONFIG_CU` status word;
- the successful `CHAIN_EXEC_NPU` failure-index word; and
- the successful `CHAIN_EXEC_NPU` failure-status word.

The deliverable is a causal, firmware-version-scoped rule suitable for emulator
tests and documentation. Raw capture files are supporting receipts, not the
research product.

## Claim Under Test

For Phoenix firmware 1.5.5.391 with SHA-256
`d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`:

- successful `CONFIG_CU` returns one little-endian word: `[SUCCESS]`; and
- successful `CHAIN_EXEC_NPU` returns three little-endian words:
  `[SUCCESS, 0, SUCCESS]`.

The chained tail words are initialized for each request. They are not stale or
unspecified data.

## Causal Basis

The physical observation cross-checks an already derived mechanism:

1. The pinned open driver defines a four-byte CONFIG response and a twelve-byte
   chained-execution response.
2. In the signed firmware image, the CONFIG handler sets its response status to
   success only after successful configuration.
3. The chained-execution handler initializes status, failure index, and failure
   status to zero before validation and execution; only error paths replace
   those values.
4. The existing signed-firmware execution guards consume `[0]` for successful
   CONFIG and `[0, 0, 0]` for successful chained execution.
5. One physical vertical pair through the real mailbox transport checks that
   the pinned NPU1 tuple delivers those same bytes.

A matching physical result strengthens this rule; it is not interpreted as a
sample from a distribution. A mismatch invalidates the proposed rule and stops
the slice for causal investigation.

## Selected Slice

### Observation point

Add one `mbox_response` tracepoint at `mailbox_get_resp()`, where the driver has
the response channel, opcode, message ID, exact byte count, and complete
`void __iomem *` response body before dispatching the callback.

The tracepoint copies the response body with `memcpy_fromio()` only while the
tracepoint is enabled. Its disabled path performs no body copy. This is the
single common observation point for synchronous and asynchronous responses.

### Derivation

Extend the existing Phase 3A parser just enough to:

- parse the new response event;
- correlate each body with exactly one request and queue-head event by channel,
  opcode, and message ID;
- reject missing, duplicated, mis-sized, or mismatched response bodies; and
- preserve the exact CONFIG and execute response words in the derived run
  result.

The existing raw trace remains in the canonical bundle. No new storage format,
capture framework, or Rust evidence schema is introduced.

### Physical execution

Rebuild and qualify the driver module with the tracepoint, then reuse the
already-approved Phase 3A vertical pair under the new pinned XRT 2.26 tuple:

- one successful direct `EXECUTE_BUFFER_CF` control; and
- one successful `CHAIN_EXEC_NPU` treatment.

The pair uses the frozen `add_one_using_dma` fixture and the existing safety,
restoration, output-oracle, and provenance gates. Its order remains the
recorded schedule order. No third warm run and no repetition campaign are part
of this slice.

## Acceptance

The characterization passes only if:

- every expected response has one correlated body of the driver-defined size;
- CONFIG returns exactly `[0]` in both arms;
- direct execute returns exactly `[0]`;
- chained execute returns exactly `[0, 0, 0]`;
- both workloads retain their existing exact host-output and clean-lifecycle
  proof; and
- teardown restores the original module and host state.

On success, replace only the now-observed Phase 3A unknowns and document the
causal firmware rule beside the existing signed-firmware guard. Do not add a
new emulator mechanism if the current firmware execution path already obeys
the rule.

## Result

Campaign `physical-response-xrt226-20260731-02` passed the complete boundary:

- both CONFIG_CU responses were exactly `[0]`;
- the CHAIN_EXEC_NPU response was exactly `[0, 0, 0]`;
- the direct EXECUTE_BUFFER_CF response was exactly `[0]`;
- both runs produced exact ordered output 2 through 65 and `PASS!`;
- neither run added a TDR or IOMMU fault; and
- explicit post-campaign restoration verified the original module hash and
  srcversion, zero active clients, the normal device node, and the original
  `power/control=auto` policy.

The preceding `-01` attempt never launched the host process because the
canonical oracle copy lacked an execute bit. It submitted no NPU request. The
fixture mode now matches its source, and preflight rejects non-executable host
oracles before privileged setup.

The signed-firmware execution guards already enforce these exact response
vectors, so the result requires no new emulator mechanism.

## Stop Boundary

This slice does not authorize:

- repetition or distribution modeling;
- cold-versus-warm claims;
- malformed, negative, recovery, or preemption traffic;
- older-firmware generalization;
- broader mailbox capture expansion;
- timing claims; or
- the Phase 3A 50+50 campaign.

After the single pair is interpreted, stop and reassess which unresolved NPU1
behavior has the highest emulator value. Further captures require a named
behavioral question that source derivation or existing evidence cannot answer.
