# Phoenix Firmware Equivalence Goal -- Design

**Date:** 2026-07-26

**Target:** Phoenix/NPU1 management firmware

**Status:** Approved in conversation; awaiting written-spec review

## Goal

Achieve provable architectural and externally observable equivalence for
Phoenix/NPU1 management firmware.

The primary target is the unmodified firmware image:

- Path: `/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin`
- Size: 248,592 bytes
- SHA-256:
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

Validation initially covers the complete legitimate host-to-NPU command
surface reachable through `amdxdna-driver` commit
`216cefececd74effcd7a88350c71b99f5ef9a215`. The driver revision pins a
reproducible surface audit and stimulus corpus; it does not define simulated
behavior and is not embedded in the emulator.

Completion requires a reproducible hardware-versus-emulator matrix with zero
unexplained divergences. Once the primary image passes, the same applicable
contract is validated against a frozen manifest of authoritative older
Phoenix/NPU1 firmware images.

## Authority and Contract Boundary

The emulator models the NPU, not a particular driver implementation.

- The emulator accepts the hardware command protocol directly: BAR accesses,
  PSP/SMU interactions, mailbox traffic, memory mappings, interrupts, resets,
  and their required ordering.
- A driver is a source of real stimuli and a way to discover surfaces that the
  current corpus missed.
- Newer or development drivers may expand validation coverage. They must not
  create driver-specific behavior in the emulator.
- Expected behavior is derived from the firmware bytes, open hardware and
  toolchain artifacts, and controlled silicon observations.
- Undocumented operations are excluded. Development-only operations are
  inventoried separately and enter an acceptance matrix only when relevant and
  verifiable on hardware.

The current local `amdxdna-driver` tree contains `AMDXDNA_DEVEL` boot-capture
instrumentation. That patch is measurement apparatus, not part of the pinned
behavioral contract.

## Definition of 100%

The primary Phoenix image is complete only when all of the following hold:

1. The exact unmodified image boots from reset without hand-forced memory
   views, forced branches, injected completion, or firmware-specific execution
   shims. A hardware-derived bank or overlay mechanism is permitted.
2. Every path reachable through the legitimate host command surface executes
   with correct Xtensa, MMU, memory-map, cache, interrupt, and peripheral
   semantics.
3. Boot and go-alive, normal commands, malformed/error responses, timeouts,
   reset, power transitions, teardown, and recovery match hardware.
4. Host-visible values and event ordering match exactly.
5. Externally observable timing matches directly where measurable; indirect
   polling and correlated side effects constrain internal timing where direct
   observation is unavailable.
6. Firmware-issued configuration, launch, reset, status, and interrupt
   transactions at the AIE-array seam match hardware.
7. The frozen primary validation matrix has zero unexplained divergences.

Dead or unreachable firmware code is not itself an acceptance gate. A path
becomes part of the contract as soon as a legitimate host stimulus can reach
it.

An explained divergence is limited to measured hardware nondeterminism with an
explicit bound and evidence. Missing observations, guessed constants, and
test-harness compensation are not explanations.

## Firmware and AIE-Array Boundary

Kernel execution is downstream of the firmware goal.

Firmware equivalence requires the management processor to emit the correct
array transactions and correctly consume array status, completion, and
interrupt results. It does not require unrelated AIE instruction or kernel
cycle-accuracy work.

End-to-end firmware validation uses a frozen corpus of kernels that already
pass independent hardware-versus-array-emulator validation. This prevents an
array defect from being misclassified as a firmware defect while preserving
the real management-processor/array interaction.

## Validation Architecture

Validation has three complementary layers.

### 1. Component Differential

Capture real host-to-NPU transactions and the associated silicon responses.
Replay the same stimuli against the emulator to isolate protocol, memory-map,
peripheral, and firmware divergences.

Each matrix row records:

- exact stimulus and ordering;
- firmware image and hardware identity;
- observable values, state changes, interrupts, and timing evidence;
- normal and relevant failure/recovery variants;
- provenance for the expected behavior; and
- emulator result and verdict.

Captured driver behavior supplies stimuli, never expected results.

### 2. Firmware-Array Seam

Compare firmware-issued array configuration, launch, reset, status, completion,
and interrupt transactions with hardware. A recording downstream component may
be used diagnostically, but it is not the ultimate acceptance case.

### 3. True End-to-End Acceptance

An unmodified compatible driver talks to the simulated NPU through the same
hardware protocol it uses on silicon. The unmodified firmware executes, drives
the real array emulator, launches an independently validated kernel, consumes
the downstream result, and returns the correct completion and output to the
driver.

This end-to-end path is the definitive acceptance case. Component replay and
seam recording exist to diagnose failures and prove coverage.

## Evidence Rules

- Internal probes are observational and must not alter firmware state,
  mappings, control flow, timing inputs, or peripheral responses.
- Every behavioral constant carries source provenance.
- Directly observable timing is compared directly.
- Internal timing may be bounded through polling, correlated registers,
  interrupts, or other externally visible effects.
- Unobservable internal microcycles remain explicitly unknown rather than
  being declared exact.
- A passing result cannot depend on a firmware-version-specific execution
  patch.

## Versioning

The primary gate is fixed to the firmware SHA above. The named driver commit
freezes the initial command-surface audit and validation corpus, not emulator
behavior.

After the primary gate passes:

1. Inventory authoritative, distributable older Phoenix/NPU1 firmware images.
2. Record each image's SHA, provenance, and known compatible host-command
   surface in a versioned manifest.
3. Freeze that manifest before compatibility implementation begins.
4. Apply the same relevant architectural and externally observable equivalence
   gate to every entry.

Newly released firmware or newly discovered development-only operations become
explicit later extensions; they do not silently move an already-frozen gate.

## Work Order

1. Freeze the primary firmware artifact, hardware identity, initial command
   surface, and independently validated kernel corpus.
2. Correct the foundational firmware architecture: Xtensa behavior, loader,
   address map, MMU/cache, memory apertures, and peripheral models.
3. Reach natural boot and go-alive using the unmodified image and genuine
   driver traffic.
4. Close the full legitimate driver-reachable contract, including errors,
   timeouts, resets, power transitions, teardown, and recovery.
5. Close the firmware-array seam and pass the true end-to-end driver →
   firmware → array-emulator → validated-kernel → driver path.
6. Tighten observable and indirectly constrained timing until the primary
   matrix has zero unexplained divergences.
7. Freeze and validate the authoritative older-Phoenix firmware manifest.

AIE kernel cycle-accuracy work remains deferred except for maintaining the
already-trusted kernel corpus required by the end-to-end firmware gate.

## Non-Goals

- AIE2P/XDNA2 firmware.
- Downstream kernel or AIE-array cycle accuracy beyond the trusted acceptance
  corpus.
- Driver emulation or driver-specific behavior.
- Undocumented operations without a hardware-verifiable contract.
- Claiming direct cycle-by-cycle equivalence inside the management Xtensa where
  no direct oracle exists.

## Proposed Tracked-Goal Objective

Achieve provable architectural and externally observable equivalence for the
unmodified Phoenix/NPU1 `1502_00` management firmware, initially covering the
complete hardware-command surface exercised by the pinned open-driver corpus.
Pass the true unmodified-driver → simulated-firmware → array-emulator →
independently-validated-kernel → driver path, plus every legitimate normal,
error, reset, power, timeout, teardown, and recovery case, with zero unexplained
hardware-versus-emulator divergences and no forced firmware execution or
driver-specific simulation. After the primary firmware SHA is green, freeze
and pass the same applicable contract for authoritative older Phoenix firmware
images. AIE2P and downstream kernel accuracy are out of scope.
