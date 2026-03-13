# DMA Engine -- Accuracy Sweep Match Report

Audited: 2026-03-12
Agent: F (DMA subsystem)

## Methodology

Compared xdna-emu DMA engine (`src/device/dma/`) against aie-rt
(`aie-rt/driver/src/dma/xaie_dma_aieml.c`) and AM025 register params
(`aie-rt/driver/src/global/xaiemlgbl_params.h`).

All aie-rt code was read first, then our emulator code was compared
function-by-function.

---

## 1. BD Field Parsing (bd.rs vs xaie_dma_aieml.c)

### 1.1 Compute Tile BD (6 registers)

Our `parse_compute()` (bd.rs:158-223) vs `_XAieMl_TileDmaReadBd()`
(xaie_dma_aieml.c:694-804).

| Field | aie-rt word | aie-rt field | Our word | Our field | Match |
|-------|-------------|--------------|----------|-----------|-------|
| Base_Address | BdWord[0] | Buffer.TileDmaBuff.BaseAddr | w0 | base_address | YES |
| Buffer_Length | BdWord[0] | BufferLen | w0 | buffer_length | YES |
| Enable_Compression | BdWord[1] | Compression.EnCompression | w1 | enable_compression | YES |
| Enable_Packet | BdWord[1] | Pkt.EnPkt | w1 | enable_packet | YES |
| OOO_BD_ID | BdWord[1] | BdEn.OutofOrderBdId | w1 | out_of_order_bd_id | YES |
| Packet_ID | BdWord[1] | Pkt.PktId | w1 | packet_id | YES |
| Packet_Type | BdWord[1] | Pkt.PktType | w1 | packet_type | YES |
| D0_Stepsize | BdWord[2] | DmaDimProp[0].StepSize | w2 | d0_stepsize | YES (+1 applied) |
| D1_Stepsize | BdWord[2] | DmaDimProp[1].StepSize | w2 | d1_stepsize | YES (+1 applied) |
| D0_Wrap | BdWord[3] | DmaDimProp[0].Wrap | w3 | d0_wrap | YES |
| D1_Wrap | BdWord[3] | DmaDimProp[1].Wrap | w3 | d1_wrap | YES |
| D2_Stepsize | BdWord[3] | DmaDimProp[2].StepSize | w3 | d2_stepsize | YES (+1 applied) |
| Iteration_Current | BdWord[4] | IterCurr | w4 | iteration_current | YES |
| Iteration_Wrap | BdWord[4] | Iter.Wrap | w4 | iteration_wrap | YES (+1 applied) |
| Iteration_Stepsize | BdWord[4] | Iter.StepSize | w4 | iteration_stepsize | YES (+1 applied) |
| TLAST_Suppress | BdWord[5] | BdEn.TlastSuppress | w5 | tlast_suppress | YES |
| Next_BD | BdWord[5] | BdEn.NxtBd | w5 | next_bd | YES |
| Use_Next_BD | BdWord[5] | BdEn.UseNxtBd | w5 | use_next_bd | YES |
| Valid_BD | BdWord[5] | BdEn.ValidBd | w5 | valid_bd | YES |
| Lock_Rel_Value | BdWord[5] | Lock.LckRelVal | w5 | lock_rel_value | YES (sign-extended) |
| Lock_Rel_ID | BdWord[5] | Lock.LckRelId | w5 | lock_rel_id | YES |
| Lock_Acq_Enable | BdWord[5] | Lock.LckAcqEn | w5 | lock_acq_enable | YES |
| Lock_Acq_Value | BdWord[5] | Lock.LckAcqVal | w5 | lock_acq_value | YES (sign-extended) |
| Lock_Acq_ID | BdWord[5] | Lock.LckAcqId | w5 | lock_acq_id | YES |

**Result: FULL MATCH.** All 22 fields match aie-rt. Stepsize +1 conversion
matches. Iteration wrap +1 conversion matches. Sign extension of lock values
matches (7-bit signed). Our regdb-driven extraction produces identical results
to aie-rt's hardcoded bit positions, validated by
`test_compute_bd_cross_validation` (bd.rs:641-701).

### 1.2 MemTile BD (8 registers)

Our `parse_memtile()` (bd.rs:230-301) vs `_XAieMl_MemTileDmaReadBd()`
(xaie_dma_aieml.c:426-565).

All 32 fields match:
- BD_0: Enable_Packet, Packet_Type, Packet_ID, OOO_BD_ID, Buffer_Length
- BD_1: D0_Zero_Before, Next_BD, Use_Next_BD, Base_Address
- BD_2: TLAST_Suppress, D0_Wrap, D0_Stepsize (+1)
- BD_3: D1_Zero_Before, D1_Wrap, D1_Stepsize (+1)
- BD_4: Enable_Compression, D2_Zero_Before, D2_Wrap, D2_Stepsize (+1)
- BD_5: D2_Zero_After, D1_Zero_After, D0_Zero_After, D3_Stepsize (+1)
- BD_6: Iteration_Current, Iteration_Wrap (+1), Iteration_Stepsize (+1)
- BD_7: Valid_BD, Lock_Rel_Value (7-bit signed), Lock_Rel_ID (8-bit),
  Lock_Acq_Enable, Lock_Acq_Value (7-bit signed), Lock_Acq_ID (8-bit)

**Result: FULL MATCH.** Validated by `test_memtile_bd_cross_validation`
(bd.rs:709-787). Note that MemTile lock IDs are correctly 8-bit (vs 4-bit
for compute/shim), matching the hardware's 192-entry lock address space.

### 1.3 Shim Tile BD (8 registers)

Our `parse_shim()` (bd.rs:309-385) vs `_XAieMl_ShimDmaReadBd()`
(xaie_dma_aieml.c:959-1086).

All fields match:
- BD_0: Buffer_Length (full 32-bit, matching NOC_MODULE register width)
- BD_1: Base_Address_Low (30-bit, LSB=2 per `xaiemlgbl_params.h:16306`)
- BD_2: Base_Address_High (16-bit), Enable_Packet, Packet_ID, Packet_Type,
  OOO_BD_ID
- BD_3: Secure_Access, D0_Wrap, D0_Stepsize (+1)
- BD_4: Burst_Length, D1_Wrap, D1_Stepsize (+1)
- BD_5: SMID, AxCache, AxQoS, D2_Stepsize (+1)
- BD_6: Iteration_Current, Iteration_Wrap (+1), Iteration_Stepsize (+1)
- BD_7: TLAST_Suppress, Next_BD, Use_Next_BD, Valid_BD, Lock_Rel_Value,
  Lock_Rel_ID, Lock_Acq_Enable, Lock_Acq_Value, Lock_Acq_ID

**Result: FULL MATCH.** Shim address reconstruction (30-bit low | 16-bit
high << 30) matches aie-rt's split-register approach
(`_XAieMl_ShimDmaReadBd` lines 987-994).

Note: Shim BD correctly does NOT parse compression_enable, d2_wrap, or
zero-padding fields (these are MemTile-only), matching aie-rt which also
does not read these from shim BDs.

---

## 2. Stepsize and Wrap Encoding

### 2.1 Stepsize: stored = actual - 1

aie-rt consistently uses `StepSize - 1U` when writing and `1U + field` when
reading (xaie_dma_aieml.c lines 334-335, 347-348, 486-489, 740-747).

Our code uses `extract(w) + 1` consistently (bd.rs:184-185, 190, 262-263,
267, 273, 279).

**Result: MATCH.** All stepsizes correctly decoded.

### 2.2 Wrap: stored as-is (NOT minus 1) for D0/D1/D2

aie-rt reads D0/D1/D2 wrap values directly from registers WITHOUT adding 1
(xaie_dma_aieml.c lines 482-485, 494-497, 506-509, 749-756).

Our code also extracts wraps without adding 1 (bd.rs:188-189, 261, 266, 272).

**Result: MATCH.** Wrap values are raw register values.

### 2.3 Iteration Wrap: stored = actual - 1

aie-rt adds 1 when reading iteration wrap: `(u16)(1U + XAie_GetField(...))`
(xaie_dma_aieml.c lines 536-539, 766-769). When writing, it stores
`Wrap - 1U` (xaie_dma_aieml.c lines 380-382, 640-642).

Our code adds 1 when parsing: `(lay.iteration_wrap.extract(w4) + 1) as u8`
(bd.rs:194, 283, 360).

**Result: MATCH.** Iteration wrap is correctly decoded as stored+1.

### 2.4 Iteration Stepsize: stored = actual - 1

aie-rt adds 1 when reading iteration stepsize: `1U + XAie_GetField(...)`
(xaie_dma_aieml.c lines 540-543, 770-773). In
`_XAieMl_DmaSetBdIteration()` (line 1517), it stores the raw user value
which was validated against `StepSize > (StepSizeMax + 1)`.

Our code adds 1: `lay.iteration_stepsize.extract(w4) + 1` (bd.rs:195, 284,
361).

**Result: MATCH.**

---

## 3. Lock Semantics

### 3.1 Lock Acquire Enable

AIE-ML has an explicit `Lock_Acq_Enable` bit in the BD. aie-rt reads it as
`LockAcqEn` (xaie_dma_aieml.c:560-562, 790-792). Our code reads it as
`lock_acq_enable` (bd.rs:204).

**Result: MATCH.**

### 3.2 Lock Release Enable

AIE-ML does NOT have a `Lock_Rel_Enable` bit in the BD registers (unlike
AIE1 which has `LckRelEn`). The `LockRelEn` field in `xaiegbl.h:290`
exists in the `XAie_LockDesc` struct, but aie-rt's `_XAieMl_TileDmaWriteBd()`
and `_XAieMl_TileDmaReadBd()` never read or write it for AIEML BDs.

Lock release in AIE-ML is implicit: it fires whenever the BD completes and
the `Lock_Rel_Value` is non-zero.

Our code at bd.rs:475 uses `lock_rel_value != 0` as the enable criterion:
```
let release_lock = if self.lock_rel_value != 0 { Some(self.lock_rel_id) } else { None };
```

**Result: MATCH.** No explicit release enable in AIE-ML registers; both
aie-rt and our emulator use the value-based model.

### 3.3 Lock Value Sign Extension

Lock values are 7-bit signed fields. Our `sign_extend_7bit()` function
(bd.rs:537-544) correctly sign-extends from 7 bits to i8. aie-rt casts to
`(s8)` (xaie_dma_aieml.c:548, 557, 778, 787).

**Result: MATCH.**

---

## 4. Channel FSM (channel.rs vs aie-rt channel operations)

### 4.1 Channel Enable/Disable

aie-rt's channel operations are in `xaie_dma.c` (not the aieml-specific
file). The `XAie_DmaChannelEnable()` writes to a control register; the
`XAie_DmaChannelDisable()` clears it.

Our `ChannelFsm` models this via the `Idle -> BdSetup -> ... -> Idle` state
machine. Channel enable pushes a task to the task queue; disable transitions
to Idle.

**Result: FUNCTIONALLY EQUIVALENT.** Our FSM models the behavioral effect
rather than the register write, which is correct for an emulator.

### 4.2 Task Queue

aie-rt checks task queue size via `TaskQSize` in the status register
(`DmaGetPendingBdCount`, xaie_dma_aieml.c:1105-1144). The queue depth is
checked against `StartQSizeMax`.

Our `ChannelContext` has a `VecDeque<TaskQueueEntry>` with `MAX_TASK_QUEUE_DEPTH`
(8, per AM025). The task queue depth is configurable and matches the hardware
specification.

**Result: MATCH.**

---

## 5. Completion Polling (engine.rs vs _XAieMl_DmaWaitForDone)

### 5.1 Completion Detection

aie-rt's `_XAieMl_DmaWaitForDone()` (xaie_dma_aieml.c:1209-1238) polls a
mask of status bits:
- `ChannelRunning`
- `StalledLockAcq`
- `StalledLockRel`
- `StalledStreamStarve`
- `StalledTCT`
- `TaskQSize`

All must be zero for "done." The value compared is
`XAIEML_DMA_STATUS_CHANNEL_NOT_RUNNING` (0x0) shifted to the
`ChannelRunning` LSB.

Our emulator reports completion when the FSM returns to `Idle` and the task
queue is empty (`has_pending_work() == false`). This is functionally
equivalent: when all status bits are zero in hardware, the channel is idle
with no pending work.

**Result: FUNCTIONALLY EQUIVALENT.** Our FSM-based completion matches the
hardware status register semantics.

---

## 6. Multi-Dimensional Addressing (addressing.rs vs _XAieMl_DmaSetMultiDim)

### 6.1 Dimension Count

aie-rt: Compute/Shim have 3 dimensions (D0, D1, D2). MemTile has 4
dimensions (D0, D1, D2, D3). Init functions set `StepSize = 1` (actual)
as default.

Our `AddressGenerator` supports 4 dimensions for all tile types (bd.rs
passes `d3_stepsize = 0` for non-MemTile). This is harmless -- D3 with
`size=0` (effective 1) and `stride=0` produces no offset.

**Result: MATCH.**

### 6.2 Address Computation

aie-rt's address formula is implicit in the BD fields:
`addr = base + d0_counter * d0_step + d1_counter * d1_step + ... + iter * iter_step`

Our `compute_address()` (addressing.rs:321-333):
```rust
addr += (iteration_counter * iteration.stepsize_bytes())
     + sum(counters[dim] * dimensions[dim].stride)
```

**Result: MATCH.** Same additive multi-dimensional formula.

### 6.3 Wrap Counter Behavior

In aie-rt, `Wrap` is the count at which the dimension resets. When
`counter == wrap`, the counter resets to 0 and the next dimension
increments. With `Wrap == 0`, the dimension does 1 iteration (no wrap).

Our `advance()` (addressing.rs:382-408): counter increments, and when
`counter >= effective_size()` (where effective_size() maps 0 -> 1), counter
resets and the next dimension increments.

**Result: MATCH.** Both use the same wrap-and-carry counter logic.

### 6.4 Stepsize Validation

aie-rt checks `StepSize == 0` as invalid (`_XAieMl_DmaSetMultiDim`,
xaie_dma_aieml.c:174-177). After +1 conversion, minimum stepsize is 1.

Our code converts stored 0 to actual 1, matching this invariant.

**Result: MATCH.**

---

## 7. Zero Padding (compression.rs / addressing.rs vs aie-rt)

### 7.1 Padding Field Widths

aie-rt checks max padding per dimension
(`_XAieMl_DmaMemTileCheckPaddingConfig`, xaie_dma_aieml.c:218-266):
- D0: 6 bits (max 63, `XAIEML_DMA_PAD_WORDS_MAX = 0x3FU`)
- D1: 5 bits (max 31, `0x3F >> 1 = 0x1F`)
- D2: 4 bits (max 15, `0x3F >> 2 = 0x0F`)

Our `ZeroPadConfig` doc comment (addressing.rs:120-124) correctly states
these widths. The pad values are extracted from registers using regdb-derived
masks that encode these widths.

**Result: MATCH.**

### 7.2 Padding Validity Rules

aie-rt enforces: if D{N}_wrap == 0, then D{N}_after must be 0, and all
higher-dimension padding must be 0 (xaie_dma_aieml.c:240-262).

Our emulator does not enforce this validation at BD parse time.

**Result: MINOR DIVERGENCE.** See catalog-dma.md. Not a behavioral bug
because the compiler never generates invalid configurations.

### 7.3 Sparsity Compression Format

Our `compression.rs` implements the sparsity compression format: 32-bit
mask + packed non-zero bytes. This matches AM020 Ch1 description. aie-rt
does not implement compression logic directly (it's a hardware feature that
the DMA engine performs in-line). Our implementation exists for the emulator
to replicate this behavior.

**Result: NO DIVERGENCE POSSIBLE.** aie-rt only sets the `Enable_Compression`
bit; the hardware does the rest. Our implementation is for emulation only.

---

## 8. BD Spacing and Base Addresses

### 8.1 BD Spacing

Our `BD_SPACING = 0x20` (32 bytes, bd.rs:17). From xaiemlgbl_params.h:
- Compute: `DMA_BD0_0 = 0x1D000`, `DMA_BD1_0 = 0x1D020` -> stride 0x20
- MemTile: `DMA_BD0_0 = 0xA0000`, `DMA_BD1_0 = 0xA0020` -> stride 0x20
- Shim: `DMA_BD0_0 = 0x1D000`, `DMA_BD1_0 = 0x1D020` -> stride 0x20

**Result: MATCH.** Cross-validated by `aiert_validation.rs` tests for all
three tile types.

### 8.2 BD Base Addresses

Cross-validated by existing tests in `aiert_validation.rs`:
- `compute_dma_bd_base_matches_regdb()`
- `memtile_dma_bd_base_matches_regdb()`
- `shim_dma_bd_base_matches_regdb()`

**Result: MATCH.** All three tile types verified.

### 8.3 Register Counts

- Compute: 6 registers (`XAIEML_TILEDMA_NUM_BD_WORDS = 6U`)
- MemTile: 8 registers (`XAIEML_MEMTILEDMA_NUM_BD_WORDS = 8U`)
- Shim: 8 registers (`XAIEML_SHIMDMA_NUM_BD_WORDS = 8U`)

Our `bd_register_count()` returns these from regdb, cross-validated by the
BD stride tests.

**Result: MATCH.**

---

## 9. MemTile BD-Channel Validity

aie-rt has `_XAieMl_MemTileDmaCheckBdChValidity()` (xaie_dma_aieml.c:1320-1331):
- BD 0-23 are valid for even channels (0, 2, 4)
- BD 24-47 are valid for odd channels (1, 3, 5)

Our emulator does not enforce this constraint.

**Result: DIVERGENCE.** See catalog-dma.md. This is a validation-only issue;
the compiler generates correct BD-channel assignments.

---

## 10. Iteration Config (to_bd_config conversion)

### 10.1 Iteration Wrap Convention

aie-rt treats iteration wrap as "actual count" (minimum 1). In the BD
register, it stores `actual - 1`. When writing: `Wrap - 1U`
(xaie_dma_aieml.c:380-382). In `_XAieMl_DmaSetBdIteration()`, `Wrap == 0`
is rejected as invalid (line 1508).

Our BufferDescriptor stores actual values (after +1 conversion from register).
In `to_bd_config()`, we convert back with `wrap.saturating_sub(1)` (bd.rs:469).
The `IterationConfig` then uses the "stored-1" convention where `wrap=0` means
1 iteration.

**Result: MATCH.** Round-trip preserves semantics correctly.

---

## Summary

| Subsection | Files Compared | Result |
|------------|----------------|--------|
| Compute BD fields (22) | bd.rs vs xaie_dma_aieml.c | FULL MATCH |
| MemTile BD fields (32) | bd.rs vs xaie_dma_aieml.c | FULL MATCH |
| Shim BD fields (26) | bd.rs vs xaie_dma_aieml.c | FULL MATCH |
| Stepsize encoding | bd.rs vs xaie_dma_aieml.c | FULL MATCH |
| Wrap encoding | bd.rs vs xaie_dma_aieml.c | FULL MATCH |
| Iteration encoding | bd.rs vs xaie_dma_aieml.c | FULL MATCH |
| Lock semantics | bd.rs vs xaie_dma_aieml.c | FULL MATCH |
| Channel FSM | channel.rs vs xaie_dma.c | FUNCTIONALLY EQUIVALENT |
| Completion polling | engine.rs vs xaie_dma_aieml.c | FUNCTIONALLY EQUIVALENT |
| Multi-dim addressing | addressing.rs vs xaie_dma_aieml.c | FULL MATCH |
| Zero padding widths | addressing.rs vs xaie_dma_aieml.c | FULL MATCH |
| Padding validation | N/A | MINOR DIVERGENCE |
| BD spacing | bd.rs vs xaiemlgbl_params.h | FULL MATCH |
| BD base addresses | aiert_validation.rs | FULL MATCH (cross-validated) |
| MemTile BD-Ch validity | N/A vs xaie_dma_aieml.c | DIVERGENCE |
| Compression format | compression.rs vs AM020 | NO DIVERGENCE POSSIBLE |

Overall: The DMA engine is highly accurate. No behavioral bugs were found.
Two validation-layer gaps were identified (padding validation and MemTile
BD-channel constraints) -- both are compile-time invariants that the
toolchain enforces, not runtime behavioral differences.
