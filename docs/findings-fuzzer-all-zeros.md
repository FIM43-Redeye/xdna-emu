# Fuzzer All-Zeros Root Cause Investigation (2026-03-01)

## Symptom
Every fuzz kernel produces all-zero output from the emulator while NPU hardware produces correct results.

## Stream Switch: VERIFIED CORRECT
- Port arrays generated from AM025 JSON match aie-rt's slave index ordering exactly
- Register bases match: compute 0x3F000, memtile 0xB0000, shim 0x3F000 (same as compute per aie-rt params)
- Circuit routes configured correctly from CDO (slave_select -> configure_local_route)
- Shim mux config parsed correctly (MM2S ch0 -> slave[5], S2MM ch0 <- master[4])
- Data DOES flow through switches: shim -> memtile confirmed via logs

## Data Flow: WHERE IT BREAKS

### Working hops (confirmed via RUST_LOG=info):
1. Shim DMA MM2S ch2 reads from host DDR (correct values: 0x00020001 = i16[1,2])
2. Shim switch slave[5] receives data, routes to master[16] (North4)
3. MemTile switch slave[11] (South4) receives, routes to master[0] (DMA0)
4. MemTile DMA S2MM ch0 receives stream data and writes to memory at 0x80400

### BROKEN: MemTile lock synchronization prevents MM2S from reading
- MemTile MM2S ch6 (logical MM2S ch0) needs lock 65 (own lock 1) with delta=-1, expected=1
- Lock 1 is initialized to 0 and NEVER changes
- S2MM ch0 BD says: acq lock 64 (own lock 0), rel lock 65 (own lock 1)
- S2MM ch0 successfully acquires lock 0 (init=2, 2-1=1, expected=1, success)
- S2MM ch0 successfully writes data to memory (confirmed via debug logs)
- **But lock 65 release never fires** -- MM2S ch6 spins forever

### Same pattern at compute tile (0,2):
- DMA S2MM ch0 acquires lock 0 (success)
- DMA MM2S ch2 waits for lock 3 forever (snapshot=0, PreconditionNotMet)

## Root Cause Hypothesis

The DMA transfer completion / lock release path has a bug. The S2MM transfer
receives all stream data and writes it to memory, but the `complete_transfer()`
function that releases the lock is never called, OR it's called but the
Transfer struct's state isn't `ReleasingLock` when it runs.

### Where to look:
- `src/device/dma/engine.rs` lines 1039-1088: The interaction between
  `do_transfer()` result, `data_transferred()` on the Transfer struct, and
  `tick()` on the timing FSM
- `src/device/dma/timing.rs` lines 223-236: The timing FSM DataTransfer phase
  counts words independently of actual data transfer
- `src/device/dma/transfer.rs` line 857: `data_transferred()` sets state to
  `ReleasingLock` only when `bytes_transferred >= total_bytes`

### Likely failure mode:
The timing FSM and the Transfer struct track progress independently. The timing
FSM counts words via `tick()` which runs every non-stall cycle. The Transfer
struct counts bytes via `data_transferred()`. If these get out of sync (e.g.,
timing FSM reaches Complete before Transfer struct reaches ReleasingLock),
then `complete_transfer()` runs but finds `transfer.state == Active` instead
of `ReleasingLock`, skipping the lock release entirely.

### Debugging next step:
Add a log line inside `complete_transfer()` that fires unconditionally (before
the `if let ReleasingLock` check) showing `transfer.state` and
`transfer.bytes_transferred` vs `transfer.total_bytes`. This will immediately
confirm whether the timing FSM outruns the transfer struct.
