"""Memtile event selection: which DMA channel and which lock the SELn trace
events watch.

Memtile DMA events (DMA_S2MM_SEL0_START_TASK, DMA_MM2S_SEL1_FINISHED_BD, ...)
and lock events (LOCK_SEL0_REL, LOCK_SEL3_ACQ_GE, ...) are selector events:
the event says "the channel / lock that selector N points at", and two tile
registers say where it points.

    DMA_Event_Channel_Selection   S2MM_Sel0/1_Channel, MM2S_Sel0/1_Channel
    Locks_Event_Selection_<n>     Lock_Select, Lock_Value      (n = 0..7)

At reset every selector points at channel 0 / lock 0, so a trace that does not
program them only ever sees channel 0 and lock 0 whatever the design does.

The register names, field names, field widths, reset values and the channel
and lock counts all come from the AM025 register database shipped with
mlir-aie; this module only builds the names. The byte offsets are resolved by
mlir-aie itself (aie.trace.reg -> aie-trace-pack-reg-writes ->
aie-inline-trace-config), so nothing here knows an address.

Spec grammar (comma separated, case-insensitive):

    S2MM_SEL0:<ch> S2MM_SEL1:<ch> MM2S_SEL0:<ch> MM2S_SEL1:<ch>
    LOCK_SEL<n>:<lock>[@<value>]

``@<value>`` sets Lock_Value, the trigger value LOCK_SELn_EQUAL_TO_VALUE
compares against; it defaults to the register's reset value.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import toolchain_paths

_DMA_REGISTER = "DMA_Event_Channel_Selection"
_DMA_SLOTS = ("S2MM_SEL0", "S2MM_SEL1", "MM2S_SEL0", "MM2S_SEL1")
_LOCK_SELECTORS = 8
_REGDB_REL = Path("lib/Dialect/AIE/Util/aie_registers_aie2.json")


@dataclass(frozen=True)
class RegWrite:
    """One ``aie.trace.reg register=... field=... value=...`` field write."""
    register: str
    field: str
    value: int


@dataclass(frozen=True)
class Selection:
    """A resolved selection request."""
    dma: dict          # slot -> channel; all four slots, or {} if no DMA slot named
    locks: dict        # LOCK_SELn -> (lock id, Lock_Value)
    writes: tuple      # tuple[RegWrite, ...], in emission order

    def as_json(self) -> dict:
        """The trace_config.json record of what each selector meant."""
        rec: dict = {}
        if self.dma:
            rec["dma_channels"] = dict(self.dma)
        if self.locks:
            rec["locks"] = {slot: {"lock": lock, "value": value}
                            for slot, (lock, value) in self.locks.items()}
        return rec


def load_memtile_registers(regdb_path: str | Path | None = None) -> dict:
    """The memory_tile module of the AM025 database as {register name: entry}."""
    path = Path(regdb_path) if regdb_path else (
        toolchain_paths.resolve_mlir_aie() / _REGDB_REL)
    data = json.loads(path.read_text())
    return {r["name"]: r for r in data["modules"]["memory_tile"]["registers"]}


def _field(regdb: dict, register: str, field: str) -> dict:
    reg = regdb.get(register)
    if reg is None:
        raise ValueError(f"register {register} not in the memory_tile register database")
    for f in reg["bit_fields"]:
        if f["name"] == field:
            return f
    raise ValueError(f"field {field} not in register {register}")


def _width(f: dict) -> int:
    lo, hi = f["bit_range"]
    return hi - lo + 1


def _reset(f: dict) -> int:
    return int(f["reset"], 0)


def channel_count(regdb: dict, direction: str) -> int:
    """DMA channels per direction, counted from the DMA_<dir>_<n>_Ctrl registers."""
    pat = re.compile(rf"DMA_{direction}_\d+_Ctrl$")
    return sum(1 for name in regdb if pat.match(name))


def lock_count(regdb: dict) -> int:
    """Locks in the tile, counted from the Lock<n>_value registers."""
    return sum(1 for name in regdb if re.fullmatch(r"Lock\d+_value", name))


def _int(text: str, what: str) -> int:
    try:
        return int(text.strip(), 0)
    except ValueError:
        raise ValueError(f"{what}: invalid number '{text}'") from None


def _check_range(value: int, limit: int, what: str) -> None:
    if not 0 <= value < limit:
        raise ValueError(f"{what}: {value} out of range (valid 0-{limit - 1})")


def parse_selection(spec: str | None, regdb: dict) -> Selection | None:
    """Parse and validate a selection spec; None when nothing is requested."""
    if not spec or not spec.strip(" ,"):
        return None

    dma: dict[str, int] = {}
    locks: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(f"selection: missing ':' in '{tok}' (expected 'SLOT:VALUE')")
        slot, rest = (p.strip() for p in tok.split(":", 1))
        slot = slot.upper()
        is_lock = re.fullmatch(rf"LOCK_SEL[0-{_LOCK_SELECTORS - 1}]", slot) is not None
        if slot not in _DMA_SLOTS and not is_lock:
            raise ValueError(
                f"selection: unknown slot '{slot}' (expected one of "
                f"{list(_DMA_SLOTS)} or LOCK_SEL0..LOCK_SEL{_LOCK_SELECTORS - 1})")
        if slot in seen:
            raise ValueError(f"selection: slot {slot} given twice")
        seen.add(slot)

        if is_lock:
            n = slot[len("LOCK_SEL"):]
            reg = f"Locks_Event_Selection_{n}"
            sel_f = _field(regdb, reg, "Lock_Select")
            val_f = _field(regdb, reg, "Lock_Value")
            lock_s, _, value_s = rest.partition("@")
            lock = _int(lock_s, slot)
            _check_range(lock, min(lock_count(regdb), 1 << _width(sel_f)), f"{slot} lock")
            value = _int(value_s, f"{slot} value") if value_s else _reset(val_f)
            _check_range(value, 1 << _width(val_f), f"{slot} value")
            locks[slot] = (lock, value)
        else:
            if "@" in rest:
                raise ValueError(f"selection: '@value' applies only to LOCK_SEL slots, not {slot}")
            direction = slot.split("_")[0]
            f = _field(regdb, _DMA_REGISTER, f"{direction}_Sel{slot[-1]}_Channel")
            channel = _int(rest, slot)
            _check_range(channel, min(channel_count(regdb, direction), 1 << _width(f)),
                         f"{slot} channel")
            dma[slot] = channel

    writes: list[RegWrite] = []
    if dma:
        # The register is rewritten whole, so slots not named go out at reset.
        for slot in _DMA_SLOTS:
            f = _field(regdb, _DMA_REGISTER, f"{slot.split('_')[0]}_Sel{slot[-1]}_Channel")
            dma.setdefault(slot, _reset(f))
        dma = {slot: dma[slot] for slot in _DMA_SLOTS}
        for slot, channel in dma.items():
            writes.append(RegWrite(_DMA_REGISTER,
                                   f"{slot.split('_')[0]}_Sel{slot[-1]}_Channel", channel))
    for slot in sorted(locks, key=lambda s: int(s[len("LOCK_SEL"):])):
        lock, value = locks[slot]
        reg = f"Locks_Event_Selection_{slot[len('LOCK_SEL'):]}"
        writes.append(RegWrite(reg, "Lock_Select", lock))
        writes.append(RegWrite(reg, "Lock_Value", value))
    locks = {s: locks[s] for s in sorted(locks, key=lambda s: int(s[len("LOCK_SEL"):]))}
    return Selection(dma=dma, locks=locks, writes=tuple(writes))
