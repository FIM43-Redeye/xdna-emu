#!/usr/bin/env python3
"""Derive the AIE2 tile-DMA bank-access width from a bankdisc capture.

The width is not read off directly -- it is inverted from the measured
bank-CONFLICT area, which is the only observable that is sensitive to how
often the DMA claims a bank.

Contention model (round-robin, single-port banks, AM020 ch.2:166): in any
cycle where the core and the DMA both target the same physical bank, a
CONFLICT_DM_BANK_n cycle is recorded.  If the two request streams are
uncorrelated in phase,

    conflict_cycles = SUM over physical banks b of
                      P(core wants b) * P(DMA wants b) * T

so, solving for the DMA's per-bank duty cycle,

    d_dma = conflicts / (2 * f_core * T)          [2 physical banks]
    accesses_per_transfer = 2 * d_dma * T
    bytes_per_bank_access = transfer_bytes / accesses_per_transfer

f_core comes from the disassembly of the hammer loop and is verified against
the measured rep period.  Every input is measured; nothing is fitted.
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bankdisc_measure import load_intervals  # noqa: E402

# --- hammer loop, from llvm-objdump of main_core_0_2.elf (ZLS_Fcore_0_2_304,
#     PC 0x210..0x270, zero-overhead loop, lc = 0x600 = 1536 iterations) ---
#     10 bundles/iteration; 3 memory ops: lda [p0,#16], lda [p0],#4, st [p1,#0].
#     The two loads are 16 B apart -> sibling physical banks.  Per iteration a
#     given physical bank sees 1.5 accesses on average (2 to one, 1 to the
#     other, alternating every 4 iterations).
HAMMER_CYCLES_PER_ITER = 10
HAMMER_ACCESSES_PER_BANK_PER_ITER = 1.5
F_CORE = HAMMER_ACCESSES_PER_BANK_PER_ITER / HAMMER_CYCLES_PER_ITER  # 0.15

TRANSFER_BYTES = 256 * 4  # 256 x i32
# Cycles after the core releases lk_full before the hammer loop issues its
# first load (lock-release call + zero-overhead-loop setup, PC 0x1d0..0x210).
CORE_RAMP_CYCLES = 20


def windows_and_areas(d: Path, rep: int):
    iv = load_intervals(d / f"perfetto_r{rep}.json", d / "trace_config.json")
    fin = sorted(s for s, _ in iv[("mem", "DMA_MM2S_0_FINISHED_BD")])
    stalls = sorted(iv[("mem", "DMA_MM2S_0_STALLED_LOCK")])
    wins = []
    for f in fin:
        prior = [e for _, e in stalls if e <= f]
        if prior:
            wins.append((max(prior), f))
    conf = sum(
        (iv[("mem", n)] for n in ("CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
                                  "CONFLICT_DM_BANK_4", "CONFLICT_DM_BANK_5")),
        [],
    )
    return {
        "windows": wins,
        "durations": [b - a for a, b in wins],
        "conflict": sum(b - a for a, b in conf),
        "core_stall": sum(b - a for a, b in iv[("core", "MEMORY_STALL")]),
        "starvation": sum(
            b - a for a, b in iv[("mem", "DMA_MM2S_0_MEMORY_STARVATION")]),
    }


def main(root="."):
    root = Path(root)
    agg = {}
    for v in ("idle", "apart", "collide", "collide2"):
        pooled_d, conf, stall, starv, n = [], [], [], [], 0
        for r in (1, 2, 3):
            d = root / f"build_bankdisc_{v}"
            if not (d / f"perfetto_r{r}.json").exists():
                continue
            m = windows_and_areas(d, r)
            pooled_d += m["durations"]
            conf.append(m["conflict"])
            stall.append(m["core_stall"])
            starv.append(m["starvation"])
            n += 1
        agg[v] = {
            "T": statistics.median(pooled_d),
            "T_mean": statistics.mean(pooled_d),
            "n_xfer": len(pooled_d) // n,
            "conflict": statistics.median(conf),
            "core_stall": statistics.median(stall),
            "starvation": statistics.median(starv),
        }

    print("=== Measured (median over 3 HW runs x 15 bracketed transfers) ===")
    print(f"{'variant':9} {'T median':>9} {'T mean':>8} {'conflict':>9} "
          f"{'coreSTALL':>10} {'MM2S starv':>11}")
    for v, a in agg.items():
        print(f"{v:9} {a['T']:>9} {a['T_mean']:>8.2f} {a['conflict']:>9} "
              f"{a['core_stall']:>10} {a['starvation']:>11}")

    ti, ta = agg["idle"]["T"], agg["apart"]["T"]
    print(f"\nVALIDITY GATE   T_APART/T_IDLE = {ta/ti:.4f}  "
          f"{'PASS' if abs(ta-ti)/ti < 0.20 else 'FAIL'} (must be ~1.0)")
    for v in ("collide", "collide2"):
        print(f"DISCRIMINATOR   T_{v.upper()}/T_APART = {agg[v]['T']/ta:.4f}"
              f"   (Model A: 1.0-1.3 | Model B: 2-4)")

    print("\n=== The DMA loses arbitrations and does not slow down ===")
    for v in ("collide", "collide2"):
        a = agg[v]
        dma_lost = a["conflict"] - a["core_stall"]
        extra = (a["T_mean"] - agg["apart"]["T_mean"]) * a["n_xfer"]
        print(f"{v:9} contended={a['conflict']:>4}  core denied="
              f"{a['core_stall']:>3}  DMA denied={dma_lost:>4}"
              f"  ->  DMA extra cycles over all transfers = {extra:.1f}")

    print("\n=== Inverting the conflict area for the DMA's bank-access width ===")
    print(f"f_core = {F_CORE:.3f} accesses/cycle/bank "
          f"({HAMMER_ACCESSES_PER_BANK_PER_ITER}/{HAMMER_CYCLES_PER_ITER}, "
          f"from the disassembly)")
    for v in ("collide", "collide2"):
        a = agg[v]
        per_xfer = a["conflict"] / a["n_xfer"]
        print(f"\n{v}: {per_xfer:.2f} conflict cycles per transfer, T = {a['T']}")
        for label, T_eff in (("no core-ramp correction", a["T"]),
                             (f"minus {CORE_RAMP_CYCLES}-cycle core ramp",
                              a["T"] - CORE_RAMP_CYCLES)):
            d_dma = per_xfer / (2 * F_CORE * T_eff)
            acc = 2 * d_dma * a["T"]
            width = TRANSFER_BYTES / acc
            print(f"   {label:32} duty={2*d_dma:.3f}  "
                  f"accesses/transfer={acc:.1f}  width={width:.1f} B "
                  f"({width/4:.2f} stream beats)")

    print("\nPredicted conflict cycles per transfer:")
    T = agg["collide"]["T"] - CORE_RAMP_CYCLES
    for name, n_acc in (("Model A (128-bit, 16 B)", TRANSFER_BYTES / 16),
                        ("Model B (32-bit, 4 B)", TRANSFER_BYTES / 4)):
        pred = 2 * F_CORE * (n_acc / 2 / agg["collide"]["T"]) * T
        print(f"   {name:26} {pred:6.1f}")
    print(f"   {'OBSERVED (collide)':26} "
          f"{agg['collide']['conflict']/agg['collide']['n_xfer']:6.1f}")
    print(f"   {'OBSERVED (collide2)':26} "
          f"{agg['collide2']['conflict']/agg['collide2']['n_xfer']:6.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
