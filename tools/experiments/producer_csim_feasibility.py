#!/usr/bin/env python3
"""C-feasibility microsim: can a FAITHFUL cycle-level model (core-priority + DMA
egress-FIFO + retry policy, NO phase constant, NO RNG) reproduce the robust HW
anchor -- dense collide ~= 1 conflict/granule (~1100/run), ~18-20 stalls,
STARVATION=0 -- ROBUSTLY across the initial phase offset?

If YES across all phi -> the colliding rate is a phase-robust attractor -> C is
feasible at CYCLE granularity (no sub-cycle re-architecture; likely a side effect
of the Part-2 core-priority fix + faithful retry timing).
If the conflict count SWINGS with phi (dodge ~0 at some phi, hammer-overshoot at
others) -> cycle-level is insufficient, the stable-1100 attractor needs sub-cycle
perturbation -> C needs re-architecture.

Model (one transfer = 256-word MM2S drain window; per-run = x16):
- Core: dense march, 1 access/cy to logical bank 0, physical sub-bank alternating
  4-on-4-off (period 8): sb_core(t) = ((t+phi)//4)%2. Throttled density d<1 via a
  duty mask. A dense core occupies SOME sub-bank every cycle.
- DMA MM2S: egress FIFO depth F. Each cycle: (1) stream pop 1 word if staged>0
  (else starvation if data remains); (2) present next in-order granule g (sub-bank
  g%2) per PRESENT policy; (3) core-priority arbitration; denied -> RETRY policy.
  Urgency: staged<=U forces the grant (core stalls).
Granule = 4 words = 1 sub-bank access, 1 cycle, brings in 4 stream words.
"""
OBJ = 256
NG = OBJ // 4          # 64 granules/transfer
REPS = 16
FIFO = 12              # archspec default (NOT HW-pinned; swept separately)


def transfer(phi, density=1.0, present="greedy", retry="hammer",
             U=0, fifo=FIFO, backoff=2):
    """Simulate ONE 256-word drain concurrent with the core. Returns
    (conflicts, stalls, starvation, drained_ok)."""
    staged = 0            # words in egress FIFO
    g = 0                 # next in-order granule to fetch
    drained = 0
    conflict = stall = starv = 0
    wait = 0              # cycles this granule has been denied (retry state)
    backoff_left = 0
    # duty: a throttled core is present on its logical bank `active` of every
    # `period` cycles; dense = present every cycle.
    def core_here(t):
        if density >= 0.999:
            return True, ((t + phi) // 4) % 2
        # throttle: present for a run of cycles then absent, keeping sub-bank
        # period-8 while present.
        period = 8
        active = max(1, round(density * period))
        present_now = (t % period) < active
        return present_now, ((t + phi) // 4) % 2

    t = 0
    while drained < OBJ and t < 100000:
        # (1) stream pop
        if staged > 0:
            staged -= 1
            drained += 1
        elif g >= NG:
            pass  # all fetched, just draining tail (can't happen: staged==0 & g==NG means done)
        else:
            starv += 1  # FIFO underflow with data still to fetch

        # (2) decide whether to present a fetch this cycle
        unfetched = NG - g
        room = staged + 4 <= fifo
        want = unfetched > 0 and room
        if present == "lazy":
            want = want and (staged <= fifo - 4)  # only top up when a full granule fits (same as greedy here)
        if present == "jit":
            want = want and (staged <= 4)          # just-in-time: only when FIFO nearly empty
        if retry == "backoff" and backoff_left > 0:
            backoff_left -= 1
            want = False

        if want:
            db = g % 2
            core_present, cb = core_here(t)
            urgent = staged <= U
            if core_present and cb == db and not urgent:
                # core-priority denies the DMA this cycle
                conflict += 1
                wait += 1
                if retry == "backoff":
                    backoff_left = backoff
            else:
                if core_present and cb == db and urgent:
                    stall += 1           # urgency forces grant over the core
                staged += 4
                g += 1
                wait = 0
        t += 1
    return conflict, stall, starv, drained == OBJ


def run(**kw):
    c = s = v = 0
    ok = True
    for _ in range(REPS):
        cc, ss, vv, o = transfer(**kw)
        c += cc; s += ss; v += vv; ok = ok and o
    return c, s, v, ok


HW = "HW collide: CONFLICT~1098-1103, STALL 18-20, STARV 0 (ROBUST across runs)"

def summarize(label, density, **kw):
    """Sweep phi 0-7; report per-run conflict/stall range and phase-robustness."""
    cs, ss, vs = [], [], []
    for phi in range(8):
        c, s, v, ok = run(phi=phi, density=density, **kw)
        cs.append(c); ss.append(s); vs.append(v)
    clo, chi = min(cs), max(cs)
    slo, shi = min(ss), max(ss)
    crob = "ROBUST" if chi <= 1.6 * max(1, clo) and clo > 0 else "SWINGS"
    srob = "ROBUST" if shi <= 1.6 * max(1, slo) else "SWINGS"
    print(f"{label:38} conflict [{clo:>4},{chi:>4}] {crob:6}  "
          f"stall [{slo:>4},{shi:>4}] {srob:6}  starv[{min(vs)},{max(vs)}]")
    return cs, ss


if __name__ == "__main__":
    print(HW)
    print(f"target: ~{NG*REPS} granules/run; ~1 conflict/granule => ~{NG*REPS}\n")

    print("--- Q1: is greedy+backoff's ~1000-conflict robustness stable vs physical params? ---")
    for fifo in (5, 8, 12, 16):
        for backoff in (1, 2, 4):
            summarize(f"DENSE greedy+backoff fifo={fifo} bo={backoff} U=1",
                      1.0, present="greedy", retry="backoff", U=1, fifo=fifo, backoff=backoff)
    print()
    print("--- Q2: per-phi detail for the central config (stall structure) ---")
    print(f"  {'phi':>3} {'conflict':>8} {'stall':>6} {'starv':>6}")
    for phi in range(8):
        c, s, v, ok = run(phi=phi, density=1.0, present="greedy", retry="backoff", U=1, backoff=2)
        print(f"  {phi:>3} {c:>8} {s:>6} {v:>6}")
    print()
    print("--- Q3: throttled densities under the same policy (HW: 68% is bimodal coin-flip) ---")
    for d in (0.5, 0.625, 0.68, 0.75, 0.875, 1.0):
        summarize(f"density={d:.3f} greedy+backoff bo=2 U=1",
                  d, present="greedy", retry="backoff", U=1, backoff=2)
