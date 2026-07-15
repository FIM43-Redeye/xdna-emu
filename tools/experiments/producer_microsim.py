#!/usr/bin/env python3
"""From-scratch behavioral microsim of the hypothesized producer-collision mechanism.

Goal: reproduce the HW K-sweep (core stalls 20/29/46/136 and CONFLICT 1103/98/141/443
per run at dwell K=4/8/16/32) from ONE mechanism with no fitted constants -- only the
known 12-word egress-FIFO depth.

Hypothesized mechanism:
  - A dense core marches, 1 store/cycle, camping physical bank A for K stores then B for
    K, alternating. A stall FREEZES the core (its store index does not advance).
  - The MM2S drains a stream at 1 word/cycle out of a 12-word egress FIFO.
  - The DMA fetches 16-byte granules (4 words) from memory IN STREAM ORDER; granule g
    targets physical bank g%2 (consecutive 16-byte granules flip bit4). It cannot fetch
    g+1 before g (in-order).
  - The core holds per-cycle bank priority. When the DMA needs granule g and the core is
    camped on g's bank, the DMA is blocked -- it force-grabs the bank (core STALLS one
    cycle) only when the FIFO is about to underflow; otherwise it waits.

The two policy knobs are physical, not fitted:
  - fetch_thresh: the DMA asserts a fetch for granule g when fifo <= fetch_thresh
    (how full it keeps the FIFO). Greedy = 8 (fetch whenever a granule fits).
  - force_thresh: it force-grabs a blocked granule when fifo <= force_thresh
    (underflow-imminent). 1 = grab only when the next drain would empty the FIFO.

Compare the shape across K; the per-transfer count times 16 transfers = per-run.
"""
FIFO_DEPTH = 12
OBJ = 256          # words per MM2S transfer
NG = OBJ // 4      # 64 granules
REPS = 16          # transfers per run

HW_STALLS = {4: 20, 8: 29, 16: 46, 32: 136}      # per run
HW_CONFLICT = {4: 1103, 8: 98, 16: 141, 32: 443}  # per run


def sim(K, fetch_thresh, force_thresh):
    """One transfer: core marches dwell-K concurrently with MM2S draining OBJ words."""
    fifo = 0        # words staged in egress FIFO
    g = 0           # next in-order granule to fetch
    s = 0           # next core store index (frozen on a stall)
    drained = 0     # words sent downstream
    conflicts = stalls = starv = 0
    t = 0
    while drained < OBJ and t < 100000:
        cb = (s // K) % 2                       # bank the core wants this cycle
        db = g % 2                              # bank the next granule needs
        wants = (g < NG) and (fifo <= fetch_thresh)  # DMA asserts a fetch?
        core_stalled = dma_fetched = False
        if wants:
            if db != cb:
                dma_fetched = True              # different banks -> both proceed
            else:
                conflicts += 1                  # same bank -> contention this cycle
                if fifo <= force_thresh:        # underflow imminent -> force-grab
                    dma_fetched = core_stalled = True
                # else core wins, DMA waits (stays on granule g)
        if dma_fetched:
            fifo += 4
            g += 1
        if not core_stalled:
            s += 1
        if fifo > 0:                            # stream drains 1 word/cy
            fifo -= 1
            drained += 1
        elif g < NG:                            # FIFO empty with data still to fetch
            starv += 1
        if core_stalled:
            stalls += 1
        t += 1
    return stalls, conflicts, starv


def run(fetch_thresh, force_thresh):
    print(f"\n=== fetch_thresh={fetch_thresh}  force_thresh={force_thresh} ===")
    print(f"{'K':>3} {'stalls/xfer':>11} {'stalls/run':>10} {'HWstall':>8} "
          f"{'conf/run':>9} {'HWconf':>7} {'starv':>6}")
    for K in (4, 8, 16, 32):
        st, cf, sv = sim(K, fetch_thresh, force_thresh)
        print(f"{K:>3} {st:>11} {st*REPS:>10} {HW_STALLS[K]:>8} "
              f"{cf*REPS:>9} {HW_CONFLICT[K]:>7} {sv:>6}")


if __name__ == "__main__":
    # greedy (retry-press) -- expected to over-count conflicts at high K
    run(fetch_thresh=8, force_thresh=1)
    # lazier fetch policies -- let the FIFO drain before topping up
    run(fetch_thresh=4, force_thresh=1)
    run(fetch_thresh=3, force_thresh=1)
    run(fetch_thresh=2, force_thresh=1)


def sim_reorder(K, R):
    """Out-of-order variant: the DMA may fetch any not-yet-fetched granule within R
    granules of the emit pointer whose bank is currently FREE (prefetch the opposite
    bank during a camp). It force-grabs (core stall) only when the granule the stream
    needs NOW (emit_g) is unfetched and sits on the core's camped bank. R = reorder
    depth in granules (the FIFO is 12 words = 3 granules)."""
    have = set()
    emit_g = 0        # next granule the stream must output (in order)
    emit_w = 0        # word within emit_g
    s = 0             # core store index (frozen on stall)
    conflicts = stalls = starv = 0
    t = 0
    while emit_g < NG and t < 200000:
        cb = (s // K) % 2
        core_stalled = False
        # choose a granule to fetch: lowest not-in-have within [emit_g, emit_g+R)
        # whose bank is free; else consider force-grab of emit_g.
        target = None
        for cand in range(emit_g, min(emit_g + R, NG)):
            if cand in have:
                continue
            if (cand % 2) != cb:        # free bank -> prefetch it
                target = cand
                break
        if target is not None:
            have.add(target)
        else:
            # nothing free to prefetch; must we force-grab to feed the stream?
            need_now = emit_g not in have
            if need_now:
                conflicts += 1          # emit_g is on the core's bank (else fetchable)
                have.add(emit_g)
                core_stalled = True
                stalls += 1
        if not core_stalled:
            s += 1
        # stream drains one word if the needed granule is present
        if emit_g in have:
            emit_w += 1
            if emit_w == 4:
                emit_w = 0
                emit_g += 1
        else:
            starv += 1
        t += 1
    return stalls, conflicts, starv


def run_reorder(R):
    print(f"\n=== REORDER model, reorder-depth R={R} granules ({R*4} words) ===")
    print(f"{'K':>3} {'stalls/run':>10} {'HWstall':>8} {'conf/run':>9} {'HWconf':>7} {'starv':>6}")
    for K in (4, 8, 16, 32):
        st, cf, sv = sim_reorder(K, R)
        print(f"{K:>3} {st*REPS:>10} {HW_STALLS[K]:>8} {cf*REPS:>9} {HW_CONFLICT[K]:>7} {sv:>6}")


for R in (3, 4, 6, 8):
    run_reorder(R)


def sim_starve(K, STARVE, FD=12):
    """AM020-faithful: round-robin realized as core-default-priority with anti-starvation.
    The DMA fetches granule g in order into a FD-word egress FIFO; when g's bank is the
    core's camped bank it CONFLICTs and retry-persists (requesting every cycle -> piles up
    CONFLICT_DM_BANK cycles) while the core wins -- UNTIL the DMA has been denied STARVE
    consecutive cycles, when anti-starvation forces the grant (core MEMORY_STALL). Tests
    whether one starvation threshold regenerates the whole curve."""
    fifo = g = s = drained = 0
    conflict = stall = starv = waited = 0
    t = 0
    while drained < OBJ and t < 200000:
        cb = (s // K) % 2
        db = g % 2
        wants = (g < NG) and (fifo + 4 <= FD)
        core_stalled = fetched = False
        if wants:
            if db != cb:
                fetched = True; waited = 0
            else:
                conflict += 1
                waited += 1
                if waited >= STARVE:          # anti-starvation forces the DMA grant
                    fetched = core_stalled = True; waited = 0
                # else core wins by default, DMA retry-persists next cycle
        else:
            waited = 0
        if fetched:
            fifo += 4; g += 1
        if not core_stalled:
            s += 1
        if fifo > 0:
            fifo -= 1; drained += 1
        elif g < NG:
            starv += 1
        if core_stalled:
            stall += 1
        t += 1
    return stall, conflict, starv


print("\n\n########## AM020-faithful: core-priority + anti-starvation + retry-persist ##########")
for STARVE in (2, 3, 4, 6, 8, 12):
    print(f"\n=== STARVE={STARVE} (DMA forced after {STARVE} denied cycles) ===")
    print(f"{'K':>3} {'stall/run':>9} {'HWstall':>8} {'conf/run':>9} {'HWconf':>7} {'starv':>6}")
    for K in (4, 8, 16, 32):
        st, cf, sv = sim_starve(K, STARVE)
        print(f"{K:>3} {st*REPS:>9} {HW_STALLS[K]:>8} {cf*REPS:>9} {HW_CONFLICT[K]:>7} {sv:>6}")


def sim_refined(K, R, STARVE):
    """Core-priority + out-of-order free-bank prefetch + retry-persist + anti-starvation.
    The DMA keeps up to R granules staged; each cycle it prefetches the lowest un-fetched
    granule in [emit, emit+R) whose bank is FREE (opposite the core's camp) -- no conflict.
    When the granule the stream needs next (emit_g) is unfetched AND on the core's camped
    bank, it can't prefetch it: it requests that bank (CONFLICT, retry-persist) and the
    core wins by priority -- UNTIL it's been denied STARVE cycles, when anti-starvation
    forces the grant (core MEMORY_STALL). emit is in-order; free-bank prefetch keeps the
    stream fed (starv~0) exactly as HW shows."""
    have = set()
    emit_g = emit_w = s = 0
    conflict = stall = starv = waited = 0
    t = 0
    while emit_g < NG and t < 300000:
        cb = (s // K) % 2
        occ = sum(1 for g in have if g >= emit_g)
        core_stalled = False
        # free-bank prefetch: lowest un-fetched granule in window on the free bank
        free_target = None
        if occ < R:
            for g in range(emit_g, min(emit_g + R, NG)):
                if g not in have and (g % 2) != cb:
                    free_target = g
                    break
        if free_target is not None:
            have.add(free_target)          # prefetch, no conflict, core also stores
            waited = 0
            s += 1
        else:
            # is the stream's next-needed granule blocked on the camped bank?
            blocking = emit_g if emit_g not in have else None
            if blocking is not None and (blocking % 2) == cb:
                conflict += 1               # DMA requests the camped bank, denied
                waited += 1
                if waited >= STARVE:        # anti-starvation forces the grant
                    have.add(blocking); waited = 0
                    core_stalled = True; stall += 1
                else:
                    s += 1                  # core wins by priority
            else:
                s += 1                      # nothing to fetch / buffer full; core stores
        # in-order stream drain
        if emit_g in have:
            emit_w += 1
            if emit_w == 4:
                emit_w = 0; emit_g += 1; waited = 0
        else:
            starv += 1
        t += 1
    return stall, conflict, starv


print("\n\n########## REFINED: core-priority + free-bank prefetch + retry-persist + anti-starve ##########")
for R in (3, 4):
    for STARVE in (3, 4, 6, 8, 12):
        print(f"\n=== R={R} granules, STARVE={STARVE} ===")
        print(f"{'K':>3} {'stall/run':>9} {'HWstall':>8} {'conf/run':>9} {'HWconf':>7} {'starv':>6}")
        for K in (4, 8, 16, 32):
            st, cf, sv = sim_refined(K, R, STARVE)
            print(f"{K:>3} {st*REPS:>9} {HW_STALLS[K]:>8} {cf*REPS:>9} {HW_CONFLICT[K]:>7} {sv:>6}")
