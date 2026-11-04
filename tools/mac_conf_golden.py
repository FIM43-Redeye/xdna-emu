#!/usr/bin/env python3
"""Build tools/golden/silicon/mac_conf.json from the MAC conf probe suites (gen_mac_conf_probe.py).

The golden holds NPU1 silicon outputs (run r1; r1 == r2 on every suite) for a sample of tiles of the
dense plans (termneg, same, flags, modes, fsingle, repro) and of the Chess sparse term_neg probe, each
with its operands, so `vector_matmul/conf_goldens.rs` can replay them through `execute_matmul`.

  python3 tools/mac_conf_golden.py <macconf-evidence-dir> [out.json]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_mac_conf_probe import cases, tile_data  # noqa: E402

SAMPLE = {"termneg": 1, "same": 1, "fsingle": 1, "repro": 1, "flags": 6, "modes": 4}
SPARSE_KEYS = {"2x8_8x8": (1, 3, 5), "2x16_16x8": (1, 2, 2), "4x16_16x8": (0, 1, 5)}


def main():
    root = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "golden/silicon/mac_conf.json"
    dense = []
    for plan, step in SAMPLE.items():
        suite = root / plan
        n = 0
        for case in cases(suite):
            meta, tiles = tile_data(suite, "r1", case)
            for t in tiles:
                n += 1
                if n % step:
                    continue
                dense.append(dict(plan=plan, kind=meta["kind"], conf=t["conf"], a=t["a"].hex(), b=t["b"].hex(),
                                  acc1=t["acc1"].hex(), acc2=t["acc2"].hex(), out=t["out"].hex()))
    sparse = []
    suite = root / "sparse"
    n = 0
    for case in cases(suite):
        meta = json.loads((suite / "cases" / case / "case.json").read_text())
        topo = json.loads((suite / "cases" / case / "topology.json").read_text())
        data = bytes.fromhex(topo["cores"][0]["args"][0]["data"])
        outb = (suite / "runs" / "r1" / case / "core.out.bin").read_bytes()
        key = SPARSE_KEYS[meta["geom"]]
        base = key[0] << 1 | key[1] << 3 | key[2] << 5
        for t, (_, conf_hi) in enumerate(meta["tiles"]):
            n += 1
            if n % 4:
                continue
            tile = data[t * 192:(t + 1) * 192]
            sparse.append(dict(geom=meta["geom"], key=list(key), conf=base | conf_hi << 8, a=tile[:64].hex(),
                               data=tile[64:128].hex(), mask=tile[128:144].hex(), out=outb[t * 128:(t + 1) * 128].hex()))
    out.write_text(json.dumps(dict(
        provenance="NPU1 silicon, tools/gen_mac_conf_probe.py suites, run r1 (== r2); dense tiles are Peano "
                   "builtins with the conf word in a register, sparse tiles Chess intrinsics (conf = key | sgn_y << 8)",
        dense=dense, sparse=sparse)))
    print(f"{len(dense)} dense, {len(sparse)} sparse tiles -> {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
