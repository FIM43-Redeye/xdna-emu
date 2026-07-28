#!/bin/bash
# Reproduce the aiesimulator producer-probe collision capture end-to-end.
# Observes AIE2 compute-tile memory-bank arbitration (core march-store vs tile
# MM2S) in AMD's cycle-accurate ISS. Read-only reference use of aietools.
set -e
cd "$(dirname "$0")"
source /home/triple/npu-work/toolchain-build/activate-npu-env.sh >/dev/null 2>&1
TL=/home/triple/npu-work/mlir-aie/install/runtime_lib/x86_64/test_lib

# 1. Compile the array (Chess) + generate aie_inc.cpp + build ps.so WITH test.cpp
#    (test.cpp defines ps_main via -Dmain(...)=ps_main(...); passing it as a host
#    source folds it into the ps.so link so ps_main is DEFINED, not undefined.)
rm -rf collide_sim.mlir.prj
aiecc.py --xchesscc --xbridge --aiesim collide_sim.mlir \
  -I"$TL/include" -L"$TL/lib" -ltest_lib test.cpp

# sanity: ps_main must be a defined symbol (T), not undefined (U)
nm -D collide_sim.mlir.prj/sim/ps/ps.so | grep ' T _Z7ps_mainz' \
  || { echo "ps_main UNDEFINED -- build broken"; exit 1; }

# 2. Run the cycle-accurate simulator to completion (VCD + profile + summary).
rm -rf aiesim_out foo.vcd
aiesimulator --pkg-dir=collide_sim.mlir.prj/sim --profile \
  --dump-vcd foo --simulation-cycle-timeout=60000 --output-dir=aiesim_out

# Outputs:
#   foo.vcd                              per-cycle waveform (whole array)
#   aiesim_out/profile_funct_7_0.txt     our core (tile_7_3) cycle/instr totals
#   aiesim_out/default.aierun_summary    flow metadata + report file index
#
# Inventory a signal:  python3 vcd_probe.py foo.vcd <name-substring> ...
# e.g.: python3 vcd_probe.py foo.vcd tile_7_3.mm.dm.conflict_
