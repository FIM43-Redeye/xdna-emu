"""Tests for tools/mlir-trace-inject.py."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
INJECTOR = REPO / "tools" / "mlir-trace-inject.py"
FIXTURES = REPO / "tools" / "tests" / "fixtures"
UNTRACED = FIXTURES / "sample_untraced.mlir"
MULTI_DEVICE = FIXTURES / "sample_multi_device.mlir"


def _run(args, check=True):
    # Use sys.executable so the subprocess inherits the same Python that is
    # running pytest.  The injector imports aie.ir which requires the ironenv
    # Python (not the system python3) -- using sys.executable ensures the test
    # works whether pytest was invoked via the activated venv or via a direct
    # path to the ironenv interpreter.
    #
    # Auto-supply --trace-config-out when we're injecting (anything that
    # writes an --out and isn't --no-op or --help). The injector errors out
    # if the flag is missing on injection paths; deriving the config path
    # from the --out parent dir keeps every test self-contained inside its
    # own tmp_path without each test needing to wire it up by hand.
    args = list(args)
    if (
        "--no-op" not in args
        and "--help" not in args
        and "--trace-config-out" not in args
        and "--out" in args
    ):
        out_path = Path(args[args.index("--out") + 1])
        args += ["--trace-config-out",
                 str(out_path.with_name("trace_config.json"))]
    return subprocess.run(
        [sys.executable, str(INJECTOR), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def test_injector_exists_and_prints_help():
    r = _run(["--help"])
    assert "usage" in r.stdout.lower() or "usage" in r.stderr.lower()


def test_injector_no_op_mode_round_trips(tmp_path):
    """With --no-op, injector should read and write the MLIR unchanged."""
    out = tmp_path / "out.mlir"
    r = _run(["--no-op", "--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"injector failed: stderr={r.stderr}"
    original = UNTRACED.read_text()
    result = out.read_text()
    # The mlir-aie parser may normalize whitespace; compare parsed structure,
    # not raw text. At minimum, tile count and op kinds should match.
    assert result.count("aie.tile") == original.count("aie.tile")
    assert result.count("aie.device") == original.count("aie.device")


ALREADY_TRACED = FIXTURES / "sample_already_traced.mlir"


def test_injector_bails_on_already_traced(tmp_path):
    """If input already has aie.trace ops, injector should refuse (exit 2)."""
    out = tmp_path / "out.mlir"
    r = _run(
        ["--input", str(ALREADY_TRACED), "--out", str(out)],
        check=False,
    )
    assert r.returncode == 2, f"expected exit 2, got {r.returncode}; stderr={r.stderr}"
    assert "already contains" in r.stderr.lower(), \
        f"stderr should cite 'already contains'; got: {r.stderr}"
    assert not out.exists(), "output file should not be written when injector refuses"


def test_no_op_round_trips_already_traced(tmp_path):
    """--no-op must not trigger the idempotency check (identity pass is always safe)."""
    out = tmp_path / "out.mlir"
    r = _run(
        ["--no-op", "--input", str(ALREADY_TRACED), "--out", str(out)],
        check=False,
    )
    assert r.returncode == 0, f"--no-op on already-traced input failed: stderr={r.stderr}"
    assert out.exists(), "output file should be written in --no-op mode"
    result = out.read_text()
    assert result.count("aie.trace") >= 1, "aie.trace op should survive round-trip"


def test_injector_adds_trace_decl_per_compute_tile(tmp_path):
    """Each non-shim tile in the input should get one aie.trace decl with
    the mandatory body (mode, packet, events, start/stop broadcasts)."""
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    # Fixture has one compute tile (0, 2). Shim tile (0, 0) is not compute.
    # Count aie.trace at the op-start level (not aie.trace.event/etc. sub-ops):
    # the decl form "aie.trace @" is uniquely the outer TraceOp (its body has
    # aie.trace.event but those are indented/nested).
    trace_count = result.count("aie.trace @")
    assert trace_count == 1, f"expected 1 aie.trace decl, got {trace_count}\n---\n{result}"
    # Symbol name should follow trace_t{col}_{row} convention.
    assert "@trace_t0_2" in result, f"sym_name missing; got:\n{result}"
    # Body must contain all the spec-mandated fields. Regressions that drop
    # the mode op, event list, or broadcast channels should fail this test.
    # Default mode is event_pc (mode 1) -- see mlir-trace-inject's
    # parse_args. To assert against a different mode, pass --trace-mode.
    assert "Event-PC" in result, "aie.trace.mode 'Event-PC' missing"
    assert "INSTR_VECTOR" in result, "INSTR_VECTOR event missing"
    assert "INSTR_EVENT_0" in result, "INSTR_EVENT_0 event missing"
    assert "INSTR_EVENT_1" in result, "INSTR_EVENT_1 event missing"
    # start broadcast=15, stop broadcast=14 (mlir-aie defaults).
    assert "15" in result and "14" in result, "broadcast channels 15/14 missing"


def test_injector_adds_runtime_sequence_trace_config(tmp_path):
    """The aie.runtime_sequence body should start with trace.host_config +
    one trace.start_config per trace decl, before existing runtime ops."""
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(UNTRACED),
        "--out", str(out),
        "--buffer-size", "16384",
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    assert "aie.trace.host_config" in result, f"host_config missing in output:\n{result}"
    assert "aie.trace.start_config" in result, f"start_config missing in output:\n{result}"
    # Start_config should reference the trace symbol by name.
    assert "@trace_t0_2" in result, "start_config should reference @trace_t0_2"
    # Custom buffer size should flow through.
    assert "16384" in result, "custom --buffer-size did not reach the output"


def test_injector_default_buffer_size_used_when_not_specified(tmp_path):
    """If --buffer-size is omitted, the default (8192) should appear."""
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    assert "aie.trace.host_config" in result
    assert "8192" in result, f"default buffer size missing:\n{result}"


def test_injector_targets_device_with_runtime_sequence(tmp_path):
    """Regression: when a module has multiple aie.device ops (e.g.
    ctrl_packet_reconfig with @base + @main), trace-inject must land the
    aie.trace decls in the device that contains the aie.runtime_sequence,
    not the first device it encounters. Otherwise orphan trace decls end up
    in the overlay device and downstream aiecc --device-name=<overlay>
    compile fails with "aie.trace ops found but no runtime_sequence
    defined"."""
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(MULTI_DEVICE), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()

    # Trace decl + runtime_sequence config must land in the same device
    # body. Split the output into per-device chunks and check which chunk
    # owns each op.
    lines = result.splitlines()
    device_boundaries = [
        i for i, line in enumerate(lines) if "aie.device(" in line
    ]
    assert len(device_boundaries) == 2, (
        f"expected 2 aie.device ops in output, got "
        f"{len(device_boundaries)}:\n{result}"
    )
    base_chunk = "\n".join(lines[device_boundaries[0]:device_boundaries[1]])
    main_chunk = "\n".join(lines[device_boundaries[1]:])

    # @base should be clean: tiles only, no trace decls or config.
    assert "aie.trace @" not in base_chunk, (
        f"trace decls leaked into @base (the overlay device):\n{base_chunk}"
    )
    assert "aie.trace.host_config" not in base_chunk
    assert "aie.trace.start_config" not in base_chunk

    # @main must carry the trace decls and runtime_sequence host/start config.
    assert "aie.trace @trace_t0_2" in main_chunk, (
        f"trace decl missing from @main:\n{main_chunk}"
    )
    assert "aie.trace.host_config" in main_chunk
    assert "aie.trace.start_config" in main_chunk


def test_injector_no_devices_have_runtime_sequence_is_clean_noop(tmp_path):
    """If NO device in the module has an aie.runtime_sequence, trace-inject
    should not leave orphan trace decls anywhere (they would break downstream
    aiecc). It can warn and exit 0 without injecting, which is a cleaner
    signal than a partially-injected output."""
    # Build a fixture on the fly: a single @overlay device with tiles only.
    fixture = tmp_path / "no_rs.mlir"
    fixture.write_text(
        "module {\n"
        "  aie.device(npu1_1col) @overlay {\n"
        "    %tile_0_0 = aie.tile(0, 0)\n"
        "    %tile_0_2 = aie.tile(0, 2)\n"
        "  }\n"
        "}\n"
    )
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(fixture), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    assert "aie.trace @" not in result, (
        "trace decls should not be injected when no device has a "
        f"runtime_sequence; output:\n{result}"
    )


# ---------------------------------------------------------------------------
# Memtile injection (#373 / Stage 2)
# ---------------------------------------------------------------------------

_FIXTURE_WITH_MEMTILE = """\
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.core(%tile_0_2) {
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
    }
  }
}
"""


def _write_memtile_fixture(tmp_path) -> Path:
    """Build a self-contained MLIR fixture with shim+memtile+compute tiles."""
    fp = tmp_path / "memtile_fixture.mlir"
    fp.write_text(_FIXTURE_WITH_MEMTILE)
    return fp


def test_memtile_injection_default_off(tmp_path):
    """Without --memtile-sweep-events, row-1 tiles must NOT receive trace ops.

    Stage 2 must preserve pre-#373 behaviour for callers that haven't opted
    in. The fixture has a memtile at (0,1); injecting with no flag should
    produce a core trace at @trace_t0_2 only -- no @trace_memtile_0_1.
    """
    fixture = _write_memtile_fixture(tmp_path)
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(fixture), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    assert "@trace_t0_2" in result, "core trace decl missing"
    assert "@trace_memtile_0_1" not in result, (
        f"memtile trace leaked without opt-in flag:\n{result}"
    )
    assert "TracePacketType.MemTile" not in result, (
        "memtile packet type appeared without opt-in"
    )


def test_memtile_injection_with_sweep_flag(tmp_path):
    """With --memtile-sweep-events all, row-1 tiles must receive a MemTile
    trace op with the upstream PORT_RUNNING DMA-channel default events,
    paired with one trace.port slot per event."""
    fixture = _write_memtile_fixture(tmp_path)
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(fixture),
        "--out", str(out),
        "--memtile-sweep-events", "all",
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()

    # Memtile trace decl should land on (0,1) with the conventional sym.
    assert "@trace_memtile_0_1" in result, (
        f"memtile trace decl missing:\n{result}"
    )
    # Packet type must be memtile, not core or shimtile. The aie dialect's
    # custom printer uses lowercase keywords for the TracePacketType enum.
    assert "type = memtile" in result, "memtile packet type missing"
    # All 8 PORT_RUNNING events should be in the body.
    for slot in range(8):
        assert f"PORT_RUNNING_{slot}" in result, (
            f"PORT_RUNNING_{slot} missing for memtile"
        )
    # Each PORT_RUNNING event needs a paired trace.port slot config.
    # The aie.trace.port op is what binds slot N to (port=DMA, channel,
    # direction). 8 events => 8 unique slot configs.
    assert result.count("aie.trace.port") == 8, (
        f"expected 8 aie.trace.port ops, got {result.count('aie.trace.port')}\n{result}"
    )
    # Both directions should appear (slots 0-3 are S2MM, 4-7 are MM2S).
    assert "S2MM" in result, "S2MM direction missing on memtile port config"
    assert "MM2S" in result, "MM2S direction missing on memtile port config"
    # Runtime sequence must reference the memtile sym in start_config.
    assert "aie.trace.start_config @trace_memtile_0_1" in result, (
        "runtime_sequence missing memtile start_config"
    )


def test_core_port_running_emits_trace_port(tmp_path):
    """Core tiles must support PORT_RUNNING_<N> events paired with an
    aie.trace.port slot config, mirroring the memtile path.

    Faithful to mlir-aie setup.py::_get_default_events_for_tile (core branch):
        PORT_RUNNING_0 -> (DMA, channel 0, master=True)  -> S2MM-0 recv
        PORT_RUNNING_1 -> (DMA, channel 0, master=False) -> MM2S-0 send
    The core layout is interleaved by direction (channel=slot//2,
    master=slot even), distinct from the memtile's grouped layout.
    """
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(UNTRACED),
        "--out", str(out),
        "--core-sweep-events", "PORT_RUNNING_0,PORT_RUNNING_1",
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    assert "@trace_t0_2" in result, "core trace decl missing"
    assert "PORT_RUNNING_0" in result and "PORT_RUNNING_1" in result, (
        f"PORT_RUNNING events missing from core body:\n{result}"
    )
    # Each PORT_RUNNING event needs a paired trace.port slot config -- this is
    # the plumbing that was previously absent on the core path.
    assert result.count("aie.trace.port") == 2, (
        f"expected 2 aie.trace.port ops, got "
        f"{result.count('aie.trace.port')}\n{result}"
    )
    # PORT_RUNNING_0 = S2MM (recv into core), PORT_RUNNING_1 = MM2S (send out).
    assert "S2MM" in result, "S2MM direction missing on core port config"
    assert "MM2S" in result, "MM2S direction missing on core port config"


def test_memtile_sel_channels_default_off(tmp_path):
    """Without --memtile-sel-channels, the runtime sequence must NOT emit
    a write to the DMA_Event_Channel_Selection register (0xA06A0).

    Leaving the register at its reset value is the safe default -- we don't
    want to clobber it for tests that aren't asking for memtile DMA SEL
    events. The reset value (all SEL slots at channel 0) matches what
    chaining_channels and similar examples assume.
    """
    fixture = _write_memtile_fixture(tmp_path)
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(fixture),
        "--out", str(out),
        "--memtile-sweep-events", "all",
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    # The register offset 0xA06A0 = 0x000A06A0 must not appear as an
    # npu_write32 target in the output. Match on the address attribute
    # rather than the bare hex so this test isn't fooled by an unrelated
    # 0xA06A0 elsewhere (e.g. in a comment).
    assert "address = 0xa06a0" not in result.lower(), (
        f"unexpected DMA_Event_Channel_Selection write without --memtile-sel-channels:\n{result}"
    )


def test_memtile_sel_channels_emits_npu_write32(tmp_path):
    """With --memtile-sel-channels, the runtime sequence must emit one
    npu_write32 per memtile to program 0xA06A0 with the packed value.

    Spec: 'S2MM_SEL1:1,MM2S_SEL1:1' -> S2MM_SEL0=0 (default), S2MM_SEL1=1,
    MM2S_SEL0=0 (default), MM2S_SEL1=1.
    Packed value: (0)<<0 | (1)<<8 | (0)<<16 | (1)<<24 = 0x01000100.
    """
    fixture = _write_memtile_fixture(tmp_path)
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(fixture),
        "--out", str(out),
        "--memtile-sweep-events", "all",
        "--memtile-sel-channels", "S2MM_SEL1:1,MM2S_SEL1:1",
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    # MLIR's standard pretty-printer emits ui32 attributes in decimal, so
    # accept either the hex form (0xA06A0 / 0x01000100) or the decimal
    # equivalents (657056 / 16777472) the printer actually writes.
    lower = result.lower()
    assert (
        "0xa06a0" in lower
        or "0x000a06a0" in lower
        or "address = 657056" in lower
    ), f"expected DMA_Event_Channel_Selection write at 0xA06A0:\n{result}"
    assert (
        "0x1000100" in lower
        or "0x01000100" in lower
        or "value = 16777472" in lower
    ), f"expected packed SEL value 0x01000100:\n{result}"


def test_memtile_sel_channels_invalid_slot_rejected(tmp_path):
    """Bad slot names produce a friendly error, not a stack trace."""
    fixture = _write_memtile_fixture(tmp_path)
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(fixture),
        "--out", str(out),
        "--memtile-sweep-events", "all",
        "--memtile-sel-channels", "S2MM_SEL2:0",  # SEL2 doesn't exist
    ], check=False)
    assert r.returncode != 0, "expected non-zero exit on invalid slot"
    assert "unknown slot" in r.stderr.lower() or "S2MM_SEL2" in r.stderr


def test_memtile_sel_channels_out_of_range_channel_rejected(tmp_path):
    """Channel numbers outside 0..5 are rejected (3-bit field)."""
    fixture = _write_memtile_fixture(tmp_path)
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(fixture),
        "--out", str(out),
        "--memtile-sweep-events", "all",
        "--memtile-sel-channels", "S2MM_SEL0:6",  # 6 is out of range
    ], check=False)
    assert r.returncode != 0, "expected non-zero exit on out-of-range channel"
    assert "out of range" in r.stderr.lower()


def test_memtile_injection_in_trace_config(tmp_path):
    """Memtile entries must appear in the trace_config.json with kind=memtile."""
    import json

    fixture = _write_memtile_fixture(tmp_path)
    out = tmp_path / "out.mlir"
    cfg_path = tmp_path / "trace_config.json"
    r = _run([
        "--input", str(fixture),
        "--out", str(out),
        "--memtile-sweep-events", "all",
        "--trace-config-out", str(cfg_path),
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    cfg = json.loads(cfg_path.read_text())
    memtile_entries = [
        t for t in cfg["tiles_traced"] if t["kind"] == "memtile"
    ]
    assert len(memtile_entries) == 1, (
        f"expected 1 memtile entry, got {memtile_entries}"
    )
    entry = memtile_entries[0]
    assert (entry["col"], entry["row"]) == (0, 1)
    # Events list should mirror the upstream defaults (8 PORT_RUNNING_X).
    assert len(entry["events"]) == 8
    assert all(e.startswith("PORT_RUNNING_") for e in entry["events"])
    # tracing.memtile_sweep should also reflect the resolved sweep when 'all'.
    assert cfg["tracing"]["memtile_sweep"] == ["all"], (
        f"unexpected memtile_sweep: {cfg['tracing']['memtile_sweep']}"
    )


# ---------------------------------------------------------------------------
# Memmod injection (#374 / Stage 3)
# ---------------------------------------------------------------------------


def test_memmod_injection_default_off(tmp_path):
    """Without --memmod-sweep-events, compute tiles must NOT receive a
    second (memmod) aie.trace decl. Stage 3 must preserve pre-#374
    behaviour for callers that haven't opted in.
    """
    out = tmp_path / "out.mlir"
    r = _run(["--input", str(UNTRACED), "--out", str(out)])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()
    # Core decl present.
    assert "@trace_t0_2" in result
    # Memmod decl absent.
    assert "@trace_mem_0_2" not in result, (
        f"memmod decl leaked without opt-in flag:\n{result}"
    )
    # Compute tile should appear exactly once in the trace_config (kind=core,
    # module=core), NOT twice with module=mem.
    cfg_path = out.with_name("trace_config.json")
    if cfg_path.exists():
        import json
        cfg = json.loads(cfg_path.read_text())
        compute_entries = [
            t for t in cfg["tiles_traced"]
            if t["kind"] == "core"
        ]
        assert len(compute_entries) == 1, (
            f"expected 1 compute entry, got {compute_entries}"
        )
        assert compute_entries[0].get("module") == "core"


def test_memmod_injection_with_sweep_flag(tmp_path):
    """With --memmod-sweep-events all, every compute tile should get a
    second aie.trace decl alongside its core decl, with packet type Mem
    and the 8 default memmod events (DMA START_TASKs + bank conflicts +
    edge detection)."""
    out = tmp_path / "out.mlir"
    r = _run([
        "--input", str(UNTRACED),
        "--out", str(out),
        "--memmod-sweep-events", "all",
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    result = out.read_text()

    # Both decls land on the same compute tile (0,2): core + memmod.
    assert "@trace_t0_2" in result, "core decl missing"
    assert "@trace_mem_0_2" in result, (
        f"memmod decl missing:\n{result}"
    )
    # Packet type for memmod is `mem` (the dialect lowercases the enum).
    assert "type = mem" in result, "memmod packet type missing"
    # All 8 memmod default events should appear.
    for ev in (
        "DMA_S2MM_0_START_TASK", "DMA_MM2S_0_START_TASK",
        "CONFLICT_DM_BANK_0", "CONFLICT_DM_BANK_1",
        "CONFLICT_DM_BANK_2", "CONFLICT_DM_BANK_3",
        "EDGE_DETECTION_EVENT_0", "EDGE_DETECTION_EVENT_1",
    ):
        assert ev in result, f"memmod default event {ev!r} missing"
    # Memmod uses GenericEvents only, no aie.trace.port slot bindings.
    # The fixture has no shim/memtile injection, so any aie.trace.port
    # in the output would be unexpected.
    assert "aie.trace.port" not in result, (
        "memmod injection should not emit trace.port ops"
    )
    # Runtime sequence must reference the memmod sym in start_config.
    assert "aie.trace.start_config @trace_mem_0_2" in result, (
        "runtime_sequence missing memmod start_config"
    )


def test_memmod_injection_in_trace_config(tmp_path):
    """Memmod entries must appear in trace_config.json as a second entry
    on the same compute tile (col, row) with module='mem'."""
    import json
    out = tmp_path / "out.mlir"
    cfg_path = tmp_path / "trace_config.json"
    r = _run([
        "--input", str(UNTRACED),
        "--out", str(out),
        "--memmod-sweep-events", "all",
        "--trace-config-out", str(cfg_path),
    ])
    assert r.returncode == 0, f"stderr={r.stderr}"
    cfg = json.loads(cfg_path.read_text())
    compute_entries = [
        t for t in cfg["tiles_traced"] if t["kind"] == "core"
    ]
    assert len(compute_entries) == 2, (
        f"expected 2 compute entries (core+memmod), got {compute_entries}"
    )
    by_module = {t["module"]: t for t in compute_entries}
    assert set(by_module.keys()) == {"core", "mem"}
    # Same tile coords on both entries.
    assert (by_module["core"]["col"], by_module["core"]["row"]) == (0, 2)
    assert (by_module["mem"]["col"], by_module["mem"]["row"]) == (0, 2)
    # Memmod events list mirrors the upstream defaults (8 GenericEvents).
    assert len(by_module["mem"]["events"]) == 8
    assert by_module["mem"]["events"][0] == "DMA_S2MM_0_START_TASK"
    # tracing.memmod_sweep should reflect the resolved sweep when 'all'.
    assert cfg["tracing"]["memmod_sweep"] == ["all"]


# ---------------------------------------------------------------------------
# Integration tests: injected MLIR survives a real aiecc.py compile
# ---------------------------------------------------------------------------

# Small, known-good aiecc test fixture with an inline core and no link_with
# deps.  Uses npu1_1col — one of the device names accepted by the installed
# mlir-aie Python bindings (unlike the newer "NPUDEVICE" alias used by some
# npu-xrt tests).  Located in mlir-aie's test/aiecc/ tree, which is
# specifically designed for aiecc.py compilation testing.
BRIDGE_TEST_MLIR = Path(
    "/home/triple/npu-work/mlir-aie/test/aiecc/cpp_npu_and_xclbin.mlir"
)


def test_injector_output_compiles_with_aiecc(tmp_path):
    """Traced MLIR should compile cleanly via aiecc.py."""
    if not BRIDGE_TEST_MLIR.exists():
        pytest.skip(f"bridge test MLIR not found: {BRIDGE_TEST_MLIR}")
    if shutil.which("aiecc.py") is None:
        pytest.skip("aiecc.py not on PATH; activate mlir-aie environment")

    traced = tmp_path / "aie-traced.mlir"
    # Inject trace ops.
    r = _run(["--input", str(BRIDGE_TEST_MLIR), "--out", str(traced)])
    assert r.returncode == 0, f"injector failed: stderr={r.stderr}"

    # Verify the injected MLIR has the expected trace structure before compile.
    traced_text = traced.read_text()
    assert "aie.trace @" in traced_text, "no trace decl in injected MLIR"
    assert "aie.trace.host_config" in traced_text, "host_config missing"
    assert "aie.trace.start_config" in traced_text, "start_config missing"

    # Compile through the full aiecc pipeline.
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    r2 = subprocess.run(
        [
            "aiecc.py",
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
            f"--xclbin-name={build_dir}/aie-traced.xclbin",
            f"--npu-insts-name={build_dir}/insts.bin",
            str(traced),
        ],
        capture_output=True,
        text=True,
        cwd=str(build_dir),
    )
    assert r2.returncode == 0, (
        f"aiecc failed:\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"
    )
    assert (build_dir / "aie-traced.xclbin").exists(), "xclbin was not produced"


def test_compiled_traced_xclbin_has_trace_buffer_slot(tmp_path):
    """The xclbin from a traced MLIR should reserve a buffer slot for the
    trace data.

    Note on 'trace' kernarg naming: the task plan expected xclbinutil to
    show a kernarg literally named "trace".  That naming convention is a
    newer mlir-aie feature (post xlnx_rel_v2025.2) not yet in the installed
    toolchain.  The installed version allocates a generic bo<N> slot for the
    trace buffer (at the arg_idx specified in trace_host_config, default=4).

    This test verifies what IS observable with the installed toolchain:
      1. xclbinutil can read the xclbin and extract its EMBEDDED_METADATA
      2. The metadata XML contains a kernel definition with at least one
         buffer argument at arg_idx >= 4 (the trace slot position), confirming
         the AIEInsertTraceFlows lowering pass processed the trace config
         and reserved the expected slot.

    When the toolchain is updated to a version that names the slot "trace",
    replace the XML arg-count assertion with 'assert "trace" in metadata_xml'.

    Baseline note: the untraced `cpp_npu_and_xclbin.mlir` produces a kernel
    with args at id=0..6 (opcode, instr, ninstr, plus the test's 4 buffers).
    The id="7" slot only exists after AIEInsertTraceFlows appends the trace
    buffer.  If this test ever starts passing on untraced MLIR (the baseline
    grows to >= 8 args for unrelated reasons), swap in an explicit
    untraced-vs-traced arg-count comparison.
    """
    if not BRIDGE_TEST_MLIR.exists():
        pytest.skip(f"bridge test MLIR not found: {BRIDGE_TEST_MLIR}")
    if shutil.which("aiecc.py") is None:
        pytest.skip("aiecc.py not on PATH; activate mlir-aie environment")
    if shutil.which("xclbinutil") is None:
        pytest.skip("xclbinutil not on PATH")

    traced = tmp_path / "aie-traced.mlir"
    r = _run(["--input", str(BRIDGE_TEST_MLIR), "--out", str(traced)])
    assert r.returncode == 0, f"injector failed: stderr={r.stderr}"

    build_dir = tmp_path / "build"
    build_dir.mkdir()
    r2 = subprocess.run(
        [
            "aiecc.py",
            "--aie-generate-xclbin",
            "--aie-generate-npu-insts",
            "--no-compile-host",
            f"--xclbin-name={build_dir}/aie-traced.xclbin",
            f"--npu-insts-name={build_dir}/insts.bin",
            str(traced),
        ],
        capture_output=True,
        text=True,
        cwd=str(build_dir),
    )
    assert r2.returncode == 0, (
        f"aiecc failed:\nSTDOUT:\n{r2.stdout}\nSTDERR:\n{r2.stderr}"
    )
    xclbin = build_dir / "aie-traced.xclbin"
    assert xclbin.exists(), "xclbin was not produced"

    # Dump the EMBEDDED_METADATA section so we can inspect kernel arg layout.
    metadata_path = tmp_path / "metadata.xml"
    info = subprocess.run(
        [
            "xclbinutil",
            f"--dump-section=EMBEDDED_METADATA:RAW:{metadata_path}",
            "--input",
            str(xclbin),
        ],
        capture_output=True,
        text=True,
    )
    assert info.returncode == 0, f"xclbinutil metadata dump failed: {info.stderr}"
    assert metadata_path.exists(), "EMBEDDED_METADATA not produced"

    metadata_xml = metadata_path.read_text()
    # The metadata must contain a kernel definition (sanity check).
    assert "<kernel" in metadata_xml, f"no kernel in EMBEDDED_METADATA:\n{metadata_xml}"
    # The trace_host_config default arg_idx=4 reserves a buffer slot at id=7
    # (after the 3 fixed args: opcode/instr/ninstr).  Verify that slot exists
    # in the metadata, confirming AIEInsertTraceFlows processed the trace config.
    # In a future mlir-aie version this arg will be named "trace"; for now it
    # appears as "bo4" at id="7".
    assert 'id="7"' in metadata_xml, (
        "expected trace buffer slot at arg id=7 (bo4) in EMBEDDED_METADATA; "
        "either AIEInsertTraceFlows did not run or arg numbering changed.\n"
        f"Metadata:\n{metadata_xml}"
    )
