"""Tests for tools/toolchain_paths.py (mirrors the Rust resolution order)."""

import pytest

import toolchain_paths as tp


def _make_mlir_aie(root):
    sentinel = root / tp.MLIR_AIE_SENTINEL
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("{}")
    return root


def test_explicit_env_wins_over_npu_work_dir(tmp_path):
    a = _make_mlir_aie(tmp_path / "a")
    b = _make_mlir_aie(tmp_path / "work" / "mlir-aie")
    env = {"MLIR_AIE_PATH": str(a), "NPU_WORK_DIR": str(b.parent)}
    assert tp.resolve_mlir_aie(env=env) == a.resolve()


def test_alias_env_is_honored(tmp_path):
    a = _make_mlir_aie(tmp_path / "a")
    assert tp.resolve_mlir_aie(env={"MLIR_AIE_DIR": str(a)}) == a.resolve()


def test_npu_work_dir_next(tmp_path):
    m = _make_mlir_aie(tmp_path / "work" / "mlir-aie")
    env = {"NPU_WORK_DIR": str(tmp_path / "work")}
    assert tp.resolve_mlir_aie(env=env) == m.resolve()


def test_explicit_path_fails_closed_even_if_discovery_would_succeed(tmp_path):
    _make_mlir_aie(tmp_path / "mlir-aie")
    start = tmp_path / "repo"
    start.mkdir()
    with pytest.raises(tp.ToolchainPathError, match="MLIR_AIE_PATH"):
        tp.resolve_mlir_aie(start, {"MLIR_AIE_PATH": str(tmp_path / "nope")})


def test_blank_env_is_an_error(tmp_path):
    with pytest.raises(tp.ToolchainPathError, match="blank"):
        tp.resolve_mlir_aie(tmp_path, {"NPU_WORK_DIR": ""})


def test_missing_sentinel_is_rejected(tmp_path):
    (tmp_path / "mlir-aie").mkdir()
    with pytest.raises(tp.ToolchainPathError, match="sentinel"):
        tp.resolve_mlir_aie(env={"MLIR_AIE_PATH": str(tmp_path / "mlir-aie")})


def test_ancestor_discovery_from_inside_a_worktree(tmp_path):
    m = _make_mlir_aie(tmp_path / "mlir-aie")
    wt = tmp_path / "xdna-emu" / ".worktrees" / "feature" / "tools"
    wt.mkdir(parents=True)
    assert tp.resolve_mlir_aie(wt, {}) == m.resolve()


def test_ancestor_discovery_skips_a_sentinel_less_namesake(tmp_path):
    m = _make_mlir_aie(tmp_path / "mlir-aie")
    (tmp_path / "xdna-emu" / ".worktrees" / "mlir-aie").mkdir(parents=True)
    start = tmp_path / "xdna-emu" / ".worktrees" / "feature"
    start.mkdir()
    assert tp.resolve_mlir_aie(start, {}) == m.resolve()


def test_nothing_found_raises(tmp_path):
    with pytest.raises(tp.ToolchainPathError, match="not found"):
        tp.resolve_mlir_aie(tmp_path, {})


def test_events_header_path(tmp_path):
    m = _make_mlir_aie(tmp_path / "mlir-aie")
    assert tp.events_header(env={"MLIR_AIE_PATH": str(m)}) == (
        m.resolve() / tp.EVENTS_HEADER_REL)


def test_workspace_dir_prefers_npu_work_dir(tmp_path):
    (tmp_path / "work" / "xdna-emu-priv").mkdir(parents=True)
    got = tp.find_workspace_dir(
        "xdna-emu-priv", tmp_path, {"NPU_WORK_DIR": str(tmp_path / "work")})
    assert got == (tmp_path / "work" / "xdna-emu-priv").resolve()


def test_workspace_dir_ancestor_discovery_from_a_worktree(tmp_path):
    (tmp_path / "xdna-emu-priv").mkdir()
    wt = tmp_path / "xdna-emu" / ".worktrees" / "x"
    wt.mkdir(parents=True)
    assert tp.find_workspace_dir("xdna-emu-priv", wt, {}) == (
        tmp_path / "xdna-emu-priv").resolve()


def test_workspace_dir_absent_is_none(tmp_path):
    assert tp.find_workspace_dir("xdna-emu-priv", tmp_path, {}) is None
    assert tp.find_workspace_dir(
        "xdna-emu-priv", tmp_path, {"NPU_WORK_DIR": str(tmp_path)}) is None


def test_try_resolve_returns_none_instead_of_raising(tmp_path):
    assert tp.try_resolve_mlir_aie(tmp_path, {}) is None
    m = _make_mlir_aie(tmp_path / "mlir-aie")
    assert tp.try_resolve_mlir_aie(tmp_path, {}) == m.resolve()


def test_workspace_root_env_wins(tmp_path):
    assert tp.workspace_root(tmp_path, {"NPU_WORK_DIR": str(tmp_path)}) == tmp_path.resolve()


def test_workspace_root_from_main_checkout_and_worktree(tmp_path):
    (tmp_path / "ws" / "xdna-emu" / ".worktrees" / "x" / "tools").mkdir(parents=True)
    ws = (tmp_path / "ws").resolve()
    assert tp.workspace_root(tmp_path / "ws" / "xdna-emu" / "tools", {}) == ws
    assert tp.workspace_root(
        tmp_path / "ws" / "xdna-emu" / ".worktrees" / "x" / "tools", {}) == ws


def test_workspace_root_missing_raises(tmp_path):
    with pytest.raises(tp.ToolchainPathError):
        tp.workspace_root(tmp_path, {})
