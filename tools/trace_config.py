#!/usr/bin/env python3
"""Load and validate trace_config.json files.

Single shared entry point for every tool in the trace pipeline. Schema lives
at tools/trace_config_schema.json; spec at
docs/archive/findings/2026-05-05-trace-config-schema.md.

Use as a library:

    from trace_config import load, dump, SCHEMA_VERSION
    cfg = load("traced/trace_config.json")
    print(cfg["buffer"]["kernel_arg_slot"])

Use from the shell to spot-check a file:

    python3 tools/trace_config.py validate path/to/trace_config.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).parent / "trace_config_schema.json"


def _load_schema() -> dict[str, Any]:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


# Cached at import time -- the schema doesn't change at runtime.
_SCHEMA = _load_schema()
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


def validate(data: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError on invalid input."""
    _VALIDATOR.validate(data)


def load(path: str | Path) -> dict[str, Any]:
    """Read trace_config.json from disk and validate it.

    Raises FileNotFoundError if the file is missing,
    json.JSONDecodeError if malformed, jsonschema.ValidationError if the
    contents don't match the schema.
    """
    with open(path) as f:
        data = json.load(f)
    validate(data)
    return data


def dump(data: dict[str, Any], path: str | Path) -> None:
    """Validate then write trace_config.json to disk.

    Validates BEFORE writing -- a malformed write would corrupt cached
    state for downstream tools.
    """
    validate(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _cli_validate(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: trace_config.py validate <path>", file=sys.stderr)
        return 2
    path = argv[0]
    try:
        cfg = load(path)
    except FileNotFoundError:
        print(f"error: {path} not found", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"error: {path}: malformed JSON: {e}", file=sys.stderr)
        return 1
    except jsonschema.ValidationError as e:
        print(f"error: {path}: schema violation: {e.message}", file=sys.stderr)
        if e.absolute_path:
            print(f"  at: {'.'.join(str(p) for p in e.absolute_path)}",
                  file=sys.stderr)
        return 1
    print(f"OK: {path} (schema_version={cfg['schema_version']}, "
          f"test_name={cfg['test_name']})")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: trace_config.py <validate> [args...]", file=sys.stderr)
        return 2
    cmd, *rest = sys.argv[1:]
    if cmd == "validate":
        return _cli_validate(rest)
    print(f"error: unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
