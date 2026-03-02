#!/usr/bin/env python3
"""MCP server that wraps aietools binaries for sandbox-safe execution.

Claude Code's sandbox isolates processes in a separate Linux network namespace,
which breaks FlexLM license validation (hostid mismatch). This MCP server runs
outside the sandbox as a separate process, giving aietools access to the host
network interfaces needed for license checks.

Protocol: MCP (Model Context Protocol) over stdio, JSON-RPC 2.0.
No external dependencies -- stdlib only.
"""

import json
import os
import subprocess
import sys
import time

# -- Configuration ----------------------------------------------------------

AIETOOLS_DIR = os.environ.get(
    "AIETOOLS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "aietools"),
)
AIETOOLS_BIN = os.path.join(AIETOOLS_DIR, "bin")

# Binaries allowed to be invoked. Anything not on this list is rejected.
ALLOWED_BINARIES = {
    "aiesimulator",
    "aiecompiler",
    "elfanalyzer",
    "eventanalyze",
    "hwanalyze",
    "mesimulator",
    "vcdanalyze",
    "vcdreader",
    "x86simulator",
    "xchesscc",
    "xchessmk",
    # Wrappers / helpers
    "loader",
    "aie_clang++",
    "aie_g++",
}

# Maximum runtime for any single invocation (seconds).
DEFAULT_TIMEOUT = 300

# -- Environment setup ------------------------------------------------------

def build_env():
    """Build environment for aietools subprocesses.

    Sets up license paths, library paths, and tool-specific variables
    that aietools binaries expect.
    """
    env = os.environ.copy()

    # License
    env.setdefault(
        "XILINXD_LICENSE_FILE",
        os.path.expanduser("~/.Xilinx/Xilinx.lic"),
    )

    # aietools paths
    env["AIETOOLS_DIR"] = AIETOOLS_DIR
    env.setdefault("XILINX_VITIS_AIETOOLS", AIETOOLS_DIR)
    env.setdefault("XILINX_VITIS", AIETOOLS_DIR)

    # Library paths -- appended (not prepended) to avoid shadowing system libs.
    aietools_lib = os.path.join(AIETOOLS_DIR, "lib", "lnx64.o")
    aietools_lib_ubuntu = os.path.join(aietools_lib, "Ubuntu")
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    if aietools_lib not in existing_ld:
        suffix = f":{aietools_lib_ubuntu}:{aietools_lib}"
        env["LD_LIBRARY_PATH"] = existing_ld + suffix if existing_ld else suffix.lstrip(":")

    # PATH -- ensure aietools/bin is present
    path = env.get("PATH", "")
    if AIETOOLS_BIN not in path:
        env["PATH"] = AIETOOLS_BIN + ":" + path

    return env


# -- MCP protocol helpers ---------------------------------------------------

def send(msg):
    """Write a JSON-RPC message to stdout."""
    raw = json.dumps(msg)
    sys.stdout.write(raw + "\n")
    sys.stdout.flush()


def send_result(id, result):
    send({"jsonrpc": "2.0", "id": id, "result": result})


def send_error(id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    send({"jsonrpc": "2.0", "id": id, "error": err})


# -- Tool implementation ----------------------------------------------------

TOOL_SCHEMA = {
    "name": "aietools_run",
    "description": (
        "Run an AMD aietools binary (aiesimulator, xchesscc, elfanalyzer, etc.) "
        "outside the sandbox so FlexLM licensing works. Returns stdout, stderr, "
        "and exit code."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "binary": {
                "type": "string",
                "description": (
                    "Name of the aietools binary to run. Must be one of: "
                    + ", ".join(sorted(ALLOWED_BINARIES))
                ),
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Command-line arguments to pass to the binary.",
                "default": [],
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command. Defaults to current directory.",
                "default": ".",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (default: {DEFAULT_TIMEOUT}).",
                "default": DEFAULT_TIMEOUT,
            },
        },
        "required": ["binary"],
    },
}


def handle_aietools_run(arguments):
    """Execute an aietools binary and return results."""
    binary = arguments.get("binary", "")
    args = arguments.get("args", [])
    cwd = arguments.get("cwd", ".")
    timeout = arguments.get("timeout", DEFAULT_TIMEOUT)

    # Validate binary name
    if binary not in ALLOWED_BINARIES:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Error: '{binary}' is not an allowed aietools binary.\n"
                        f"Allowed: {', '.join(sorted(ALLOWED_BINARIES))}"
                    ),
                }
            ],
            "isError": True,
        }

    # Resolve binary path
    bin_path = os.path.join(AIETOOLS_BIN, binary)
    if not os.path.exists(bin_path):
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: binary not found at {bin_path}",
                }
            ],
            "isError": True,
        }

    # Resolve working directory
    cwd = os.path.expanduser(cwd)
    if not os.path.isdir(cwd):
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: working directory does not exist: {cwd}",
                }
            ],
            "isError": True,
        }

    # Run the binary
    cmd = [bin_path] + args
    env = build_env()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        text_parts = []
        text_parts.append(f"exit_code: {result.returncode}")
        if result.stdout:
            text_parts.append(f"--- stdout ---\n{result.stdout}")
        if result.stderr:
            text_parts.append(f"--- stderr ---\n{result.stderr}")
        if not result.stdout and not result.stderr:
            text_parts.append("(no output)")

        return {
            "content": [{"type": "text", "text": "\n".join(text_parts)}],
            "isError": result.returncode != 0,
        }

    except subprocess.TimeoutExpired:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: command timed out after {timeout}s\ncmd: {' '.join(cmd)}",
                }
            ],
            "isError": True,
        }
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"Error: {type(e).__name__}: {e}\ncmd: {' '.join(cmd)}",
                }
            ],
            "isError": True,
        }


# -- MCP message dispatch ---------------------------------------------------

SERVER_INFO = {
    "name": "aietools",
    "version": "1.0.0",
}

CAPABILITIES = {
    "tools": {},  # We support tools
}


def handle_message(msg):
    """Dispatch a JSON-RPC request."""
    method = msg.get("method", "")
    id = msg.get("id")
    params = msg.get("params", {})

    if method == "initialize":
        send_result(id, {
            "protocolVersion": "2024-11-05",
            "serverInfo": SERVER_INFO,
            "capabilities": CAPABILITIES,
        })
    elif method == "notifications/initialized":
        pass  # Client notification, no response needed
    elif method == "tools/list":
        send_result(id, {"tools": [TOOL_SCHEMA]})
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        if tool_name == "aietools_run":
            result = handle_aietools_run(arguments)
            send_result(id, result)
        else:
            send_error(id, -32601, f"Unknown tool: {tool_name}")
    elif method == "ping":
        send_result(id, {})
    else:
        # Unknown method -- ignore notifications, error on requests
        if id is not None:
            send_error(id, -32601, f"Unknown method: {method}")


# -- Main loop --------------------------------------------------------------

def main():
    """Read JSON-RPC messages from stdin, dispatch, respond on stdout."""
    # Redirect our own stderr so it doesn't interfere with the protocol
    log_path = os.environ.get("AIETOOLS_MCP_LOG", "")
    if log_path:
        sys.stderr = open(log_path, "a")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            handle_message(msg)
        except json.JSONDecodeError as e:
            # Protocol error -- can't even parse the request
            send_error(None, -32700, f"Parse error: {e}")
        except Exception as e:
            # Internal error
            send_error(
                msg.get("id") if isinstance(msg, dict) else None,
                -32603,
                f"Internal error: {e}",
            )


if __name__ == "__main__":
    main()
