# SPDX-License-Identifier: MIT
"""Per-mode byte-stream decoders.

Each module in this package consumes a payload byte stream (already
de-interleaved per tile) and yields ``TraceCommand`` instances.  Modes
share the Start/Stop/Sync/Repeat opcodes; they differ in how event
firings are encoded (cycles in mode 0, PC value in mode 1, branch
records in mode 2).
"""
