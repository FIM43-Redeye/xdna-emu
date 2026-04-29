#!/usr/bin/env bash
# PostToolUse hook: auto-run rustfmt on Edit/Write of *.rs files.
#
# Reads the tool's stdin JSON, extracts the file path, and rewrites
# the file with rustfmt's output -- but ONLY if it's a *.rs file.
#
# Critical: uses rustfmt's stdin mode (`--emit stdout < file`) instead
# of `rustfmt file`, because the latter follows `mod` declarations and
# would format the entire module tree on every edit to lib.rs / mod.rs.
# Stdin mode treats the input as a standalone source unit.
#
# Silent on success. Surfaces stderr only on failure.

set -u

f=$(jq -r '.tool_response.filePath // .tool_input.file_path // empty')

# No-op for non-Rust paths and empty input.
case "$f" in
    *.rs) ;;
    *)    exit 0 ;;
esac

# File must exist (Edit/Write should have produced it, but be defensive).
if [ ! -f "$f" ]; then
    exit 0
fi

tmp=$(mktemp) || exit 0
err=$(mktemp) || { rm -f "$tmp"; exit 0; }

if rustfmt --emit stdout < "$f" > "$tmp" 2> "$err"; then
    # Replace only if rustfmt produced output and output differs.
    if [ -s "$tmp" ] && ! cmp -s "$tmp" "$f"; then
        mv "$tmp" "$f"
    else
        rm -f "$tmp"
    fi
else
    rm -f "$tmp"
    echo "rustfmt failed on $f:" >&2
    sed 's/^/  /' < "$err" >&2
fi

rm -f "$err"
exit 0
