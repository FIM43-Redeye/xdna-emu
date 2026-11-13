#!/usr/bin/env bash
# Exercises scripts/lib/kernel-log.sh against fake journalctl / dmesg binaries
# on PATH.  No hardware, no privileges.  Usage: scripts/test-kernel-log.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(mktemp -d)"
trap 'rm -rf "$ROOT"' EXIT
BIN="$ROOT/bin"
mkdir -p "$BIN"

# Fake journalctl: JFAKE=ok|fail|empty.  Log is $ROOT/journal (one line each);
# the cursor is the line number.  Understands -n N, --show-cursor, --after-cursor=.
cat > "$BIN/journalctl" <<'FAKE'
#!/usr/bin/env bash
case "${JFAKE:-ok}" in
  fail) exit 1 ;;
  empty) exit 0 ;;
esac
n="" show=0 after=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) n="$2"; shift ;;
    --show-cursor) show=1 ;;
    --after-cursor=*) after="${1#*=}" ;;
  esac
  shift
done
total=$(wc -l < "$JLOG")
if [[ "$n" == 0 ]]; then
  [[ $show -eq 1 ]] && echo "-- cursor: $total"
  exit 0
fi
tail -n +"$((after + 1))" "$JLOG" | { [[ -n "$n" ]] && tail -n "$n" || cat; }
FAKE
# Fake dmesg: DFAKE=ok|fail|empty, log in $DLOG.
cat > "$BIN/dmesg" <<'FAKE'
#!/usr/bin/env bash
case "${DFAKE:-ok}" in
  fail) echo "dmesg: read kernel buffer failed: Operation not permitted" >&2; exit 1 ;;
  empty) exit 0 ;;
esac
cat "$DLOG"
FAKE
chmod +x "$BIN/journalctl" "$BIN/dmesg"

export PATH="$BIN:$PATH" JLOG="$ROOT/journal" DLOG="$ROOT/dmesg" KLOG_SETTLE_SEC=0
printf 'boot line\nsecond line\n' > "$JLOG"
printf 'boot line\nsecond line\n' > "$DLOG"

# shellcheck source=lib/kernel-log.sh
source "$HERE/lib/kernel-log.sh"

fails=0
check() {  # check DESCRIPTION EXPECTED ACTUAL
  if [[ "$2" == "$3" ]]; then echo "ok   - $1"; else echo "FAIL - $1 (want '$2', got '$3')"; fails=$((fails + 1)); fi
}
rc() { "$@" >/dev/null 2>&1 && echo 0 || echo 1; }

# Backend selection and fail-closed behaviour.
JFAKE=ok DFAKE=ok klog_selfcheck; check "journal preferred" journal "$KLOG_BACKEND"
JFAKE=fail DFAKE=ok klog_selfcheck; check "dmesg fallback when journal fails" dmesg "$KLOG_BACKEND"
JFAKE=empty DFAKE=ok klog_selfcheck; check "empty journal falls back to dmesg" dmesg "$KLOG_BACKEND"
check "both unreadable fails closed" 1 "$(JFAKE=fail DFAKE=fail rc klog_selfcheck)"
check "restricted dmesg + no journal access (empty) fails closed" 1 "$(JFAKE=empty DFAKE=fail rc klog_selfcheck)"
check "both empty fails closed" 1 "$(JFAKE=empty DFAKE=empty rc klog_selfcheck)"
msg="$(JFAKE=fail DFAKE=fail klog_selfcheck 2>&1 || true)"
check "failure message names the fix" 1 "$([[ "$msg" == *systemd-journal* ]] && echo 1 || echo 0)"

# Journal backend: cursor-based counting.
JFAKE=ok DFAKE=fail klog_selfcheck
mark="$(klog_mark)"
check "journal mark is a cursor token" "journal:2" "$mark"
printf 'amdxdna: aie2_tdr_work: tdr\nIO_PAGE_FAULT at 0x0\nIO_PAGE_FAULT again\nunrelated\n' >> "$JLOG"
check "journal tdr count" 1 "$(klog_count_since "$mark" aie2_tdr_work)"
check "journal iommu count" 2 "$(klog_count_since "$mark" IO_PAGE_FAULT)"
mark2="$(klog_mark)"
check "no new lines counts zero" 0 "$(klog_count_since "$mark2" 'aie2_tdr_work|IO_PAGE_FAULT')"
export JFAKE=fail
check "reader failing after the run is an error, not zero" 1 "$(rc klog_since "$mark2")"
export JFAKE=ok

# dmesg backend: line-count marks.
JFAKE=fail DFAKE=ok klog_selfcheck
dmark="$(klog_mark)"
check "dmesg mark is a line count" "dmesg:2" "$dmark"
printf 'aie2_tdr_work: tdr\nIO_PAGE_FAULT\n' >> "$DLOG"
check "dmesg tdr count" 1 "$(klog_count_since "$dmark" aie2_tdr_work)"
check "dmesg iommu count" 1 "$(klog_count_since "$dmark" IO_PAGE_FAULT)"
export DFAKE=fail
check "dmesg failing after the run is an error" 1 "$(rc klog_since "$dmark")"
export DFAKE=ok
check "mark without backend is an error" 1 "$(unset KLOG_BACKEND; rc klog_mark)"
check "garbage mark is an error" 1 "$(rc klog_since garbage)"

if [[ $fails -ne 0 ]]; then echo "$fails check(s) failed"; exit 1; fi
echo "all kernel-log checks passed"
