# shellcheck shell=bash
# scripts/lib/kernel-log.sh -- read the kernel log of the current boot, for
# decisions that must not silently see an empty log.  Source it; do not run it.
#
# Why this exists: with kernel.dmesg_restrict=1 an unprivileged `dmesg` fails.
# A caller that discards stderr and greps the output then sees "no TDR, no
# IO_PAGE_FAULT" on every run, so the hardware-run guard never fires.  Here:
#   * `journalctl -k -b` is preferred (readable by the adm / systemd-journal
#     groups), `dmesg` is the fallback;
#   * a reader that fails OR returns no lines is treated as unreadable;
#   * every function returns non-zero when the log cannot be read, and callers
#     must treat that as "unknown", never as "clean".
#
# API (all functions are exported for use in xargs/parallel subshells):
#   klog_selfcheck        pick a backend and prove it yields >= 1 line; 0 / 1
#   klog_mark             print an opaque position token for the end of the log
#   klog_since MARK       print kernel lines logged after MARK (after a settle)
#   klog_count_since MARK PATTERN
#                         print the number of lines after MARK matching the
#                         extended regex PATTERN
# Environment:
#   KLOG_BACKEND          "journal" or "dmesg"; set by klog_selfcheck
#   KLOG_SETTLE_SEC       seconds to wait before reading new lines (default 1);
#                         kernel -> journald delivery is asynchronous

KLOG_SETTLE_SEC="${KLOG_SETTLE_SEC:-1}"

_klog_journal_lines() {  # at most N (default 1) lines of this boot, or fail
  journalctl -k -b -q --no-pager -n "${1:-1}" 2>/dev/null
}

klog_selfcheck() {
  local out
  out="$(_klog_journal_lines 1)" || out=""
  if [[ -n "$out" ]]; then
    KLOG_BACKEND=journal
  else
    out="$(dmesg 2>/dev/null | head -n 1)" || out=""
    if [[ -n "$out" ]]; then
      KLOG_BACKEND=dmesg
    else
      unset KLOG_BACKEND
      echo "kernel log unreadable: 'journalctl -k -b' and 'dmesg' both returned nothing." >&2
      echo "  Fix: add this user to the 'adm' or 'systemd-journal' group (re-login), or" >&2
      echo "  set kernel.dmesg_restrict=0.  Hardware runs are refused without a readable log." >&2
      return 1
    fi
  fi
  export KLOG_BACKEND
}

# Token is "journal:<cursor>" or "dmesg:<line count>".  The cursor survives
# journal rotation; the dmesg count is the best a ring buffer allows.
klog_mark() {
  local cursor n
  case "${KLOG_BACKEND:-}" in
    journal)
      cursor="$(journalctl -k -b -q --no-pager -n 0 --show-cursor 2>/dev/null \
                | sed -n 's/^-- cursor: //p' | tail -n 1)"
      [[ -n "$cursor" ]] || { echo "klog_mark: no journal cursor" >&2; return 1; }
      echo "journal:$cursor" ;;
    dmesg)
      n="$(dmesg 2>/dev/null | wc -l)"
      [[ "$n" -gt 0 ]] || { echo "klog_mark: dmesg returned nothing" >&2; return 1; }
      echo "dmesg:$n" ;;
    *) echo "klog_mark: no backend; call klog_selfcheck first" >&2; return 1 ;;
  esac
}

klog_since() {
  local mark="$1" kind="${1%%:*}" val="${1#*:}" out
  [[ -n "$mark" && "$mark" == *:* ]] || { echo "klog_since: bad mark" >&2; return 1; }
  sleep "$KLOG_SETTLE_SEC"
  case "$kind" in
    journal)
      # Exit status is the reader's own: a failure must not look like "no lines".
      journalctl -k -b -q --no-pager --after-cursor="$val" 2>/dev/null ;;
    dmesg)
      out="$(dmesg 2>/dev/null)" || return 1
      [[ -n "$out" ]] || return 1
      printf '%s\n' "$out" | tail -n +"$((val + 1))" ;;
    *) echo "klog_since: bad mark kind '$kind'" >&2; return 1 ;;
  esac
}

klog_count_since() {
  local lines
  lines="$(klog_since "$1")" || return 1
  printf '%s\n' "$lines" | grep -cE -- "$2" || true
}

export -f _klog_journal_lines klog_selfcheck klog_mark klog_since klog_count_since
export KLOG_SETTLE_SEC
