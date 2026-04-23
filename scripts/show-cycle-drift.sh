#!/usr/bin/env bash
# show-cycle-drift.sh -- sort bridge-test cycle-drift results by severity.
#
# Usage:
#   scripts/show-cycle-drift.sh [--results DIR] [--top N]
#
# Default: read build/bridge-test-results/latest/, print all results.
# --top N limits to the N highest |log(ratio)| entries.

set -euo pipefail

RESULTS="build/bridge-test-results/latest"
TOP=0  # 0 = unlimited

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) RESULTS="$2"; shift 2 ;;
    --top)     TOP="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$RESULTS" ]]; then
  echo "Results dir not found: $RESULTS" >&2
  exit 1
fi

# Emit: |log(ratio)|  test.variant.compiler  tag
awk_script='
function abs(x) { return x < 0 ? -x : x }
function logr(r)  { if (r <= 0) return 999; return log(r)/log(10) }
BEGIN { }
{
  # Line format: "<tag> <file>"
  tag=$1
  file=$2
  # Extract <safe>[.<variant>].<compiler> from filename.
  name=file
  sub(/.*\//, "", name)
  sub(/\.cycle\.result$/, "", name)
  # Parse ratio if present in DRIFT(ratio=X,...) or MATCH(X).
  ratio=0
  if (match(tag, /ratio=[0-9.]+/)) {
    r=substr(tag, RSTART+6, RLENGTH-6)
    ratio=r+0
  } else if (match(tag, /MATCH\(([0-9.]+)\)/, m)) {
    ratio=m[1]+0
  }
  sev=abs(logr(ratio))
  # Bugs / compare errors float to the top.
  if (tag ~ /TRACE_BUG|COMPARE-ERR/) sev=10
  # EMPTY / NO_DATA: expected artifacts, not bugs. Below any real drift
  # (max |log10(ratio)| in [0.5, 2.0] bounds is ~0.6), above MATCH baseline.
  else if (tag ~ /EMPTY|NO_DATA/) sev=1
  printf("%.4f  %-48s  %s\n", sev, name, tag)
}
'

mapfile -t lines < <(
  find -L "$RESULTS" -maxdepth 1 -name '*.cycle.result' -print 2>/dev/null |
  while IFS= read -r f; do
    printf "%s %s\n" "$(tr -d '[:space:]' < "$f")" "$f"
  done |
  awk "$awk_script" |
  sort -rn
)

if [[ ${#lines[@]} -eq 0 ]]; then
  echo "No .cycle.result files found under $RESULTS"
  exit 0
fi

printf "%-8s  %-48s  %s\n" "|log|" "TEST" "RESULT"
printf "%-8s  %-48s  %s\n" "--------" "------------------------------------------------" "------"

count=0
for line in "${lines[@]}"; do
  echo "$line"
  count=$(( count + 1 ))
  if [[ "$TOP" -gt 0 && "$count" -ge "$TOP" ]]; then
    break
  fi
done
