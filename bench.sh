#!/usr/bin/env bash
# bench.sh — COM6 thermally-aware benchmark harness
#
# Why this exists: on a 15W TDP laptop, thermal noise at n>=4096 is ±30%.
# v109/v110 experiments produced bogus "conclusions" because single runs
# were compared across different thermal states. This harness enforces
# fixed cooldowns and reports a distribution so signal can be told from
# noise.
#
# Usage:
#   ./bench.sh -e EXE -n SIZE [-m MODE] [-r RUNS] [-c COOLDOWN_SEC]
#   ./bench.sh -e com6_v108.exe -n 4096 -m mt -r 5 -c 120
#
#   -e EXE        path to com6_vN.exe (required)
#   -n SIZE       matrix size (required)
#   -m MODE       mt or 1t (default: mt)
#   -r RUNS       number of runs (default: 5)
#   -c COOLDOWN   seconds to sleep between runs (default: 90)
#   -l LABEL      label for this run (printed in summary)
#
# Aborts if any lingering com6_v*.exe processes are detected (per
# feedback_com6_check_rogue_procs.md — rogue background compute
# destroyed multi-session measurement validity).

set -euo pipefail

EXE=""
SIZE=""
MODE="mt"
RUNS=5
COOLDOWN=90
LABEL=""

while getopts "e:n:m:r:c:l:h" opt; do
    case $opt in
        e) EXE="$OPTARG" ;;
        n) SIZE="$OPTARG" ;;
        m) MODE="$OPTARG" ;;
        r) RUNS="$OPTARG" ;;
        c) COOLDOWN="$OPTARG" ;;
        l) LABEL="$OPTARG" ;;
        h) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown opt"; exit 1 ;;
    esac
done

[[ -z "$EXE"  ]] && { echo "error: -e EXE required"; exit 1; }
[[ -z "$SIZE" ]] && { echo "error: -n SIZE required"; exit 1; }
[[ ! -x "$EXE" && ! -f "$EXE" ]] && { echo "error: $EXE not found/executable"; exit 1; }
[[ "$MODE" != "mt" && "$MODE" != "1t" ]] && { echo "error: -m must be mt or 1t"; exit 1; }

# Normalize EXE so it runs under Git Bash (which doesn't have . in PATH).
# If user passed a bare name like "com6_v108.exe", prepend "./".
case "$EXE" in
    /*|./*|../*) ;;  # absolute or already-relative — leave alone
    *) EXE="./$EXE" ;;
esac

# Rogue-process guard
rogues=$(tasklist 2>/dev/null | grep -i 'com6_v' | grep -v "$(basename "$EXE")" || true)
# Also check for ANY com6 — even the same binary shouldn't be running a second copy
any_com6=$(tasklist 2>/dev/null | grep -ic 'com6_v' || true)
if [[ -n "$rogues" ]]; then
    echo "=== ROGUE PROCESS GUARD ==="
    echo "Detected lingering com6_v*.exe processes. Kill them first:"
    echo "$rogues"
    echo ""
    echo "Fix: tasklist | grep -i com6   # find PIDs"
    echo "     taskkill //PID <pid> //F  # kill each"
    exit 2
fi

LABEL="${LABEL:-$(basename "$EXE") n=$SIZE $MODE}"

echo "===================================================================="
echo "  bench.sh | $LABEL"
echo "  exe=$EXE size=$SIZE mode=$MODE runs=$RUNS cooldown=${COOLDOWN}s"
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="

declare -a GFS=()
for i in $(seq 1 "$RUNS"); do
    if (( i > 1 )); then
        printf "  cooling %ds... " "$COOLDOWN"
        sleep "$COOLDOWN"
        echo "done"
    fi

    t_start=$(date '+%H:%M:%S')
    out=$("$EXE" "$SIZE" "$MODE" 2>&1 | tr -d '\r')
    # Parse the line containing "GF": extract the float before "GF"
    gf=$(echo "$out" | awk '/GF/ { for(i=1;i<=NF;i++) if($i=="GF") print $(i-1) }' | head -1)
    if [[ -z "$gf" ]]; then
        echo "  run $i [$t_start] PARSE FAIL: $out"
        continue
    fi
    GFS+=("$gf")
    printf "  run %d [%s] %8.2f GF\n" "$i" "$t_start" "$gf"
done

# Summary stats
if (( ${#GFS[@]} == 0 )); then
    echo "no successful runs"; exit 3
fi

# Sort, extract min/median/max/best; awk for mean/stddev
printf "%s\n" "${GFS[@]}" | sort -g > /tmp/bench_gfs.$$
min=$(head -1 /tmp/bench_gfs.$$)
max=$(tail -1 /tmp/bench_gfs.$$)
n=${#GFS[@]}
# Median: middle element (or avg of two middles if even count)
if (( n % 2 == 1 )); then
    median=$(sed -n "$((n/2 + 1))p" /tmp/bench_gfs.$$)
else
    m1=$(sed -n "$((n/2))p" /tmp/bench_gfs.$$)
    m2=$(sed -n "$((n/2 + 1))p" /tmp/bench_gfs.$$)
    median=$(awk "BEGIN{printf \"%.2f\", ($m1+$m2)/2}")
fi
mean=$(awk 'BEGIN{s=0;n=0} {s+=$1;n++} END{printf "%.2f", s/n}' /tmp/bench_gfs.$$)
stddev=$(awk -v m="$mean" 'BEGIN{s=0;n=0} {d=$1-m; s+=d*d; n++} END{printf "%.2f", (n>1?sqrt(s/(n-1)):0)}' /tmp/bench_gfs.$$)
spread_pct=$(awk "BEGIN{if($mean>0) printf \"%.1f\", 100*($max-$min)/$mean; else print \"inf\"}")
spread_abs=$(awk "BEGIN{printf \"%.2f\", $max-$min}")
rm -f /tmp/bench_gfs.$$

echo "===================================================================="
echo "  $LABEL — $n runs"
echo "  best=${max}  median=${median}  min=${min}  mean=${mean}  stddev=${stddev}"
echo "  spread=${spread_abs} GF (${spread_pct}% of mean)"
echo "  end: $(date '+%Y-%m-%d %H:%M:%S')"
echo "===================================================================="
