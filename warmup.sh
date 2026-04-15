#!/usr/bin/env bash
# warmup.sh — push the CPU into a stable high-P-state before measurement.
#
# Why this exists: on Windows laptops with aggressive power management,
# the CPU can sit at <1 GHz after 30+ seconds of idle. The first few
# matmul runs then read 3-11 GF (vs the 60-90 GF the kernel can sustain
# once clocks stabilize), which contaminates the "best-of-N" numbers.
#
# This script runs short matmuls repeatedly until the measured GFLOPS
# plateaus, at which point the CPU is in steady high-P-state (or full
# turbo if thermals allow) and subsequent benchmarks are comparable.
#
# Usage:
#   ./warmup.sh                     # default: probe com6_v108.exe at n=512
#   ./warmup.sh -e com6_v114.exe    # specify exe
#   ./warmup.sh -n 1024             # different probe size
#   ./warmup.sh -t 10               # max 10 probes (default 8)
#   ./warmup.sh -g 50               # gate: stop once >= 50 GF observed
#
# Exit status 0 when plateau reached, 1 if never above gate after N
# probes (laptop probably in thermal limit already — further benchmarks
# won't reproduce).

set -euo pipefail

EXE="com6_v108.exe"
SIZE=512
MAX_TRIES=8
GATE=40     # consider "warm" once this GFLOPS threshold is reached
VERBOSE=0

while getopts "e:n:t:g:vh" opt; do
    case $opt in
        e) EXE="$OPTARG" ;;
        n) SIZE="$OPTARG" ;;
        t) MAX_TRIES="$OPTARG" ;;
        g) GATE="$OPTARG" ;;
        v) VERBOSE=1 ;;
        h) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown opt"; exit 1 ;;
    esac
done

case "$EXE" in
    /*|./*|../*) ;;
    *) EXE="./$EXE" ;;
esac

[[ ! -f "$EXE" ]] && { echo "warmup: $EXE not found"; exit 2; }

echo "warmup: probing $EXE at n=$SIZE, gate=$GATE GF, max=$MAX_TRIES tries"

prev=0
plateau_hits=0
for i in $(seq 1 "$MAX_TRIES"); do
    out=$("$EXE" "$SIZE" mt 2>&1 | tr -d '\r')
    gf=$(echo "$out" | awk '/GF/ { for(i=1;i<=NF;i++) if($i=="GF") print $(i-1) }' | head -1)
    if [[ -z "$gf" ]]; then
        echo "warmup: parse fail at try $i: $out" >&2
        continue
    fi
    (( VERBOSE )) && printf "  probe %d: %.1f GF\n" "$i" "$gf"

    # Done if we crossed the gate
    ge=$(awk "BEGIN{print ($gf >= $GATE) ? 1 : 0}")
    if [[ "$ge" == "1" ]]; then
        printf "warmup: reached %.1f GF (gate=%d) on probe %d — CPU ready\n" \
               "$gf" "$GATE" "$i"
        exit 0
    fi

    # Plateau detection: 2 consecutive probes within 5% of each other
    # AND below the gate → thermally limited, stop trying
    if [[ "$prev" != "0" ]]; then
        delta_pct=$(awk "BEGIN{d=($gf-$prev); if(d<0)d=-d; if($prev>0) printf \"%.1f\", 100*d/$prev; else print 999}")
        converged=$(awk "BEGIN{print ($delta_pct < 5.0) ? 1 : 0}")
        if [[ "$converged" == "1" ]]; then
            plateau_hits=$((plateau_hits + 1))
            if (( plateau_hits >= 2 )); then
                printf "warmup: plateaued at %.1f GF below gate %d — thermally limited\n" \
                       "$gf" "$GATE"
                exit 1
            fi
        else
            plateau_hits=0
        fi
    fi
    prev="$gf"
done

printf "warmup: exhausted %d tries, last probe %.1f GF — below gate %d\n" \
       "$MAX_TRIES" "$prev" "$GATE"
exit 1
