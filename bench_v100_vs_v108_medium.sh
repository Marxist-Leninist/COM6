#!/usr/bin/env bash
# Extend bench_v100_vs_v108_512.sh up to 1024 and 2048 MT.
#
# Why these sizes only: memory/project_com6_platform_limit.md documents that
# at n>=4096, thermal noise on the 15W i7-10510U exceeds ±30%, so any
# measurable pool-vs-fork-join advantage would drown in drift. At 1024 and
# 2048 the matmul is still short enough for fork-join overhead to be
# proportionally visible, but long enough that thermal noise is ~10-15%.
#
# Abort if any rogue com6_v*.exe is running (feedback_com6_check_rogue_procs).

set -u

cool() { sleep "$1"; }

rogues=$(tasklist 2>/dev/null | grep -i 'com6_v' || true)
if [ -n "$rogues" ]; then
  echo "ABORT: rogue com6 procs detected:"
  echo "$rogues"
  exit 1
fi

parse_gf() {
  echo "$1" | grep -oE '[0-9]+\.[0-9]+ GF' | head -1 | awk '{print $1}'
}

run_size() {
  local n="$1"
  local rounds="$2"
  local cooldown="$3"

  echo "============================================================"
  echo "v100 (pthreads pool) vs v108 (OpenMP+IC-par) — $n MT"
  echo "interleaved, ${cooldown}s cooldown, $rounds rounds each"
  echo "start: $(date '+%F %T')"
  echo "============================================================"

  local V100=() V108=()

  echo "initial cooldown 60s..."
  cool 60

  for r in $(seq 1 "$rounds"); do
    out=$(./com6_v100.exe "$n" mt 2>&1)
    gf=$(parse_gf "$out")
    printf "  round %d  v100 = %s GF\n" "$r" "$gf"
    V100+=("$gf")
    cool "$cooldown"

    out=$(./com6_v108.exe "$n" mt 2>&1)
    gf=$(parse_gf "$out")
    printf "  round %d  v108 = %s GF\n" "$r" "$gf"
    V108+=("$gf")
    cool "$cooldown"
  done

  echo "------------------------------------------------------------"
  printf "v100 (n=$n): %s\n" "${V100[*]}"
  printf "v108 (n=$n): %s\n" "${V108[*]}"

  local best_v100 best_v108 med_v100 med_v108
  best_v100=$(printf "%s\n" "${V100[@]}" | sort -g | tail -1)
  best_v108=$(printf "%s\n" "${V108[@]}" | sort -g | tail -1)
  # upper median of N samples = element at index (N/2)+1 after sort
  local mid=$(( rounds / 2 + 1 ))
  med_v100=$(printf "%s\n" "${V100[@]}" | sort -g | sed -n "${mid}p")
  med_v108=$(printf "%s\n" "${V108[@]}" | sort -g | sed -n "${mid}p")

  printf "BEST    n=$n  v100=%s  v108=%s\n" "$best_v100" "$best_v108"
  printf "MEDIAN  n=$n  v100=%s  v108=%s\n" "$med_v100" "$med_v108"
  echo "end: $(date '+%F %T')"
}

# 1024: 4 rounds × 2 × ~50ms + 30s cooldown × 8 = ~4 min
run_size 1024 4 30
echo ""
# 2048: 4 rounds × 2 × ~200ms + 45s cooldown × 8 = ~6 min
run_size 2048 4 45

echo ""
echo "DONE: $(date '+%F %T')"
