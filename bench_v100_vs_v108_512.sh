#!/usr/bin/env bash
# Interleaved fair bench: v100 (pthreads pool) vs v108 (OpenMP champion) at 512 MT
# Per memory feedback_com6_check_rogue_procs.md, abort if any com6_v*.exe is running.

set -u

cool() { sleep "$1"; }

rogues=$(tasklist 2>/dev/null | grep -i 'com6_v' || true)
if [ -n "$rogues" ]; then
  echo "ABORT: rogue com6 procs detected:"
  echo "$rogues"
  exit 1
fi

# Parse "X.Y GF" out of either v100 "(GF)" or v108 "GF (8 threads)" form.
parse_gf() {
  echo "$1" | grep -oE '[0-9]+\.[0-9]+ GF' | head -1 | awk '{print $1}'
}

echo "============================================================"
echo "v100 (pthreads pool) vs v108 (OpenMP+IC-par) — 512 MT"
echo "interleaved, 45s cooldown between every run, 6 rounds each"
echo "start: $(date '+%F %T')"
echo "============================================================"

V100=()
V108=()

# Initial cool-down to start both from comparable thermal state
echo "initial cooldown 60s..."
cool 60

for r in 1 2 3 4 5 6; do
  out=$(./com6_v100.exe 512 mt 2>&1)
  gf=$(parse_gf "$out")
  printf "  round %d  v100 = %s GF  | %s\n" "$r" "$gf" "$out"
  V100+=("$gf")
  cool 45

  out=$(./com6_v108.exe 512 mt 2>&1)
  gf=$(parse_gf "$out")
  printf "  round %d  v108 = %s GF  | %s\n" "$r" "$gf" "$out"
  V108+=("$gf")
  cool 45
done

echo "------------------------------------------------------------"
printf "v100: %s\n" "${V100[*]}"
printf "v108: %s\n" "${V108[*]}"

best_v100=$(printf "%s\n" "${V100[@]}" | sort -g | tail -1)
best_v108=$(printf "%s\n" "${V108[@]}" | sort -g | tail -1)
med_v100=$(printf "%s\n" "${V100[@]}" | sort -g | sed -n '4p')   # 4th of 6 = upper median
med_v108=$(printf "%s\n" "${V108[@]}" | sort -g | sed -n '4p')

printf "BEST    v100=%s  v108=%s\n" "$best_v100" "$best_v108"
printf "MEDIAN  v100=%s  v108=%s\n" "$med_v100" "$med_v108"
echo "end: $(date '+%F %T')"
