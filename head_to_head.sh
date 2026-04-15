#!/bin/bash
# Head-to-head v108 vs v115 on the sizes where OMP overhead matters.
# Long inter-test sleep + alternating to control thermal bias.
SLEEP=10
echo "=== sequence: v115 -> sleep -> v108 -> sleep, repeat at 512 then 1024 ==="
for size in 512 1024; do
  echo ""
  echo "--- size=$size ---"
  for trial in 1 2 3; do
    echo "Trial $trial:"
    sleep $SLEEP
    ./com6_v115.exe $size mt 2>&1
    sleep $SLEEP
    ./com6_v108.exe $size mt 2>&1
  done
done
