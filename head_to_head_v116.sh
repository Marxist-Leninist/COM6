#!/bin/bash
# Fair v116 vs v108 head-to-head with 30s cooldowns to manage 15W TDP throttling.
# Alternates A/B/A/B/A/B for matched thermal positions; reports best of 3.
SLEEP=30
echo "=== Initial 30s cool-down ===" ; sleep $SLEEP
for size in 512 1024; do
  echo ""
  echo "=================== size=$size ==================="
  for trial in 1 2 3; do
    echo "[$trial] v116:"
    ./com6_v116.exe $size mt
    sleep $SLEEP
    echo "[$trial] v108:"
    ./com6_v108.exe $size mt
    sleep $SLEEP
  done
done
