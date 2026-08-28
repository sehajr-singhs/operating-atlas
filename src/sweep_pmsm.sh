#!/bin/bash
cd "$(dirname "$0")"
echo "=== pmsm sweep starting $(date) ==="
for s in 0 1 2 3 4; do
  for a in mono raw invariant raw+inv naive activity random; do
    echo "--dataset pmsm --arm $a --seed $s --k 3 --epochs 30 --threads 4 --bs 16384 --lr 4e-3"
  done
done | xargs -P 4 -I{} sh -c 'OMP_NUM_THREADS=4 python -u job.py {} 2>&1'
echo "=== pmsm sweep done $(date) ==="
