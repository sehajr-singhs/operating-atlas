#!/bin/bash
# Sequential so the sweeps never oversubscribe each other: running 8 jobs x 2
# threads against a concurrent 6-worker sim pool made a 493 s job take 6503 s.
cd "$(dirname "$0")"
for ds in cmapss pmsm; do
  echo "=== $ds sweep starting $(date) ==="
  for s in 0 1 2 3 4; do
    for a in mono raw invariant raw+inv naive activity random; do
      echo "--dataset $ds --arm $a --seed $s --k 3 --epochs 40 --threads 3"
    done
  done | xargs -P 5 -I{} sh -c 'OMP_NUM_THREADS=3 python -u job.py {} 2>&1'
  echo "=== $ds sweep done $(date) ==="
done
