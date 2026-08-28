#!/bin/bash
# usage: sweep.sh <dataset> <epochs> <nseeds> <parallel>
DS=$1; EP=$2; NS=$3; P=$4
for s in $(seq 0 $((NS-1))); do
  for a in mono raw invariant raw+inv naive activity random; do
    echo "--dataset $DS --arm $a --seed $s --k 3 --epochs $EP --threads 2"
  done
done | xargs -P $P -I{} sh -c 'OMP_NUM_THREADS=2 python -u job.py {} 2>&1'
