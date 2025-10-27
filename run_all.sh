#!/usr/bin/env bash
set -e

OUT="outputs/sweeps_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# Depths, losses, optimizers, dropouts to explore
DEPTHS=("2" "3" "4")
LOSSES=("nll" "l1" "l2" "l1+l2")
OPTIMS=("sgd" "adam" "rmsprop")
DROPS=("0.0" "0.5")

for d in "${DEPTHS[@]}"; do
  for l in "${LOSSES[@]}"; do
    for o in "${OPTIMS[@]}"; do
      for do in "${DROPS[@]}"; do
        echo ">>> depth=$d loss=$l opt=$o drop=$do"
        python3 src/forgetting_mlp.py \
          --depth "$d" --loss "$l" --optimizer "$o" --dropout "$do" \
          --outdir "$OUT" --verbosity 2
      done
    done
  done
done
