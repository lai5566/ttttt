#!/bin/bash
# MLA-GAN K-fold BD ir20 batch=8 — 400ep seed7（前置 kfold_bd_ir20.json,已含於包內）
set -euo pipefail; cd "$(dirname "$0")/code"
NPROC=${NPROC:-1}; if [ "$NPROC" -gt 1 ]; then LAUNCHER="torchrun --standalone --nproc_per_node=$NPROC"; else LAUNCHER="python"; fi
EPOCHS=${EPOCHS:-400}; SEED=${SEED:-7}
OUTDIR=${OUTDIR:-../output_ir20_kfold/run_seed${SEED}}
$LAUNCHER train_kfoldbd.py --ir 20 --bd-method kfold_json --epochs "$EPOCHS" --batch-size 8 --seed "$SEED" --output-dir "$OUTDIR" "$@"
echo "[DONE] kfold ir20 bs8 -> $OUTDIR/best_model.pth"
