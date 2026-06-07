#!/bin/bash
# MLA-GAN K-fold BD ir5 batch=16 — 400ep seed7（前置 kfold_bd_ir5.json,已含於包內）
set -euo pipefail; cd "$(dirname "$0")/code"
NPROC=${NPROC:-1}; if [ "$NPROC" -gt 1 ]; then LAUNCHER="torchrun --standalone --nproc_per_node=$NPROC"; else LAUNCHER="python"; fi
EPOCHS=${EPOCHS:-400}; SEED=${SEED:-7}
OUTDIR=${OUTDIR:-../output_ir5_kfold_bs16/run_seed${SEED}}
$LAUNCHER train_kfoldbd.py --ir 5 --bd-method kfold_json --epochs "$EPOCHS" --batch-size 16 --seed "$SEED" --output-dir "$OUTDIR" "$@"
echo "[DONE] kfold ir5 bs16 -> $OUTDIR/best_model.pth"
