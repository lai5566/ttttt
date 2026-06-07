#!/bin/bash
# MLA-GAN K-fold BD ir5 batch=8 — 400ep seed7（前置 kfold_bd_ir5.json,已含於包內）
set -euo pipefail; cd "$(dirname "$0")/code"
EPOCHS=${EPOCHS:-400}; SEED=${SEED:-7}
OUTDIR=${OUTDIR:-../output_ir5_kfold/run_seed${SEED}}
python train_kfoldbd.py --ir 5 --bd-method kfold_json --epochs "$EPOCHS" --batch-size 8 --seed "$SEED" --output-dir "$OUTDIR" "$@"
echo "[DONE] kfold ir5 bs8 -> $OUTDIR/best_model.pth"
