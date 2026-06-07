#!/bin/bash
# MLA-GAN K-fold BD ir20 batch=8 — 400ep seed7（前置 kfold_bd_ir20.json,已含於包內）
set -euo pipefail; cd "$(dirname "$0")/code"
EPOCHS=${EPOCHS:-400}; SEED=${SEED:-7}
OUTDIR=${OUTDIR:-../output_ir20_kfold/run_seed${SEED}}
python train_kfoldbd.py --ir 20 --bd-method kfold_json --epochs "$EPOCHS" --batch-size 8 --seed "$SEED" --output-dir "$OUTDIR" "$@"
echo "[DONE] kfold ir20 bs8 -> $OUTDIR/best_model.pth"
