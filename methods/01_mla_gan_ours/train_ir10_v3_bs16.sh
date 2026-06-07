#!/bin/bash
# MLA-GAN IR=10 訓練（原版 / 非 K-fold / batch=16）— 400ep, seed=7。
# ┌─[BD 標注]──────────────────────────────────────────────────────────┐
# │ 非 K-fold。on-the-fly 'rank' BD,由 resnet50_v3_ir10 引導分類器現算。│
# │ batch=16 版(矩陣 bs16 cell)。輸出到 output_ir10_v3_bs16/,不蓋 bs8。│
# │ 對照:train_ir10_v3.sh = bs8 原版;train_ir10.sh = K-fold。         │
# └────────────────────────────────────────────────────────────────────┘
set -euo pipefail
cd "$(dirname "$0")/code"

EPOCHS=${EPOCHS:-400}
SEED=${SEED:-7}
OUTDIR=${OUTDIR:-../output_ir10_v3_bs16/run_seed${SEED}}

python train_ir10_v3.py \
  --epochs "$EPOCHS" --batch-size 16 --seed "$SEED" \
  --output-dir "$OUTDIR" "$@"
echo "[DONE] IR=10 原版/非kfold/bs16 -> $OUTDIR/best_model.pth"
