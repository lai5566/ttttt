#!/bin/bash
# MLA-GAN IR=10 訓練（原版 / 非 K-fold）— 400ep, batch=8, seed=7。
# ┌─[BD 標注]──────────────────────────────────────────────────────────┐
# │ 非 K-fold。on-the-fly 'rank' BD,由 resnet50_v3_ir10 引導分類器現算。│
# │ 對照:同目錄 train_ir10.sh 才是 K-fold(train_kfoldbd.py,讀          │
# │       kfold_bd.json)。本版輸出到獨立的 output_ir10_v3/,不蓋 kfold。  │
# └────────────────────────────────────────────────────────────────────┘
set -euo pipefail
cd "$(dirname "$0")/code"
NPROC=${NPROC:-1}; if [ "$NPROC" -gt 1 ]; then LAUNCHER="torchrun --standalone --nproc_per_node=$NPROC"; else LAUNCHER="python"; fi

EPOCHS=${EPOCHS:-400}
SEED=${SEED:-7}
OUTDIR=${OUTDIR:-../output_ir10_v3/run_seed${SEED}}

$LAUNCHER train_ir10_v3.py \
  --epochs "$EPOCHS" --batch-size 8 --seed "$SEED" \
  --output-dir "$OUTDIR" "$@"
echo "[DONE] IR=10 原版/非kfold -> $OUTDIR/best_model.pth"
