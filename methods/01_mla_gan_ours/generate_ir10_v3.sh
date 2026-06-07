#!/bin/bash
# IR=10 原版(非kfold)生成 — 從 output_ir10_v3 checkpoint，pure_boundary(bd≈0.05) → generated_ir10_v3/class{1,2}
# 對照:generate_ir10_kfold.sh 從 kfold checkpoint 生成。生成 profile 相同,只差訓練。
set -euo pipefail
cd "$(dirname "$0")/code"

CKPT=${CKPT:-../output_ir10_v3/run_seed7/best_model.pth}
DST=${DST:-../generated_ir10_v3}
TARGET=${TARGET:-3540}

if [[ ! -f "$CKPT" ]]; then
  echo "[FATAL] checkpoint 不存在: $CKPT(先跑 train_ir10_v3.sh)" >&2; exit 1
fi
if [[ -e "$DST" && -z "${FORCE:-}" ]]; then
  echo "[ABORT] $DST 已存在,不覆蓋;要重生請 FORCE=1 或換 DST=" >&2; exit 1
fi

python generate_bd_mix.py --ir 10 --ckpt "$CKPT" \
  --profile pure_boundary --target "$TARGET" --out-dir "$DST"
echo "[DONE] IR=10 原版/非kfold pure_boundary -> $DST"
