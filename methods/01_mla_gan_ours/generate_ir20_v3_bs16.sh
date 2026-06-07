#!/bin/bash
# IR=20 原版(非kfold/bs16)生成 — 從 output_ir20_v3_bs16 checkpoint，pure_boundary → generated_ir20_v3_bs16/class{1,2}
set -euo pipefail
cd "$(dirname "$0")/code"

CKPT=${CKPT:-../output_ir20_v3_bs16/run_seed7/best_model.pth}
DST=${DST:-../generated_ir20_v3_bs16}
TARGET=${TARGET:-3540}

if [[ ! -f "$CKPT" ]]; then
  echo "[FATAL] checkpoint 不存在: $CKPT(先跑 train_ir20_v3_bs16.sh)" >&2; exit 1
fi
if [[ -e "$DST" && -z "${FORCE:-}" ]]; then
  echo "[ABORT] $DST 已存在,不覆蓋;要重生請 FORCE=1 或換 DST=" >&2; exit 1
fi

python generate_bd_mix.py --ir 20 --ckpt "$CKPT" \
  --profile pure_boundary --target "$TARGET" --out-dir "$DST"
echo "[DONE] IR=20 原版/非kfold/bs16 pure_boundary -> $DST"
