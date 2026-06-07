#!/bin/bash
# kfold ir20 bs16 生成（pure_boundary）→ generated_ir20_kfold_bs16
set -euo pipefail; cd "$(dirname "$0")/code"
CKPT=${CKPT:-../output_ir20_kfold_bs16/run_seed7/best_model.pth}
DST=${DST:-../generated_ir20_kfold_bs16}; TARGET=${TARGET:-3540}
[ -f "$CKPT" ] || { echo "[FATAL] 無 ckpt: $CKPT(先 train_ir20_kfold_bs16.sh)"; exit 1; }
[ -d "$DST" ] && [ -z "${FORCE:-}" ] && { echo "[ABORT] $DST 已存在"; exit 1; } || true
python generate_bd_mix.py --ir 20 --ckpt "$CKPT" --profile pure_boundary --target "$TARGET" --out-dir "$DST"
echo "[DONE] kfold ir20 bs16 gen -> $DST"
