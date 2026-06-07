#!/bin/bash
# kfold ir5 bs8 生成（pure_boundary）→ generated_ir5_kfold
set -euo pipefail; cd "$(dirname "$0")/code"
CKPT=${CKPT:-../output_ir5_kfold/run_seed7/best_model.pth}
DST=${DST:-../generated_ir5_kfold}; TARGET=${TARGET:-3540}
[ -f "$CKPT" ] || { echo "[FATAL] 無 ckpt: $CKPT(先 train_ir5_kfold.sh)"; exit 1; }
[ -d "$DST" ] && [ -z "${FORCE:-}" ] && { echo "[ABORT] $DST 已存在"; exit 1; } || true
python generate_bd_mix.py --ir 5 --ckpt "$CKPT" --profile pure_boundary --target "$TARGET" --out-dir "$DST"
echo "[DONE] kfold ir5 bs8 gen -> $DST"
