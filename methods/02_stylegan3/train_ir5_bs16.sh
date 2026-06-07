#!/bin/bash
# StyleGAN3 (StudioGAN) ir5 batch=16 訓練 — 只吃 CT($dr),不用 mask
set -euo pipefail; cd "$(dirname "$0")/code"
CFG=${CFG:-../configs/sg3_ir5_bs16.yaml}
SAVE=${SAVE:-../results_ir5_bs16/run_seed7}
if ls "$SAVE/checkpoints"/model=G-best-* >/dev/null 2>&1 && [ "${FORCE:-0}" != 1 ]; then echo "[skip] $SAVE 已有 best G"; exit 0; fi
python src/main.py -t -cfg "$CFG" -data ../../../data/ct_256_ir5 -save "$SAVE" "$@"
echo "[DONE] sg3 ir5 bs16 -> $SAVE"
