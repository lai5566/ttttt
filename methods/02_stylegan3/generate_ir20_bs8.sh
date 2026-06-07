#!/bin/bash
# StyleGAN3 ir20 bs8 生成 → generated_ir20_bs8/class{0,1,2}
set -euo pipefail; cd "$(dirname "$0")/code"
CFG=${CFG:-../configs/sg3_ir20_bs8.yaml}
CKPT_DIR=${CKPT_DIR:-../results_ir20_bs8/run_seed7/checkpoints}
DST=${DST:-../generated_ir20_bs8}; TARGET=${TARGET:-3540}
[ -d "$CKPT_DIR" ] || { echo "[FATAL] 無 ckpt: $CKPT_DIR(先 train_ir20_bs8.sh)"; exit 1; }
python generate_by_class.py -cfg "$CFG" -ckpt "$CKPT_DIR" -o "$DST" --all -n $TARGET --individual "$@"
echo "[DONE] sg3 ir20 bs8 gen -> $DST"
