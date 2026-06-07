#!/bin/bash
# gateA ir20 bs8：從 v3(rank)G + QualityGate 生成 → generated_ir20_gateA
set -euo pipefail; cd "$(dirname "$0")/code"
CKPT=${CKPT:-../output_ir20_v3/run_seed7/best_model.pth}
DST=${DST:-../generated_ir20_gateA}; TARGET=${TARGET:-3540}
[ -f "$CKPT" ] || { echo "[FATAL] 無 v3 ckpt: $CKPT(gateA 需先有 v3 G,先 train_ir20_v3.sh)"; exit 1; }
[ -d "$DST" ] && [ -z "${FORCE:-}" ] && { echo "[ABORT] $DST 已存在"; exit 1; } || true
python generate_gate.py --ir 20 --ckpt "$CKPT" --out-dir "$DST" --target "$TARGET"
echo "[DONE] gateA ir20 bs8 gen -> $DST"
