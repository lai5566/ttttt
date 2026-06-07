#!/bin/bash
# 通用下游評估:run_eval_generic.sh <生成圖目錄名> <syn-name> <IR> <BS>
# → eval/results/eval_<syn-name>_5arch_5seed/。含 Real-only,5arch×5seed×20ep。
set -uo pipefail
cd "$(dirname "$0")"
M=$(pwd)
ARCHS="resnet50 efficientnet_b0 efficientnet_b3 vit_small_patch16_224 densenet121"
GEN=$1; NAME=$2; IR=$3; BS=${4:-8}
mkdir -p "$M/logs"
cd ../../eval
if [ "$IR" = "10" ]; then ROOT=../data/ct_256; else ROOT=../data/ct_256_ir${IR}; fi
SYN=../methods/01_mla_gan_ours/$GEN
OUT=results/eval_${NAME}_5arch_5seed
[ -d "$SYN" ] || { echo "[skip] 無生成圖 $SYN"; exit 0; }
[ -f "$OUT/results.csv" ] && { echo "[skip] $OUT 已存在"; exit 0; }
echo "==== eval $NAME (ir$IR) ===="
python evaluate_multi_v3.py --syn-dirs "$SYN" --syn-names "$NAME" \
  --train-root "$ROOT/train" --val-root "$ROOT/val" --test-root "$ROOT/test" \
  --archs $ARCHS --runs 5 --epochs 20 --output-dir "$OUT" > "$M/logs/eval_${NAME}.log" 2>&1 \
  || { echo "[FATAL] eval $NAME(見 logs/eval_${NAME}.log)"; exit 1; }
echo "[done] $NAME -> eval/$OUT"
