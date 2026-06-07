#!/bin/bash
# v3(原版/非 K-fold)下游評估 — eval/evaluate_multi_v3.py, 5arch×5seed, 20ep。
# 對應 generated_ir{5,10,20}_v3[_bs16];輸出到 eval/results/eval_mlagan_v3_<tag>_5arch_5seed/。
# 含 Real-only(未加 --no-real,與 run_eval_6arch.sh 一致)。
#
# 前置:先 train_ir*_v3[_bs16].sh → generate_ir*_v3[_bs16].sh 產生生成圖。
# 用法:
#   bash run_eval_v3.sh                 # 全部 6 cell(IR 5/10/20 × bs 8/16)
#   bash run_eval_v3.sh "5"             # 只 ir5(bs8+bs16)
#   bash run_eval_v3.sh "5 20" "8"      # ir5+ir20 的 bs8
set -uo pipefail
cd "$(dirname "$0")"
M=$(pwd)
ARCHS="resnet50 efficientnet_b0 efficientnet_b3 vit_small_patch16_224 densenet121"  # convnext_tiny 移除(v3 下崩潰)
IRS=${1:-"5 10 20"}
BSS=${2:-"8 16"}
mkdir -p "$M/logs"
cd ../../eval

for IR in $IRS; do
  # ir10 資料根用 ct_256(與 config_ir10 一致);ir5/ir20 用 ct_256_irX
  if [ "$IR" = "10" ]; then ROOT=../data/ct_256; else ROOT=../data/ct_256_ir${IR}; fi
  for BS in $BSS; do
    if [ "$BS" = "8" ]; then
      SUF="";       NAME=mlagan_v3_ir${IR};       TAG=ir${IR}
    else
      SUF="_bs16";  NAME=mlagan_v3_ir${IR}_bs16;  TAG=ir${IR}_bs16
    fi
    SYN=../methods/01_mla_gan_ours/generated_ir${IR}_v3${SUF}
    OUT=results/eval_mlagan_v3_${TAG}_5arch_5seed
    if [ ! -d "$SYN" ]; then echo "[skip] 生成圖不存在: $SYN(先跑 generate_ir${IR}_v3${SUF}.sh)"; continue; fi
    if [ -f "$OUT/results.csv" ]; then echo "[skip] $OUT 已存在(不覆蓋)"; continue; fi
    echo "==== v3 eval $TAG (5 archs, 5 seed, 20ep) ===="
    python evaluate_multi_v3.py \
      --syn-dirs "$SYN" --syn-names "$NAME" \
      --train-root "$ROOT/train" --val-root "$ROOT/val" --test-root "$ROOT/test" \
      --archs $ARCHS --runs 5 --epochs 20 \
      --output-dir "$OUT" > "$M/logs/eval_${TAG}_v3_5arch.log" 2>&1 \
      || { echo "[FATAL] eval $TAG(見 logs/eval_${TAG}_v3_5arch.log)"; exit 1; }
    echo "[done] $TAG -> eval/$OUT"
  done
done
echo "==== V3 EVAL DONE ===="
