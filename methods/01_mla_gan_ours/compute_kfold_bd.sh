#!/bin/bash
# 重算 K-fold out-of-fold BD（重訓 aux classifier）→ data/classifiers/kfold_bd_<tag>.json
# 樣本少不切太多 fold：ir5(minority 708)→K=5；ir20(minority 177)→K=3。
# ir10 已有 data/classifiers/kfold_bd.json，預設不重算（傳 ir10 參數可重算）。
set -euo pipefail
cd "$(dirname "$0")/code"

IR=${1:-all}   # all | ir5 | ir20 | ir10

run() { echo "== kfold BD $1 (K=$2) =="; python kfold_bd_compute.py --train-root "../../../data/$3/train" --k "$2" --tag "$1"; }

case "$IR" in
  ir5)  run ir5  5  ct_256_ir5 ;;
  ir20) run ir20 3  ct_256_ir20 ;;
  ir10) run ir10 5  ct_256 ;;
  all)  run ir5 5 ct_256_ir5; run ir20 3 ct_256_ir20 ;;
  *) echo "usage: $0 [all|ir5|ir20|ir10]"; exit 1 ;;
esac
echo "[DONE] kfold BD -> ../../../data/classifiers/"
