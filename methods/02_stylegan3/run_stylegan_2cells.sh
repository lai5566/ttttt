#!/bin/bash
# ============================================================================
# run_stylegan_2cells.sh — 只跑 StyleGAN3 的 ir5_bs8 + ir10_bs16，
#   每個訓練用 DDP「吃滿所有 GPU」（不像 per-cell master 每 cell 只給 1 卡）。
#   流程：train(-DDP, 多卡) → generate → eval。wandb 已改可選，不裝也能跑。
#
# 用法（repo 根目錄或本目錄都行）：
#   bash methods/02_stylegan3/run_stylegan_2cells.sh
#   # 背景跑 + R2 備份（推薦）：先在背景開 stateless 同步，再跑本腳本
#   nohup bash methods/02_stylegan3/run_stylegan_2cells.sh > logs/sg2.out 2>&1 &
#
# 先決條件（StudioGAN 依賴；wandb 不必裝）：
#   pip install h5py pyyaml kornia tqdm six matplotlib seaborn ninja click
#
# 可調：CELLS="5:8 10:16"（預設）；FORCE=1 強制重訓
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
M=$(pwd); mkdir -p "$M/logs"
TS(){ date '+%Y-%m-%d %H:%M:%S'; }
LOG="$M/logs/two_cells.log"
say(){ echo "[$(TS)] $*" | tee -a "$LOG"; }

# 自動安裝依賴（冪等，已裝齊就秒過）
ROOT="$(cd "$M/../.." && pwd)"
[ -f "$ROOT/setup_env.sh" ] && bash "$ROOT/setup_env.sh"

# 不限制 CUDA_VISIBLE_DEVICES → main.py 看得到所有 GPU → -DDP 會把它們都吃滿
NGPU=$(python3 -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)
{ [ -z "$NGPU" ] || [ "$NGPU" -lt 1 ]; } && NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -lt 1 ] && { echo "[FATAL] 偵測不到 GPU"; exit 1; }

# 防 OMP 超賣：DDP 會起 NGPU 個 process，每個分到 ncores/NGPU 條執行緒 + 關空轉
NCORES=$(nproc 2>/dev/null || echo 8)
OMP=$(( NCORES / NGPU )); [ "$OMP" -lt 2 ] && OMP=2; [ "$OMP" -gt 8 ] && OMP=8
export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP
export OMP_WAIT_POLICY=PASSIVE
say "GPU x${NGPU} | OMP_NUM_THREADS=${OMP} WAIT=PASSIVE | DDP 多卡訓練"

ARCHS="resnet50 efficientnet_b0 efficientnet_b3 vit_small_patch16_224 densenet121"
CELLS=${CELLS:-"5:8 10:16"}

run_cell() {
  local ir=$1 bs=$2 tag="ir${ir}_bs${bs}"
  local dr="ct_256_ir${ir}"; [ "$ir" = 10 ] && dr="ct_256"
  local save="results_ir${ir}_bs${bs}/run_seed7" gen="generated_ir${ir}_bs${bs}" name="stylegan3_ir${ir}_bs${bs}"
  say "════ ${tag} 開始 ════"
  {
    echo "==== [$(TS)] ${tag} TRAIN（DDP，${NGPU} GPU）===="
    if ls "$save/checkpoints"/model=G-best-* >/dev/null 2>&1 && [ "${FORCE:-0}" != 1 ]; then
      echo "[skip train] 已有 best G"
    else
      bash "train_${tag}.sh" -DDP        # -DDP 透傳給 main.py → mp.spawn 每卡一 process
    fi

    echo "==== [$(TS)] ${tag} GENERATE ===="
    if [ -d "$gen" ]; then echo "[skip gen] $gen 已存在"; else bash "generate_${tag}.sh"; fi

    echo "==== [$(TS)] ${tag} EVAL ===="
    out="results/eval_${name}_5arch_5seed"
    if [ -f "../../eval/$out/results.csv" ]; then echo "[skip eval] 已有結果"
    elif [ ! -d "$gen" ]; then echo "[skip eval] 無生成圖（train/generate 失敗？）"
    else ( cd ../../eval && python evaluate_multi_v3.py \
            --syn-dirs "../methods/02_stylegan3/$gen" --syn-names "$name" \
            --train-root "../data/$dr/train" --val-root "../data/$dr/val" --test-root "../data/$dr/test" \
            --archs $ARCHS --runs 5 --epochs 20 --output-dir "$out" ); fi
  } 2>&1 | tee -a "$M/logs/cell_${tag}.log"
  say "════ ${tag} 完成 ════"
}

say "★ StyleGAN3 2-cell（${CELLS}）DDP 多卡開始"
for c in $CELLS; do
  ir=${c%%:*}; bs=${c##*:}
  run_cell "$ir" "$bs"
done
say "★ 全部完成。結果：eval/results/eval_stylegan3_*（CSV + full_report.txt）"
