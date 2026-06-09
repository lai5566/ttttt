#!/bin/bash
# ============================================================================
# run_stylegan_all.sh — StyleGAN3(StudioGAN)多卡自動排程,ir5/10/20 × bs8/16
#   只吃 CT(無 mask);每槽 = 一個 cell 端到端 train→generate→eval。
#   N卡×PER_GPU semaphore;skip 守衛可續跑。eval 用 999 的 evaluate_multi_v3。
# 用法:
#   bash run_stylegan_all.sh
#   GPUS="0,1" PER_GPU=2 IRS="5 10 20" BSS="8 16" bash run_stylegan_all.sh
# 需求:StudioGAN deps(torch/torchvision/numpy/scipy/pyyaml/h5py 等)
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"; M=$(pwd); mkdir -p "$M/logs"; TS(){ date +%H:%M:%S; }; LOG="$M/logs/master.log"
IRS=${IRS:-"5 10 20"}; BSS=${BSS:-"8 16"}
ARCHS="resnet50 efficientnet_b0 efficientnet_b3 vit_small_patch16_224 densenet121"

if [ -n "${GPUS:-}" ]; then IFS=',' read -ra GA <<< "$GPUS"; else mapfile -t GA < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi
NGPU=${#GA[@]}; [ "$NGPU" -eq 0 ] && { echo "[FATAL] 無 GPU"; exit 1; }
PER_GPU=${PER_GPU:-1}
if [ "$PER_GPU" = auto ]; then FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null|sort -n|head -1); PER_GPU=$(( ${FREE:-10000}/10000 )); [ "$PER_GPU" -lt 1 ]&&PER_GPU=1; fi
SLOTS=$((NGPU*PER_GPU))
# ── 防 CPU 執行緒超賣（純效能、不改結果）：依並行槽數分配每 process 執行緒 + 關空轉 ──
if [ -z "${OMP_NUM_THREADS:-}" ]; then
  NCORES=$(nproc 2>/dev/null || echo 8)
  OMP=$(( NCORES / SLOTS )); [ "$OMP" -lt 1 ] && OMP=1; [ "$OMP" -gt 8 ] && OMP=8
  export OMP_NUM_THREADS=$OMP MKL_NUM_THREADS=$OMP OPENBLAS_NUM_THREADS=$OMP NUMEXPR_NUM_THREADS=$OMP
fi
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
echo "[$(TS)] GPU x$NGPU=${GA[*]} PER_GPU=$PER_GPU SLOTS=$SLOTS | IRS=$IRS BSS=$BSS | OMP=$OMP_NUM_THREADS WAIT=$OMP_WAIT_POLICY" | tee -a "$LOG"
FIFO=$(mktemp -u); mkfifo "$FIFO"; exec 9<>"$FIFO"; rm -f "$FIFO"
for g in "${GA[@]}"; do for ((k=0;k<PER_GPU;k++)); do echo "$g" >&9; done; done

run_cell() {
  local ir=$1 bs=$2 tag="ir${ir}_bs${bs}"
  local dr="ct_256_ir${ir}"; [ "$ir" = 10 ] && dr="ct_256"
  local save="results_ir${ir}_bs${bs}/run_seed7" gen="generated_ir${ir}_bs${bs}" name="stylegan3_ir${ir}_bs${bs}"
  local gpu; read -u 9 gpu
  (
    echo "[$(TS)] GPU$gpu ▶ $tag" | tee -a "$LOG"
    {
      echo "==== [$(TS)] $tag TRAIN ===="
      if ls "$save/checkpoints"/model=G-best-* >/dev/null 2>&1; then echo "[skip train]"; else CUDA_VISIBLE_DEVICES=$gpu bash "train_ir${ir}_bs${bs}.sh"; fi
      echo "==== [$(TS)] $tag GENERATE ===="
      if [ -d "$gen" ]; then echo "[skip gen]"; else CUDA_VISIBLE_DEVICES=$gpu bash "generate_ir${ir}_bs${bs}.sh"; fi
      echo "==== [$(TS)] $tag EVAL ===="
      out="results/eval_${name}_5arch_5seed"
      if [ -f "../../eval/$out/results.csv" ]; then echo "[skip eval]"
      elif [ ! -d "$gen" ]; then echo "[skip eval] 無生成圖"
      else ( cd ../../eval && CUDA_VISIBLE_DEVICES=$gpu python evaluate_multi_v3.py \
              --syn-dirs "../methods/02_stylegan3/$gen" --syn-names "$name" \
              --train-root "../data/$dr/train" --val-root "../data/$dr/val" --test-root "../data/$dr/test" \
              --archs $ARCHS --runs 5 --epochs 20 --output-dir "$out" ); fi
    } > "$M/logs/cell_${tag}.log" 2>&1
    echo "[$(TS)] GPU$gpu ■ DONE $tag (rc=$?)" | tee -a "$LOG"
    echo "$gpu" >&9
  ) &
}
for ir in $IRS; do for bs in $BSS; do run_cell "$ir" "$bs"; done; done
wait
echo "[$(TS)] ★ STYLEGAN3 ALL DONE" | tee -a "$LOG"
