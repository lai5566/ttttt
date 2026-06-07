#!/bin/bash
# ============================================================================
# master_run_all.sh — 多卡自動排程,跑「方法 × IR × batch」整個矩陣
#   方法:v3(rank,非kfold) / kfold / gateA(從 v3 G + QualityGate,不另訓)
#   單槽 = 一個 cell 端到端(train→generate→eval);N卡×PER_GPU 個並行槽,
#         FIFO token semaphore → 有槽空出立刻接下一個 → 自動最大化 GPU 利用率。
#   gateA 依賴 v3 G → 分兩波:Wave1 跑 v3/kfold,wait,Wave2 跑 gateA。
#   skip 守衛:已完成 train/generate/eval 自動跳過 → 可中斷續跑。
#
# 用法:
#   bash master_run_all.sh                         # v3+kfold+gateA × ir5/10/20 × bs8/16
#   METHODS="v3 kfold" IRS="5" BSS="8 16" bash master_run_all.sh
#   GPUS="0,1" PER_GPU=3 bash master_run_all.sh    # 2×32GB 建議
#   PER_GPU=auto bash master_run_all.sh            # 依空閒 VRAM 估(8GB/槽)
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
M=$(pwd); mkdir -p "$M/logs"; TS(){ date +%H:%M:%S; }; LOG="$M/logs/master.log"

METHODS=${METHODS:-"v3 kfold gateA"}
IRS=${IRS:-"5 10 20"}
BSS=${BSS:-"8 16"}

# ---- GPU ----
if [ -n "${GPUS:-}" ]; then IFS=',' read -ra GPUARR <<< "$GPUS"
else mapfile -t GPUARR < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); fi
NGPU=${#GPUARR[@]}; [ "$NGPU" -eq 0 ] && { echo "[FATAL] 無 GPU"; exit 1; }
PER_GPU=${PER_GPU:-1}
if [ "$PER_GPU" = "auto" ]; then
  SLOT_MB=${SLOT_MB:-8000}
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
  PER_GPU=$(( ${FREE:-8000} / SLOT_MB )); [ "$PER_GPU" -lt 1 ] && PER_GPU=1
  echo "[$(TS)] PER_GPU=auto → ${FREE}MB/${SLOT_MB}MB = $PER_GPU /卡" | tee -a "$LOG"
fi
SLOTS=$((NGPU*PER_GPU))
echo "[$(TS)] GPU x$NGPU=${GPUARR[*]} PER_GPU=$PER_GPU SLOTS=$SLOTS | METHODS=$METHODS IRS=$IRS BSS=$BSS" | tee -a "$LOG"

# ---- semaphore ----
FIFO=$(mktemp -u); mkfifo "$FIFO"; exec 9<>"$FIFO"; rm -f "$FIFO"
for g in "${GPUARR[@]}"; do for ((k=0;k<PER_GPU;k++)); do echo "$g" >&9; done; done

run_cell() {
  local method=$1 ir=$2 bs=$3
  local suf=""; [ "$bs" = 16 ] && suf="_bs16"
  local bstag=""; [ "$bs" = 16 ] && bstag="_bs16"
  local train gen gendir name ckpt
  case $method in
    v3)    train="train_ir${ir}_v3${suf}.sh";    gen="generate_ir${ir}_v3${suf}.sh";    gendir="generated_ir${ir}_v3${suf}";    name="mlagan_v3_ir${ir}${bstag}";    ckpt="output_ir${ir}_v3${suf}/run_seed7/best_model.pth" ;;
    kfold) train="train_ir${ir}_kfold${suf}.sh"; gen="generate_ir${ir}_kfold${suf}.sh"; gendir="generated_ir${ir}_kfold${suf}"; name="mlagan_kfold_ir${ir}${bstag}"; ckpt="output_ir${ir}_kfold${suf}/run_seed7/best_model.pth" ;;
    gateA) train="";                              gen="generate_ir${ir}_gateA${suf}.sh"; gendir="generated_ir${ir}_gateA${suf}"; name="mlagan_gateA_ir${ir}${bstag}"; ckpt="" ;;
    *) echo "[WARN] 未知方法 $method"; return ;;
  esac
  local tag="${method}_ir${ir}${bstag}"
  local gpu; read -u 9 gpu
  (
    echo "[$(TS)] GPU$gpu ▶ $tag" | tee -a "$LOG"
    {
      if [ -n "$train" ]; then
        echo "==== [$(TS)] $tag TRAIN ===="
        if [ -f "$ckpt" ]; then echo "[skip train] $ckpt 已存在"; else CUDA_VISIBLE_DEVICES=$gpu bash "$train"; fi
      fi
      echo "==== [$(TS)] $tag GENERATE ===="
      if [ -d "$gendir" ]; then echo "[skip gen] $gendir 已存在"; else CUDA_VISIBLE_DEVICES=$gpu bash "$gen"; fi
      echo "==== [$(TS)] $tag EVAL ===="
      CUDA_VISIBLE_DEVICES=$gpu bash run_eval_generic.sh "$gendir" "$name" "$ir" "$bs"
    } > "$M/logs/cell_${tag}.log" 2>&1
    echo "[$(TS)] GPU$gpu ■ DONE $tag (rc=$?)" | tee -a "$LOG"
    echo "$gpu" >&9
  ) &
}

# Wave 1: v3 / kfold(有訓練的)
echo "[$(TS)] === WAVE 1: train 方法(v3/kfold)===" | tee -a "$LOG"
for me in $METHODS; do [ "$me" = gateA ] && continue
  for ir in $IRS; do for bs in $BSS; do run_cell "$me" "$ir" "$bs"; done; done
done
wait
echo "[$(TS)] === WAVE 1 DONE ===" | tee -a "$LOG"

# Wave 2: gateA(複用 v3 G,需 wave1 的 v3 已訓完)
if echo "$METHODS" | grep -qw gateA; then
  echo "[$(TS)] === WAVE 2: gateA(從 v3 G 生成)===" | tee -a "$LOG"
  for ir in $IRS; do for bs in $BSS; do run_cell gateA "$ir" "$bs"; done; done
  wait
  echo "[$(TS)] === WAVE 2 DONE ===" | tee -a "$LOG"
fi
echo "[$(TS)] ★ ALL DONE | 結果:eval/results/eval_mlagan_*  log:logs/cell_*.log" | tee -a "$LOG"
