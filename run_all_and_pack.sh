#!/bin/bash
# ============================================================================
# run_all_and_pack.sh — 頂層總控（在目標多卡機上跑）
#
#   依序跑完兩個方法的完整矩陣，每個方法內部用 FIFO semaphore「一卡多 cell」並行
#   → 自動最大化 GPU 利用率；全部跑完後把結果打包成 tar.gz 帶回。
#
#     方法01  MLA-GAN     v3 + kfold + gateA  × ir5/10/20 × bs8/16  (train→generate→eval)
#     方法02  StyleGAN3   ir5/10/20 × bs8/16                        (train→generate→eval)
#
#   • 兩個方法「依序」跑（不同時），避免兩個排程器搶同一批 GPU 造成超賣/OOM。
#   • 兩個子排程器都有 skip 守衛 → 可中斷續跑（重跑自動跳過已完成的 cell）。
#   • 用 `set -uo pipefail`（不含 -e）：單一 cell/方法失敗不會中斷全局，
#     最後仍會把已產出的結果打包。
#
# 用法：
#   bash run_all_and_pack.sh                                   # 自動偵測全部 GPU
#   GPUS="0,1" bash run_all_and_pack.sh                        # 指定用哪些卡
#   GPUS="0,1" PER_GPU_1=3 PER_GPU_2=2 bash run_all_and_pack.sh
#   METHODS="01" bash run_all_and_pack.sh                      # 只跑方法01
#   # 背景跑（推薦，可登出 SSH）：
#   nohup bash run_all_and_pack.sh > logs/run_all.out 2>&1 &
#   tail -f logs/run_all.log
#
# 可調環境變數：
#   GPUS       用哪些卡（逗號清單，如 "0,1"）；空 = 自動偵測全部
#   PER_GPU_1  方法01 每卡並行 cell 數（預設 3；mlagan 單實例 ~5GB，32GB 卡可塞 3）
#   PER_GPU_2  方法02 每卡並行 cell 數（預設 2；stylegan 較大）
#   METHODS    要跑哪些方法（預設 "01 02"）
#   PACK       跑完是否打包（預設 1；設 0 不打包）
#   PACK_DIR   打包輸出目錄（預設此包根目錄）
# ============================================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/run_all.log"
TS(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(TS)] $*" | tee -a "$LOG"; }

# 自動載入 R2 憑證（r2_env.sh 為 gitignored，不入庫；見 r2_env.sh.example）
if [ -f "$ROOT/r2_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$ROOT/r2_env.sh"
  say "[R2] 已載入 r2_env.sh 憑證"
fi

GPUS=${GPUS:-}
PER_GPU_1=${PER_GPU_1:-3}
PER_GPU_2=${PER_GPU_2:-2}
METHODS=${METHODS:-"01 02"}
INCREMENTAL_UPLOAD=${INCREMENTAL_UPLOAD:-1}   # 1=邊跑邊上傳（一有結果就傳）/ 0=不增量
# 增量上傳開著時，預設不再做結尾的大打包（避免重複上傳）；要結尾整包就 PACK=1
if [ "$INCREMENTAL_UPLOAD" = 1 ]; then PACK=${PACK:-0}; else PACK=${PACK:-1}; fi
PACK_DIR=${PACK_DIR:-$ROOT}

# ---- 前置檢查 ----
if ! command -v nvidia-smi >/dev/null 2>&1; then
  say "[WARN] 找不到 nvidia-smi —— 這台機器有 GPU 嗎？子排程器會嘗試自動偵測，可能失敗。"
fi
if [ -z "$GPUS" ]; then
  DET=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
  say "未指定 GPUS，自動偵測到 GPU：${DET:-（偵測不到）}"
else
  say "使用指定 GPU：$GPUS"
fi

START=$(date +%s)
say "★ 開始 run_all_and_pack | METHODS=$METHODS PER_GPU_1=$PER_GPU_1 PER_GPU_2=$PER_GPU_2"

# ---- 啟動 GPU 狀態通知監控（背景；發 hi/run/stop 到 Discord）----
MONITOR=${MONITOR:-1}     # 1=啟動監控（預設）/ 0=不啟動
MON_PID=""
if [ "$MONITOR" = 1 ] && [ -f "$ROOT/0_gpu_notify_monitor.py" ] && command -v python3 >/dev/null 2>&1; then
  python3 -c 'import requests' 2>/dev/null || { say "[monitor] 安裝 requests ..."; pip install -q requests 2>>"$LOG" || pip3 install -q requests 2>>"$LOG" || say "[monitor][WARN] requests 安裝失敗，監控可能無法發通知"; }
  python3 "$ROOT/0_gpu_notify_monitor.py" >> "$ROOT/logs/gpu_monitor.log" 2>&1 &
  MON_PID=$!
  say "[monitor] GPU 通知監控已啟動（PID ${MON_PID}，log: logs/gpu_monitor.log）"
fi
# ---- 啟動增量上傳監看（背景；一有完成且穩定的結果就立刻傳 R2）----
WATCH_PID=""
if [ "$INCREMENTAL_UPLOAD" = 1 ] && [ -f "$ROOT/watch_and_upload.py" ] && command -v python3 >/dev/null 2>&1; then
  if [ -z "${R2_ACCESS_KEY_ID:-}" ]; then
    say "[upload] 未設定 R2 憑證 → 不啟動增量上傳（先設好 r2_env.sh）"
  else
    python3 -c 'import boto3' 2>/dev/null || { pip install -q boto3 2>>"$LOG" || pip3 install -q boto3 2>>"$LOG" || true; }
    python3 "$ROOT/watch_and_upload.py" >> "$ROOT/logs/watch_upload.log" 2>&1 &
    WATCH_PID=$!
    say "[upload] 增量上傳監看已啟動（PID ${WATCH_PID}，log: logs/watch_upload.log）"
  fi
fi

# 不論正常結束或被中斷，都嘗試收掉背景程序
stop_bg(){
  [ -n "$MON_PID" ]   && kill "$MON_PID"   2>/dev/null && say "[monitor] 已停止 GPU 監控（PID ${MON_PID}）"
  [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null && say "[upload] 已停止增量上傳監看（PID ${WATCH_PID}）"
  MON_PID=""; WATCH_PID=""
}
trap stop_bg EXIT INT TERM

run_01(){
  local sub="$ROOT/methods/01_mla_gan_ours/master_run_all.sh"
  [ -f "$sub" ] || { say "[ERR] 找不到 ${sub}，略過方法01"; return; }
  say "──── 方法01 MLA-GAN 開始（v3+kfold+gateA × ir5/10/20 × bs8/16，PER_GPU=${PER_GPU_1}）────"
  ( cd "$ROOT/methods/01_mla_gan_ours" && GPUS="$GPUS" PER_GPU="$PER_GPU_1" bash master_run_all.sh )
  say "──── 方法01 結束（rc=$?）────"
}

run_02(){
  local sub="$ROOT/methods/02_stylegan3/run_stylegan_all.sh"
  [ -f "$sub" ] || { say "[ERR] 找不到 ${sub}，略過方法02"; return; }
  say "──── 方法02 StyleGAN3 開始（ir5/10/20 × bs8/16，PER_GPU=${PER_GPU_2}）────"
  say "      （注意：StyleGAN3 需 StudioGAN 依賴 h5py/click/ninja 等，首次請先 pip 補齊）"
  ( cd "$ROOT/methods/02_stylegan3" && GPUS="$GPUS" PER_GPU="$PER_GPU_2" bash run_stylegan_all.sh )
  say "──── 方法02 結束（rc=$?）────"
}

for me in $METHODS; do
  case "$me" in
    01) run_01 ;;
    02) run_02 ;;
    *)  say "[WARN] 未知方法 '$me'（略過）" ;;
  esac
done

ELAPSED=$(( $(date +%s) - START ))
say "★ 訓練/生成/評估全部完成，耗時 $((ELAPSED/3600))h$(((ELAPSED%3600)/60))m$((ELAPSED%60))s"

# 增量上傳：最後補掃一次，把最後一輪剛完成、還沒被監看器掃到的結果也傳上去
if [ "$INCREMENTAL_UPLOAD" = 1 ] && [ -f "$ROOT/watch_and_upload.py" ] && [ -n "${R2_ACCESS_KEY_ID:-}" ]; then
  say "[upload] 最後補掃一次（WATCH_ONCE，穩定門檻放寬到 30s）..."
  WATCH_ONCE=1 WATCH_STABLE=30 python3 "$ROOT/watch_and_upload.py" >> "$ROOT/logs/watch_upload.log" 2>&1 || true
  say "[upload] 補掃完成（明細見 logs/watch_upload.log）"
fi

# ============================================================================
# 打包 + 上傳 R2（伺服器對公網快；你在台灣從 R2 公開連結下載，繞開慢的直連）
#   分兩包、小的先傳：
#     1) results_<host>_<ts>.tar.gz  結果+日誌+權重（重要、較小，gzip）
#     2) generated_<host>_<ts>.tar   全部生成影像（數十 GB，store 不壓縮省 CPU）
#
#   上傳憑證（在伺服器設為環境變數，勿寫死）：
#     R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET
#   控制：
#     R2_UPLOAD=auto（預設，有憑證才傳）/ 1 強制 / 0 只打包不傳
#     PACK_GEN=1（預設，打包生成影像）/ 0 不打包影像（只要結果時用）
# ============================================================================
if [ "$PACK" = 1 ]; then
  shopt -s nullglob
  HOST=$(hostname -s 2>/dev/null || echo host)
  STAMP=$(date '+%Y%m%d_%H%M%S')
  R2_UPLOAD=${R2_UPLOAD:-auto}
  PACK_GEN=${PACK_GEN:-1}
  : "${R2_PUBLIC_BASE:=https://pub-f3129530dfcd42b9ad0c77f15dba3245.r2.dev}"
  export R2_PUBLIC_BASE
  ARCHIVES=()

  # ---- 包1：結果 + 日誌 + 權重（gzip）----
  say "──── 打包 [1/2] 結果+日誌+權重 ────"
  R_LIST=()
  radd(){ for p in "$@"; do [ -e "$ROOT/$p" ] && R_LIST+=("$p"); done; }
  radd "eval/results" "logs" "methods/01_mla_gan_ours/logs" "methods/02_stylegan3/logs"
  for d in "$ROOT"/methods/01_mla_gan_ours/output_*; do radd "methods/01_mla_gan_ours/$(basename "$d")"; done
  for d in "$ROOT"/methods/02_stylegan3/results_*;   do radd "methods/02_stylegan3/$(basename "$d")"; done
  RES_TAR="$PACK_DIR/results_${HOST}_${STAMP}.tar.gz"
  if [ ${#R_LIST[@]} -gt 0 ]; then
    say "  → $RES_TAR"
    if tar -czf "$RES_TAR" -C "$ROOT" "${R_LIST[@]}" 2>>"$LOG"; then
      ARCHIVES+=("$RES_TAR"); say "  完成（$(du -h "$RES_TAR" 2>/dev/null | cut -f1)）"
    else say "[WARN] 結果打包有警告/錯誤（見 ${LOG}）"; [ -e "$RES_TAR" ] && ARCHIVES+=("$RES_TAR"); fi
  else
    say "  [跳過] 找不到結果（還沒跑出來？）"
  fi

  # ---- 包2：生成影像（store，不壓縮；PNG 本來就壓過，gzip 沒意義且耗 CPU）----
  if [ "$PACK_GEN" = 1 ]; then
    say "──── 打包 [2/2] 生成影像（store，數十 GB 可能要一會）────"
    G_LIST=()
    gadd(){ for p in "$@"; do [ -e "$ROOT/$p" ] && G_LIST+=("$p"); done; }
    for d in "$ROOT"/methods/01_mla_gan_ours/generated_*; do gadd "methods/01_mla_gan_ours/$(basename "$d")"; done
    for d in "$ROOT"/methods/02_stylegan3/generated_*;    do gadd "methods/02_stylegan3/$(basename "$d")"; done
    GEN_TAR="$PACK_DIR/generated_${HOST}_${STAMP}.tar"
    if [ ${#G_LIST[@]} -gt 0 ]; then
      say "  → $GEN_TAR"
      if tar -cf "$GEN_TAR" -C "$ROOT" "${G_LIST[@]}" 2>>"$LOG"; then
        ARCHIVES+=("$GEN_TAR"); say "  完成（$(du -h "$GEN_TAR" 2>/dev/null | cut -f1)）"
      else say "[WARN] 影像打包有警告/錯誤（見 ${LOG}）"; [ -e "$GEN_TAR" ] && ARCHIVES+=("$GEN_TAR"); fi
    else
      say "  [跳過] 找不到生成影像"
    fi
  else
    say "──── PACK_GEN=0：不打包生成影像 ────"
  fi

  # ---- 上傳 R2 ----
  if [ ${#ARCHIVES[@]} -eq 0 ]; then
    say "[R2] 沒有可上傳的包"
  elif [ "$R2_UPLOAD" = 0 ]; then
    say "[R2] R2_UPLOAD=0，只打包不上傳。檔案："
    printf '  - %s\n' "${ARCHIVES[@]}" | tee -a "$LOG"
  elif [ "$R2_UPLOAD" = auto ] && [ -z "${R2_ACCESS_KEY_ID:-}" ]; then
    say "[R2] 未設定 R2 憑證 → 不上傳（已打包，見下）。要上傳請設好 R2_* 後手動："
    printf '  R2_KEY="$(basename %s)" python3 %s/upload_to_r2.py %s\n' "$RES_TAR" "$ROOT" "$RES_TAR" | tee -a "$LOG"
    printf '  - %s\n' "${ARCHIVES[@]}" | tee -a "$LOG"
  else
    command -v python3 >/dev/null 2>&1 || say "[R2][WARN] 無 python3，無法上傳"
    python3 -c 'import boto3' 2>/dev/null || { say "[R2] 安裝 boto3 ..."; pip install -q boto3 2>>"$LOG" || pip3 install -q boto3 2>>"$LOG" || say "[R2][WARN] boto3 安裝失敗，請手動 pip install boto3"; }
    for a in "${ARCHIVES[@]}"; do
      say "[R2] 上傳 $(basename "$a") ..."
      R2_KEY="$(basename "$a")" python3 "$ROOT/upload_to_r2.py" "$a" 2>&1 | tee -a "$LOG" \
        || say "[R2][WARN] $(basename "$a") 上傳失敗（見 ${LOG}），檔案仍在 $a"
    done
    say "[R2] ✅ 上傳流程結束（下載連結見上方 [R2] 公開下載連結）"
  fi
fi

say "✅ run_all_and_pack 全部結束"
