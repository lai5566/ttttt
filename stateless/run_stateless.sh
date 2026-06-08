#!/bin/bash
# ============================================================================
# run_stateless.sh — 「運算/儲存分離」一鍵執行（Vast.ai 等可隨時被回收的機器）
#
#   核心理念：租來的主機 = 無狀態運算節點，硬碟只是暫存；R2 才是唯一真理。
#   流程：
#     1) bootstrap：安裝 rclone + 從 R2 拉回先前所有進度/結果
#     2) 背景常駐：每 SYNC_EVERY_MIN 分鐘把本地 → R2 增量同步
#     3) 跑 master（run_all_and_pack.sh）；skip 守衛跳過已完成的 cell，
#        未完成的訓練 cell 由 latest.pth 自動續訓（不從頭來、loss 不飆高）
#     4) 結尾再同步一次
#   機器被回收 → R2 有最新進度；租新機器重跑本腳本即接續。
#
# 用法（在伺服器 repo 根目錄；憑證先放好 r2_env.sh）：
#   bash stateless/run_stateless.sh
#   # 背景跑（推薦）：
#   nohup bash stateless/run_stateless.sh > logs/stateless.out 2>&1 &
#   tail -f logs/r2_sync.log logs/run_all.log
#
# 可調：
#   SYNC_EVERY_MIN  R2 同步間隔分鐘（預設 10）
#   GPUS PER_GPU_1 PER_GPU_2 METHODS  → 透傳給 run_all_and_pack.sh
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
mkdir -p logs
[ -f "$ROOT/r2_env.sh" ] && source "$ROOT/r2_env.sh"

# 1) 拉回先前進度
bash "$HERE/bootstrap.sh" || { echo "[stateless][ERR] bootstrap 失敗，中止"; exit 1; }

# 2) 背景 R2 同步常駐
SYNC_EVERY_MIN=${SYNC_EVERY_MIN:-10} bash "$HERE/sync_daemon.sh" >> "$ROOT/logs/r2_sync.log" 2>&1 &
SYNC_PID=$!
echo "[stateless] R2 背景同步啟動（PID ${SYNC_PID}，每 ${SYNC_EVERY_MIN:-10} 分鐘，log: logs/r2_sync.log）"
# 結束/中斷時收掉同步常駐，並做最後一次同步
final_sync(){
  kill "$SYNC_PID" 2>/dev/null
  echo "[stateless] 最後同步一次到 R2 ..."
  bash "$HERE/sync_to_r2.sh" || echo "[stateless][WARN] 最後同步出錯（進度多半已在前幾輪上去）"
}
trap final_sync EXIT INT TERM

# 3) 跑 master：R2 同步已負責備份，故關掉 run_all 的增量上傳與結尾大打包
INCREMENTAL_UPLOAD=0 PACK=0 bash "$ROOT/run_all_and_pack.sh"

echo "[stateless] master 結束。"
