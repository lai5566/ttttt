#!/bin/bash
# 背景常駐：每 SYNC_EVERY_MIN 分鐘把本地狀態同步到 R2。
# 由 run_stateless.sh 自動於背景啟動；也可單獨跑：
#   source r2_env.sh; SYNC_EVERY_MIN=10 bash stateless/sync_daemon.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EVERY=$(( ${SYNC_EVERY_MIN:-10} * 60 ))
echo "[sync-daemon] 每 ${SYNC_EVERY_MIN:-10} 分鐘同步一次到 R2（Ctrl+C 結束）"
while true; do
  bash "$HERE/sync_to_r2.sh" || echo "[sync-daemon] 本輪同步出錯，下輪重試"
  sleep "$EVERY"
done
