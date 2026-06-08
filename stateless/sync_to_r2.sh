#!/bin/bash
# 把本地的訓練/結果「狀態」同步到 R2（rclone sync，增量，只傳變動的檔）。
# 同步內容 = 續訓與最終結果所需的一切：
#   methods/*/output_* | results_* | generated_* | eval/results | logs
# R2 為唯一真理：本地刪掉的（例如 keep-latest 清掉的舊 checkpoint）R2 也會跟著刪。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# 載入憑證與 rclone remote
[ -f "$ROOT/r2_env.sh" ] && source "$ROOT/r2_env.sh"
# shellcheck source=/dev/null
source "$HERE/r2_rclone_env.sh" || exit 1
cd "$ROOT"

FILTERS=(
  --include "methods/01_mla_gan_ours/output_*/**"
  --include "methods/01_mla_gan_ours/generated_*/**"
  --include "methods/02_stylegan3/results_*/**"
  --include "methods/02_stylegan3/generated_*/**"
  --include "eval/results/**"
  --include "logs/**"
)

echo "[sync $(date '+%H:%M:%S')] 本地 → ${R2_REMOTE}"
rclone sync "$ROOT" "$R2_REMOTE" "${FILTERS[@]}" \
  --transfers "${RCLONE_TRANSFERS:-8}" --checkers "${RCLONE_CHECKERS:-16}" \
  --fast-list --stats-one-line --stats 30s "$@"
rc=$?
echo "[sync $(date '+%H:%M:%S')] 結束（rc=${rc}）"
exit $rc
