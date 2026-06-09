#!/bin/bash
# 把本地的訓練/結果「狀態」同步到 R2。分兩段，避免被幾萬張小 PNG 拖死：
#   (A) 會變動的小東西（checkpoint/結果/日誌）→ rclone sync（checksum，確保最新）
#   (B) 不可變的大量生成影像 → rclone copy --ignore-existing
#       （已在 R2 的就跳過、不重查 checksum；只傳新檔）→ 快非常多
# 用大量 --checkers/--transfers 加速列舉。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

[ -f "$ROOT/r2_env.sh" ] && source "$ROOT/r2_env.sh"
# shellcheck source=/dev/null
source "$HERE/r2_rclone_env.sh" || exit 1
cd "$ROOT"

CK="${RCLONE_CHECKERS:-64}"; TR="${RCLONE_TRANSFERS:-16}"
COMMON=(--fast-list --checkers "$CK" --transfers "$TR" --stats-one-line --stats 30s)

echo "[sync $(date '+%H:%M:%S')] (A) checkpoint/結果/日誌 → ${R2_REMOTE}"
rclone sync "$ROOT" "$R2_REMOTE" "${COMMON[@]}" \
  --include "methods/01_mla_gan_ours/output_*/**" \
  --include "methods/02_stylegan3/results_*/**" \
  --include "eval/results/**" \
  --include "logs/**"
rcA=$?

echo "[sync $(date '+%H:%M:%S')] (B) 生成影像（只傳新檔，不重查）→ ${R2_REMOTE}"
rclone copy "$ROOT" "$R2_REMOTE" "${COMMON[@]}" --ignore-existing \
  --include "methods/01_mla_gan_ours/generated_*/**" \
  --include "methods/02_stylegan3/generated_*/**"
rcB=$?

rc=$(( rcA | rcB ))
echo "[sync $(date '+%H:%M:%S')] 結束（rcA=${rcA} rcB=${rcB}）"
exit $rc
