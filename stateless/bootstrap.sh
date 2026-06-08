#!/bin/bash
# 新機器啟動的第一個動作：安裝 rclone（若無）→ 從 R2 把先前所有進度/結果拉回本地。
# 拉回後，master 的 skip 守衛會自動跳過已完成的 cell，未完成的 cell 由 latest.pth 續訓。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# 1) 確保 rclone 存在
if ! command -v rclone >/dev/null 2>&1; then
  echo "[bootstrap] 安裝 rclone ..."
  if command -v sudo >/dev/null 2>&1; then
    curl -fsSL https://rclone.org/install.sh | sudo bash
  else
    curl -fsSL https://rclone.org/install.sh | bash
  fi
  command -v rclone >/dev/null 2>&1 || { echo "[bootstrap][ERR] rclone 安裝失敗，請手動安裝後重試"; exit 1; }
fi
echo "[bootstrap] rclone $(rclone version 2>/dev/null | head -1)"

# 2) 載入憑證與 rclone remote
[ -f "$ROOT/r2_env.sh" ] && source "$ROOT/r2_env.sh"
# shellcheck source=/dev/null
source "$HERE/r2_rclone_env.sh" || exit 1

# 3) 從 R2 拉回 state（copy 不刪本地；只補本地缺的/較舊的）
echo "[bootstrap] 從 ${R2_REMOTE} 拉回先前進度到 ${ROOT} ..."
rclone copy "$R2_REMOTE" "$ROOT" \
  --transfers "${RCLONE_TRANSFERS:-8}" --checkers "${RCLONE_CHECKERS:-16}" \
  --fast-list --stats-one-line --stats 30s

echo "[bootstrap] 完成。已恢復的訓練輸出："
ls -d "$ROOT"/methods/01_mla_gan_ours/output_* 2>/dev/null | sed 's/^/  /' \
  || echo "  （R2 上尚無進度 → 全新開始）"
