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

# 3) 從 R2 拉回 state，分兩段（避免被幾萬張小圖拖死）：
#    (A) checkpoint/結果/日誌：照常拉（量小、要最新）
#    (B) 生成影像：--ignore-existing，本機已有的就跳過、不重查 → 同機重啟近乎瞬間
CK="${RCLONE_CHECKERS:-64}"; TR="${RCLONE_TRANSFERS:-16}"
COMMON=(--fast-list --checkers "$CK" --transfers "$TR" --stats-one-line --stats 30s)
echo "[bootstrap] (A) 拉 checkpoint/結果/日誌 ← ${R2_REMOTE} ..."
rclone copy "$R2_REMOTE" "$ROOT" "${COMMON[@]}" \
  --include "methods/01_mla_gan_ours/output_*/**" \
  --include "methods/02_stylegan3/results_*/**" \
  --include "eval/results/**" --include "logs/**"
echo "[bootstrap] (B) 拉生成影像（跳過本機已有的）← ${R2_REMOTE} ..."
rclone copy "$R2_REMOTE" "$ROOT" "${COMMON[@]}" --ignore-existing \
  --include "methods/01_mla_gan_ours/generated_*/**" \
  --include "methods/02_stylegan3/generated_*/**"

echo "[bootstrap] 完成。已恢復的訓練輸出："
ls -d "$ROOT"/methods/01_mla_gan_ours/output_* 2>/dev/null | sed 's/^/  /' \
  || echo "  （R2 上尚無進度 → 全新開始）"
