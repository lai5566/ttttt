#!/bin/bash
# 由 R2_* 環境變數產生 rclone 的 R2 remote 設定（remote 名稱：r2）。
# 需先 source r2_env.sh（提供 R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_BUCKET
# 與 R2_TOKEN 或 R2_SECRET_ACCESS_KEY）。本檔以 source 方式使用。
#
# 產出：
#   RCLONE_CONFIG_R2_*  rclone 的 R2 remote（透過環境變數設定，不需 rclone.conf）
#   R2_REMOTE           r2:<bucket>/<prefix>，預設 prefix=state

: "${R2_ACCOUNT_ID:?需要 R2_ACCOUNT_ID（先 source r2_env.sh）}"
: "${R2_ACCESS_KEY_ID:?需要 R2_ACCESS_KEY_ID（先 source r2_env.sh）}"
: "${R2_BUCKET:?需要 R2_BUCKET（先 source r2_env.sh）}"

# Secret：直接給 R2_SECRET_ACCESS_KEY，或由 R2_TOKEN 算 sha256
if [ -n "${R2_SECRET_ACCESS_KEY:-}" ]; then
  _R2_SECRET="$R2_SECRET_ACCESS_KEY"
elif [ -n "${R2_TOKEN:-}" ]; then
  _R2_SECRET="$(printf '%s' "$R2_TOKEN" | sha256sum | awk '{print $1}')"
else
  echo "[rclone-env] 缺 R2_SECRET_ACCESS_KEY 或 R2_TOKEN" >&2
  return 1 2>/dev/null || exit 1
fi

export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$_R2_SECRET"
export RCLONE_CONFIG_R2_ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export RCLONE_CONFIG_R2_NO_CHECK_BUCKET=true
export RCLONE_CONFIG_R2_ACL=private

export R2_REMOTE="r2:${R2_BUCKET}/${R2_STATE_PREFIX:-state}"
echo "[rclone-env] R2 remote 就緒 → ${R2_REMOTE}"
