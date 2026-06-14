#!/bin/bash
# ============================================================================
# setup_env.sh — 一鍵安裝所有 Python 依賴（冪等：已裝齊就秒過，torch 已內建就跳過）。
#   被各執行入口（run_stateless / run_all_and_pack / run_stylegan_2cells）自動呼叫，
#   也可手動：bash setup_env.sh
#   wandb 為選用（程式已改可選），故不在此安裝。
# ============================================================================
set -uo pipefail

# 核心(方法01+eval+工具) + 方法02 StudioGAN 依賴
PKGS="numpy scipy scikit-learn pillow timm boto3 requests h5py pyyaml kornia tqdm six matplotlib seaborn ninja click"

PY=$(command -v python3 || command -v python || echo python3)

# 快速檢查：全部能 import 就跳過（避免每次重跑都 pip）
if "$PY" - <<'EOF' 2>/dev/null
import importlib.util, sys
mods = ['numpy','scipy','sklearn','PIL','timm','boto3','requests','h5py','yaml',
        'kornia','tqdm','six','matplotlib','seaborn','click','ninja']
missing = [m for m in mods if importlib.util.find_spec(m) is None]
sys.exit(1 if missing else 0)
EOF
then
  echo "[setup] Python 依賴已齊全，跳過安裝"
else
  echo "[setup] 安裝依賴中（首次較久）..."
  "$PY" -m pip install -q --root-user-action=ignore $PKGS 2>&1 | tail -5 \
    || "$PY" -m pip install -q --break-system-packages $PKGS 2>&1 | tail -5 \
    || echo "[setup][WARN] 部分依賴安裝失敗，請看上面訊息手動補"
fi

# torch 通常 GPU 映像已內建；只檢查、不自動裝（CUDA 版本要對）
"$PY" -c "import torch,torchvision;print('[setup] torch',torch.__version__,'| cuda',torch.cuda.is_available())" 2>/dev/null \
  || echo "[setup][WARN] 偵測不到 torch/torchvision！請依你的 CUDA 版本到 pytorch.org 手動安裝。"
echo "[setup] 完成"
