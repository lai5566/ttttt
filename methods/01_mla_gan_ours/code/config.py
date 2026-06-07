"""
MLA-GAN 設定檔：集中管理所有超參數與路徑。

[v3-fix] Paths default to env vars for portability across machines:
  MLAGAN_DATASET_ROOT  — 真實 CT/mask 資料集根目錄（必須）
  MLAGAN_CLASSIFIER    — 預訓練分類器 .pth（預設 ./classifiers/efficientnet_b0_best.pth）
  MLAGAN_OUTPUT        — 輸出目錄（預設 ./output）

If env vars are not set, defaults below are used. The classifier and output
defaults are RELATIVE to this config.py file's location, so they remain
correct after copying the directory to another machine.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent


@dataclass
class MLAGANConfig:
    """MLA-GAN 完整設定。"""

    # ═══ 路徑 ═══
    # Dataset path: must be set externally on the slow machine
    dataset_root: str = os.environ.get(
        'MLAGAN_DATASET_ROOT',
        str(Path(__file__).resolve().parents[3] / 'data'),
    )
    ct_dir: str = "ct_256"
    mask_dir: str = "masks"
    # Classifier and output: relative to this directory by default
    pretrained_classifier: str = os.environ.get(
        'MLAGAN_CLASSIFIER',
        str(_HERE / 'classifiers' / 'efficientnet_b0_best.pth'),
    )
    # [v3-fix] Classifier backbone — 'efficientnet_b0' (default, feat_dim=1280)
    # 或 'resnet50' (feat_dim=2048, ResNet50 v3 引導分類器用)
    classifier_arch: str = os.environ.get('MLAGAN_CLASSIFIER_ARCH', 'efficientnet_b0')
    output_dir: str = os.environ.get(
        'MLAGAN_OUTPUT',
        str(_HERE / 'output'),
    )
    resume_checkpoint: str = ""    # 接續訓練的 checkpoint 路徑（空字串=從零開始）

    # ═══ 資料集 ═══
    img_size: int = 256
    num_classes: int = 3
    minority_classes: list = field(default_factory=lambda: [1, 2])  # Ischemic, Hemorrhagic

    # ═══ 模型：Generator ═══
    z_dim: int = 128
    c_dim: int = 3       # one-hot 類別數
    bd_dim: int = 1       # boundary distance 維度（v2: bd 已解耦到 BoundaryDistanceModulator）
    w_dim: int = 256      # w-space 維度
    enc_out_channels: int = 256   # context encoder 輸出通道數
    context_dropout: float = 0.3  # 恢復 0.3：0.5 導致 D_gap 翻轉率 48%
    mask_expand_px: int = 16      # mask 外擴像素數
    blur_sigma: float = 4.0       # soft mask 高斯模糊 sigma
    blur_kernel_size: int = 17    # 高斯核大小

    # ═══ Ablation switches (架構元件) ═══
    use_bd_modulator: bool = True    # A1: False -> bd 注入 mapping，刪 self.bd_mod
    use_gated_conv: bool = True      # A7: False -> HardMaskEncoder
    use_skip: bool = True            # A8: False -> decoder skip_ch=0
    # [v3-fix] 預設 = 真 BD 設定（rank percentile of logit margin + class-conditional guided sampling）
    bd_compute_method: str = 'rank'  # 'rank' (Issue 3 fix) | 'sigmoid' (legacy) | 'kfold_json' (新預設方法)
    bd_sampling: str = 'guided'      # B1/B2/B3/B5: random/fixed_low/fixed_high/real
    # K-fold out-of-fold BD 來源 json（bd_compute_method='kfold_json*' 時讀此檔）。
    # 空字串 fallback 到 ir10 預設 data/classifiers/kfold_bd.json；各 IR config 覆寫。
    kfold_bd_json: str = ""

    # ═══ 模型：Discriminator ═══
    global_d_base_ch: int = 32    # Global D 基礎通道數
    local_d_base_ch: int = 64     # Local D 基礎通道數
    local_crop_size: int = 64     # Local D 裁切大小

    # ═══ 訓練 ═══
    batch_size: int = 8
    epochs: int = 400
    lr: float = 2e-4
    dlr: float = 1e-4             # TTUR: D lr = G lr / 2，防止 D 壓倒 G
    betas: tuple = (0.0, 0.99)
    d_steps: int = 1              # Hinge + SN 不需多步 critic（WGAN-GP 才需要 5）
    grad_clip: float = 5.0        # 梯度裁剪，防止 NaN

    # ═══ 損失函數權重 ═══
    lambda_local: float = 0.5     # Local D 損失權重（原 2.0，降低避免 local 主導梯度）
    lambda_gp: float = 10.0       # WGAN-GP 梯度懲罰
    lambda_bd: float = 1.0        # boundary distance 回歸
    lambda_fd: float = 0.5        # mode-seeking diversity（1.0→0.5，讓重建信號主導）
    lambda_rec: float = 1.0       # mask 外重建損失（5.0→1.0，降低對 z 表現力的壓制）
    lambda_mask_guide: float = 30.0  # mask 內紋理引導（10.0→30.0，讓重建信號主導）
    fd_margin: float = 0.5        # feature diversity loss margin
    lambda_r1: float = 10.0       # R1 正則化（防止 D 過擬合 real images）
    r1_interval: int = 16         # R1 lazy regularization 間隔

    # ═══ ADA（自適應判別器增強）═══
    ada_target_rt: float = 0.6    # 目標 r_t 值
    ada_adjust_speed: float = 0.01
    ada_max_p: float = 0.8

    # ═══ 生成 ═══
    target_per_class: int = 3540  # 每類目標數量（與 Normal 類對齊）
    # boundary-guided sampling 比例
    bd_near_ratio: float = 0.5    # 50% 靠近邊界
    bd_mid_ratio: float = 0.3     # 30% 中等距離
    bd_far_ratio: float = 0.2     # 20% 真實分布

    # ═══ 下游分類器 ═══
    clf_lr: float = 1e-3
    clf_weight_decay: float = 1e-4
    clf_epochs: int = 50
    clf_alpha_mask: float = 0.7   # Lesion-Aware Pooling 的 mask 權重

    # ═══ 日誌 ═══
    log_interval: int = 100       # 每 N 步列印日誌
    save_interval: int = 100      # 每 N epoch 儲存 checkpoint

    @property
    def ct_train_dir(self) -> Path:
        return Path(self.dataset_root) / self.ct_dir / "train"

    @property
    def ct_val_dir(self) -> Path:
        return Path(self.dataset_root) / self.ct_dir / "val"

    @property
    def ct_test_dir(self) -> Path:
        return Path(self.dataset_root) / self.ct_dir / "test"

    @property
    def mask_train_dir(self) -> Path:
        return Path(self.dataset_root) / self.mask_dir / "train"
