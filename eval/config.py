"""
MLA-GAN 設定檔：集中管理所有超參數與路徑。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MLAGANConfig:
    """MLA-GAN 完整設定。"""

    # ═══ 路徑 ═══
    # Resolved at runtime relative to this file: 999_out/eval/config.py
    # parent => 999_out/eval; parents[1] => 999_out
    dataset_root: str = str(Path(__file__).resolve().parents[1] / "data")
    ct_dir: str = "ct_256"
    mask_dir: str = "masks"
    pretrained_classifier: str = str(Path(__file__).resolve().parents[1] / "data" / "classifiers" / "efficientnet_b0_best.pth")
    output_dir: str = str(Path(__file__).resolve().parent / "results")
    resume_checkpoint: str = ""    # 接續訓練的 checkpoint 路徑（空字串=從零開始）

    # ═══ 資料集 ═══
    img_size: int = 256
    num_classes: int = 3
    minority_classes: list = field(default_factory=lambda: [1, 2])  # Ischemic, Hemorrhagic

    # ═══ 模型：Generator ═══
    z_dim: int = 128
    c_dim: int = 3       # one-hot 類別數
    bd_dim: int = 1       # boundary distance 維度
    w_dim: int = 256      # w-space 維度
    enc_out_channels: int = 256   # context encoder 輸出通道數
    context_dropout: float = 0.3  # 恢復 0.3：0.5 導致 D_gap 翻轉率 48%
    mask_expand_px: int = 16      # mask 外擴像素數
    blur_sigma: float = 4.0       # soft mask 高斯模糊 sigma
    blur_kernel_size: int = 17    # 高斯核大小

    # ═══ 模型：Discriminator ═══
    global_d_base_ch: int = 32    # Global D 基礎通道數
    local_d_base_ch: int = 64     # Local D 基礎通道數
    local_crop_size: int = 64     # Local D 裁切大小

    # ═══ 訓練 ═══
    batch_size: int = 8
    epochs: int = 5000
    lr: float = 2e-4
    betas: tuple = (0.0, 0.99)
    d_steps: int = 1              # Hinge + SN 不需多步 critic（WGAN-GP 才需要 5）

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
    save_interval: int = 500      # 每 N epoch 儲存 checkpoint

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
