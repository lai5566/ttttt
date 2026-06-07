"""
MLA-GAN 工具函式：ADA、boundary sampling、分類器載入。
"""

import torch
import torch.nn as nn
from torchvision import models


# ═══════════════════════════════════════════════
# 預訓練分類器載入
# ═══════════════════════════════════════════════

class CTClassifier(nn.Module):
    """
    CT 分類器（1 通道灰階輸入）。

    支援的 backbone:
      - 'efficientnet_b0' (預設, feat_dim=1280)
      - 'resnet50' (feat_dim=2048)

    輸入: [B, 1, 256, 256]，範圍 [-1, 1]
    輸出: [B, num_classes] logits

    EfficientNet-B0 reference: Test acc=0.9595, macro_F1=0.9411
    """

    def __init__(self, num_classes: int = 3, arch: str = 'efficientnet_b0'):
        super().__init__()
        self.arch = arch
        if arch == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=None)
            # 第一層: 3ch → 1ch
            self.backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False,
            )
            in_features = self.backbone.classifier[1].in_features  # 1280
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        elif arch == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            # 第一層: 3ch → 1ch
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False,
            )
            in_features = self.backbone.fc.in_features  # 2048
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported arch: {arch} (expected 'efficientnet_b0' or 'resnet50')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_pretrained_classifier(path: str, num_classes: int = 3,
                               device: str = 'cuda',
                               arch: str = None) -> nn.Module:
    """
    載入預訓練的 CT 分類器。

    Args:
        path: checkpoint 路徑
        num_classes: 類別數
        device: 裝置
        arch: backbone 架構 ('efficientnet_b0' or 'resnet50')。
              若為 None，從 ckpt['arch'] 讀取，否則 fallback 'efficientnet_b0'。
    Returns:
        凍結的分類器模型（eval 模式）
    """
    ckpt = torch.load(path, map_location='cpu')
    if arch is None:
        arch = ckpt.get('arch', 'efficientnet_b0')
    model = CTClassifier(num_classes=num_classes, arch=arch)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_feature_extractor(classifier: nn.Module) -> nn.Module:
    """
    從分類器中提取特徵提取器（去掉最後分類層）。

    依 classifier.arch 切換：
      - EfficientNet-B0: features → avgpool → flatten → [B, 1280]
      - ResNet50:        conv1..layer4 → avgpool → flatten → [B, 2048]
    """
    arch = getattr(classifier, 'arch', 'efficientnet_b0')
    backbone = classifier.backbone

    if arch == 'efficientnet_b0':
        class _EfficientNetFeatureExtractor(nn.Module):
            def __init__(self, features, avgpool):
                super().__init__()
                self.features = features
                self.avgpool = avgpool

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                return torch.flatten(x, 1)  # [B, 1280]

        feat_ext = _EfficientNetFeatureExtractor(backbone.features, backbone.avgpool)
    elif arch == 'resnet50':
        class _ResNet50FeatureExtractor(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.conv1 = backbone.conv1
                self.bn1 = backbone.bn1
                self.relu = backbone.relu
                self.maxpool = backbone.maxpool
                self.layer1 = backbone.layer1
                self.layer2 = backbone.layer2
                self.layer3 = backbone.layer3
                self.layer4 = backbone.layer4
                self.avgpool = backbone.avgpool

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                return torch.flatten(x, 1)  # [B, 2048]

        feat_ext = _ResNet50FeatureExtractor(backbone)
    else:
        raise ValueError(f"Unsupported arch: {arch}")

    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False
    return feat_ext


def get_feature_dim(arch: str) -> int:
    """Return feature dim for the given backbone."""
    return {'efficientnet_b0': 1280, 'resnet50': 2048}[arch]


# ═══════════════════════════════════════════════
# ADA（自適應判別器增強）
# ═══════════════════════════════════════════════

class AdaptiveAugmentation:
    """
    Adaptive Discriminator Augmentation (Karras et al., NeurIPS 2020)。

    根據 D 的過擬合程度自動調整增強機率 p：
    r_t = E[sign(D(real))]
    r_t > target → D 太強 → 增大 p
    r_t < target → D 太弱 → 減小 p
    """

    def __init__(self, target_rt: float = 0.6, adjust_speed: float = 0.01,
                 max_p: float = 0.8):
        self.target_rt = target_rt
        self.adjust_speed = adjust_speed
        self.p = 0.0
        self.max_p = max_p

    def update(self, d_real_scores: torch.Tensor):
        """根據 D 的 real score 更新增強機率。"""
        rt = torch.sign(d_real_scores).mean().item()
        if rt > self.target_rt:
            self.p = min(self.p + self.adjust_speed, self.max_p)
        else:
            self.p = max(self.p - self.adjust_speed, 0.0)

    def augment(self, img: torch.Tensor) -> torch.Tensor:
        """對影像做隨機增強（只用於 D 的輸入）。"""
        if self.p <= 0:
            return img

        B = img.shape[0]
        augmented = img.clone()

        for i in range(B):
            if torch.rand(1).item() < self.p:
                aug_type = torch.randint(0, 4, (1,)).item()
                if aug_type == 0:    # 水平翻轉
                    augmented[i] = torch.flip(augmented[i], dims=[2])
                elif aug_type == 1:  # 垂直翻轉
                    augmented[i] = torch.flip(augmented[i], dims=[1])
                elif aug_type == 2:  # 90° 旋轉
                    augmented[i] = torch.rot90(augmented[i], 1, [1, 2])
                elif aug_type == 3:  # 亮度微調
                    augmented[i] = augmented[i] + torch.randn(1, device=img.device) * 0.1

        return augmented


# ═══════════════════════════════════════════════
# Boundary-guided sampling
# ═══════════════════════════════════════════════

def sample_boundary_target(bd_stats: dict, batch_size: int,
                           strategy: str = 'guided',
                           near_ratio: float = 0.5,
                           mid_ratio: float = 0.3,
                           far_ratio: float = 0.2,
                           class_label: torch.Tensor = None) -> torch.Tensor:
    """
    Boundary-target sampling with selectable strategy (for ablations).

    [v3-fix B2]: When `class_label` is provided AND `bd_stats['per_class']`
    exists, sample target_bd from each sample's class-conditional distribution.
    This avoids the contradictory conditioning where a class=2 sample would
    receive a target_bd shaped from class=1's margin distribution.

    strategy:
      'guided'      : 3-zone mixture per (near/mid/far)_ratio (default).
      'random'      : uniform U(0, 1) — ablation B1.
      'fixed_low'   : constant 0.05    — ablation B2.
      'fixed_high'  : constant 0.95    — ablation B3.
      'real'        : caller must handle (pass real_bd directly) — ablation B5.

    Args:
        bd_stats: dict with 'mean', 'std', 'p25', and optionally 'per_class'.
        batch_size: B
        near_ratio, mid_ratio, far_ratio: zone proportions for 'guided' strategy.
        class_label: [B] long tensor of class indices. If given and
                     bd_stats['per_class'] is present, use per-class stats.
    Returns:
        targets: [B, 1]
    """
    if strategy == 'random':
        return torch.rand(batch_size, 1)
    if strategy == 'fixed_low':
        return torch.full((batch_size, 1), 0.05)
    if strategy == 'fixed_high':
        return torch.full((batch_size, 1), 0.95)
    if strategy == 'real':
        raise ValueError(
            "strategy='real' must be handled in caller (pass real_bd directly)"
        )

    # 'guided' default
    near_thr = near_ratio
    mid_thr = near_ratio + mid_ratio

    use_per_class = (class_label is not None and 'per_class' in bd_stats)

    targets = []
    for i in range(batch_size):
        # [v3-fix B2] pick per-class stats when available, else fall back global
        if use_per_class:
            cls = int(class_label[i].item())
            stats = bd_stats['per_class'].get(cls, bd_stats)
        else:
            stats = bd_stats

        r = torch.rand(1).item()
        if r < near_thr:                                   # near boundary
            t = torch.rand(1) * stats['p25']
        elif r < mid_thr:                                  # mid range
            # use p50 (median) as upper end of mid when available
            p50 = stats.get('p50', stats['mean'])
            span = abs(p50 - stats['p25'])
            base = stats['p25']
            t = base + torch.rand(1) * span
        else:                                              # far from boundary
            # use p75-1.0 range when available, else gaussian
            p75 = stats.get('p75', None)
            if p75 is not None:
                span = abs(1.0 - p75)
                t = p75 + torch.rand(1) * span
            else:
                t = torch.randn(1) * stats['std'] + stats['mean']
            t = t.clamp(min=0.0, max=1.0)
        targets.append(t.flatten()[:1])
    return torch.stack(targets)  # [B, 1]
