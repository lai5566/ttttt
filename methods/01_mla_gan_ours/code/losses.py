"""
MLA-GAN 損失函數：Hinge Loss (SN-GAN)、Boundary Distance、Feature Diversity、重建損失。

D 已使用 spectral norm 約束 Lipschitz 常數，改用 hinge loss 替代 WGAN-GP。
（GP 與 SN 衝突：SN 使梯度 norm ≈ 1，GP loss ≈ 0，無法有效約束。）

核心損失循環：
  G 接收 d_b_target → 生成 fake → D Branch2 預測 d_b'
  → MSE(d_b', d_b_target) → G 學會控制邊界位置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryDistanceComputer:
    """
    計算樣本到分類器 decision boundary 的距離。

    使用 logit margin: d(x) = max_logit(x) - 2nd_max_logit(x)
    d(x) 大 → 離邊界遠 → 分類器很確信 → 增強價值低
    d(x) 小 → 靠近邊界 → 分類器不確定 → 增強價值高

    method='sigmoid' (legacy): sigmoid(raw) — saturates near 1.0 for confident
                               classifiers (Issue 3 — diagnosed: 75%+ samples
                               > 0.99 with our EfficientNet-B0).
    method='raw'    (default): return raw logit margin; caller should apply
                               percentile-rank normalization across the dataset
                               (handled in dataset.precompute_boundary_distances).
    """

    def __init__(self, pretrained_classifier: nn.Module, method: str = 'raw'):
        assert method in ('sigmoid', 'raw'), method
        self.classifier = pretrained_classifier
        self.classifier.eval()
        self.method = method
        for p in self.classifier.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(images)
        sorted_logits, _ = logits.sort(dim=1, descending=True)
        raw_dist = sorted_logits[:, 0] - sorted_logits[:, 1]
        if self.method == 'sigmoid':
            return torch.sigmoid(raw_dist).unsqueeze(1)
        return raw_dist.unsqueeze(1)


class ModeSeekingLoss(nn.Module):
    """
    Mode-seeking loss (Mao et al., CVPR 2019)。

    永遠最大化 feat_dist/z_dist，沒有 margin 上限。
    比 margin-based 好：ratio 超過 margin 後仍有梯度推動多樣性。
    """

    def __init__(self, feature_extractor: nn.Module, **_kwargs):
        super().__init__()
        self.feat_ext = feature_extractor
        self.feat_ext.eval()
        for p in self.feat_ext.parameters():
            p.requires_grad = False

    def forward(self, img1: torch.Tensor, img2: torch.Tensor,
                z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # 不用 no_grad — 梯度要流回 G
        f1 = self.feat_ext(img1)  # [B, feat_dim]
        f2 = self.feat_ext(img2)  # [B, feat_dim]

        feat_dist = F.pairwise_distance(f1, f2, p=2)  # [B]
        z_dist = F.pairwise_distance(z1, z2, p=2)      # [B]

        # 最小化 z_dist/feat_dist → 最大化 feat_dist/z_dist
        # 用 log 壓縮避免 feat_dist≈0 時 loss 爆炸
        loss = torch.log(z_dist / (feat_dist + 1e-4) + 1.0).mean()
        return loss


# 向後相容別名
FeatureDiversityLoss = ModeSeekingLoss


class MLAGANLoss:
    """
    MLA-GAN 完整損失函數（Hinge Loss + SN-GAN）。

    L_D = hinge_real + hinge_fake (global)
          + λ_local · (hinge_real + hinge_fake) (local)
          + λ_bd · L_bd_regress
    L_G = -D_fake_global
          + λ_local · (-D_fake_local)
          + λ_bd · L_bd_guide + λ_fd · L_feat_div + λ_rec · L_rec
    """

    def __init__(self, lambda_local: float = 0.5,
                 lambda_bd: float = 1.0, lambda_fd: float = 2.0,
                 lambda_rec: float = 5.0, lambda_mask_guide: float = 0.5,
                 lambda_r1: float = 10.0, r1_interval: int = 16,
                 **_kwargs):
        self.lambda_local = lambda_local
        self.lambda_bd = lambda_bd
        self.lambda_fd = lambda_fd
        self.lambda_rec = lambda_rec
        self.lambda_mask_guide = lambda_mask_guide
        self.lambda_r1 = lambda_r1
        self.r1_interval = r1_interval  # 每 N 步才算一次（lazy reg）

    # ─── Discriminator Loss ───
    def d_loss(self, D: nn.Module, real_img: torch.Tensor,
               fake_img: torch.Tensor, class_label: torch.Tensor,
               mask: torch.Tensor, real_bd: torch.Tensor,
               target_bd: torch.Tensor = None,
               global_step: int = 0) -> tuple:
        """
        Hinge loss discriminator + R1 正則化 + boundary distance 回歸。
        R1 用 lazy regularization（每 r1_interval 步才算一次）。

        [v3-fix B3]: 如果 target_bd 提供，D 額外學習「fake (條件於 target_bd)
        的 bd_pred 應 ≈ target_bd」這個契約。沒有此項時 D 對 fake 的
        bd_pred 是 undefined，會送 G 噪音梯度。
        """
        # R1 需要 real_img 的梯度
        do_r1 = (self.lambda_r1 > 0 and
                 global_step % self.r1_interval == 0)

        if do_r1:
            real_img = real_img.detach().requires_grad_(True)

        d_real_g, d_real_l, bd_pred_real = D(real_img, class_label, mask)
        d_fake_g, d_fake_l, bd_pred_fake = D(fake_img.detach(), class_label, mask)

        # Hinge loss
        loss_global = (F.relu(1.0 - d_real_g) + F.relu(1.0 + d_fake_g)).mean()
        loss_local = (F.relu(1.0 - d_real_l) + F.relu(1.0 + d_fake_l)).mean()

        # Boundary distance 回歸（real）
        loss_bd_regress = F.mse_loss(bd_pred_real, real_bd)

        # [v3-fix B3] BD 回歸（fake → target_bd）：教 D 學會 fake 的 bd 契約
        loss_bd_regress_fake = torch.tensor(0.0, device=real_img.device)
        if target_bd is not None:
            loss_bd_regress_fake = F.mse_loss(bd_pred_fake, target_bd)

        total = (loss_global
                 + self.lambda_local * loss_local
                 + self.lambda_bd * loss_bd_regress
                 + 0.5 * self.lambda_bd * loss_bd_regress_fake)

        # R1 正則化（只對 real images 的 Global D score）
        r1_val = 0.0
        if do_r1:
            grad_real = torch.autograd.grad(
                outputs=d_real_g.sum(),
                inputs=real_img,
                create_graph=True,
            )[0]
            r1 = grad_real.square().sum(dim=[1, 2, 3]).mean()
            # lazy reg 補償：乘以 interval
            total = total + self.lambda_r1 * 0.5 * r1 * self.r1_interval
            r1_val = r1.item()

        return total, {
            'd_global': loss_global.item(),
            'd_local': loss_local.item(),
            'd_bd': loss_bd_regress.item(),
            'd_bd_fake': float(loss_bd_regress_fake.item() if torch.is_tensor(loss_bd_regress_fake) else loss_bd_regress_fake),
            'd_r1': r1_val,
            'd_real_score': d_real_g.mean().item(),
            'd_fake_score': d_fake_g.mean().item(),
        }

    # ─── Generator Loss ───
    def g_loss(self, D: nn.Module, fake_img: torch.Tensor,
               class_label: torch.Tensor, mask: torch.Tensor,
               real_img: torch.Tensor = None,
               mask_soft: torch.Tensor = None,
               mask_hard: torch.Tensor = None,
               target_bd: torch.Tensor = None,
               diversity_loss_fn: FeatureDiversityLoss = None,
               z1: torch.Tensor = None, z2: torch.Tensor = None,
               fake_img2: torch.Tensor = None):
        """Generator 總損失。"""
        device = fake_img.device

        # 1. 對抗損失: G 要讓 D(fake) 盡量大
        d_fake_g, d_fake_l, bd_pred_fake = D(fake_img, class_label, mask)
        l_adv = -d_fake_g.mean()
        l_local = -d_fake_l.mean()

        # 2. Boundary distance guidance
        l_bd = torch.tensor(0.0, device=device)
        if target_bd is not None:
            l_bd = F.mse_loss(bd_pred_fake, target_bd)

        # 3. Feature diversity loss
        l_fd = torch.tensor(0.0, device=device)
        if diversity_loss_fn is not None and fake_img2 is not None:
            l_fd = diversity_loss_fn(fake_img, fake_img2, z1, z2)

        # 4. 重建損失（mask 外區域不應被改變）
        l_rec = torch.tensor(0.0, device=device)
        if real_img is not None and mask_soft is not None:
            l_rec = F.l1_loss(
                fake_img * (1 - mask_soft),
                real_img * (1 - mask_soft),
            )

        # 5. Mask 區域紋理引導（讓 G 學會正確亮度和基本紋理）
        l_mask_guide = torch.tensor(0.0, device=device)
        _mask = mask_hard if mask_hard is not None else mask
        if real_img is not None and _mask is not None and _mask.sum() > 0:
            mask_sum = _mask.sum() + 1e-8
            l_mask_guide = (torch.abs(fake_img - real_img) * _mask).sum() / mask_sum

        total = (l_adv
                 + self.lambda_local * l_local
                 + self.lambda_bd * l_bd
                 + self.lambda_fd * l_fd
                 + self.lambda_rec * l_rec
                 + self.lambda_mask_guide * l_mask_guide)

        return total, {
            'g_adv': l_adv.item(),
            'g_local': l_local.item(),
            'g_bd': l_bd.item(),
            'g_fd': l_fd.item(),
            'g_mask_guide': l_mask_guide.item(),
            'g_rec': l_rec.item(),
        }
