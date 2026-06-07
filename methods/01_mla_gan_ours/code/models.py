"""
MLA-GAN v2 模型定義：Generator（~8.7M 參數）+ Discriminator（~2M 參數）。

Generator v2 三項改進：
  1. bd 解耦：BoundaryDistanceModulator 直接調變 AdaIN（不經 MappingNetwork）
  2. Gated Conv Encoder：取代 hard masking，提供 skip features
  3. FFC Decoder：FFT 全局感受野 + skip connection，消除條紋偽影

資料流：
  GatedContextEncoder → MappingNetwork(z+class) + BoundaryDistanceModulator(bd)
  → Bottleneck Fusion → FFCAdaINDecoder(skip + w + bd_delta) → Soft Compositing

Discriminator（未修改）：
  GlobalDiscriminator（Projection D + Boundary Distance Regressor）
  + LocalDiscriminator（Lesion ROI 64×64）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════
# Generator 組件
# ═══════════════════════════════════════════════

class GatedConv2d(nn.Module):
    """
    Gated Convolution (Yu et al., DeepFill v2).

    每個空間位置學一個 gate ∈ [0,1]：
    - mask 內像素：gate → ~0（自動忽略無效區域）
    - mask 外像素：gate → ~1（正常使用）
    - 邊緣像素：gate → 中間值（利用部分資訊）
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 4,
                 stride: int = 2, padding: int = 1):
        super().__init__()
        self.conv_feature = nn.utils.spectral_norm(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding))
        self.conv_gate = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_feature(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return self.norm(feat * gate)


class GatedContextEncoder(nn.Module):
    """
    Gated 上下文編碼器：用 Gated Conv 取代 hard masking。

    輸入：concat(img, mask) → [B, 2, 256, 256]（不做 hard mask）
    輸出：bottleneck [B, 256, 16, 16] + skip_features [e0, e1, e2]

    Gated Conv 自動學會忽略 mask 區域，保留邊緣部分資訊。
    Skip features 供 decoder 使用高解析度參考。

    防記憶設計：
    1. 參數量 ~800K（gated conv 雙卷積但仍輕量）
    2. 訓練時 context dropout → 強迫 z 學語義
    """

    def __init__(self, out_channels: int = 256, context_dropout: float = 0.3):
        super().__init__()
        # 256→128→64→32→16, channels: 2→64→128→256→256
        self.enc0 = GatedConv2d(2, 64, 4, 2, 1)       # 256→128, ch=64
        self.enc1 = GatedConv2d(64, 128, 4, 2, 1)      # 128→64,  ch=128
        self.enc2 = GatedConv2d(128, 256, 4, 2, 1)     # 64→32,   ch=256
        self.enc3 = GatedConv2d(256, out_channels, 4, 2, 1)  # 32→16, ch=256
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.context_dropout_rate = context_dropout

    def forward(self, img: torch.Tensor, mask_expanded: torch.Tensor):
        """
        Args:
            img:           [B, 1, 256, 256] — 原始灰階 CT
            mask_expanded: [B, 1, 256, 256] — 外擴過的 mask
        Returns:
            bottleneck:    [B, 256, 16, 16]
            skip_features: [e0(128×128, 64ch), e1(64×64, 128ch), e2(32×32, 256ch)]
        """
        x = torch.cat([img, mask_expanded], dim=1)  # [B, 2, 256, 256] — no hard mask

        e0 = self.act(self.enc0(x))    # [B, 64,  128, 128]
        e1 = self.act(self.enc1(e0))   # [B, 128,  64,  64]
        e2 = self.act(self.enc2(e1))   # [B, 256,  32,  32]
        e3 = self.act(self.enc3(e2))   # [B, 256,  16,  16]

        # Context dropout：訓練時隨機清零 bottleneck，強迫 z 學語義
        if self.training:
            drop = torch.bernoulli(torch.tensor(self.context_dropout_rate, device=e3.device))
            e3 = e3 * (1.0 - drop)

        skip_features = [e0, e1, e2]  # 128×128, 64×64, 32×32
        return e3, skip_features


class HardMaskEncoder(nn.Module):
    """
    Ablation A7: 取代 GatedContextEncoder 的硬遮罩 baseline。

    輸入：concat(img * (1 - mask_expanded), mask_expanded)
    輸出：bottleneck [B, out_channels, 16, 16] + skip_features [e0, e1, e2]

    與 GatedContextEncoder 介面完全一致（同樣 4 個下採樣 stage、同樣 channel
    pyramid），差異只在內部用 Conv2d + InstanceNorm 取代 GatedConv2d。
    """

    def __init__(self, out_channels: int = 256, context_dropout: float = 0.3):
        super().__init__()
        self.enc0 = self._block(2, 64)
        self.enc1 = self._block(64, 128)
        self.enc2 = self._block(128, 256)
        self.enc3 = self._block(256, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.context_dropout_rate = context_dropout

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1)),
            nn.InstanceNorm2d(out_ch),
        )

    def forward(self, img: torch.Tensor, mask_expanded: torch.Tensor):
        masked_img = img * (1 - mask_expanded)
        x = torch.cat([masked_img, mask_expanded], dim=1)

        e0 = self.act(self.enc0(x))
        e1 = self.act(self.enc1(e0))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))

        if self.training:
            drop = torch.bernoulli(torch.tensor(
                self.context_dropout_rate, device=e3.device))
            e3 = e3 * (1.0 - drop)

        return e3, [e0, e1, e2]


class MappingNetwork(nn.Module):
    """
    映射網路：z + class_embed (+ bd if use_bd_in_mapping) → w.

    use_bd_in_mapping=False (default v2) — bd 解耦到 BoundaryDistanceModulator
    use_bd_in_mapping=True  (A1 ablation) — bd 回到 mapping concat
    """

    def __init__(self, z_dim: int = 128, c_dim: int = 3,
                 w_dim: int = 256, c_embed_dim: int = 64,
                 use_bd_in_mapping: bool = False, bd_dim: int = 1):
        super().__init__()
        self.class_embed = nn.Embedding(c_dim, c_embed_dim)
        nn.init.normal_(self.class_embed.weight, std=1.0)
        self.use_bd_in_mapping = use_bd_in_mapping
        self.bd_dim = bd_dim
        input_dim = z_dim + c_embed_dim + (bd_dim if use_bd_in_mapping else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, w_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(w_dim, w_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(w_dim, w_dim),
        )

    def forward(self, z: torch.Tensor, class_onehot: torch.Tensor,
                bd: torch.Tensor = None) -> torch.Tensor:
        class_idx = class_onehot.argmax(dim=1)
        c_emb = self.class_embed(class_idx)
        if self.use_bd_in_mapping:
            assert bd is not None, "use_bd_in_mapping=True 需傳 bd"
            return self.net(torch.cat([z, c_emb, bd], dim=1))
        return self.net(torch.cat([z, c_emb], dim=1))


class BoundaryDistanceModulator(nn.Module):
    """
    bd 的獨立影響通道：bd → 每層 AdaIN 的 (Δγ, Δβ)。

    與 w 完全解耦：
    - w 控制「風格/多樣性」（由 z + class 決定）
    - bd 控制「離決策邊界多遠」（獨立通道）

    加法調變：gamma = gamma_w + Δγ_bd, beta = beta_w + Δβ_bd
    """

    def __init__(self, num_layers: int = 4,
                 layer_channels: tuple = (256, 128, 64, 32)):
        super().__init__()
        self.layer_channels = layer_channels

        # bd scalar → high-dim embedding
        self.bd_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Per-layer Δγ, Δβ projections
        self.delta_layers = nn.ModuleList()
        for ch in layer_channels[:num_layers]:
            self.delta_layers.append(nn.Linear(128, ch * 2))

    def forward(self, bd: torch.Tensor):
        """
        Args:
            bd: [B, 1] boundary distance
        Returns:
            deltas: list of (Δγ, Δβ) per decoder layer
                    Δγ: [B, ch, 1, 1], Δβ: [B, ch, 1, 1]
        """
        h = self.bd_embed(bd)  # [B, 128]
        deltas = []
        for i, layer in enumerate(self.delta_layers):
            ch = self.layer_channels[i]
            delta = layer(h)  # [B, ch*2]
            delta_gamma = delta[:, :ch].unsqueeze(-1).unsqueeze(-1)
            delta_beta = delta[:, ch:].unsqueeze(-1).unsqueeze(-1)
            deltas.append((delta_gamma, delta_beta))
        return deltas



class FFCBlock(nn.Module):
    """
    Fast Fourier Convolution Block (LaMa, Suvorov et al., WACV 2022).

    spatial branch: 3×3 conv（局部細節）
    frequency branch: FFT → 1×1 conv → IFFT（全局結構，消除條紋偽影）
    兩分支交互：spatial↔frequency 互相融合。
    """

    def __init__(self, channels: int, ratio_freq: float = 0.5):
        super().__init__()
        self.ch_s = int(channels * (1 - ratio_freq))  # spatial channels
        self.ch_f = channels - self.ch_s               # frequency channels

        # Spatial branch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(self.ch_s, self.ch_s, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Frequency branch: operates on real+imag concatenated
        self.freq_conv = nn.Sequential(
            nn.Conv2d(self.ch_f * 2, self.ch_f * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Cross-branch interaction
        self.s2f = nn.Conv2d(self.ch_s, self.ch_f, 1, 1, 0)
        self.f2s = nn.Conv2d(self.ch_f, self.ch_s, 1, 1, 0)

        self.norm = nn.InstanceNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_s = x[:, :self.ch_s]   # spatial part
        x_f = x[:, self.ch_s:]   # frequency part

        # Spatial path + freq→spatial cross
        out_s = self.spatial_conv(x_s) + self.f2s(x_f)

        # Frequency path: FFT → conv → IFFT + spatial→freq cross
        # 強制 float32 避免 ComplexHalf 數值溢出
        B, C, H, W = x_f.shape
        x_f_fp32 = x_f.float()
        fft = torch.fft.rfft2(x_f_fp32, norm='ortho')
        fft_cat = torch.cat([fft.real, fft.imag], dim=1)  # [B, C*2, H, W//2+1]
        fft_out = self.freq_conv(fft_cat.to(x_f.dtype)).float()
        real, imag = fft_out.chunk(2, dim=1)
        fft_processed = torch.complex(real, imag)
        out_f = torch.fft.irfft2(fft_processed, s=(H, W), norm='ortho').to(x_f.dtype)
        out_f = out_f + self.s2f(x_s)

        out = torch.cat([out_s, out_f], dim=1)
        return self.norm(out)


class FFCDecoderBlock(nn.Module):
    """
    單個 decoder block: upsample → concat skip → FFC → AdaIN(w + bd_delta)。

    FFC 提供全局感受野（消除條紋），skip 提供高頻細節。
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, w_dim: int = 256):
        super().__init__()
        self.has_skip = skip_ch > 0

        # PixelShuffle upsample (避免 bilinear 條紋)
        self.up_conv = nn.utils.spectral_norm(nn.Conv2d(in_ch, in_ch * 4, 3, 1, 1))
        self.pixel_shuffle = nn.PixelShuffle(2)

        # Skip fusion: 1×1 conv 降回 in_ch
        if self.has_skip:
            self.skip_conv = nn.Conv2d(in_ch + skip_ch, in_ch, 1, 1, 0)

        # FFC block (spatial + frequency dual path)
        self.ffc = FFCBlock(in_ch)

        # Channel reduce
        self.channel_reduce = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        # AdaIN (from w)
        self.inst_norm = nn.InstanceNorm2d(out_ch, affine=False)
        self.adain_fc = nn.Linear(w_dim, out_ch * 2)

    def forward(self, x: torch.Tensor, skip_feat, w: torch.Tensor,
                bd_delta) -> torch.Tensor:
        """
        Args:
            x:         [B, in_ch, H, W]
            skip_feat: [B, skip_ch, H*2, W*2] or None
            w:         [B, w_dim]
            bd_delta:  (Δγ, Δβ) from BoundaryDistanceModulator, or None
        """
        # Upsample
        x = self.pixel_shuffle(self.up_conv(x))

        # Skip connection
        if self.has_skip and skip_feat is not None:
            x = torch.cat([x, skip_feat], dim=1)
            x = self.skip_conv(x)

        # FFC
        x = self.ffc(x)

        # Channel reduce
        x = self.channel_reduce(x)

        # AdaIN with bd modulation
        x = self.inst_norm(x)
        params = self.adain_fc(w)
        ch = x.shape[1]
        gamma = params[:, :ch].unsqueeze(-1).unsqueeze(-1)
        beta = params[:, ch:].unsqueeze(-1).unsqueeze(-1)

        if bd_delta is not None:
            d_gamma, d_beta = bd_delta
            gamma = gamma + d_gamma
            beta = beta + d_beta

        x = (1 + gamma) * x + beta
        return F.leaky_relu(x, 0.2)


class FFCAdaINDecoder(nn.Module):
    """
    完整 Decoder: 16×16 → 256×256。
    每層: PixelShuffle up → skip concat → FFC → AdaIN(w + bd) → LeakyReLU。
    """

    def __init__(self, w_dim: int = 256, enc_out_channels: int = 256):
        super().__init__()
        # 16→32→64→128→256
        #                    in_ch, skip_ch, out_ch
        self.block0 = FFCDecoderBlock(enc_out_channels, 256, 128, w_dim)  # 16→32, skip=e2(32×32,256ch)
        self.block1 = FFCDecoderBlock(128, 128, 64, w_dim)   # 32→64, skip=e1(64×64,128ch)
        self.block2 = FFCDecoderBlock(64, 64, 32, w_dim)     # 64→128, skip=e0(128×128,64ch)
        self.block3 = FFCDecoderBlock(32, 0, 32, w_dim)      # 128→256, no skip

        self.to_img = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, bottleneck: torch.Tensor, skip_features: list,
                w: torch.Tensor, bd_deltas: list) -> torch.Tensor:
        """
        Args:
            bottleneck:    [B, 256, 16, 16]
            skip_features: [e0(128×128,64ch), e1(64×64,128ch), e2(32×32,256ch)]
            w:             [B, w_dim]
            bd_deltas:     list of 4 (Δγ, Δβ)
        Returns:
            [B, 1, 256, 256]
        """
        e0, e1, e2 = skip_features

        x = self.block0(bottleneck, e2, w, bd_deltas[0])  # 16→32
        x = self.block1(x, e1, w, bd_deltas[1])           # 32→64
        x = self.block2(x, e0, w, bd_deltas[2])           # 64→128
        x = self.block3(x, None, w, bd_deltas[3])         # 128→256

        return self.to_img(x)


class MLAGenerator(nn.Module):
    """
    MLA-GAN v2 生成器：三項架構改進。

    改進 1: bd 解耦 — BoundaryDistanceModulator 直接調變 AdaIN γ/β
    改進 2: Gated Conv Encoder — 自動學忽略 mask，保留 skip features
    改進 3: FFC Decoder — FFT 全局感受野 + skip connection

    資料流：
    1. GatedContextEncoder: concat(img, mask) → bottleneck + skip_features
    2. MappingNetwork: z + class → w（bd 不再進入）
    3. BoundaryDistanceModulator: bd → per-layer (Δγ, Δβ)
    4. Bottleneck: context + w injection → fused [256, 16, 16]
    5. FFCAdaINDecoder: skip + FFC + AdaIN(w + bd_delta) → generated
    6. Soft compositing

    總參數量：~3.3M（+~0.5M from gated conv + FFC + bd_mod）
    """

    def __init__(self, z_dim: int = 128, c_dim: int = 3, bd_dim: int = 1,
                 w_dim: int = 256, enc_out_channels: int = 256,
                 context_dropout: float = 0.3, blur_sigma: float = 4.0,
                 blur_kernel_size: int = 17, mask_expand_px: int = 16,
                 use_bd_modulator: bool = True,
                 use_gated_conv: bool = True,
                 use_skip: bool = True,
                 encoder_hard_mask: bool = False):
        super().__init__()

        self.mask_expand_px = mask_expand_px
        self.use_bd_modulator = use_bd_modulator
        self.use_gated_conv = use_gated_conv
        self.use_skip = use_skip
        # A7': Gated Conv + hard mask input (confound control for A7).
        self.encoder_hard_mask = encoder_hard_mask

        # Encoder: gated (default) or hard-mask (A7 ablation)
        if use_gated_conv:
            self.context_enc = GatedContextEncoder(
                out_channels=enc_out_channels,
                context_dropout=context_dropout,
            )
        else:
            self.context_enc = HardMaskEncoder(
                out_channels=enc_out_channels,
                context_dropout=context_dropout,
            )

        # Mapping: if NOT using bd_mod (A1 ablation), bd is concatenated into mapping
        self.mapping = MappingNetwork(
            z_dim, c_dim, w_dim,
            use_bd_in_mapping=(not use_bd_modulator),
            bd_dim=bd_dim,
        )

        # Boundary Distance Modulator (skipped when use_bd_modulator=False)
        # Decoder per-layer channels: block0=128, block1=64, block2=32, block3=32
        if use_bd_modulator:
            self.bd_mod = BoundaryDistanceModulator(
                num_layers=4,
                layer_channels=(128, 64, 32, 32),
            )
        else:
            self.bd_mod = None

        # Bottleneck fusion（保留：雙路徑 context + w 融合）
        self.context_norm = nn.InstanceNorm2d(enc_out_channels)
        self.z_to_spatial = nn.Linear(w_dim, enc_out_channels * 4 * 4)
        self.z_spatial_norm = nn.InstanceNorm2d(enc_out_channels)
        self.bottleneck_fuse = nn.Sequential(
            nn.Conv2d(enc_out_channels * 2, enc_out_channels, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # FFC Decoder with skip connections（改進 3）
        self.decoder = FFCAdaINDecoder(w_dim, enc_out_channels)

        # Soft mask 的高斯模糊核（不可學習）
        self.register_buffer(
            'blur_kernel',
            self._make_gaussian_kernel(sigma=blur_sigma, size=blur_kernel_size),
        )

    def _make_gaussian_kernel(self, sigma: float, size: int) -> torch.Tensor:
        """預計算高斯模糊核，用於 soft mask 生成。"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel = g.outer(g)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def _soften_mask(self, mask: torch.Tensor) -> tuple:
        """
        將二值 mask 外擴並高斯模糊，產生 soft 過渡帶。

        Returns:
            mask_expanded: 外擴版（給 encoder 標記 mask 位置）
            mask_soft:     模糊版（給 compositing，平滑邊界）
        """
        ep = self.mask_expand_px
        mask_expanded = F.max_pool2d(
            mask, kernel_size=ep * 2 + 1, stride=1, padding=ep,
        )
        pad = self.blur_kernel.shape[-1] // 2
        mask_soft = F.conv2d(mask, self.blur_kernel, padding=pad)
        mask_soft = torch.clamp(mask_soft, 0, 1)
        return mask_expanded, mask_soft

    def forward(self, img: torch.Tensor, mask: torch.Tensor,
                z: torch.Tensor, class_onehot: torch.Tensor,
                boundary_dist: torch.Tensor):
        """
        Args:
            img:           [B, 1, 256, 256] — 原始灰階 CT
            mask:          [B, 1, 256, 256] — 二值 lesion mask
            z:             [B, 128]         — 隨機噪聲
            class_onehot:  [B, 3]           — 類別 one-hot
            boundary_dist: [B, 1]           — 目標 boundary distance

        Returns:
            output:    [B, 1, 256, 256] — 合成影像
            mask_soft: [B, 1, 256, 256] — soft mask（供 loss 使用）
            generated: [B, 1, 256, 256] — raw 生成內容（供 debug）
        """
        # Step 1: 準備 mask
        mask_expanded, mask_soft = self._soften_mask(mask)

        # Step 2: Encoder (gated or hard-mask, both return same interface)
        # A7': 若 encoder_hard_mask=True，先把 lesion 區置 0 再餵 encoder
        img_for_enc = img * (1.0 - mask_expanded) if self.encoder_hard_mask else img
        bottleneck, skip_features = self.context_enc(img_for_enc, mask_expanded)
        if not self.use_skip:
            skip_features = [None, None, None]  # A8: drop skip connections

        # Step 3: Mapping (bd routed through here if bd_mod disabled)
        if self.use_bd_modulator:
            w = self.mapping(z, class_onehot)
        else:
            w = self.mapping(z, class_onehot, bd=boundary_dist)

        # Step 4: Boundary modulation deltas (None if disabled)
        if self.use_bd_modulator:
            bd_deltas = self.bd_mod(boundary_dist)
        else:
            bd_deltas = [None, None, None, None]

        # Step 5: 注入 w 到 bottleneck（雙空間路徑，量級對等）
        B = img.shape[0]
        ctx_norm = self.context_norm(bottleneck)
        z_feat = self.z_to_spatial(w).view(B, -1, 4, 4)
        z_feat = F.interpolate(z_feat, size=(16, 16), mode='bilinear', align_corners=False)
        z_feat = self.z_spatial_norm(z_feat)
        fused = self.bottleneck_fuse(
            torch.cat([ctx_norm, z_feat], dim=1)
        )  # [B, 256, 16, 16]

        # Step 6: FFC Decode with skip + bd modulation
        generated = self.decoder(fused, skip_features, w, bd_deltas)

        # Step 7: 可微分 soft compositing
        output = img * (1 - mask_soft) + generated * mask_soft

        return output, mask_soft, generated


# ═══════════════════════════════════════════════
# Discriminator 組件
# ═══════════════════════════════════════════════

class GlobalDiscriminator(nn.Module):
    """
    全圖 256×256 雙任務判別器。

    Branch 1 — Projection D (Miyato & Koyama, ICLR 2018):
      score = φ(x)ᵀ · embed(y) + ψ(x)

    Branch 2 — Boundary Distance Regressor:
      預測影像到分類器 decision boundary 的距離，
      為 G 提供更豐富的梯度信號。
    """

    def __init__(self, num_classes: int = 3, base_ch: int = 32):
        super().__init__()

        # 共享 Backbone: 256 → 4×4
        self.backbone = nn.Sequential(
            # 256 → 128
            nn.utils.spectral_norm(nn.Conv2d(1, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 → 64
            nn.utils.spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 → 32
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 → 16
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 → 8
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 → 4
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 8, base_ch * 8, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        feat_dim = base_ch * 8  # 256

        # Branch 1: Projection Real/Fake
        self.phi = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(feat_dim * 4 * 4, feat_dim)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.psi = nn.utils.spectral_norm(nn.Linear(feat_dim, 1))
        self.class_embed = nn.utils.spectral_norm(
            nn.Embedding(num_classes, feat_dim)
        )

        # Branch 2: Boundary Distance Regressor
        self.boundary_head = nn.Sequential(
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(feat_dim * 4 * 4, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, img: torch.Tensor, class_label: torch.Tensor,
                mask: torch.Tensor = None):
        """
        Args:
            img:         [B, 1, 256, 256]
            class_label: [B] — 整數類別標籤
            mask:        未使用，保留介面一致性
        Returns:
            rf_score: [B, 1] — real/fake 分數
            bd_pred:  [B, 1] — 預測的 boundary distance
        """
        feat_map = self.backbone(img)  # [B, 256, 4, 4]

        # Branch 1: Projection discriminator
        phi_x = self.phi(feat_map)                          # [B, 256]
        class_emb = self.class_embed(class_label)           # [B, 256]
        proj = (phi_x * class_emb).sum(dim=1, keepdim=True) # [B, 1]
        uncond = self.psi(phi_x)                             # [B, 1]
        rf_score = proj + uncond

        # Branch 2: Boundary distance regression
        bd_pred = self.boundary_head(feat_map)  # [B, 1]

        return rf_score, bd_pred


class LocalDiscriminator(nn.Module):
    """
    局部判別器：動態裁切 lesion 周圍 ROI（64×64）進行判別。

    只看病灶區域，專注判斷局部病灶真實性。
    裁切操作是可微分的（tensor indexing 不影響梯度流）。
    """

    def __init__(self, crop_size: int = 64, base_ch: int = 64):
        super().__init__()
        self.crop_size = crop_size

        # 輕量判別器: 64×64 → scalar
        self.net = nn.Sequential(
            # 64 → 32
            nn.utils.spectral_norm(nn.Conv2d(1, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 → 16
            nn.utils.spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 → 8
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 → 4
            nn.utils.spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # 4×4 → 1
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(base_ch * 4 * 4 * 4, 1)),
        )

    def crop_lesion_roi(self, img: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
        """
        以 mask 中心動態裁切 crop_size×crop_size 區域。
        """
        B = img.shape[0]
        cs = self.crop_size
        img_size = img.shape[-1]
        crops = []

        for i in range(B):
            m = mask[i, 0]  # [H, W]
            ys, xs = torch.where(m > 0.5)

            if len(ys) > 0:
                cy = int(ys.float().mean().item())
                cx = int(xs.float().mean().item())
            else:
                # Normal 類：隨機選一個位置
                cy = torch.randint(cs // 2, img_size - cs // 2, (1,)).item()
                cx = torch.randint(cs // 2, img_size - cs // 2, (1,)).item()

            # 確保裁切不超出邊界
            y1 = max(0, min(cy - cs // 2, img_size - cs))
            x1 = max(0, min(cx - cs // 2, img_size - cs))
            crops.append(img[i:i + 1, :, y1:y1 + cs, x1:x1 + cs])

        return torch.cat(crops, dim=0)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img:  [B, 1, 256, 256]
            mask: [B, 1, 256, 256]
        Returns:
            score: [B, 1]
        """
        crops = self.crop_lesion_roi(img, mask)  # [B, 1, 64, 64]
        return self.net(crops)


class MLADiscriminator(nn.Module):
    """
    雙尺度判別器：Global D + Local D。

    Global D: 整張 256×256，判斷全局真實性 + 類別條件 + BD 回歸
    Local D:  只看 lesion 周圍 64×64，判斷局部病灶品質
    """

    def __init__(self, num_classes: int = 3, global_base_ch: int = 32,
                 local_base_ch: int = 64, local_crop_size: int = 64):
        super().__init__()
        self.global_d = GlobalDiscriminator(
            num_classes=num_classes, base_ch=global_base_ch,
        )
        self.local_d = LocalDiscriminator(
            crop_size=local_crop_size, base_ch=local_base_ch,
        )

    def forward(self, img: torch.Tensor, class_label: torch.Tensor,
                mask: torch.Tensor):
        """
        Returns:
            rf_score:    [B, 1] — Global real/fake 分數
            local_score: [B, 1] — Local real/fake 分數
            bd_pred:     [B, 1] — 預測的 boundary distance
        """
        rf_score, bd_pred = self.global_d(img, class_label, mask)
        local_score = self.local_d(img, mask)
        return rf_score, local_score, bd_pred
