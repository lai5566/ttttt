#!/usr/bin/env python3
"""
MLA-GAN 訓練腳本。

Phase 0: 載入預訓練分類器 → 預計算 boundary distance
Phase 1: 訓練 MLA-GAN（200 epochs）
  - D: 5 步 per G 1 步
  - WGAN-GP + ADA + boundary regression + feature diversity

用法：
    python train.py
    python train.py --epochs 100 --batch-size 16
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from config import MLAGANConfig
from models import MLAGenerator, MLADiscriminator
from losses import BoundaryDistanceComputer, FeatureDiversityLoss, MLAGANLoss
from dataset import MLADataset, precompute_boundary_distances
from utils import (
    load_pretrained_classifier, get_feature_extractor,
    AdaptiveAugmentation, sample_boundary_target,
)


def train(config: MLAGANConfig):
    """MLA-GAN 完整訓練流程。"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(config.output_dir, exist_ok=True)

    # ═══ GPU 效能優化 ═══
    torch.backends.cudnn.benchmark = True   # 固定輸入尺寸，啟用 cuDNN autotuner
    scaler_G = GradScaler('cuda')            # AMP: fp16 混合精度
    scaler_D = GradScaler('cuda')

    # ═══ Phase 0: 載入預訓練分類器 + 預計算 Boundary Distance ═══
    print("=" * 60)
    print("Phase 0: 載入預訓練分類器 & 預計算 boundary distance")
    print("=" * 60)

    classifier = load_pretrained_classifier(
        config.pretrained_classifier, config.num_classes, device,
        arch=getattr(config, 'classifier_arch', 'efficientnet_b0'),
    )
    feat_extractor = get_feature_extractor(classifier)

    # 先建立不增強的 dataset 用於預計算 boundary distance
    temp_dataset = MLADataset(config, augment=False)
    bd_method = getattr(config, 'bd_compute_method', 'rank')
    bd_map, bd_stats = precompute_boundary_distances(
        temp_dataset, classifier, device, method=bd_method,
    )

    # 儲存 bd_stats 供生成時使用
    bd_stats_path = Path(config.output_dir) / 'bd_stats.json'
    with open(bd_stats_path, 'w') as f:
        json.dump(bd_stats, f, indent=2)
    print(f"Boundary distance 統計已儲存至 {bd_stats_path}")

    # ═══ Phase 1: 建立模型和資料集 ═══
    print("\n" + "=" * 60)
    print("Phase 1: 訓練 MLA-GAN")
    print("=" * 60)

    # 帶增強的 dataset（使用預計算的 bd）
    dataset = MLADataset(config, precomputed_bd=bd_map, augment=True)
    dataset.to_device(device)  # 整個 dataset 搬到 GPU，零傳輸開銷
    N = len(dataset)
    num_batches = N // config.batch_size

    # 模型
    G = MLAGenerator(
        z_dim=config.z_dim, c_dim=config.c_dim, bd_dim=config.bd_dim,
        w_dim=config.w_dim, enc_out_channels=config.enc_out_channels,
        context_dropout=config.context_dropout, blur_sigma=config.blur_sigma,
        blur_kernel_size=config.blur_kernel_size,
        mask_expand_px=config.mask_expand_px,
        use_bd_modulator=getattr(config, 'use_bd_modulator', True),
        use_gated_conv=getattr(config, 'use_gated_conv', True),
        use_skip=getattr(config, 'use_skip', True),
        encoder_hard_mask=getattr(config, 'encoder_hard_mask', False),
    ).to(device)

    D = MLADiscriminator(
        num_classes=config.num_classes,
        global_base_ch=config.global_d_base_ch,
        local_base_ch=config.local_d_base_ch,
        local_crop_size=config.local_crop_size,
    ).to(device)

    # 計算參數量
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())
    print(f"Generator 參數量: {g_params:,} ({g_params/1e6:.1f}M)")
    print(f"Discriminator 參數量: {d_params:,} ({d_params/1e6:.1f}M)")
    print(f"總參數量: {(g_params + d_params):,} ({(g_params + d_params)/1e6:.1f}M)")

    # torch.compile：融合 kernel，減少 launch overhead
    # D 的 inplace LeakyReLU 與 compile 不相容，只 compile G
    if hasattr(torch, 'compile'):
        print("啟用 torch.compile（G only）")
        G = torch.compile(G)

    # 優化器（TTUR: D lr < G lr，防止 D 壓倒較大的 G）
    d_lr = getattr(config, 'dlr', config.lr)
    opt_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=config.betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=d_lr, betas=config.betas)
    print(f"TTUR: G_lr={config.lr}, D_lr={d_lr}")

    # 損失函數
    loss_fn = MLAGANLoss(
        lambda_local=config.lambda_local,
        lambda_bd=config.lambda_bd,
        lambda_fd=config.lambda_fd,
        lambda_rec=config.lambda_rec,
        lambda_mask_guide=config.lambda_mask_guide,
        lambda_r1=config.lambda_r1,
        r1_interval=config.r1_interval,
    )
    feat_div_loss = FeatureDiversityLoss(feat_extractor, margin=config.fd_margin)

    # ADA
    ada = AdaptiveAugmentation(
        target_rt=config.ada_target_rt,
        adjust_speed=config.ada_adjust_speed,
        max_p=config.ada_max_p,
    )

    # ═══ Resume from checkpoint ═══
    start_epoch = 0
    global_step = 0
    best_score_gap = -999
    log_history = []

    if config.resume_checkpoint:
        print(f"從 checkpoint 接續訓練: {config.resume_checkpoint}")
        ckpt = torch.load(config.resume_checkpoint, map_location=device)
        # 處理 torch.compile 的 _orig_mod. 前綴
        g_state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['G_state_dict'].items()}
        d_state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['D_state_dict'].items()}
        # 載入到未 compile 的模型再 compile
        G_inner = G._orig_mod if hasattr(G, '_orig_mod') else G
        G_inner.load_state_dict(g_state)
        D.load_state_dict(d_state)
        if 'opt_G_state_dict' in ckpt:
            opt_G.load_state_dict(ckpt['opt_G_state_dict'])
            opt_D.load_state_dict(ckpt['opt_D_state_dict'])
        if 'ada_p' in ckpt:
            ada.p = ckpt['ada_p']
        start_epoch = ckpt.get('epoch', 0)
        global_step = start_epoch * num_batches
        print(f"  從 epoch {start_epoch} 接續，global_step={global_step}")
        # 載入既有日誌
        log_path = Path(config.output_dir) / 'training_log.json'
        if log_path.exists():
            with open(log_path) as f:
                log_history = json.load(f)
            print(f"  載入既有日誌 {len(log_history)} 條")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        G.train()
        D.train()

        # 固定權重（由 config 控制）
        loss_fn.lambda_mask_guide = config.lambda_mask_guide
        loss_fn.lambda_fd = config.lambda_fd

        # 每 epoch 重新打亂 index（純 GPU 操作）
        perm = torch.randperm(N, device=device)

        for batch_idx in range(num_batches):
            idx = perm[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size]
            real_img, mask, label, real_bd = dataset.get_batch(idx)
            B = real_img.shape[0]
            class_onehot = F.one_hot(label, config.num_classes).float()

            # ─── [v3-fix B1+B2] 在 D 步驟之前先採樣一次 target_bd ───
            # B1: D 與 G 都用同一個 target_bd 生成 fake，保持對抗對稱
            # B2: sample_boundary_target 接收 class_label，做 per-class 取樣
            bd_strategy = getattr(config, 'bd_sampling', 'guided')
            if bd_strategy == 'real':
                target_bd = real_bd  # B5: use the real precomputed BD
            else:
                target_bd = sample_boundary_target(
                    bd_stats, B, strategy=bd_strategy,
                    near_ratio=config.bd_near_ratio,
                    mid_ratio=config.bd_mid_ratio,
                    far_ratio=config.bd_far_ratio,
                    class_label=label,            # [v3-fix B2] class-conditional
                ).to(device)

            # ─── 訓練 D（d_steps 步）───
            for _ in range(config.d_steps):
                z = torch.randn(B, config.z_dim, device=device)
                with torch.no_grad(), autocast('cuda'):
                    # [v3-fix B1] D-step 用 target_bd 生成 fake（原本用 real_bd）
                    fake_img, mask_soft, _ = G(
                        real_img, mask, z, class_onehot, target_bd,
                    )

                # ADA 只增強 D 的輸入
                real_aug = ada.augment(real_img)
                fake_aug = ada.augment(fake_img.float())

                # R1 需要 create_graph，不能用 AMP scaler
                # [v3-fix B3] 傳 target_bd 給 d_loss，讓 D 學會 fake 的 bd 契約
                d_loss, d_logs = loss_fn.d_loss(
                    D, real_aug, fake_aug, label, mask, real_bd,
                    target_bd=target_bd,
                    global_step=global_step,
                )

                opt_D.zero_grad()
                d_loss.backward()
                if hasattr(config, 'grad_clip') and config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), config.grad_clip)
                opt_D.step()

            # 更新 ADA
            with torch.no_grad():
                d_real_scores, _, _ = D(real_img, label, mask)
                ada.update(d_real_scores)

            # ─── 訓練 G ───
            z1 = torch.randn(B, config.z_dim, device=device)
            z2 = torch.randn(B, config.z_dim, device=device)

            with autocast('cuda'):
                fake_img1, mask_soft1, _ = G(
                    real_img, mask, z1, class_onehot, target_bd,
                )
                fake_img2, _, _ = G(
                    real_img, mask, z2, class_onehot, target_bd,
                )

                g_loss, g_logs = loss_fn.g_loss(
                    D, fake_img1, label, mask,
                    real_img=real_img,
                    mask_soft=mask_soft1,
                    mask_hard=mask,
                    target_bd=target_bd,
                    diversity_loss_fn=feat_div_loss,
                    z1=z1, z2=z2, fake_img2=fake_img2,
                )

            opt_G.zero_grad()
            scaler_G.scale(g_loss).backward()
            if hasattr(config, 'grad_clip') and config.grad_clip > 0:
                scaler_G.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G.parameters(), config.grad_clip)
            scaler_G.step(opt_G)
            scaler_G.update()

            global_step += 1

            # ─── 日誌 ───
            if global_step % config.log_interval == 0:
                score_gap = d_logs['d_real_score'] - d_logs['d_fake_score']
                log_entry = {
                    'epoch': epoch,
                    'step': global_step,
                    'score_gap': score_gap,
                    'ada_p': ada.p,
                    **{f'd_{k}': v for k, v in d_logs.items()},
                    **{f'g_{k}': v for k, v in g_logs.items()},
                }
                log_history.append(log_entry)

                print(f"[E{epoch:03d} S{global_step:05d}] "
                      f"D_gap={score_gap:.3f} "
                      f"G_adv={g_logs['g_adv']:.3f} "
                      f"G_local={g_logs['g_local']:.3f} "
                      f"G_bd={g_logs['g_bd']:.3f} "
                      f"G_fd={g_logs['g_fd']:.3f} "
                      f"G_rec={g_logs['g_rec']:.3f} "
                      f"G_mg={g_logs.get('g_mask_guide',0):.3f} "
                      f"D_r1={d_logs.get('d_r1',0):.3f} "
                      f"ADA_p={ada.p:.2f}")

        # ─── Epoch 結束：儲存 checkpoint ───
        epoch_time = time.time() - epoch_start

        if (epoch + 1) % config.save_interval == 0 or epoch == config.epochs - 1:
            ckpt_path = Path(config.output_dir) / f"checkpoint_epoch{epoch + 1:03d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_G_state_dict': opt_G.state_dict(),
                'opt_D_state_dict': opt_D.state_dict(),
                'ada_p': ada.p,
                'bd_stats': bd_stats,
                'config': vars(config),
            }, ckpt_path)
            print(f"  Checkpoint 已儲存: {ckpt_path}")

        # 追蹤最佳模型（以 score gap 穩定性作為指標）
        if len(log_history) > 0:
            recent_gaps = [e['score_gap'] for e in log_history[-10:]]
            avg_gap = sum(recent_gaps) / len(recent_gaps)
            if 0 < avg_gap < best_score_gap * 3 or best_score_gap < 0:
                best_score_gap = avg_gap
                best_path = Path(config.output_dir) / "best_model.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'bd_stats': bd_stats,
                    'config': vars(config),
                }, best_path)

        print(f"  Epoch {epoch + 1}/{config.epochs} 完成 "
              f"({epoch_time:.1f}s, ADA_p={ada.p:.2f})")

    # ═══ 儲存訓練日誌 ═══
    log_path = Path(config.output_dir) / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(log_history, f, indent=2)
    print(f"\n訓練完成！日誌已儲存至 {log_path}")
    print(f"最佳模型：{Path(config.output_dir) / 'best_model.pth'}")

    return G, D, bd_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLA-GAN 訓練')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None,
                        help='接續訓練的 checkpoint 路徑')
    parser.add_argument('--seed', type=int, default=7,
                        help='隨機種子（必須是 7/49/91/133/175 之一）')
    parser.add_argument('--lambda-bd', type=float, default=None,
                        help='覆寫 config.lambda_bd (D-based BD regression loss weight)')
    parser.add_argument('--bd-method', default='kfold_json',
                        choices=['kfold_json', 'kfold_json_perclass'],
                        help='K-fold BD normalize mode (global or per-class)')
    parser.add_argument('--ir', type=int, default=10, choices=[5, 10, 20],
                        help='失衡比變體：選 MLAGANConfigIR{5,10,20}ResNet50（預設 10）')
    parser.add_argument('--bd-near-ratio', type=float, default=None,
                        help='Override config.bd_near_ratio (default 0.5)')
    parser.add_argument('--bd-mid-ratio', type=float, default=None,
                        help='Override config.bd_mid_ratio (default 0.3)')
    parser.add_argument('--bd-far-ratio', type=float, default=None,
                        help='Override config.bd_far_ratio (default 0.2)')
    args = parser.parse_args()

    # 設置 seed
    import random, numpy as np
    assert args.seed in (7, 49, 91, 133, 175), \
        f"seed 必須是 7/49/91/133/175 其一，得到 {args.seed}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[Seed] set to {args.seed}")

    if args.ir == 5:
        from config_ir5_resnet50 import MLAGANConfigIR5ResNet50 as _Cfg
    elif args.ir == 20:
        from config_ir20_resnet50 import MLAGANConfigIR20ResNet50 as _Cfg
    else:
        from config_ir10_resnet50 import MLAGANConfigIR10ResNet50 as _Cfg
    config = _Cfg()
    print(f'[IR] variant=ir{args.ir}, ct_dir={config.ct_dir}, kfold_bd_json={config.kfold_bd_json}')
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.resume is not None:
        config.resume_checkpoint = args.resume
    # Override BD method to load K-fold from JSON
    config.bd_compute_method = args.bd_method
    print(f'[BD method] {args.bd_method}')
    if args.bd_near_ratio is not None:
        config.bd_near_ratio = args.bd_near_ratio
    if args.bd_mid_ratio is not None:
        config.bd_mid_ratio = args.bd_mid_ratio
    if args.bd_far_ratio is not None:
        config.bd_far_ratio = args.bd_far_ratio
    print(f'[BD sampling] near={config.bd_near_ratio}, mid={config.bd_mid_ratio}, far={config.bd_far_ratio}')
    if args.lambda_bd is not None:
        config.lambda_bd = args.lambda_bd
        print(f'[OVERRIDE] config.lambda_bd = {args.lambda_bd}')

    train(config)
