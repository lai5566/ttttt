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
from contextlib import nullcontext
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
from ddp_utils import (
    setup_ddp, cleanup_ddp, unwrap_model, broadcast_module,
    average_gradients, all_reduce_mean, make_epoch_perm,
)


def train(config: MLAGANConfig):
    """MLA-GAN 完整訓練流程（支援單卡與多卡 DDP）。"""

    # ═══ 分散式初始化（單卡時退化為 no-op，行為與原本一致）═══
    ddp = setup_ddp()
    device = ddp.device
    distributed, world_size, is_main = ddp.distributed, ddp.world_size, ddp.is_main
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

    # 儲存 bd_stats 供生成時使用（只在主 rank 寫檔，避免多卡競寫）
    bd_stats_path = Path(config.output_dir) / 'bd_stats.json'
    if is_main:
        with open(bd_stats_path, 'w') as f:
            json.dump(bd_stats, f, indent=2)
        print(f"Boundary distance 統計已儲存至 {bd_stats_path}")

    # ═══ Phase 1: 建立模型和資料集 ═══
    print("\n" + "=" * 60)
    print("Phase 1: 訓練 MLA-GAN")
    print("=" * 60)

    # 帶增強的 dataset（使用預計算的 bd）
    dataset = MLADataset(config, precomputed_bd=bd_map, augment=True)
    dataset.to_device(device)  # 整個 dataset 搬到 GPU，零傳輸開銷（資料集小，每卡複製一份）
    N = len(dataset)
    num_batches = N // config.batch_size

    # ── 多卡：全域 batch 維持 config.batch_size 不變，平均切到各卡 ──
    # 每步取出同一個全域 batch，再切出本 rank 的 local_bs 張，DDP/手動 all-reduce
    # 平均梯度後 ≡ 單卡 batch=config.batch_size，訓練動態與單卡一致。
    if distributed:
        assert config.batch_size % world_size == 0, (
            f"batch_size({config.batch_size}) 必須能被 world_size({world_size}) 整除"
            f"（全域 batch 不變、平均切到各卡）"
        )
        local_bs = config.batch_size // world_size
    else:
        local_bs = config.batch_size

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

    # ═══ 多卡：純手動資料平行（刻意不使用 DDP wrapper）═══
    # 本訓練迴圈有兩個對 torch DDP 不友善的特性：
    #   1) G 每步 forward 兩次（fake_img1/fake_img2 供 diversity loss）→ 不符合 DDP
    #      「一次 forward 對一次 backward」的假設，會 reducer 報錯或誤算。
    #   2) D-loss 的 R1 用 create_graph=True（double backward）→ 與 DDP reducer 不相容。
    # 因此 G 與 D 皆「不包 DDP」，改在每次 backward 後用 average_gradients() 手動
    # all-reduce 梯度（等效於 DDP 的梯度平均），版本相容性最佳、行為最可預測。
    # 初始權重用 broadcast_module 對齊（取代 DDP 建構時的自動廣播）。
    if distributed:
        broadcast_module(G, src=0)
        broadcast_module(D, src=0)
        if is_main:
            print("[DDP] 多卡：G/D 皆用手動梯度 all-reduce"
                  "（避開 R1 double-backward 與 G 多次 forward 對 DDP 的限制）")

    # torch.compile：融合 kernel，減少 launch overhead
    # D 的 inplace LeakyReLU 與 compile 不相容，只 compile G。
    # 多卡時停用 compile：多卡路徑改用 fp32 + 手動 all-reduce，為穩健性與可預測性
    # 不疊加 compile；多卡本身已有並行加速。單卡維持原本 compile 行為不變。
    if hasattr(torch, 'compile') and not distributed:
        print("啟用 torch.compile（G only）")
        G = torch.compile(G)
    elif distributed and is_main:
        print("[DDP] 多卡模式停用 torch.compile（改用 fp32 + 手動 all-reduce，穩健性優先）")

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
        if is_main:
            print(f"從 checkpoint 接續訓練: {config.resume_checkpoint}")
        ckpt = torch.load(config.resume_checkpoint, map_location=device)
        # 同時剝除 torch.compile (_orig_mod.) 與 DDP (module.) 前綴
        def _strip(sd):
            return {k.replace('_orig_mod.', '').replace('module.', ''): v
                    for k, v in sd.items()}
        # unwrap_model 取回 compile/DDP 包裝下的原始模型再載入（各 rank 載入相同權重）
        unwrap_model(G).load_state_dict(_strip(ckpt['G_state_dict']))
        unwrap_model(D).load_state_dict(_strip(ckpt['D_state_dict']))
        if 'opt_G_state_dict' in ckpt:
            opt_G.load_state_dict(ckpt['opt_G_state_dict'])
            opt_D.load_state_dict(ckpt['opt_D_state_dict'])
        if 'ada_p' in ckpt:
            ada.p = ckpt['ada_p']
        start_epoch = ckpt.get('epoch', 0)
        global_step = start_epoch * num_batches
        if is_main:
            print(f"  從 epoch {start_epoch} 接續，global_step={global_step}")
            # 載入既有日誌（只在主 rank）
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

        # 每 epoch 重新打亂 index
        # 多卡時用「跨 rank 一致」的洗牌（make_epoch_perm），這樣每個全域 batch
        # 才能被一致地切分到各 rank；單卡維持原本的 torch.randperm 行為。
        if distributed:
            perm = make_epoch_perm(N, epoch, device)
        else:
            perm = torch.randperm(N, device=device)

        for batch_idx in range(num_batches):
            # 取全域 batch 的 index，再切出本 rank 負責的那一段（全域 batch 不變）
            g_idx = perm[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size]
            idx = g_idx[ddp.rank * local_bs:(ddp.rank + 1) * local_bs]
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
                # 多卡：手動把 D 梯度跨 rank 平均（D 未包 DDP，閃開 R1 double-backward）
                if distributed:
                    average_gradients(D, world_size)
                if hasattr(config, 'grad_clip') and config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(D.parameters(), config.grad_clip)
                opt_D.step()

            # 更新 ADA（多卡：跨 rank 平均 r_t，讓所有 rank 的增強機率 p 一致）
            with torch.no_grad():
                d_real_scores, _, _ = D(real_img, label, mask)
                rt = torch.sign(d_real_scores).mean().item()
                if distributed:
                    rt = all_reduce_mean(rt, world_size, device)
                ada.update_rt(rt)

            # ─── 訓練 G ───
            z1 = torch.randn(B, config.z_dim, device=device)
            z2 = torch.randn(B, config.z_dim, device=device)

            # 多卡走 fp32（不用 AMP scaler，閃開跨 rank GradScaler scale/inf 不同步）；
            # 單卡維持原本 autocast + GradScaler 行為不變。
            g_amp = nullcontext() if distributed else autocast('cuda')
            with g_amp:
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
            if distributed:
                # fp32 路徑：一般 backward → 手動跨 rank 平均 G 梯度 → step
                g_loss.backward()
                average_gradients(G, world_size)
                if hasattr(config, 'grad_clip') and config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(G.parameters(), config.grad_clip)
                opt_G.step()
            else:
                scaler_G.scale(g_loss).backward()
                if hasattr(config, 'grad_clip') and config.grad_clip > 0:
                    scaler_G.unscale_(opt_G)
                    torch.nn.utils.clip_grad_norm_(G.parameters(), config.grad_clip)
                scaler_G.step(opt_G)
                scaler_G.update()

            global_step += 1

            # ─── 日誌（只在主 rank 記錄/列印，其餘 rank 靜默）───
            if is_main and global_step % config.log_interval == 0:
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

        # ─── Epoch 結束：儲存 checkpoint（只在主 rank，避免多卡競寫）───
        epoch_time = time.time() - epoch_start

        # 存檔前先 unwrap，去掉 DDP(module.)/compile(_orig_mod.) 前綴，
        # 確保 checkpoint 與單卡、與下游 generate 腳本完全相容。
        if is_main:
            if (epoch + 1) % config.save_interval == 0 or epoch == config.epochs - 1:
                ckpt_path = Path(config.output_dir) / f"checkpoint_epoch{epoch + 1:03d}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'G_state_dict': unwrap_model(G).state_dict(),
                    'D_state_dict': unwrap_model(D).state_dict(),
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
                        'G_state_dict': unwrap_model(G).state_dict(),
                        'D_state_dict': unwrap_model(D).state_dict(),
                        'bd_stats': bd_stats,
                        'config': vars(config),
                    }, best_path)

            print(f"  Epoch {epoch + 1}/{config.epochs} 完成 "
                  f"({epoch_time:.1f}s, ADA_p={ada.p:.2f})")

    # ═══ 儲存訓練日誌（只在主 rank）═══
    if is_main:
        log_path = Path(config.output_dir) / 'training_log.json'
        with open(log_path, 'w') as f:
            json.dump(log_history, f, indent=2)
        print(f"\n訓練完成！日誌已儲存至 {log_path}")
        print(f"最佳模型：{Path(config.output_dir) / 'best_model.pth'}")

    # ═══ 分散式清理（單卡時為 no-op）═══
    cleanup_ddp(ddp)

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

    config = MLAGANConfig()
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

    train(config)
