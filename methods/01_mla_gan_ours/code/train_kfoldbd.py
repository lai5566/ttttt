#!/usr/bin/env python3
"""
MLA-GAN K-fold BD 訓練入口。

與 train_ir{5,10,20}_v3.py 的差異只在 __main__：本檔用 K-fold out-of-fold BD
（bd_compute_method='kfold_json*'，讀 kfold_bd_{ir}.json）。

訓練流程本身（含「多卡 DDP 支援」）統一由 train.py 的 train() 提供——本檔直接
import，不再複製一份，避免兩處邏輯不同步。

用法：
    # 單卡
    python train_kfoldbd.py --ir 5 --epochs 400 --batch-size 8 --seed 7
    # 多卡（全域 batch 仍為 8，平均切到各卡）
    torchrun --standalone --nproc_per_node=4 \\
        train_kfoldbd.py --ir 5 --epochs 400 --batch-size 8 --seed 7
"""

import argparse

import torch

from train import train


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
