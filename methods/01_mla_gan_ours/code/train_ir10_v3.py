"""IR=10 + ResNet50 訓練入口（原版 / 非 K-fold）。

BD 由 config.pretrained_classifier(resnet50_v3_ir10)現算,method='rank'(base config 預設),
非 K-fold。對照:train_kfoldbd.py --ir 10 才是 K-fold(讀 kfold_bd.json)。

用法：
  python train_ir10_v3.py --epochs 400 --batch-size 8 --seed 7
"""
import argparse
import torch

from config_ir10_resnet50 import MLAGANConfigIR10ResNet50
from train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLA-GAN IR=10 + ResNet50 (原版/非kfold) 訓練')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    import random, numpy as np
    assert args.seed in (7, 49, 91, 133, 175), f"seed 必須 7/49/91/133/175,得到 {args.seed}"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"[Seed] {args.seed}")

    config = MLAGANConfigIR10ResNet50()
    if args.epochs is not None: config.epochs = args.epochs
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.lr is not None: config.lr = args.lr
    if args.output_dir is not None: config.output_dir = args.output_dir
    if args.resume is not None: config.resume_checkpoint = args.resume

    print(f"[IR=10 原版/非kfold] bd_method={getattr(config, 'bd_compute_method', 'rank')}")
    print(f"[IR=10] ct_dir={config.ct_dir}, classifier={config.pretrained_classifier} ({config.classifier_arch})")
    print(f"[IR=10] output_dir={config.output_dir}")
    train(config)
