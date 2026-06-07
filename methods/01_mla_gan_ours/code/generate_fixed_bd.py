"""Generate syn samples at a FIXED target_bd value (sweep over bd levels).

用既有 A3 G, 不訓練, 生成 3 批 syn:
  - bd = 0.05  (低 = 假設邊界附近, 困難)
  - bd = 0.50  (中等)
  - bd = 0.95  (高 = 典型樣本, 容易)

之後分別 eval 看哪批對 downstream 有幫助.
若三批差不多 → BD value 對下游無關緊要 → BD 概念可能是 placebo
若顯著不同 → BD value 真有意義
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR))

from config_ir5_resnet50 import MLAGANConfigIR5ResNet50    # noqa: E402
from config_ir10_resnet50 import MLAGANConfigIR10ResNet50  # noqa: E402
from config_ir20_resnet50 import MLAGANConfigIR20ResNet50  # noqa: E402
from models import MLAGenerator  # noqa: E402
from dataset import MLADataset  # noqa: E402


def _cfg_for_ir(ir):
    return {5: MLAGANConfigIR5ResNet50, 10: MLAGANConfigIR10ResNet50,
            20: MLAGANConfigIR20ResNet50}[ir]()


def save_png(img: torch.Tensor, path: str):
    arr = ((img + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    arr = (arr[0] * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(path)


@torch.no_grad()
def generate_at_bd(ckpt_path, out_dir, fixed_bd, cfg, target=3540, batch=16, device='cuda'):
    G = MLAGenerator(
        z_dim=cfg.z_dim, c_dim=cfg.c_dim, bd_dim=cfg.bd_dim,
        w_dim=cfg.w_dim, enc_out_channels=cfg.enc_out_channels,
        context_dropout=0.0,
        blur_sigma=cfg.blur_sigma, blur_kernel_size=cfg.blur_kernel_size,
        mask_expand_px=cfg.mask_expand_px,
        use_bd_modulator=getattr(cfg, 'use_bd_modulator', True),
        use_gated_conv=getattr(cfg, 'use_gated_conv', True),
        use_skip=getattr(cfg, 'use_skip', True),
    ).to(device)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = {k.replace('_orig_mod.', ''): v for k, v in ck['G_state_dict'].items()}
    G.load_state_dict(state)
    G.eval()

    dataset = MLADataset(cfg, augment=False)

    for cls in cfg.minority_classes:
        save_cls_dir = Path(out_dir) / f'class{cls}'
        save_cls_dir.mkdir(parents=True, exist_ok=True)
        idxs = [i for i, s in enumerate(dataset.samples) if s[2] == cls]
        n_idxs = len(idxs)
        if n_idxs == 0:
            continue
        rng = np.random.default_rng(7)
        saved = 0
        t0 = time.time()
        while saved < target:
            chosen = rng.choice(idxs, size=min(batch, target - saved), replace=True)
            imgs = []
            masks = []
            for i in chosen:
                img, mask, _, _ = dataset[i]
                imgs.append(img.unsqueeze(0))
                masks.append(mask.unsqueeze(0))
            imgs = torch.cat(imgs).to(device)
            masks = torch.cat(masks).to(device)
            B = imgs.shape[0]
            cls_oh = F.one_hot(torch.tensor([cls] * B, device=device),
                               num_classes=cfg.num_classes).float()
            z = torch.randn(B, cfg.z_dim, device=device)
            tbd = torch.full((B, 1), float(fixed_bd), device=device)
            syn, _, _ = G(imgs, masks, z, cls_oh, tbd)
            for j in range(B):
                save_png(syn[j], str(save_cls_dir / f'syn_{saved+j+1:05d}.png'))
            saved += B
        print(f'  class{cls}: {saved} saved ({time.time()-t0:.1f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--bd', type=float, required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--target', type=int, default=3540)
    ap.add_argument('--ir', type=int, default=10, choices=[5, 10, 20],
                    help='失衡比變體：選 ir{5,10,20} config（決定 mask/真實影像來源）')
    args = ap.parse_args()

    cfg = _cfg_for_ir(args.ir)
    print(f'Generating at fixed bd={args.bd} → {args.out_dir} (ir{args.ir}, ct_dir={cfg.ct_dir})')
    generate_at_bd(args.ckpt, args.out_dir, args.bd, cfg, target=args.target)


if __name__ == '__main__':
    main()
