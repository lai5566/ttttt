"""Generate syn with controlled BD distribution from a trained G.

5 種 BD 分布比例:
  pure_typical:  100% bd ~ 0.95
  typical_heavy: 80% bd ~ 0.9 + 20% bd ~ 0.2
  mid:           uniform bd ∈ [0, 1]
  boundary_heavy: 20% bd ~ 0.9 + 80% bd ~ 0.2
  pure_boundary: 100% bd ~ 0.05
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR))

from config_ir5_resnet50 import MLAGANConfigIR5ResNet50    # noqa
from config_ir10_resnet50 import MLAGANConfigIR10ResNet50  # noqa
from config_ir20_resnet50 import MLAGANConfigIR20ResNet50  # noqa
from models import MLAGenerator  # noqa
from dataset import MLADataset  # noqa


def _cfg_for_ir(ir):
    return {5: MLAGANConfigIR5ResNet50, 10: MLAGANConfigIR10ResNet50,
            20: MLAGANConfigIR20ResNet50}[ir]()


MIX_PROFILES = {
    'pure_typical':   lambda B, rng: torch.full((B, 1), 0.95) + torch.randn(B, 1) * 0.02,
    'typical_heavy':  lambda B, rng: torch.tensor(np.where(rng.random(B) < 0.8,
                                                            0.9 + rng.standard_normal(B)*0.05,
                                                            0.2 + rng.standard_normal(B)*0.05),
                                                   dtype=torch.float32).unsqueeze(-1),
    'mid':            lambda B, rng: torch.tensor(rng.random(B), dtype=torch.float32).unsqueeze(-1),
    'boundary_heavy': lambda B, rng: torch.tensor(np.where(rng.random(B) < 0.2,
                                                            0.9 + rng.standard_normal(B)*0.05,
                                                            0.2 + rng.standard_normal(B)*0.05),
                                                   dtype=torch.float32).unsqueeze(-1),
    'pure_boundary':  lambda B, rng: torch.full((B, 1), 0.05) + torch.randn(B, 1) * 0.02,
}


def save_png(img, path):
    arr = ((img + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    arr = (arr[0] * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(path)


@torch.no_grad()
def generate(ckpt, out_dir, profile, cfg, target=3540, batch=16, seed=7, device='cuda', z_temp=None):
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
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    state = {k.replace('_orig_mod.', ''): v for k, v in ck['G_state_dict'].items()}
    G.load_state_dict(state)
    G.eval()

    ds = MLADataset(cfg, augment=False)
    rng = np.random.default_rng(seed)
    profile_fn = MIX_PROFILES[profile]

    for cls in cfg.minority_classes:
        T = (z_temp or {}).get(cls, 1.0)
        save_dir = Path(out_dir) / f'class{cls}'
        save_dir.mkdir(parents=True, exist_ok=True)
        idxs = [i for i, s in enumerate(ds.samples) if s[2] == cls]
        saved = 0
        t0 = time.time()
        while saved < target:
            chosen = rng.choice(idxs, size=min(batch, target - saved), replace=True)
            imgs, masks = [], []
            for i in chosen:
                img, mask, _, _ = ds[i]
                imgs.append(img.unsqueeze(0))
                masks.append(mask.unsqueeze(0))
            imgs = torch.cat(imgs).to(device)
            masks = torch.cat(masks).to(device)
            B = imgs.shape[0]
            cls_oh = F.one_hot(torch.tensor([cls]*B, device=device), num_classes=cfg.num_classes).float()
            z = torch.randn(B, cfg.z_dim, device=device) * T          # z 溫度旋鈕(per-class)
            tbd = profile_fn(B, rng).to(device).clamp(0, 1)
            syn, _, _ = G(imgs, masks, z, cls_oh, tbd)
            for j in range(B):
                save_png(syn[j], str(save_dir / f'syn_{saved+j+1:05d}.png'))
            saved += B
        print(f'  class{cls}: {saved} saved (T={T}, {time.time()-t0:.1f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--profile', required=True, choices=list(MIX_PROFILES.keys()))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--target', type=int, default=3540)
    ap.add_argument('--ir', type=int, default=10, choices=[5, 10, 20],
                    help='失衡比變體：選 ir{5,10,20} config（決定 mask/真實影像來源）')
    ap.add_argument('--z-temp', type=float, nargs='*', default=None,
                    help='per-class z 溫度，順序對應 cfg.minority_classes；單值=全類共用；省略=全 1.0')
    a = ap.parse_args()
    cfg = _cfg_for_ir(a.ir)
    z_temp = None
    if a.z_temp:
        mc = cfg.minority_classes
        vals = a.z_temp * len(mc) if len(a.z_temp) == 1 else a.z_temp
        assert len(vals) == len(mc), f'--z-temp 需 1 或 {len(mc)} 個值(minority_classes={mc})'
        z_temp = dict(zip(mc, vals))
    print(f'Profile: {a.profile} → {a.out_dir} (ir{a.ir}, ct_dir={cfg.ct_dir}, z_temp={z_temp})')
    generate(a.ckpt, a.out_dir, a.profile, cfg, target=a.target, z_temp=z_temp)


if __name__ == '__main__':
    main()
