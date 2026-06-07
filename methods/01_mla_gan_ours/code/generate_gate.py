"""消融:MLA-GAN 生成 + 三級 QualityGate 過濾(對照純生成)。

複用 generate_bd_mix 的生成(同 G、同 pure_boundary profile),唯一新增變量 =
QualityGate 過濾(L1 D分數 / L2 BD / L3 NN novelty)。用以實證「gate 是否真的
提升下游」——若 gate 圖下游沒更好(甚至更差),即證 gate 機制無益。

用法:
    python generate_gate.py --ir 20 --ckpt ../methods/01_mla_gan_ours/output_ir20/run_seed7/best_model.pth \
        --out-dir ../methods/01_mla_gan_ours/generated_ir20_gate --target 3540
    --skip-d-gate 跳過 L1(D 太強時)
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_CODE = Path('/workspace/666_in0526/999_mlagan_Export_2026/methods/01_mla_gan_ours/code')
sys.path.insert(0, str(_CODE))

from config_ir5_resnet50 import MLAGANConfigIR5ResNet50    # noqa
from config_ir10_resnet50 import MLAGANConfigIR10ResNet50  # noqa
from config_ir20_resnet50 import MLAGANConfigIR20ResNet50  # noqa
from models import MLAGenerator, MLADiscriminator  # noqa
from dataset import MLADataset  # noqa
from losses import BoundaryDistanceComputer  # noqa
from utils import load_pretrained_classifier, get_feature_extractor  # noqa


def _cfg_for_ir(ir):
    return {5: MLAGANConfigIR5ResNet50, 10: MLAGANConfigIR10ResNet50,
            20: MLAGANConfigIR20ResNet50}[ir]()


def save_png(img, path):
    arr = ((img + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    Image.fromarray((arr[0] * 255.0).astype(np.uint8), mode='L').save(path)


class QualityGate:
    """三級過濾(同 ablation/generate_old_v1.py),batch 版。"""
    def __init__(self, D, bd_computer, feat_ext, device, skip_d=False):
        self.D, self.bd_computer, self.feat_ext = D, bd_computer, feat_ext
        self.device, self.skip_d = device, skip_d
        self.tau1 = self.tau2 = self.tau3 = None
        self.real_features = None

    @torch.no_grad()
    def compute_thresholds(self, ds, batch=64):
        d_scores, bd_values, feats = [], [], []
        n = len(ds.samples)
        for s in range(0, n, batch):
            idx = list(range(s, min(s + batch, n)))
            imgs, masks, labels, _ = ds.get_batch(idx) if hasattr(ds, 'get_batch') else _manual_batch(ds, idx)
            imgs, masks, labels = imgs.to(self.device), masks.to(self.device), labels.to(self.device)
            rf, _, _ = self.D(imgs, labels, masks)
            d_scores.append(rf.squeeze(-1).cpu())
            bd_values.append(self.bd_computer.compute(imgs).cpu())
            feats.append(self.feat_ext(imgs).cpu())
        d_scores = torch.cat(d_scores).flatten()
        bd_values = torch.cat(bd_values).flatten()
        feats = torch.cat(feats)
        self.tau1 = torch.quantile(d_scores, 0.25).item()
        self.tau2 = torch.quantile(bd_values, 0.75).item()
        dists = torch.cdist(feats, feats); dists.fill_diagonal_(float('inf'))
        self.tau3 = torch.quantile(dists.min(1).values, 0.25).item()
        self.real_features = feats.to(self.device)
        print(f'[QualityGate] τ₁(D)={self.tau1:.3f}, τ₂(BD)={self.tau2:.3f}, τ₃(NN)={self.tau3:.3f}')

    @torch.no_grad()
    def pass_mask(self, syn, label, mask):
        """回傳 [B] bool:通過全部三級。"""
        B = syn.shape[0]
        ok = torch.ones(B, dtype=torch.bool, device=self.device)
        if not self.skip_d and self.tau1 is not None:
            rf, _, _ = self.D(syn, label, mask)
            ok &= (rf.squeeze(-1) > self.tau1)
        bd = self.bd_computer.compute(syn).flatten()
        ok &= (bd < self.tau2)
        feat = self.feat_ext(syn)
        nn = torch.cdist(feat, self.real_features).min(1).values
        ok &= (nn > self.tau3)
        return ok


def _manual_batch(ds, idx):
    imgs = torch.stack([ds[i][0] for i in idx])
    masks = torch.stack([ds[i][1] for i in idx])
    labels = torch.tensor([ds.samples[i][2] for i in idx])
    return imgs, masks, labels, None


@torch.no_grad()
def generate(ckpt, out_dir, cfg, target, skip_d, batch=16, seed=7, device='cuda'):
    G = MLAGenerator(z_dim=cfg.z_dim, c_dim=cfg.c_dim, bd_dim=cfg.bd_dim, w_dim=cfg.w_dim,
                     enc_out_channels=cfg.enc_out_channels, context_dropout=0.0,
                     blur_sigma=cfg.blur_sigma, blur_kernel_size=cfg.blur_kernel_size,
                     mask_expand_px=cfg.mask_expand_px,
                     use_bd_modulator=getattr(cfg, 'use_bd_modulator', True),
                     use_gated_conv=getattr(cfg, 'use_gated_conv', True),
                     use_skip=getattr(cfg, 'use_skip', True)).to(device)
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    G.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ck['G_state_dict'].items()})
    G.eval()

    D = MLADiscriminator(num_classes=cfg.num_classes,
                         global_base_ch=getattr(cfg, 'global_d_base_ch', 32),
                         local_base_ch=getattr(cfg, 'local_d_base_ch', 64),
                         local_crop_size=getattr(cfg, 'local_crop_size', 64)).to(device)
    D.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ck['D_state_dict'].items()})
    D.eval()

    clf = load_pretrained_classifier(cfg.pretrained_classifier, cfg.num_classes, device, arch='resnet50')
    bd_computer = BoundaryDistanceComputer(clf)
    feat_ext = get_feature_extractor(clf)

    ds = MLADataset(cfg, augment=False)
    gate = QualityGate(D, bd_computer, feat_ext, device, skip_d=skip_d)
    gate.compute_thresholds(ds)

    rng = np.random.default_rng(seed)
    for cls in cfg.minority_classes:
        save_dir = Path(out_dir) / f'class{cls}'
        save_dir.mkdir(parents=True, exist_ok=True)
        idxs = [i for i, s in enumerate(ds.samples) if s[2] == cls]
        saved = attempt = 0
        t0 = time.time()
        while saved < target:
            chosen = rng.choice(idxs, size=batch, replace=True)
            imgs = torch.stack([ds[i][0] for i in chosen]).to(device)
            masks = torch.stack([ds[i][1] for i in chosen]).to(device)
            B = imgs.shape[0]
            cls_oh = F.one_hot(torch.tensor([cls] * B, device=device), cfg.num_classes).float()
            z = torch.randn(B, cfg.z_dim, device=device)
            tbd = (torch.full((B, 1), 0.05) + torch.randn(B, 1) * 0.02).to(device).clamp(0, 1)  # pure_boundary
            syn, _, _ = G(imgs, masks, z, cls_oh, tbd)
            lbl = torch.tensor([cls] * B, device=device)
            ok = gate.pass_mask(syn, lbl, masks)
            for j in range(B):
                if ok[j] and saved < target:
                    save_png(syn[j], str(save_dir / f'syn_{saved+1:05d}.png'))
                    saved += 1
            attempt += B
            if attempt % 2000 == 0:
                print(f'  class{cls}: {saved}/{target} (attempt {attempt}, pass {saved/attempt:.1%})')
            if attempt > 5000 and saved < 10:   # 安全閥(同原版)
                print('  ⚠ pass rate 極低 → 門檻砍半')
                gate.tau1 *= 0.5; gate.tau3 *= 0.5
        print(f'  class{cls}: {saved} saved, {attempt} attempts, pass {saved/attempt:.1%} ({time.time()-t0:.1f}s)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ir', type=int, required=True, choices=[5, 10, 20])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--target', type=int, default=3540)
    ap.add_argument('--skip-d-gate', action='store_true')
    a = ap.parse_args()
    cfg = _cfg_for_ir(a.ir)
    print(f'[GATE] ir{a.ir} → {a.out_dir} (ct_dir={cfg.ct_dir}, skip_d={a.skip_d_gate})')
    generate(a.ckpt, a.out_dir, cfg, a.target, a.skip_d_gate)


if __name__ == '__main__':
    main()
