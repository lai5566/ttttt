"""K-fold out-of-fold BD（參數化:ir5 / ir10 / ir20）

每個 train 樣本的 BD 來自「沒看過它的 aux classifier」，避免 aux overfit train
導致 BD 飽和。輔助分類器在每折都「重訓」一顆全新的 ResNet50(1-ch)。

樣本少時用較小 K(ir20 minority=177 → K=3)以維持每折 hold-out 不過小、aux 仍訓得動。

Steps:
  1. StratifiedKFold 切 train（保持 class 平衡）
  2. fold k：在 {全 train} - {fold k} 重訓 fresh aux classifier
  3. 對 fold k 預測 → out-of-fold logits / BD(top1-top2 margin)/ loss / pred
  4. 存 kfold_bd_<tag>.json，印分布 / per-class / Spearman(BD,loss) / 混淆矩陣

用法:
  python kfold_bd_compute.py --train-root ../../../data/ct_256_ir5/train  --k 5 --tag ir5
  python kfold_bd_compute.py --train-root ../../../data/ct_256_ir20/train --k 3 --tag ir20
輸出: <repo>/data/classifiers/kfold_bd_<tag>.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms as T
from sklearn.model_selection import StratifiedKFold
from PIL import Image

DEVICE = 'cuda'
_REPO = Path(__file__).resolve().parents[3]  # 999_mlagan_Export_2026


class CT1ch(torch.utils.data.Dataset):
    def __init__(self, root, classes=(0, 1, 2)):
        self.samples = []
        root = Path(root)
        for cls in classes:
            d = root / f'class{cls}'
            if not d.exists():
                continue
            for p in sorted(d.glob('*.png')):
                self.samples.append((str(p.resolve()), cls))  # 絕對路徑，與 MLADataset 一致
        self.tx = T.Compose([T.Grayscale(), T.Resize((256, 256)),
                             T.ToTensor(), T.Normalize([0.5], [0.5])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, l = self.samples[i]
        return self.tx(Image.open(p)), l


def build_resnet50_1ch(num_classes=3):
    """與本 repo aux classifier(resnet50_v3)同 arch:1-ch 輸入、ResNet50。"""
    m = models.resnet50(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(2048, num_classes)
    return m


def train_aux_on_fold(train_subset, epochs=20, lr=1e-3, seed=7):
    """在 train 子集上重訓一顆 fresh aux classifier。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    labels = np.array([s[1] for s in train_subset.dataset.samples])[train_subset.indices]
    counts = np.bincount(labels)
    w = 1.0 / counts[labels]
    loader = DataLoader(train_subset, batch_size=32,
                        sampler=WeightedRandomSampler(w, len(w)), num_workers=4)
    model = build_resnet50_1ch().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-root', required=True)
    ap.add_argument('--k', type=int, required=True, help='folds(樣本少用小 K)')
    ap.add_argument('--tag', required=True, help='ir5 / ir10 / ir20，決定輸出檔名')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out-dir', default=str(_REPO / 'data' / 'classifiers'))
    args = ap.parse_args()

    full_ds = CT1ch(args.train_root)
    N = len(full_ds)
    labels = np.array([s[1] for s in full_ds.samples])
    print(f'[{args.tag}] Train: {N} samples, dist={np.bincount(labels).tolist()}, K={args.k}')

    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=7)
    folds = list(skf.split(np.zeros(N), labels))

    oof_bds = np.zeros(N)
    oof_losses = np.zeros(N)
    oof_preds = np.zeros(N, dtype=np.int64)

    print(f'\n=== Step 1: K-fold ({args.k}) out-of-fold BD ===')
    for k, (train_idx, val_idx) in enumerate(folds):
        tr_lab = labels[train_idx]
        print(f'\n  Fold {k+1}/{args.k}: train {len(train_idx)} '
              f'(per-class {np.bincount(tr_lab).tolist()}), predict {len(val_idx)}')
        model = train_aux_on_fold(Subset(full_ds, train_idx), epochs=args.epochs, lr=args.lr)

        val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=64, num_workers=4)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                all_logits.append(model(x.to(DEVICE)).cpu())
                all_labels.extend(y.tolist())
        all_logits = torch.cat(all_logits)
        all_labels = torch.tensor(all_labels)

        sorted_l, _ = all_logits.sort(dim=1, descending=True)
        margins = (sorted_l[:, 0] - sorted_l[:, 1]).numpy()
        losses = F.cross_entropy(all_logits, all_labels, reduction='none').numpy()
        preds = all_logits.argmax(dim=1).numpy()

        oof_bds[val_idx] = margins
        oof_losses[val_idx] = losses
        oof_preds[val_idx] = preds
        print(f'  Fold {k+1} held-out acc={(preds == all_labels.numpy()).mean():.4f}, '
              f'mean margin={margins.mean():.4f}')
        del model
        torch.cuda.empty_cache()

    out_path = Path(args.out_dir) / f'kfold_bd_{args.tag}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        'tag': args.tag, 'k': args.k, 'train_root': args.train_root,
        'paths': [s[0] for s in full_ds.samples], 'labels': labels.tolist(),
        'oof_bd': oof_bds.tolist(), 'oof_loss': oof_losses.tolist(),
        'oof_pred': oof_preds.tolist(),
    }, open(out_path, 'w'))
    print(f'\n[saved] {out_path}')

    print(f'\n=== Step 2: K-fold BD distribution ===')
    print(f'Overall: mean={oof_bds.mean():.4f}, std={oof_bds.std():.4f}, '
          f'min={oof_bds.min():.4f}, max={oof_bds.max():.4f}')
    print(f'p25={np.percentile(oof_bds,25):.4f}, p50={np.percentile(oof_bds,50):.4f}, '
          f'p75={np.percentile(oof_bds,75):.4f}')
    for cls in [0, 1, 2]:
        m = labels == cls
        if m.sum() == 0:
            continue
        print(f'  class{cls} (n={m.sum()}): acc={(oof_preds[m]==cls).mean():.4f}, '
              f'BD mean={oof_bds[m].mean():.4f}, std={oof_bds[m].std():.4f}, '
              f'p25={np.percentile(oof_bds[m],25):.4f}, loss mean={oof_losses[m].mean():.4f}')

    try:
        from scipy.stats import spearmanr
        print(f'\n  Spearman corr(K-fold BD, loss) = {spearmanr(oof_bds, oof_losses)[0]:.4f}')
    except Exception as e:
        print(f'  [spearman skipped] {e}')

    print(f'\n  Out-of-fold overall acc = {(oof_preds == labels).mean():.4f}')
    for t in [0, 1, 2]:
        print(f'    class{t}: {[int(((labels==t)&(oof_preds==p)).sum()) for p in [0,1,2]]}')


if __name__ == '__main__':
    main()
