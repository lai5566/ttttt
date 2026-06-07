"""
MLA-GAN 資料集：載入成對的 CT 影像 + Lesion Mask，
並預計算 boundary distance。
"""

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from config import MLAGANConfig


class MLADataset(Dataset):
    """
    少數類 CT + Mask 成對資料集。

    只載入少數類（class1: Ischemic, class2: Hemorrhagic），
    每類 354 張真實影像（排除 _aug_ 增強影像）。

    回傳：
        img:           [1, 256, 256] — 灰階 CT，正規化到 [-1, 1]
        mask:          [1, 256, 256] — 二值 lesion mask {0, 1}
        label:         int           — 類別標籤（1 或 2）
        boundary_dist: [1]           — 預計算的 boundary distance
    """

    def __init__(self, config: MLAGANConfig, precomputed_bd: dict = None,
                 augment: bool = True):
        """
        Args:
            config: 設定
            precomputed_bd: {filepath: bd_value} 預計算的 boundary distance
            augment: 是否做資料增強（翻轉）
        """
        self.config = config
        self.augment = augment
        self.samples = []  # [(ct_path, mask_path, label)]

        ct_root = config.ct_train_dir
        mask_root = config.mask_train_dir

        for cls in config.minority_classes:
            ct_dir = ct_root / f"class{cls}"
            mask_dir = mask_root / f"class{cls}"

            if not ct_dir.exists():
                raise FileNotFoundError(f"CT 目錄不存在: {ct_dir}")
            if not mask_dir.exists():
                raise FileNotFoundError(f"Mask 目錄不存在: {mask_dir}")

            # 只載入真實影像（排除 _aug_ 增強）
            ct_files = sorted([
                f for f in os.listdir(ct_dir)
                if f.endswith('.png') and '_aug_' not in f
            ])

            for fname in ct_files:
                ct_path = ct_dir / fname
                mask_path = mask_dir / fname
                if mask_path.exists():
                    self.samples.append((str(ct_path), str(mask_path), cls))

        print(f"[MLADataset] 載入 {len(self.samples)} 筆樣本 "
              f"(classes: {config.minority_classes})")

        # 預計算的 boundary distance
        self.bd_map = precomputed_bd or {}

        # 預載入所有影像，堆成連續 tensor
        imgs, masks, labels, bds = [], [], [], []
        print("[MLADataset] 預載入所有影像...")
        for ct_path, mask_path, label in self.samples:
            img = Image.open(ct_path).convert('L')
            img = img.resize((config.img_size, config.img_size), Image.LANCZOS)
            img = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
            imgs.append(torch.from_numpy(img).unsqueeze(0))

            msk = Image.open(mask_path).convert('L')
            msk = msk.resize((config.img_size, config.img_size), Image.NEAREST)
            msk = (np.array(msk, dtype=np.float32) > 127).astype(np.float32)
            masks.append(torch.from_numpy(msk).unsqueeze(0))

            labels.append(label)
            bds.append(self.bd_map.get(ct_path, 0.5))

        # 堆成 [N, 1, H, W] 連續 tensor
        self.all_imgs = torch.stack(imgs)           # [N, 1, H, W]
        self.all_masks = torch.stack(masks)         # [N, 1, H, W]
        self.all_labels = torch.tensor(labels, dtype=torch.long)  # [N]
        self.all_bds = torch.tensor(bds, dtype=torch.float32).unsqueeze(1)  # [N, 1]
        print(f"[MLADataset] 預載入完成，共 {len(self.samples)} 筆")

    def to_device(self, device: str) -> 'MLADataset':
        """將整個 dataset 搬到 GPU，消除每 batch 的 CPU→GPU 傳輸。"""
        self.all_imgs = self.all_imgs.to(device)
        self.all_masks = self.all_masks.to(device)
        self.all_labels = self.all_labels.to(device)
        self.all_bds = self.all_bds.to(device)
        print(f"[MLADataset] 已搬至 {device}，"
              f"VRAM 佔用 ≈ {self.all_imgs.nbytes / 1024**2:.0f} MB")
        return self

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return (self.all_imgs[idx], self.all_masks[idx],
                self.all_labels[idx], self.all_bds[idx])

    def get_batch(self, indices: torch.Tensor):
        """直接用 index tensor 取 batch，零開銷（純 GPU indexing）。"""
        imgs = self.all_imgs[indices]
        masks = self.all_masks[indices]
        # 隨機翻轉（整個 batch 同時操作，比逐張快）
        if self.augment:
            if torch.rand(1, device=imgs.device).item() > 0.5:
                imgs = imgs.flip(3)
                masks = masks.flip(3)
            if torch.rand(1, device=imgs.device).item() > 0.5:
                imgs = imgs.flip(2)
                masks = masks.flip(2)
        return imgs, masks, self.all_labels[indices], self.all_bds[indices]


def precompute_boundary_distances(dataset: MLADataset,
                                  classifier: torch.nn.Module,
                                  device: str = 'cuda',
                                  method: str = 'rank') -> dict:
    """
    預計算所有樣本的 boundary distance.

    [v3-fix B2]: 同時計算 per-class bd_stats，讓 sample_boundary_target
                 可以做 class-conditional 取樣，避免 class 與 target_bd
                 相互矛盾的問題。

    Args:
        method:
          'sigmoid' — legacy: sigmoid(logit_margin); saturates near 1.0
                      for confident classifiers (Issue 3 problem).
          'rank'   — Issue 3 fix: compute raw logit margins for all samples,
                      then percentile-rank to uniform [0, 1]. This guarantees
                      well-spread BD targets (mean ≈ 0.5, p25 ≈ 0.25).
    Returns:
        bd_map: {ct_path: bd_value}
        bd_stats: {
            'mean','std','min','p25','method',  # global (legacy)
            'per_class': {cls: {'mean','std','min','p25','p50','p75'}},  # NEW
        }
    """
    from losses import BoundaryDistanceComputer
    import numpy as np

    raw_method = 'sigmoid' if method == 'sigmoid' else 'raw'
    bd_computer = BoundaryDistanceComputer(classifier, method=raw_method)
    paths = []
    raw_values = []
    labels = []

    classifier.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            ct_path = dataset.samples[i][0]
            img, mask, label, _ = dataset[i]
            img_gpu = img.unsqueeze(0).to(device)
            raw = bd_computer.compute(img_gpu).item()
            paths.append(ct_path)
            raw_values.append(raw)
            labels.append(int(label.item() if torch.is_tensor(label) else label))

    raw_arr = np.array(raw_values, dtype=np.float64)
    labels_arr = np.array(labels)
    if method == 'rank':
        # Percentile rank to uniform [0, 1]: rank/N for tied-fair ordering.
        from scipy.stats import rankdata
        bd_values = rankdata(raw_arr, method='average') / len(raw_arr)
    elif method in ('kfold_json', 'kfold_json_perclass'):
        # Load K-fold out-of-fold BD (unsaturated)
        # method='kfold_json':         global rank-normalize to [0, 1]
        # method='kfold_json_perclass': per-class rank-normalize to [0, 1]
        import json
        from scipy.stats import rankdata
        # 從 config 取 IR 對應的 kfold json（空字串時 fallback ir10 預設）。
        kfold_path = getattr(dataset.config, 'kfold_bd_json', '') \
            or '/workspace/666_in0526/999_mlagan_Export_2026/data/classifiers/kfold_bd.json'
        print(f'[K-fold BD] loading out-of-fold BD from {kfold_path}')
        d = json.load(open(kfold_path))
        # 用 classX/檔名 後綴當 key，對「絕對/相對路徑」差異不敏感（避免全 miss → BD 退化成常數）。
        import os as _os
        def _suffix(p):
            return _os.path.join(_os.path.basename(_os.path.dirname(p)), _os.path.basename(p))
        path_to_kfold = {_suffix(p): v for p, v in zip(d['paths'], d['oof_bd'])}
        raw_arr = np.array([path_to_kfold.get(_suffix(p), np.nan) for p in paths], dtype=np.float64)
        n_miss = int(np.isnan(raw_arr).sum())
        if n_miss:
            print(f'[K-fold BD][WARN] {n_miss}/{len(paths)} 樣本在 json 找不到對應 BD（path key 不符）')
            raw_arr = np.nan_to_num(raw_arr, nan=0.0)
        assert n_miss < len(paths), \
            'K-fold BD 全部 miss：json 與 dataset 的 path key 完全不符，請檢查 kfold_bd_json'
        if method == 'kfold_json_perclass':
            # Per-class rank-normalize: 每個 class 自己 [0, 1]
            bd_values = np.zeros_like(raw_arr)
            for cls in np.unique(labels_arr):
                m = labels_arr == cls
                bd_values[m] = rankdata(raw_arr[m], method='average') / m.sum()
            print(f'[K-fold BD per-class] each class rank-normalized to [0,1] independently')
        else:
            bd_values = rankdata(raw_arr, method='average') / len(raw_arr)
            print(f'[K-fold BD global] global rank-normalize to [0,1]')
        print(f'  mean={bd_values.mean():.4f}, p25={np.percentile(bd_values, 25):.4f}, p75={np.percentile(bd_values, 75):.4f}')
    else:  # 'sigmoid' legacy: raw is already in [0, 1]
        bd_values = raw_arr

    bd_map = dict(zip(paths, bd_values.tolist()))

    all_bd = torch.tensor(bd_values)
    bd_stats = {
        'mean': all_bd.mean().item(),
        'std': all_bd.std().item(),
        'min': all_bd.min().item(),
        'p25': all_bd.quantile(0.25).item(),
        'method': method,
    }

    # [v3-fix B2] per-class statistics
    per_class = {}
    for cls in sorted(set(labels)):
        mask_cls = labels_arr == cls
        bd_cls = torch.tensor(bd_values[mask_cls])
        per_class[int(cls)] = {
            'mean': bd_cls.mean().item(),
            'std': bd_cls.std().item() if len(bd_cls) > 1 else 0.0,
            'min': bd_cls.min().item(),
            'p25': bd_cls.quantile(0.25).item(),
            'p50': bd_cls.quantile(0.50).item(),
            'p75': bd_cls.quantile(0.75).item(),
            'count': int(mask_cls.sum()),
        }
    bd_stats['per_class'] = per_class

    print(f"[Boundary Distance] method={method}  "
          f"mean={bd_stats['mean']:.3f}, std={bd_stats['std']:.3f}, "
          f"min={bd_stats['min']:.3f}, p25={bd_stats['p25']:.3f}")
    for cls, ps in per_class.items():
        print(f"  └─ class {cls} (n={ps['count']}): "
              f"mean={ps['mean']:.3f}, p25={ps['p25']:.3f}, "
              f"p50={ps['p50']:.3f}, p75={ps['p75']:.3f}")

    return bd_map, bd_stats
