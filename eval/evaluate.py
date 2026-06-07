#!/usr/bin/env python3
"""
MLA-GAN 下游評估腳本。

比較三種策略的分類效果：
1. Real-only（不平衡基線）
2. Real + MLA-GAN 合成（標準 ResNet-18）
3. Real + MLA-GAN 合成（Lesion-Aware Classifier, mask-weighted pooling）

用法：
    python evaluate.py --synthetic-dir output/generated/
    python evaluate.py --synthetic-dir output/generated/ --epochs 50
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from config import MLAGANConfig


# ═══════════════════════════════════════════════
# 資料集
# ═══════════════════════════════════════════════

class CTEvalDataset(Dataset):
    """
    CT 影像評估資料集。
    支援載入真實影像 + 合成影像的混合資料集。
    """

    def __init__(self, root_dirs: list, img_size: int = 256):
        """
        Args:
            root_dirs: 資料夾列表，每個含 class0/, class1/, class2/ 子目錄
            img_size: 影像大小
        """
        self.img_size = img_size
        self.samples = []  # [(path, label)]
        self.masks = {}    # {path: mask_path} 選填

        for root_dir in root_dirs:
            root = Path(root_dir)
            for cls_dir in sorted(root.iterdir()):
                if not cls_dir.is_dir():
                    continue
                # 從目錄名稱推斷類別
                cls_name = cls_dir.name
                if cls_name.startswith('class'):
                    cls_idx = int(cls_name.replace('class', '').split('_')[0])
                else:
                    continue

                for fname in sorted(os.listdir(cls_dir)):
                    if fname.endswith(('.png', '.jpg')):
                        self.samples.append((str(cls_dir / fname), cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        # ResNet 需要 3 通道
        img = img.repeat(3, 1, 1)
        return img, label


class CTEvalDatasetWithMask(Dataset):
    """
    帶 mask 的 CT 影像評估資料集（供 LesionAwareClassifier 使用）。
    """

    def __init__(self, root_dirs: list, mask_dirs: list = None,
                 img_size: int = 256):
        self.img_size = img_size
        self.samples = []  # [(path, label)]
        self.mask_map = {}  # {img_path: mask_path}

        for root_dir in root_dirs:
            root = Path(root_dir)
            for cls_dir in sorted(root.iterdir()):
                if not cls_dir.is_dir():
                    continue
                cls_name = cls_dir.name
                if cls_name.startswith('class'):
                    cls_idx = int(cls_name.replace('class', '').split('_')[0])
                else:
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.endswith(('.png', '.jpg')):
                        self.samples.append((str(cls_dir / fname), cls_idx))

        # 建立 mask 對照表
        if mask_dirs:
            for mask_dir in mask_dirs:
                mroot = Path(mask_dir)
                for cls_dir in sorted(mroot.iterdir()):
                    if not cls_dir.is_dir():
                        continue
                    for fname in sorted(os.listdir(cls_dir)):
                        if fname.endswith(('.png', '.jpg')):
                            self.mask_map[fname] = str(cls_dir / fname)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        fname = Path(path).name

        # 載入影像
        img = Image.open(path).convert('L')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]

        # 載入 mask（如果有）
        mask = torch.zeros(1, self.img_size, self.img_size)
        if fname in self.mask_map:
            m = Image.open(self.mask_map[fname]).convert('L')
            m = m.resize((self.img_size, self.img_size), Image.NEAREST)
            m = np.array(m, dtype=np.float32)
            mask = torch.from_numpy((m > 127).astype(np.float32)).unsqueeze(0)

        return img, mask, label


# ═══════════════════════════════════════════════
# 分類器
# ═══════════════════════════════════════════════

class LesionAwareClassifier(nn.Module):
    """
    Lesion-Aware 分類器：mask-weighted feature pooling。

    feature = α · GAP(fm × mask) + (1-α) · GAP(fm × (1-mask))
    α = 0.7 → 病灶 2% 像素放大到 70% 的 feature 權重

    這是讓 GAN 增強「真正有效」的關鍵拼圖。
    標準 GAP 讓 98% 背景主導特徵 → 分類器看不到病灶差異。
    """

    def __init__(self, num_classes: int = 3, alpha_mask: float = 0.7):
        super().__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        # 取到 avgpool 之前的所有層
        self.features = nn.Sequential(*list(base.children())[:-2])
        feat_dim = 512  # ResNet-18 最後一層通道數

        self.alpha_mask = alpha_mask
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x:    [B, 3, 256, 256]
            mask: [B, 1, 256, 256] — lesion mask（Normal 類為全零）
        Returns:
            logits: [B, num_classes]
        """
        feat_map = self.features(x)  # [B, 512, 8, 8]

        if mask is not None and mask.sum() > 0:
            H, W = feat_map.shape[2], feat_map.shape[3]
            mask_small = F.interpolate(
                mask.float(), size=(H, W), mode='bilinear', align_corners=False,
            )

            lesion_feat = self.pool(feat_map * mask_small).flatten(1)     # [B, 512]
            bg_feat = self.pool(feat_map * (1 - mask_small)).flatten(1)   # [B, 512]

            # 有 mask 的樣本使用加權 pooling，無 mask 的使用標準 GAP
            has_mask = (mask_small.sum(dim=[2, 3]) > 0).float().unsqueeze(1)  # [B, 1, 1]
            has_mask = has_mask.squeeze(-1)  # [B, 1]

            features = (self.alpha_mask * lesion_feat * has_mask
                        + (1 - self.alpha_mask * has_mask) * bg_feat)
        else:
            features = self.pool(feat_map).flatten(1)

        return self.classifier(features)


# ═══════════════════════════════════════════════
# 訓練與評估
# ═══════════════════════════════════════════════

def train_classifier(model, train_loader, val_loader, config, device,
                     use_mask=False):
    """
    訓練分類器。

    Args:
        use_mask: 是否使用 mask（LesionAwareClassifier 需要）
    """
    optimizer = optim.Adam(
        model.parameters(), lr=config.clf_lr,
        weight_decay=config.clf_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_state = None

    for epoch in range(config.clf_epochs):
        model.train()
        correct = 0
        total = 0

        for batch in train_loader:
            if use_mask:
                imgs, masks, labels = batch
                imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                outputs = model(imgs, masks)
            else:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        # 驗證
        val_acc = evaluate_classifier(model, val_loader, device, use_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{config.clf_epochs}: "
                  f"train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")

    # 載入最佳權重
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc


def evaluate_classifier(model, loader, device, use_mask=False):
    """評估分類器準確度。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            if use_mask:
                imgs, masks, labels = batch
                imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                outputs = model(imgs, masks)
            else:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def get_detailed_report(model, loader, device, use_mask=False):
    """取得詳細的分類報告。"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if use_mask:
                imgs, masks, labels = batch
                imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
                outputs = model(imgs, masks)
            else:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(
        all_labels, all_preds,
        target_names=['Normal', 'Ischemic', 'Hemorrhagic'],
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds)
    return report, cm


def evaluate(config: MLAGANConfig, synthetic_dir: str):
    """完整下游評估流程。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_root = config.dataset_root
    ct_dir = config.ct_dir

    train_root = str(Path(dataset_root) / ct_dir / 'train')
    val_root = str(Path(dataset_root) / ct_dir / 'val')
    test_root = str(Path(dataset_root) / ct_dir / 'test')
    mask_train = str(config.mask_train_dir)

    results = {}

    # ═══ 實驗 1: Real-only Baseline（標準 ResNet-18）═══
    print("\n" + "=" * 60)
    print("實驗 1: Real-only Baseline（標準 ResNet-18）")
    print("=" * 60)

    train_ds = CTEvalDataset([train_root])
    val_ds = CTEvalDataset([val_root])
    test_ds = CTEvalDataset([test_root])

    # 加權採樣處理類別不平衡
    labels = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)

    model1 = models.resnet18(weights='IMAGENET1K_V1')
    model1.fc = nn.Linear(model1.fc.in_features, config.num_classes)
    model1 = model1.to(device)

    best_val = train_classifier(model1, train_loader, val_loader, config, device)
    test_acc = evaluate_classifier(model1, test_loader, device)
    report, cm = get_detailed_report(model1, test_loader, device)

    results['real_only'] = {
        'val_acc': best_val,
        'test_acc': test_acc,
        'report': report,
        'confusion_matrix': cm.tolist(),
    }
    print(f"  Real-only: val={best_val:.2f}%, test={test_acc:.2f}%")
    print(f"  Per-class F1: Normal={report['Normal']['f1-score']:.3f}, "
          f"Ischemic={report['Ischemic']['f1-score']:.3f}, "
          f"Hemorrhagic={report['Hemorrhagic']['f1-score']:.3f}")

    # ═══ 實驗 2: Real + MLA-GAN（標準 ResNet-18）═══
    print("\n" + "=" * 60)
    print("實驗 2: Real + MLA-GAN Synthetic（標準 ResNet-18）")
    print("=" * 60)

    train_ds2 = CTEvalDataset([train_root, synthetic_dir])
    labels2 = [s[1] for s in train_ds2.samples]
    class_counts2 = np.bincount(labels2)
    weights2 = 1.0 / class_counts2[labels2]
    sampler2 = WeightedRandomSampler(weights2, len(weights2))
    train_loader2 = DataLoader(train_ds2, batch_size=32, sampler=sampler2, num_workers=4)

    model2 = models.resnet18(weights='IMAGENET1K_V1')
    model2.fc = nn.Linear(model2.fc.in_features, config.num_classes)
    model2 = model2.to(device)

    best_val2 = train_classifier(model2, train_loader2, val_loader, config, device)
    test_acc2 = evaluate_classifier(model2, test_loader, device)
    report2, cm2 = get_detailed_report(model2, test_loader, device)

    results['mlagan_standard'] = {
        'val_acc': best_val2,
        'test_acc': test_acc2,
        'report': report2,
        'confusion_matrix': cm2.tolist(),
    }
    print(f"  MLA-GAN + ResNet: val={best_val2:.2f}%, test={test_acc2:.2f}%")
    print(f"  Per-class F1: Normal={report2['Normal']['f1-score']:.3f}, "
          f"Ischemic={report2['Ischemic']['f1-score']:.3f}, "
          f"Hemorrhagic={report2['Hemorrhagic']['f1-score']:.3f}")

    # ═══ 實驗 3: Real + MLA-GAN（Lesion-Aware Classifier）═══
    print("\n" + "=" * 60)
    print("實驗 3: Real + MLA-GAN Synthetic（Lesion-Aware Classifier）")
    print("=" * 60)

    train_ds3 = CTEvalDatasetWithMask(
        [train_root, synthetic_dir],
        mask_dirs=[mask_train],
    )
    val_ds3 = CTEvalDatasetWithMask(
        [val_root], mask_dirs=[mask_train],
    )
    test_ds3 = CTEvalDatasetWithMask(
        [test_root], mask_dirs=[mask_train],
    )

    train_loader3 = DataLoader(train_ds3, batch_size=32, shuffle=True, num_workers=4)
    val_loader3 = DataLoader(val_ds3, batch_size=32, shuffle=False, num_workers=4)
    test_loader3 = DataLoader(test_ds3, batch_size=32, shuffle=False, num_workers=4)

    model3 = LesionAwareClassifier(
        num_classes=config.num_classes,
        alpha_mask=config.clf_alpha_mask,
    ).to(device)

    best_val3 = train_classifier(
        model3, train_loader3, val_loader3, config, device, use_mask=True,
    )
    test_acc3 = evaluate_classifier(model3, test_loader3, device, use_mask=True)
    report3, cm3 = get_detailed_report(model3, test_loader3, device, use_mask=True)

    results['mlagan_lesion_aware'] = {
        'val_acc': best_val3,
        'test_acc': test_acc3,
        'report': report3,
        'confusion_matrix': cm3.tolist(),
    }
    print(f"  MLA-GAN + Lesion-Aware: val={best_val3:.2f}%, test={test_acc3:.2f}%")
    print(f"  Per-class F1: Normal={report3['Normal']['f1-score']:.3f}, "
          f"Ischemic={report3['Ischemic']['f1-score']:.3f}, "
          f"Hemorrhagic={report3['Hemorrhagic']['f1-score']:.3f}")

    # ═══ 結果摘要 ═══
    print("\n" + "=" * 60)
    print("結果摘要")
    print("=" * 60)
    print(f"{'方法':<35} {'Val Acc':>10} {'Test Acc':>10}")
    print("-" * 60)
    print(f"{'Real-only (baseline)':<35} {best_val:.2f}% {test_acc:.2f}%")
    print(f"{'+ MLA-GAN (standard ResNet)':<35} {best_val2:.2f}% {test_acc2:.2f}%")
    print(f"{'+ MLA-GAN (Lesion-Aware)':<35} {best_val3:.2f}% {test_acc3:.2f}%")

    delta_std = test_acc2 - test_acc
    delta_la = test_acc3 - test_acc
    print(f"\n改善: Standard={delta_std:+.2f}pp, Lesion-Aware={delta_la:+.2f}pp")

    # 儲存結果
    results_path = Path(config.output_dir) / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n結果已儲存至 {results_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLA-GAN 下游評估')
    parser.add_argument('--synthetic-dir', type=str, required=True,
                        help='合成影像目錄')
    parser.add_argument('--epochs', type=int, default=None,
                        help='分類器訓練 epochs')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    config = MLAGANConfig()
    if args.epochs is not None:
        config.clf_epochs = args.epochs
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    evaluate(config, args.synthetic_dir)
