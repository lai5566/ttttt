#!/usr/bin/env python3
"""
[v3] 多模型 × 多資料集 × 多次重複 下游評估 — 對資料敏感版

相對 v2 的改動(目的:讓不同 GAN 合成資料的差異不被分類器優化壓平):
  1. AdamW → SGD + momentum       (拆掉 per-parameter 自適應)
  2. 拿掉 CosineAnnealingLR        (拆掉後期 lr→0 的平滑收斂)
  3. weight_decay 0.01 → 0         (拆掉隱性正則化)
  4. 拿掉 AMP / GradScaler         (拆掉 fp16 數值平滑)
  5. epochs 20 → 10                (避免大家都跑到天花板)
  6. lr 1e-4 → 1e-2                (SGD 對應的尺度)
  7. best-of-val → last-3 SWA      (拆掉「挑運氣最好的 epoch」)
  8. ResNet-18 為主 backbone        (對資料敏感、領域標準小型 baseline)
  9. weights=None (random init)    (拆掉 ImageNet 強先驗;可 --pretrained 切回)
 10. 拿掉 ImageNet normalize        (用原始 [0,1] 像素分佈)
 11. 報告新增 Paired Wilcoxon test  (利用同 seed 配對降 noise)

用法:
    # 預設:無 pretrain、無 normalize、SGD、最敏感版
    python evaluate_multi_v3.py \
        --syn-dirs dir1 dir2 dir3 \
        --syn-names mlagan stylegan reacgan \
        --output-dir output/final_eval \
        --runs 5

    # 若需 ImageNet pretrain(較弱版本就用 IMAGENET1K_V1):
    python evaluate_multi_v3.py --pretrained ...
"""

import argparse
import csv
import gc
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

from config import MLAGANConfig
from evaluate import CTEvalDataset


IMG_SIZE = 224


class CTEvalDataset224(CTEvalDataset):
    """resize 到 224。[v3] 拿掉 ImageNet normalize,使用 [0,1] 原始像素。"""

    def __init__(self, root_dirs, img_size=224):
        super().__init__(root_dirs, img_size=256)
        # ─── 原 v2: ToPILImage + Resize + ToTensor + Normalize(ImageNet stats)
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.Resize((img_size, img_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])
        # [v3] 拿掉 normalize:模型直接看 [0,1] 像素,
        #      合成圖的像素分佈瑕疵會直接傳到 logits
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert('L')
        img = img.resize((256, 256), Image.LANCZOS)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        img = self.transform(img)
        return img, label


def build_model(arch, num_classes=3, pretrained=False):
    """[v3] 預設 pretrained=False (random init),讓資料品質決定特徵學習。
    若需 weak pretrain,可 --pretrained 切回 IMAGENET1K_V1。"""
    try:
        import timm
        # ─── 原 v2: pretrained=True
        return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
    except Exception:
        pass
    # ─── 原 v2: weights='IMAGENET1K_V1' 一律 pretrain
    weights = 'IMAGENET1K_V1' if pretrained else None
    from torchvision import models
    if arch == 'resnet18':
        m = models.resnet18(weights=weights); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == 'resnet50':
        m = models.resnet50(weights=weights); m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=weights); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == 'efficientnet_b3':
        m = models.efficientnet_b3(weights=weights); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == 'convnext_tiny':
        m = models.convnext_tiny(weights=weights); m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif arch == 'densenet121':
        m = models.densenet121(weights=weights); m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    else:
        raise ValueError(f'Unknown: {arch}')
    return m


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval(model, train_loader, val_loader, test_loader, device,
                   # ─── 原 v2: lr=1e-4, weight_decay=0.01, epochs=20
                   lr=1e-2, weight_decay=0.0, epochs=20):
    """[v3] 對資料敏感版訓練:SGD + 固定 lr + 無 AMP + 無 best-of-val + SWA-lite。"""
    # ─── 原 v2: AdamW (適應性、寬容、會修補爛資料的梯度噪聲)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    # ─── 原 v2: CosineAnnealing 把後期 lr→0,所有條件平滑收斂
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # (整個拿掉,固定 lr 跑到底)

    criterion = nn.CrossEntropyLoss()

    # ─── 原 v2: AMP 混合精度,數值範圍壓縮會吞掉 0.5% 量級差異
    # scaler = torch.amp.GradScaler('cuda')

    # ─── 原 v2: best_val_acc / best_state (挑運氣最好的 epoch)
    # best_val_acc = 0; best_state = None
    last_states = []  # 改用最後 3 個 epoch 權重平均(SWA-lite)

    for ep in range(epochs):
        model.train()
        for imgs, lbs in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbs = lbs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            # ─── 原 v2: with torch.amp.autocast('cuda'): loss = criterion(model(imgs), lbs)
            #            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss = criterion(model(imgs), lbs)
            loss.backward()
            optimizer.step()

        # val 純監測,不再決定 checkpoint
        model.eval()
        correct = total = 0
        # ─── 原 v2: with torch.no_grad(), torch.amp.autocast('cuda'):
        with torch.no_grad():
            for imgs, lbs in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbs = lbs.to(device, non_blocking=True)
                correct += (model(imgs).argmax(1) == lbs).sum().item()
                total += lbs.size(0)
        val_acc = correct / total
        print(f'    ep{ep+1:02d}/{epochs} val_acc={val_acc:.4f}', flush=True)

        # ─── 原 v2: scheduler.step()
        # ─── 原 v2: if val_acc > best_val_acc: best_val_acc = val_acc; best_state = ...
        if ep >= epochs - 3:
            last_states.append(
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            )

    # ─── 原 v2: model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    # 改用最後 3 個 epoch 權重平均
    avg_state = {}
    for k in last_states[0]:
        stacked = torch.stack([s[k].float() for s in last_states])
        avg_state[k] = stacked.mean(0).to(last_states[0][k].dtype).to(device)
    model.load_state_dict(avg_state)

    model.eval()
    preds, labs, probs = [], [], []
    # ─── 原 v2: with torch.no_grad(), torch.amp.autocast('cuda'):
    with torch.no_grad():
        for imgs, lbs in test_loader:
            logits = model(imgs.to(device, non_blocking=True))
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())   # [v3+] AUC 用機率
            labs.extend(lbs.numpy())
    # ─── 原 v2: return np.array(preds), np.array(labs), best_val_acc
    return np.array(preds), np.array(labs), np.array(probs), val_acc  # +probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--syn-dirs', nargs='*', default=[])
    parser.add_argument('--syn-names', nargs='*', default=[])
    parser.add_argument('--output-dir', type=str, default='output/final_eval')
    # ─── 原 v2: default=20
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--no-real', action='store_true')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--start-run', type=int, default=0,
                        help='Run index to start from (seeds = (start_run+i)*42+7). Use to skip already-completed seeds.')
    parser.add_argument('--archs', nargs='+', default=None,
                        help='Subset of architectures. Default: resnet18.')
    # [v3 新增]
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet IMAGENET1K_V1 pretrained weights (default: False).')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='SGD learning rate (default: 1e-2).')
    # 可選：覆寫真實訓練資料 root（給 IR=20 等變體用，預設沿用 cfg.ct_dir）
    parser.add_argument('--train-root', type=str, default=None,
                        help='Override real training data root (e.g. data/ct_256_ir20/train).')
    parser.add_argument('--val-root', type=str, default=None,
                        help='Override real validation data root.')
    parser.add_argument('--test-root', type=str, default=None,
                        help='Override real test data root.')
    args = parser.parse_args()

    assert len(args.syn_dirs) == len(args.syn_names)

    cfg = MLAGANConfig()
    device = 'cuda'
    os.makedirs(args.output_dir, exist_ok=True)

    train_root = args.train_root or str(Path(cfg.dataset_root) / cfg.ct_dir / 'train')
    val_root   = args.val_root   or str(Path(cfg.dataset_root) / cfg.ct_dir / 'val')
    test_root  = args.test_root  or str(Path(cfg.dataset_root) / cfg.ct_dir / 'test')

    dataset_configs = []
    if not args.no_real:
        dataset_configs.append(('Real-only', [train_root]))
    for name, syn_dir in zip(args.syn_names, args.syn_dirs):
        dataset_configs.append((f'+{name}', [train_root, syn_dir]))

    val_ds = CTEvalDataset224([val_root])
    test_ds = CTEvalDataset224([test_root])
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # ─── 原 v2: archs = ['resnet50', 'efficientnet_b0', 'efficientnet_b3',
    #                     'convnext_tiny', 'vit_small_patch16_224', 'densenet121']
    archs = ['resnet18']  # [v3] 預設單一 ResNet-18,對資料品質敏感
    if args.archs:
        archs = args.archs
    # convnext_tiny 在 v3 setup(from-scratch/SGD/no-norm)下崩潰,永久排除 — 2026-06-06
    if 'convnext_tiny' in archs:
        print('[v3] 略過 convnext_tiny(v3 setup 下崩潰,永久排除)')
        archs = [a for a in archs if a != 'convnext_tiny']

    csv_rows = []
    total = len(archs) * len(dataset_configs) * args.runs
    done = 0

    for arch in archs:
        for ds_name, ds_dirs in dataset_configs:
            for run_local in range(args.runs):
                done += 1
                run = args.start_run + run_local
                seed = run * 42 + 7
                set_seed(seed)
                print(f'[{done}/{total}] {arch} — {ds_name} — run {run+1}/{args.runs} '
                      f'(seed={seed}, pretrained={args.pretrained})', flush=True)

                train_ds = CTEvalDataset224(ds_dirs)
                labels = [s[1] for s in train_ds.samples]
                counts = np.bincount(labels)
                print(f'  Train class counts: {dict(enumerate(counts.tolist()))} total={len(labels)}', flush=True)
                w = 1.0 / counts[labels]
                train_loader = DataLoader(train_ds, batch_size=32,
                                          sampler=WeightedRandomSampler(w, len(w)),
                                          num_workers=4, pin_memory=True)

                # ─── 原 v2: build_model(arch, cfg.num_classes)
                model = build_model(arch, cfg.num_classes, pretrained=args.pretrained).to(device)
                # ─── 原 v2: lr=1e-4, weight_decay=0.01
                preds, labs, probs, last_val = train_and_eval(
                    model, train_loader, val_loader, test_loader, device,
                    lr=args.lr, weight_decay=0.0, epochs=args.epochs,
                )

                acc = accuracy_score(labs, preds)
                report = classification_report(labs, preds,
                                               target_names=['class0', 'class1', 'class2'],
                                               digits=4, output_dict=True, zero_division=0)
                # [v3+] AUC (One-vs-Rest, per-class + macro) + 混淆矩陣
                try:
                    auc_pc = roc_auc_score(labs, probs, multi_class='ovr',
                                           average=None, labels=[0, 1, 2])
                    auc_macro = float(np.mean(auc_pc))
                except Exception as e:
                    print(f'  [warn] AUC 計算失敗: {e}', flush=True)
                    auc_pc = [float('nan')] * 3
                    auc_macro = float('nan')
                cm = confusion_matrix(labs, preds, labels=[0, 1, 2])  # 列=true, 欄=pred
                print(f'  acc={acc:.4f} auc_macro={auc_macro:.4f} (last_val={last_val:.4f})', flush=True)

                row = {'model': arch, 'dataset': ds_name, 'run': run+1, 'seed': seed,
                       'accuracy': acc, 'val_acc': last_val, 'auc_macro': auc_macro}
                for cls in ['class0', 'class1', 'class2']:
                    for metric in ['precision', 'recall', 'f1-score']:
                        row[f'{cls}_{metric}'] = report[cls][metric]
                for i, cls in enumerate(['class0', 'class1', 'class2']):
                    row[f'{cls}_auc'] = float(auc_pc[i])
                for avg in ['macro avg', 'weighted avg']:
                    for metric in ['precision', 'recall', 'f1-score']:
                        row[f'{avg.replace(" ","_")}_{metric}'] = report[avg][metric]
                for i in range(3):           # 混淆矩陣攤平:cm_<true><pred>
                    for j in range(3):
                        row[f'cm_{i}{j}'] = int(cm[i][j])
                csv_rows.append(row)

                del model, train_ds, train_loader
                gc.collect()
                torch.cuda.empty_cache()

    # ═══ 存 CSV ═══
    csv_path = Path(args.output_dir) / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    # ═══ 統計摘要 ═══
    import pandas as pd
    df = pd.DataFrame(csv_rows)

    ds_names = [d[0] for d in dataset_configs]
    report_lines = []

    setup_str = (f'pretrained={args.pretrained}, no-normalize, SGD lr={args.lr}, '
                 f'wd=0, no-AMP, no-scheduler, last-3-epoch-avg')
    report_lines.append('=' * 100)
    report_lines.append(f'[v3] RESULTS SUMMARY ({args.runs} runs, {args.epochs} epochs)')
    report_lines.append(f'Setup: {setup_str}')
    report_lines.append('=' * 100)
    report_lines.append('')

    # Accuracy mean±std
    col_w = max(14, max(len(d) + 2 for d in ds_names))
    header = f'{"Model":<25}' + ''.join(f'{d:<{col_w}}' for d in ds_names)
    report_lines.append('Accuracy (mean ± std):')
    report_lines.append(header)
    report_lines.append('-' * len(header))
    for arch in archs:
        cells = ''
        for ds in ds_names:
            vals = df[(df['model'] == arch) & (df['dataset'] == ds)]['accuracy'].values
            if len(vals) > 0:
                cells += f'{vals.mean():.4f}±{vals.std():.4f}'.ljust(col_w)
            else:
                cells += 'N/A'.ljust(col_w)
        report_lines.append(f'{arch:<25}{cells}')

    # Δ vs Real-only
    if 'Real-only' in ds_names:
        report_lines.append('')
        report_lines.append('Δ Accuracy vs Real-only (mean):')
        header2 = f'{"Model":<25}' + ''.join(f'{d:<{col_w}}' for d in ds_names if d != 'Real-only')
        report_lines.append(header2)
        report_lines.append('-' * len(header2))
        for arch in archs:
            real_vals = df[(df['model'] == arch) & (df['dataset'] == 'Real-only')]['accuracy'].values
            if len(real_vals) == 0:
                continue
            real_mean = real_vals.mean()
            cells = ''
            for ds in ds_names:
                if ds == 'Real-only':
                    continue
                vals = df[(df['model'] == arch) & (df['dataset'] == ds)]['accuracy'].values
                if len(vals) > 0:
                    delta = vals.mean() - real_mean
                    cells += f'{delta:+.4f}'.ljust(col_w)
                else:
                    cells += 'N/A'.ljust(col_w)
            report_lines.append(f'{arch:<25}{cells}')

    # [v3 新增] Paired Wilcoxon test vs Real-only(同 seed 配對)
    if 'Real-only' in ds_names and args.runs >= 3:
        try:
            from scipy.stats import wilcoxon
            report_lines.append('')
            report_lines.append('Paired Wilcoxon p-value vs Real-only (same-seed pairs, * = p<0.05):')
            header_w = f'{"Model":<25}' + ''.join(
                f'{d:<{col_w}}' for d in ds_names if d != 'Real-only'
            )
            report_lines.append(header_w)
            report_lines.append('-' * len(header_w))
            for arch in archs:
                real_vals = df[(df['model'] == arch) & (df['dataset'] == 'Real-only')]\
                              .sort_values('seed')['accuracy'].values
                if len(real_vals) < 3:
                    continue
                cells = ''
                for ds in ds_names:
                    if ds == 'Real-only':
                        continue
                    cur_vals = df[(df['model'] == arch) & (df['dataset'] == ds)]\
                                  .sort_values('seed')['accuracy'].values
                    if len(cur_vals) == len(real_vals):
                        try:
                            _, p = wilcoxon(cur_vals, real_vals)
                            sig = ' *' if p < 0.05 else ''
                            cells += f'p={p:.3f}{sig}'.ljust(col_w)
                        except ValueError:  # 全相同會丟錯
                            cells += 'p=1.000'.ljust(col_w)
                    else:
                        cells += 'n/a'.ljust(col_w)
                report_lines.append(f'{arch:<25}{cells}')
        except ImportError:
            report_lines.append('  (scipy not installed — skipping Wilcoxon test)')

    # Per-class F1
    for cls, cls_label in [('class1', 'Ischemic F1'), ('class2', 'Hemorrhagic F1')]:
        report_lines.append('')
        report_lines.append(f'{cls_label} (mean ± std):')
        report_lines.append(header)
        report_lines.append('-' * len(header))
        for arch in archs:
            cells = ''
            for ds in ds_names:
                vals = df[(df['model'] == arch) & (df['dataset'] == ds)][f'{cls}_f1-score'].values
                if len(vals) > 0:
                    cells += f'{vals.mean():.4f}±{vals.std():.4f}'.ljust(col_w)
                else:
                    cells += 'N/A'.ljust(col_w)
            report_lines.append(f'{arch:<25}{cells}')

    # 最佳結果
    report_lines.append('')
    report_lines.append('=' * 60)
    report_lines.append('Best Results:')
    report_lines.append('=' * 60)
    for ds in ds_names:
        best_arch = None
        best_mean = -1
        for arch in archs:
            vals = df[(df['model'] == arch) & (df['dataset'] == ds)]['accuracy'].values
            if len(vals) > 0 and vals.mean() > best_mean:
                best_mean = vals.mean()
                best_arch = arch
        report_lines.append(f'  {ds}: {best_arch} = {best_mean:.4f}')

    # 存報告
    report_path = Path(args.output_dir) / 'full_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    # 印出摘要
    print()
    for line in report_lines:
        print(line)

    print(f'\nCSV: {csv_path}')
    print(f'Report: {report_path}')


if __name__ == '__main__':
    main()