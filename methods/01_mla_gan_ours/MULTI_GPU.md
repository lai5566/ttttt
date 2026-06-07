# MLA-GAN 多卡（DDP）訓練說明

本方法（`01_mla_gan_ours`）已支援單機多卡訓練。**單卡用法完全不變**；多卡只需用
`torchrun` 啟動，或在既有 `.sh` 腳本前加 `GPUS=N`。

## 快速開始

```bash
cd methods/01_mla_gan_ours

# 單卡（與原本一致）
./train_ir5_v3.sh

# 單一訓練用 4 張卡跑 DDP（全域 batch 仍為 8，平均切到各卡 → 每卡 2 張）
NPROC=4 ./train_ir5_v3.sh

# K-fold 入口同理
NPROC=4 ./train_ir5_kfold.sh
```

所有 `train_*.sh`（v3 與 kfold、bs8 與 bs16）都支援 `NPROC` 環境變數：
`NPROC=1`（預設）走 `python`；`NPROC>1` 自動改用 `torchrun --standalone --nproc_per_node=$NPROC`。

> **`NPROC` vs `GPUS`**：`NPROC` 是「單一訓練用幾張卡跑 DDP」；而 `master_run_all.sh`
> 的 `GPUS`（如 `GPUS="0,1"`）是「用哪些卡、一卡一個 cell 並行跑整個矩陣」。兩者
> 不同用途、互不衝突。跑整個實驗矩陣請優先用 master（並行多個獨立 cell 通常比對單一
> 訓練開 DDP 更能榨滿 GPU）。

直接用 torchrun 也可以：

```bash
cd methods/01_mla_gan_ours/code
torchrun --standalone --nproc_per_node=4 \
    train_ir5_v3.py --epochs 400 --batch-size 8 --seed 7 \
    --output-dir ../output_ir5_v3/run_seed7
```

## 設計重點

| 項目 | 做法 | 原因 |
|------|------|------|
| **Batch 語義** | 全域 batch = `--batch-size` **不變**，每步切成 `batch_size/world_size` 張分給各卡 | 訓練動態與單卡一致，超參（lambda、ADA、R1）不必重調。**需 `batch_size` 能被卡數整除** |
| **梯度同步** | G 與 D 皆**不包 `DistributedDataParallel`**，改用手動 `all_reduce` 平均梯度 | 規避兩個對 DDP 不友善的特性：① G 每步 forward 兩次（diversity loss）；② D-loss 的 R1 用 `create_graph=True`（double backward） |
| **混合精度** | 單卡維持 AMP；**多卡走 fp32** | 避免各卡 `GradScaler` 的 scale/inf 狀態不同步導致參數發散 |
| **torch.compile** | 單卡啟用；多卡停用 | 穩健性優先（多卡已有並行加速） |
| **資料切分** | 每卡複製整個 dataset（約 185MB），用「跨 rank 一致的洗牌」切同一個全域 batch | 資料集小，複製成本可忽略；切分一致才能重現單卡的 batch 組成 |
| **ADA `p`** | 每步把 `r_t` 跨 rank 平均後更新 | 各卡增強強度一致 |
| **存檔 / 日誌** | 只在 rank 0 寫檔；存檔前 `unwrap_model` 去掉包裝前綴 | 避免多卡競寫；checkpoint 與單卡、與 `generate_*.py` 完全相容 |

## 注意事項

- **卡數整除**：`batch_size` 必須能被卡數整除（例：batch=8 → 可用 1/2/4/8 卡）。否則啟動時 assert 報錯。
- **每卡 batch 過小**：8 卡時每卡只有 1 張，BN 類層統計會較不穩；建議 ≤4 卡，或視情況改用「每卡固定 batch」語義（需另行調整）。
- **checkpoint 通用**：多卡存的 checkpoint 可直接用單卡 resume 或丟給 `generate_*.py`，反之亦然。
- **相關檔案**：核心邏輯在 `code/train.py`（`train()`）與 `code/ddp_utils.py`；`train_kfoldbd.py` 直接 `from train import train` 共用同一份邏輯。
