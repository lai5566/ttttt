# 運算/儲存分離 + 斷點續訓（Vast.ai 等可被回收的機器）

## 核心理念
租來的 GPU 主機 = **無狀態運算節點**，硬碟只是暫存（Cache）；**Cloudflare R2 才是唯一真理**。
機器隨時可能被回收 → 進度持續同步到 R2 → 換新機器拉回即接續，不怕資料消失。

```
  [運算主機] ──每 N 分鐘 rclone sync──▶ [R2: zxx/state/]  ◀──bootstrap 拉回── [新機器]
   訓練 + 定期存 latest.pth(含 opt+RNG+epoch)         唯一真理        續訓不從頭、loss 不飆高
```

## 一次設定（每台新機器）
```bash
git clone https://github.com/lai5566/ttttt.git && cd ttttt
# 放 R2 憑證（見 r2_env.sh.example；token 那行填你的 cfat_）
cp r2_env.sh.example r2_env.sh && nano r2_env.sh
```

## 一鍵跑（拉回 → 背景同步 → 訓練 → 收尾同步）
```bash
nohup bash stateless/run_stateless.sh > logs/stateless.out 2>&1 &
tail -f logs/r2_sync.log logs/run_all.log
```
機器被回收後，**租新機器重跑同一行**即自動接續。

## 它做了什麼
1. **bootstrap**（`stateless/bootstrap.sh`）：裝 rclone + `rclone copy r2:zxx/state → 本地`，把先前所有進度/結果拉回。
2. **背景同步**（`stateless/sync_daemon.sh`）：每 `SYNC_EVERY_MIN`（預設 10）分鐘 `rclone sync 本地 → R2`，只傳變動。
3. **跑 master**：`run_all_and_pack.sh`（增量上傳/大打包關閉，改由 rclone 負責）。
   - **完成標記**：每個 cell 跑完才寫 `.train_done` / `.gen_done`；沒標記的會被重跑。
   - **自動續訓**：未完成的訓練 cell，`train.py` 自動從 `latest.pth`（或最新 `checkpoint_epoch`）接續。
4. **收尾**：再同步一次，確保最後進度都上 R2。

## 斷點設計（train.py）
- `latest.pth` 每 `CKPT_EVERY_MIN`（預設 20）分鐘存一次，內含 **模型 + 優化器 + RNG(torch/cuda/numpy/python) + epoch**，原子寫入（不會被同步抓到半截檔）。
- 續訓會還原 RNG → 資料順序/噪聲與未中斷時一致，**loss 不會突然飆高**。
- `checkpoint_epoch*.pth` 里程碑只保留最新 `CKPT_KEEP`（預設 3）個，舊的自動刪（連帶 R2 也刪，省空間）。

## 可調環境變數
| 變數 | 預設 | 說明 |
|---|---|---|
| `SYNC_EVERY_MIN` | 10 | R2 背景同步間隔（分鐘） |
| `CKPT_EVERY_MIN` | 20 | latest.pth 存檔間隔（分鐘） |
| `CKPT_KEEP` | 3 | 保留幾個里程碑 checkpoint |
| `R2_STATE_PREFIX` | state | R2 上存放進度的前綴（`zxx/<prefix>/`） |
| `GPUS` `PER_GPU_1` `PER_GPU_2` `METHODS` | — | 透傳給 master |

## 手動操作
```bash
source r2_env.sh
bash stateless/sync_to_r2.sh            # 立刻同步一次本地 → R2
bash stateless/bootstrap.sh            # 只拉回，不跑訓練
rclone ls r2:zxx/state | head          # 看 R2 上有什麼（需先 source stateless/r2_rclone_env.sh）
```

## 注意
- StyleGAN3（方法02）的 checkpoints 也會被同步/拉回，靠 master 的 G-best skip 守衛做粗粒度續跑；MLA-GAN（方法01）有 epoch 級細粒度續訓。
- R2 為唯一真理：本地刪掉的（含舊 checkpoint）同步後 R2 也會刪。要保留就別讓 keep-latest 清掉。
