# MLA-GAN 訓練包(V4 rank / 非 kfold「v3」整套 ir5/10/20)

**打包日期**:2026-06-07 ｜ **自包含、可直接在別台機器跑**

此包用於**以「修好的 mask」重訓** mlagan（V4 = rank BD、非 kfold）。
**masks 已是修復版**（class2/Hemorrhagic 原本 ~11–28% 全黑 mask 已補上病灶；ir10 的 `_aug_` 廢檔已排除）。

## 內容
```
methods/01_mla_gan_ours/
  code/                     models/losses/train/dataset/utils/config*/generate_bd_mix + train_ir{5,10,20}_v3.py
  train_ir{5,10,20}_v3.sh           訓練 bs8（rank BD,非 kfold）
  train_ir{5,10,20}_v3_bs16.sh      訓練 bs16
  generate_ir{5,10,20}_v3[_bs16].sh 生成（pure_boundary）
  run_eval_v3.sh                    下游評估（5arch×5seed）
data/
  ct_256/(ir10) ct_256_ir5/ ct_256_ir20/(symlink→ct_256)
  masks/(ir10,base) masks_ir5/ masks_ir20/(symlink→masks)   ← 已修復
  classifiers/resnet50_v3_ir{5,10,20}/                       ← 引導分類器
eval/  evaluate_multi_v3.py config.py evaluate.py
```
> **已排除**：kfold（train_kfoldbd/kfold_bd_compute/kfold JSON）、z_Discard 的 `_aug_` mask。要乾淨 V4 baseline 就用這包。

## 環境需求
`torch torchvision timm numpy scipy scikit-learn pillow`（CUDA GPU）

## 路徑（已自動處理）
config 用相對路徑,只要保持此包目錄結構即自動指向包內 `data/`。
若要改資料位置:`export MLAGAN_DATASET_ROOT=/your/data`。

## ★ 一鍵全跑（多卡自動排程,推薦）
```bash
cd methods/01_mla_gan_ours
bash master_run_all.sh                  # 方法×IR×batch 全矩陣:v3+kfold+gateA × ir5/10/20 × bs8/16
```
**方法**:`v3`(rank,非kfold)/ `kfold`(K-fold OOF BD)/ `gateA`(從 v3 G + QualityGate,不另訓)。
- 矩陣 = 3 方法 × 3 IR × 2 batch;gateA 不訓練(複用 v3 G)→ master 分兩波(先 v3/kfold,再 gateA)。
- 子集:`METHODS="v3 kfold" IRS="5" BSS="8 16" bash master_run_all.sh`
- **自動偵測 N 張 GPU,同時跑 N 個 cell(一卡一 cell 端到端),有卡空出立刻接下一個** → 自動最大化利用率。
- skip 守衛:已完成的 train/generate/eval 自動跳過,**可中斷續跑**。
- 進度:`logs/master.log`(總覽)+ `logs/cell_<tag>.log`(各 cell)。
- 自訂:
  - `CELLS="5:8 5:16 10:16 20:16" bash master_run_all.sh`  只跑這 4 cell
  - `GPUS="0,1,2" bash master_run_all.sh`  只用指定卡
  - **`PER_GPU=2 bash master_run_all.sh`  每張卡同時跑 2 個 cell(吃滿卡)**
    - mlagan 很小(單實例 ~5GB / 24GB),`PER_GPU=2~3` 對 GAN 訓練常提升吞吐;不需改任何模型 code(獨立程序共用一張卡)。
    - 注意:單實例 util 已 ~86%,增益會遞減;`PER_GPU` 太大會 OOM。建議先 `PER_GPU=2` 觀察 `nvidia-smi`。
    - eval(下游分類器)較吃算力,`PER_GPU` 增益較小。
- 背景跑:`nohup bash master_run_all.sh > logs/master.out 2>&1 &`

## 跑法（單一 cell,以 ir5 bs8 為例,手動）
```bash
cd methods/01_mla_gan_ours
bash train_ir5_v3.sh        # 400ep → output_ir5_v3/run_seed7/best_model.pth
bash generate_ir5_v3.sh     # pure_boundary 3540×2 → generated_ir5_v3/class{1,2}
bash run_eval_v3.sh "5" "8" # → eval/results/eval_mlagan_v3_ir5_5arch_5seed/
```
ir10/ir20、bs16 同理（換對應腳本;run_eval_v3.sh 參數 "10 20" "16" 等）。

## 一鍵 4-cell（ir5 bs8 + ir5/10/20 bs16）
原 `run_pipeline_v3_4cells.sh` 未包含;可自行串:
```bash
cd methods/01_mla_gan_ours
for s in train_ir5_v3 train_ir5_v3_bs16 train_ir10_v3_bs16 train_ir20_v3_bs16; do bash $s.sh; done
for s in generate_ir5_v3 generate_ir5_v3_bs16 generate_ir10_v3_bs16 generate_ir20_v3_bs16; do bash $s.sh; done
bash run_eval_v3.sh "5" "8 16"; bash run_eval_v3.sh "10 20" "16"
```

## 注意
- 訓練只用 minority class（1,2）;eval 用全 3 類（含 class0）。
- seed 限 7/49/91/133/175。
- 此包 mask = 修復後;之前的結果都是修復前,需重訓才反映 mask 修復效果。

## 你的硬體:2 × 32GB VRAM — 建議
6 個 cell、記憶體寬裕(單實例 ~5GB)。直接讓全部一次跑:
```bash
GPUS="0,1" PER_GPU=3 bash master_run_all.sh      # 每卡 3 cell(~15GB/32GB),6 cell 一輪全完成
# 或自動:
GPUS="0,1" PER_GPU=auto bash master_run_all.sh   # 依空閒 VRAM 估(8GB/槽 → 約 4/卡)
```
記憶體不是瓶頸(32GB);compute 才是。PER_GPU=3 期間訓練階段吞吐↑,eval 階段時間共享(不會更差)。先看 nvidia-smi 沒 OOM 再考慮加大。

## StyleGAN3（methods/02_stylegan3,只吃 CT,無 mask）
另一條對照線(StudioGAN backbone)。**不用 lesion mask**,直接吃 CT(`ct_256*`,已在包內)。
```bash
cd methods/02_stylegan3
GPUS="0,1" PER_GPU=2 bash run_stylegan_all.sh    # ir5/10/20 × bs8/16,train→generate→eval
```
- 6 config 已 template(`configs/sg3_ir{5,10,20}_bs{8,16}.yaml`,只差 batch_size/name)。
- ir→CT:ir5=ct_256_ir5、ir10=ct_256、ir20=ct_256_ir20。
- eval 共用 999 的 `evaluate_multi_v3.py`(5arch×5seed)→ `eval/results/eval_stylegan3_*`。
- **需求**:StudioGAN deps(torch/torchvision/numpy/scipy/pyyaml/h5py/click/ninja 等),首次在目標機 `pip` 補齊;`total_steps=125000`(step-based,非 epoch),時間視卡而定。
- 注意:mask 修復**不影響** stylegan3(它本來就只吃 CT)→ 跑它是為了完整 benchmark,不是為 mask。
