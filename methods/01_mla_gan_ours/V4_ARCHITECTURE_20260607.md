# V4(methods/01_mla_gan_ours)架構調查報告 + 與 V3/V2 差異

**日期**:2026-06-07
**範圍**:`methods/01_mla_gan_ours`(版本對應:**V4**;其 code = 8888 V4)的模型結構、邏輯,與 **V3/V2 的差異**。
**方法**:直接讀 code 並實測參數量(`code/models.py` / `losses.py` / `config.py` / `train.py` / `dataset.py` / `utils.py`)。
**結論先講**:**V4 與 V2/V3 共用完全相同的 backbone**。V4 相對 V3 = **把 multi-dim BD 介面移除、bd_dim 固定回 1**(models.py 僅 31 行 diff),BD 改用 **rank(百分位)** + **class-conditional guided 取樣(B1/B2/B3/B5 修正)**;並提供 method-01 專屬的 **K-fold OOF BD**(「ours」的進一步改良,`train_kfoldbd.py` 預設)。

> ⚠️ 命名提醒:本 repo 內 `_v3` 後綴(如 `train_ir5_v3.sh`、`eval_mlagan_v3_*`)指 **resnet50-v3 引導分類器 setup**,**不是版本 V3**。本 repo = **V4**。
> 對照:V2 見 `mlagan_v2/V2_ARCHITECTURE_20260607.md`、V3 見 `mlagan_v3/V3_ARCHITECTURE_20260607.md`。

---

## 0. 一句話總覽(粗體=與 V3 不同)

```
GatedContextEncoder(img+mask, skip) → MappingNetwork(z+class) + BoundaryDistanceModulator(**bd[B,1]**)
   → Bottleneck Fusion → FFCAdaINDecoder(skip+AdaIN(w)+bd_delta) → Soft Compositing      ← 全同 V2/V3
判別:GlobalD(Projection + **BD regressor 輸出 [B,1]**) + LocalD(ROI 64×64)
BD 計算:**rank**(logit margin 百分位→均勻[0,1])  或  **kfold_json**(OOF BD,ours)   ← V3 是 softmax[B,3]
取樣:**guided + class-conditional(per-class)**,B1/B2/B3/B5 修正                        ← V3 是 'real'
```

---

## 1. V4 vs V3 差異表(本報告重點)

| 面向 | **V3** | **V4** | code 位置(V4) |
|---|---|---|---|
| **bd_dim** | 3 | **1**(移除 multi-dim 參數介面) | `config.py:55`;`models.py` diff |
| **BD 計算法** | softmax(logits)→[B,3] | **rank**:raw logit margin → 百分位 rank → 均勻[0,1](Issue 3 fix);另有 **kfold_json** | `config.py:68`、`dataset.py:144` |
| **D 的 BD regressor** | `Linear(128→3)` | **`Linear(128→1)`** | `models.py`(GlobalD) |
| **BD modulator embed** | `Linear(3→64)` | **`Linear(1→64)`** | `models.py`(BD modulator) |
| **BD 取樣** | `'real'` | **`'guided'` + class-conditional(per-class)** | `config.py:69`、`utils.py:202`、`train.py:188-220` |
| **multi-dim/ablation 介面** | bd_dim 可配 | **移除**(bd_dim 寫死 1) | models.py 31 行 diff |
| **Generator 參數量** | 8,730,577 | **8,730,449**(= V2) | 實測 |
| **Discriminator 參數量** | 6,071,205 | **6,070,947**(= V2) | 實測 |

> V4 參數量**完全等於 V2**(bd_dim=1);V3 因 bd_dim=3 多 +128/+258。→ 三版同 backbone,差異純在 **BD 維度 + 計算法 + 取樣**。

---

## 2. V4 的 BD 機制(核心)

### 2.1 rank BD(`dataset.py:132-150`,`losses.py` method='raw')
- `BoundaryDistanceComputer(method='raw')` 回傳**原始 logit margin**(max−2nd_max);
- `precompute_boundary_distances(method='rank')` 對全資料集做**百分位 rank → 均勻 [0,1]**(mean≈0.5, p25≈0.25)。
- 動機(**Issue 3 fix**):V2 sigmoid-margin 對自信分類器飽和(75%+ >0.99);V3 softmax multi-dim 有 bug→負結果;**V4 rank 保證 BD 均勻分布、不飽和**。

### 2.2 class-conditional guided 取樣(`utils.py:202-251`,`train.py:188-220`)
`sample_boundary_target(strategy='guided', class_label=…)`:
- **guided**(預設):3-zone near/mid/far 混合;**[v3-fix B2] 用 per-class bd_stats**(各類別自己的邊界分布)。
- ablation 策略:`random`(B1)、`fixed_low`=0.05(B2)、`fixed_high`=0.95(B3)、`real`(B5)。

### 2.3 B1/B2/B3/B5 修正(`train.py:188-220`,V4 相對 V3 的對抗循環修正)
| 修正 | 內容 |
|---|---|
| **B1** | D 步驟**之前先採樣 target_bd**,D 與 G 用同一個 target_bd 生成 fake(對抗對稱;原本 D 用 real_bd) |
| **B2** | `sample_boundary_target` 接收 `class_label` → **per-class(class-conditional)取樣** |
| **B3** | **把 target_bd 傳給 `d_loss`**,讓 D 學會 fake 的 bd 契約 |
| **B5** | 可選 `target_bd = real_bd`(bd_sampling='real') |

### 2.4 K-fold OOF BD(⚠️ 是「消融方法」,非乾淨版本;且誤改正式版)
- `bd_compute_method ∈ {'kfold_json','kfold_json_perclass'}`(`dataset.py:180`):載入 **K-fold out-of-fold BD**(每樣本 BD 來自「沒看過它」的 aux,解飽和),`train_kfoldbd.py` 走此。
- **正名(2026-06-07,使用者)**:kfold 只是 **BD 消融的一個方法**(消融總表家族 D),**不是 V4 之上的乾淨「ours」**。它被**錯誤地直接改進正式版 V4 檔案**(`config.py`/`dataset.py`/`train_kfoldbd.py`/kfold JSON),**污染了 V4 baseline**。比較乾淨 V4 時應走 `train.py`/`train_*_v3.py`(rank,非 kfold)。

---

## 3. 完全相同於 V2/V3 的部分(共用 backbone)

- **GatedContextEncoder**:concat(img,mask)=2ch、4 層 GatedConv、skip[e0,e1,e2]、context_dropout=0.3。
- **MappingNetwork**:z[128]+class_embed[64]→w[256](bd 預設不進)。
- **BoundaryDistanceModulator**:bd→per-layer (Δγ,Δβ) 加法解耦調變 AdaIN(V4 bd 為 1 維)。
- **FFCAdaINDecoder**:FFC(FFT)+ PixelShuffle + skip + AdaIN,16→256。
- **Discriminator**:GlobalD(Projection + BD regressor)+ LocalD(ROI 64×64)。
- **損失**:Hinge + R1(lazy 16)+ λ_bd·BD回歸/導引 + λ_fd·mode-seeking + λ_rec + λ_mask_guide。權重同 V2/V3(lambda_local=0.5、lambda_bd=1.0、lambda_fd=0.5、lambda_rec=1.0、lambda_mask_guide=30.0、lambda_r1=10.0)。
- **soft compositing**、ADA(target_rt=0.6/max_p=0.8)。
- 保留 `use_bd_modulator`(A1)等開關(從 V3 沿用)。

---

## 4. 三版本 BD 機制速查(V2 → V3 → V4)

| | **V2** | **V3** | **V4** |
|---|---|---|---|
| bd_dim | 1 | 3 | 1 |
| BD 計算 | sigmoid-margin [B,1] | softmax [B,3] | **rank [B,1]**(或 kfold OOF) |
| BD 取樣 | near/mid/far | real | **guided + class-conditional** |
| 對抗循環 | 基本 | 基本 | **B1/B2/B3/B5 修正** |
| 參數量(G/D) | 8,730,449 / 6,070,947 | +128 / +258 | 8,730,449 / 6,070,947(=V2) |
| backbone | Gated+FFC+dualD+Hinge/R1 | 同 | 同 |

---

## 5. 版本定位(來自 `mlagan_history.md`)

- **V4 = 正式版/修正版**:V3 的 multi-dim BD(bd_dim=3)+ 數個 BD bug → 負結果;V4 **回退 bd_dim=1 + rank + class-conditional(B1/B2/B3)** 後轉正。
- V3→V4 `models.py` 僅 ~31 行 diff(純移除 multi-dim 介面),其餘改動在 `dataset.py`/`utils.py`/`train.py`(rank、per-class、B-fixes)與 `config.py`(預設值)。
- ⚠️ kfold **不是** V4 之上的乾淨版本,而是消融方法且誤改了正式版檔案(見 §2.4 正名)。

---

## 6. 小瑕疵記錄

- 同 V2/V3:`MLAGenerator` docstring 參數量、`MLAGANLoss` L_G 公式漏 mask_guide 等註解問題沿用。
- `config.py:69` 註解把 sampling 策略標 `B1/B2/B3/B5: random/fixed_low/fixed_high/real`,與 train.py 的「B1/B2/B3 對抗循環修正」是**不同的 B 編號系統**(一個是取樣策略名、一個是循環修正名),易混淆,已於本報告分開說明。
