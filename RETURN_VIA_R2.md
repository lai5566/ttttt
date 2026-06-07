# 結果回程：透過 Cloudflare R2

## 為什麼這樣做

- 運算伺服器（美國）**對公網很快**，但「伺服器 → 台灣」直連只有 ~100kb（scp/ssh 很慢）。
- 解法：伺服器**快速上傳 R2**（中轉），你在台灣**從 R2 公開連結下載**（正常寬頻），繞開慢的直連。
- 為什麼不用 GitHub 裝大檔：GitHub 單檔上限 100MB、repo 不該放數十 GB。R2 是物件儲存，無此限制。

```
  [美國伺服器] ──快──> [Cloudflare R2] ──你的正常寬頻──> [台灣你的電腦]
         （跑訓練、打包、上傳）          （公開連結下載）
```

## 流程

### 去程（程式 + 資料 → 伺服器）：用 GitHub
```bash
# 在伺服器
git clone https://github.com/lai5566/ttttt.git
cd ttttt    # 或對應目錄
```

### 在伺服器設 R2 憑證（環境變數，勿寫死進檔案）

到 Cloudflare 後台 → R2 → **Manage R2 API Tokens** 建一組 token，會給你
Access Key ID / Secret Access Key，帳號 ID 在 R2 總覽頁。

```bash
export R2_ACCOUNT_ID="你的帳號ID"
export R2_ACCESS_KEY_ID="AccessKeyID"
export R2_SECRET_ACCESS_KEY="SecretAccessKey"
export R2_BUCKET="你的bucket名稱"
# 公開下載前綴（已預設為你給的，可覆寫）
export R2_PUBLIC_BASE="https://pub-f3129530dfcd42b9ad0c77f15dba3245.r2.dev"
```

### 一鍵：跑全部 → 打包 → 自動上傳 R2
```bash
nohup bash run_all_and_pack.sh > logs/run_all.out 2>&1 &
tail -f logs/run_all.log
```
跑完會自動：
1. 打包 **包1**`results_<host>_<ts>.tar.gz`（結果+日誌+權重，較小，先傳）
2. 打包 **包2**`generated_<host>_<ts>.tar`（全部生成影像，數十 GB）
3. 兩包都上傳 R2，log 會印出每包的**公開下載連結**。

### 回程（台灣下載）
從 log 裡的 `[R2] 公開下載連結` 複製網址，在台灣用瀏覽器或：
```bash
curl -LO "https://pub-f3129530dfcd42b9ad0c77f15dba3245.r2.dev/results_<host>_<ts>.tar.gz"
curl -LO "https://pub-f3129530dfcd42b9ad0c77f15dba3245.r2.dev/generated_<host>_<ts>.tar"
# 解開
tar -xzf results_*.tar.gz
tar -xf  generated_*.tar
```

## 常用調整（環境變數）

| 變數 | 預設 | 說明 |
|---|---|---|
| `R2_UPLOAD` | `auto` | `auto`=有憑證才傳 / `1`=強制 / `0`=只打包不傳 |
| `PACK_GEN` | `1` | `1`=打包生成影像 / `0`=只要結果不打包影像 |
| `R2_CHUNK_MB` | `64` | multipart 分段大小 MB |
| `R2_CONCURRENCY` | `8` | 並行分段數（連線快可調高） |
| `GPUS` `PER_GPU_1` `PER_GPU_2` | — | 見 `run_all_and_pack.sh` 開頭 |

## 只想手動傳某個檔
```bash
R2_KEY="myfile.tar" python3 upload_to_r2.py /path/to/myfile.tar
```

## 注意
- 打包包2（數十 GB）需要**伺服器有等量的暫存磁碟空間**。空間不夠就先 `PACK_GEN=0` 只傳結果，影像之後再處理。
- 只要結果不要影像：`PACK_GEN=0 bash run_all_and_pack.sh`。
- 上傳需 `boto3`（腳本會自動 `pip install`；或先手動裝）。
