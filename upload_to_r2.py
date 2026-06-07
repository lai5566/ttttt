#!/usr/bin/env python3
"""把（可能數十 GB 的）檔案上傳到 Cloudflare R2（S3 相容 API）。

用途：運算伺服器對公網很快，但「伺服器→台灣」直連只有 ~100kb。故把結果上傳到
R2（伺服器端快），使用者再從 R2 公開連結在台灣下載（正常寬頻），繞開慢的直連。

針對「快速連線 + 大檔」調校：大分段 multipart（預設 64MB）+ 多執行緒並行
（預設 8）+ 多次重試 + 進度/速率/ETA 列印。自動處理 >5GB（multipart 必需）。

憑證/設定一律從環境變數讀取（不寫死、不入庫）：
  R2_ACCOUNT_ID         Cloudflare 帳號 ID（組成上傳 endpoint）
  R2_ACCESS_KEY_ID      R2 API token 的 Access Key ID
  R2_SECRET_ACCESS_KEY  R2 API token 的 Secret Access Key
  R2_BUCKET             目標 bucket 名稱
選填：
  R2_PUBLIC_BASE        公開下載前綴（例 https://pub-xxx.r2.dev）；給了就印完整下載連結
  R2_KEY                上傳後物件 key（預設 = 檔名）
  R2_CHUNK_MB           multipart 分段大小 MB（預設 64）
  R2_CONCURRENCY        並行分段數（預設 8；連線快可調高）

用法：
  python upload_to_r2.py <local_file>
離開碼：0=成功 / 2=參數或檔案錯 / 3=缺憑證 / 4=缺 boto3 / 5=上傳失敗
"""
import os
import sys
import time


def main():
    if len(sys.argv) < 2:
        print("用法: python upload_to_r2.py <local_file>")
        sys.exit(2)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"[R2] 檔案不存在: {path}")
        sys.exit(2)

    need = ['R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'R2_BUCKET']
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        print(f"[R2] 缺少環境變數: {', '.join(missing)} → 無法上傳")
        sys.exit(3)

    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
    except ImportError:
        print("[R2] 需要 boto3，請先 `pip install boto3`")
        sys.exit(4)

    account = os.environ['R2_ACCOUNT_ID']
    bucket = os.environ['R2_BUCKET']
    key = os.environ.get('R2_KEY') or os.path.basename(path)
    chunk_mb = int(os.environ.get('R2_CHUNK_MB', '64'))
    concurrency = int(os.environ.get('R2_CONCURRENCY', '8'))
    endpoint = f"https://{account}.r2.cloudflarestorage.com"

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
        region_name='auto',
        config=Config(
            retries={'max_attempts': 10, 'mode': 'adaptive'},
            connect_timeout=60, read_timeout=300,
            s3={'addressing_style': 'path'},
        ),
    )

    # 快速連線 + 大檔：大分段 + 多執行緒並行，最大化吞吐
    cfg = TransferConfig(
        multipart_threshold=chunk_mb * 1024 * 1024,
        multipart_chunksize=chunk_mb * 1024 * 1024,
        max_concurrency=concurrency,
        use_threads=True,
    )

    size = os.path.getsize(path)
    size_mb = size / 1024 / 1024
    print(f"[R2] 上傳 {path} ({size_mb:.1f} MB) → s3://{bucket}/{key}", flush=True)
    print(f"[R2] endpoint={endpoint}  分段={chunk_mb}MB  並行={concurrency}", flush=True)

    state = {'done': 0, 'last_pct': -5, 't0': time.time()}

    def cb(n):
        state['done'] += n
        pct = int(state['done'] * 100 / size) if size else 100
        if pct >= state['last_pct'] + 5:
            state['last_pct'] = pct
            el = max(time.time() - state['t0'], 1e-6)
            rate = state['done'] / 1024 / el  # KB/s
            eta = (size - state['done']) / 1024 / max(rate, 1e-6)
            print(f"[R2]   {pct:3d}%  {state['done']/1024/1024:.0f}/{size_mb:.0f} MB  "
                  f"{rate:.0f} KB/s  ETA {eta/60:.0f} min", flush=True)

    # 整體重試：multipart 內部 part 會自重試；這層保險整個流程斷掉再來
    for attempt in range(1, 4):
        try:
            s3.upload_file(path, bucket, key, Config=cfg, Callback=cb)
            break
        except Exception as e:  # noqa
            print(f"[R2] 第 {attempt}/3 次上傳失敗：{e}", flush=True)
            if attempt == 3:
                sys.exit(5)
            state['done'] = 0
            state['last_pct'] = -5
            state['t0'] = time.time()
            time.sleep(10)

    print(f"[R2] ✅ 上傳完成: s3://{bucket}/{key}", flush=True)
    base = os.environ.get('R2_PUBLIC_BASE', '').rstrip('/')
    if base:
        print(f"[R2] 公開下載連結: {base}/{key}", flush=True)


if __name__ == '__main__':
    main()
