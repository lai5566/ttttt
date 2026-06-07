#!/usr/bin/env python3
"""
增量上傳監看器：一有「完成且穩定」的結果就立刻上傳 R2（邊跑邊傳）。

運作方式
────────
定時掃描以下「結果單元」目錄，每個視為獨立可上傳單位：
  eval/results/*                          每個 cell 的評估輸出（小、最重要）
  methods/01_mla_gan_ours/output_*        MLA-GAN 訓練輸出（best_model.pth / json）
  methods/01_mla_gan_ours/generated_*     MLA-GAN 生成影像（大）
  methods/02_stylegan3/results_*          StyleGAN3 checkpoints
  methods/02_stylegan3/generated_*        StyleGAN3 生成影像（大）

對每個單元：
  - 若「穩定」（最近 WATCH_STABLE 秒內無任何檔案變動 = 該 cell 寫完了）
    且尚未上傳過 → tar（影像用 store 不壓縮、其餘 gzip）→ 呼叫 upload_to_r2.py
    上傳到 R2，key = 相對路徑 + 副檔名（保留資料夾結構）。
  - 已上傳的記在 .r2_upload_state.json，不重複傳。

不碰正在跑的 orchestrator；可與 run_all_and_pack.sh 同時跑。

環境變數（憑證同 upload_to_r2.py，建議 source r2_env.sh）：
  R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / (R2_SECRET_ACCESS_KEY 或 R2_TOKEN) / R2_BUCKET
  R2_PUBLIC_BASE（選填，印下載連結）
  WATCH_POLL    掃描間隔秒（預設 120）
  WATCH_STABLE  判定「寫完」需幾秒無變動（預設 300）
  WATCH_ONCE=1  只掃一輪就結束（給結尾做最後一次補掃用）

用法：
  source r2_env.sh
  nohup python3 watch_and_upload.py > logs/watch_upload.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = os.path.join(ROOT, '.r2_upload_state.json')
POLL = int(os.environ.get('WATCH_POLL', '120'))
STABLE = int(os.environ.get('WATCH_STABLE', '300'))
ONCE = os.environ.get('WATCH_ONCE', '0') == '1'

# (相對 glob, 是否為大影像目錄)
UNIT_GLOBS = [
    ('eval/results/*', False),
    ('methods/01_mla_gan_ours/output_*', False),
    ('methods/01_mla_gan_ours/generated_*', True),
    ('methods/02_stylegan3/results_*', False),
    ('methods/02_stylegan3/generated_*', True),
]


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_state():
    if os.path.isfile(STATE_PATH):
        try:
            return json.load(open(STATE_PATH))
        except Exception:
            return {}
    return {}


def save_state(state):
    tmp = STATE_PATH + '.tmp'
    json.dump(state, open(tmp, 'w'), indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_PATH)


def iter_units():
    import glob
    for pat, is_img in UNIT_GLOBS:
        for abs_path in glob.glob(os.path.join(ROOT, pat)):
            if os.path.isdir(abs_path):
                rel = os.path.relpath(abs_path, ROOT)
                yield rel, abs_path, is_img


def max_mtime(path):
    """目錄內所有檔案的最新 mtime；空目錄回傳 0。"""
    newest = 0.0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                m = os.path.getmtime(os.path.join(dirpath, f))
                if m > newest:
                    newest = m
            except OSError:
                pass
    return newest


def has_files(path):
    for _dp, _d, files in os.walk(path):
        if files:
            return True
    return False


def upload(rel, abs_path, is_img, state):
    ext = '.tar' if is_img else '.tar.gz'
    key = rel + ext                                   # 保留資料夾結構當 R2 key
    safe = rel.replace('/', '__') + ext
    tarpath = os.path.join('/tmp', safe)
    flags = '-cf' if is_img else '-czf'               # 影像 store、其餘 gzip
    log(f"打包 {rel} → {tarpath}")
    rc = subprocess.call(['tar', flags, tarpath, '-C', ROOT, rel])
    if rc != 0:
        log(f"[WARN] tar 失敗 rc={rc}：{rel}（下輪重試）")
        return False
    env = dict(os.environ, R2_KEY=key)
    log(f"上傳 {rel} → R2 key={key}")
    rc = subprocess.call([sys.executable, os.path.join(ROOT, 'upload_to_r2.py'), tarpath], env=env)
    try:
        os.remove(tarpath)
    except OSError:
        pass
    if rc != 0:
        log(f"[WARN] 上傳失敗 rc={rc}：{rel}（下輪重試）")
        return False
    state[rel] = {'uploaded_at': datetime.now().isoformat(), 'key': key}
    save_state(state)
    log(f"✅ 已上傳並記錄：{rel}")
    return True


def sweep(state):
    now = time.time()
    for rel, abs_path, is_img in iter_units():
        if rel in state:
            continue
        if not has_files(abs_path):
            continue
        idle = now - max_mtime(abs_path)
        if idle < STABLE:
            log(f"… {rel} 仍在寫入（{int(idle)}s 前才變動，需 ≥{STABLE}s），暫不傳")
            continue
        upload(rel, abs_path, is_img, state)


def main():
    # 預檢憑證（缺就直接退出，免得空轉）
    if not (os.environ.get('R2_SECRET_ACCESS_KEY') or os.environ.get('R2_TOKEN')) \
            or not os.environ.get('R2_ACCESS_KEY_ID'):
        log("[ERR] 未設定 R2 憑證（R2_ACCESS_KEY_ID + R2_TOKEN/R2_SECRET_ACCESS_KEY）。"
            "請先 `source r2_env.sh` 再啟動。")
        sys.exit(3)

    log(f"啟動增量上傳監看：每 {POLL}s 掃一次，穩定門檻 {STABLE}s，once={ONCE}")
    state = load_state()
    if state:
        log(f"已上傳記錄 {len(state)} 筆（不重複傳）")
    while True:
        try:
            sweep(state)
        except Exception as e:  # noqa
            log(f"[WARN] 掃描出錯：{e}")
        if ONCE:
            log("WATCH_ONCE=1：補掃完成，結束。")
            break
        time.sleep(POLL)


if __name__ == '__main__':
    main()
