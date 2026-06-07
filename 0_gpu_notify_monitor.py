#!/usr/bin/env python3
"""
GPU 狀態切換通知 daemon（純 nvidia-smi 監控）

行為：
  啟動         → 發一則「hi + 現在時間（東八區 UTC+8）」。
  GPU 開始運算 → 發一則「run」。
  GPU 停止     → 發一則「stop」。
  交替規則：發過 run 後下一則只會是 stop；發過 stop 後下一則只會是 run。
  → 每次狀態改變只發一則，不洗版。

Webhook：環境變數 GPU_NOTIFY_WEBHOOK 優先，否則用檔內寫死的值。
run_all_and_pack.sh 會在 run 開始時自動於背景啟動本監控、結束時停止。
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

import requests

# ============================ 設定區 ============================
# Webhook：環境變數 GPU_NOTIFY_WEBHOOK 優先，否則用下面寫死的。
WEBHOOK_URL    = os.environ.get("GPU_NOTIFY_WEBHOOK", "").strip() or \
    "https://discord.com/api/webhooks/1510510909347467314/r168thMxW_1f8f9Zdxhc1AUqzkT7QHZ9l3wKvAkwYw9iVAeKAlajVmwTHN47hEZ1ZD-4"
POLL_INTERVAL  = int(os.environ.get("GPU_NOTIFY_POLL", "300"))   # 每幾秒查一次 GPU
IDLE_THRESHOLD = int(os.environ.get("GPU_NOTIFY_IDLE", "5"))     # 使用率 >= 此值(%) 視為 running
GPU_INDICES    = None      # None = 監控全部；或指定 [0] / [0, 1]
MENTION        = os.environ.get("GPU_NOTIFY_MENTION", "")        # 例如 "@here"，留空不標記
# ===============================================================

COLOR_GREEN  = 0x2ECC71   # run
COLOR_RED    = 0xE74C3C   # stop
COLOR_BLUE   = 0x3498DB   # hi
TZ8 = timezone(timedelta(hours=8))   # 東八區
HOSTNAME = socket.gethostname()


def now_str():
    """本地 log 時間。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def now8_str():
    """東八區（UTC+8）時間字串。"""
    return datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S (UTC+8)")


def query_gpus():
    """回傳 [{'index': int, 'util': int, 'mem': int}, ...]；失敗時拋出例外。"""
    output = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        stderr=subprocess.STDOUT,
    )
    gpus = []
    for line in output.decode().strip().splitlines():
        if not line.strip():
            continue
        idx, util, mem = [x.strip() for x in line.split(",")]
        idx, util, mem = int(idx), int(util), int(mem)
        if GPU_INDICES is None or idx in GPU_INDICES:
            gpus.append({"index": idx, "util": util, "mem": mem})
    return gpus


def send_discord(title, desc, color=COLOR_BLUE):
    """發送 Discord embed；失敗只印 log，不讓 daemon 崩潰。"""
    if not WEBHOOK_URL or "XXX" in WEBHOOK_URL:
        print(f"[{now_str()}] [警告] GPU_NOTIFY_WEBHOOK 未設定，略過通知：{title}")
        return False
    payload = {
        "embeds": [{
            "title": title,
            "description": desc,
            "color": color,
            "footer": {"text": f"{HOSTNAME} · gpu_notify_monitor"},
            "timestamp": datetime.now().astimezone().isoformat(),
        }]
    }
    if MENTION:
        payload["content"] = MENTION
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        r.raise_for_status()
        print(f"[{now_str()}] [通知已送] {title}")
        return True
    except Exception as e:
        print(f"[{now_str()}] [通知失敗] {title} -> {e}")
        return False


def fmt_gpu(g):
    return f"GPU{g['index']} · 使用率 {g['util']}% · 記憶體 {g['mem']} MiB"


def run(idle_threshold, poll_interval):
    # 每張卡上一次「已通知」的狀態："running" / "stopped" / None（尚未通知過）
    last = {}

    # 啟動：發一則 hi + 東八區時間
    send_discord("👋 hi", f"GPU 監控已啟動\n時間：{now8_str()}", COLOR_BLUE)

    while True:
        try:
            gpus = query_gpus()
        except Exception as e:
            print(f"[{now_str()}] nvidia-smi 失敗：{e}")
            time.sleep(poll_interval)
            continue

        for g in gpus:
            i = g["index"]
            cur = "running" if g["util"] >= idle_threshold else "stopped"
            if cur != last.get(i):
                # 狀態改變才發，天然交替 run ↔ stop，不洗版
                if cur == "running":
                    send_discord(f"🟢 GPU{i} run", f"{fmt_gpu(g)}\n時間：{now8_str()}", COLOR_GREEN)
                else:
                    send_discord(f"🔴 GPU{i} stop", f"{fmt_gpu(g)}\n時間：{now8_str()}", COLOR_RED)
                last[i] = cur

        time.sleep(poll_interval)


def main():
    p = argparse.ArgumentParser(description="GPU 狀態切換通知（run / stop）")
    p.add_argument("--test", action="store_true", help="立刻送一則測試訊息後結束")
    p.add_argument("--idle-threshold", type=int, default=IDLE_THRESHOLD, help="使用率 >= 此值(%%)視為 running")
    p.add_argument("--poll", type=int, default=POLL_INTERVAL, help="每幾秒查一次")
    args = p.parse_args()

    if args.test:
        ok = send_discord(
            "✅ GPU 監控測試訊息",
            f"如果你看到這則訊息，代表 webhook 設定正確。\n主機：{HOSTNAME}\n時間：{now8_str()}",
            COLOR_BLUE,
        )
        sys.exit(0 if ok else 1)

    print(f"[{now_str()}] 啟動 GPU 監控："
          f"running 門檻 >={args.idle_threshold}%，每 {args.poll}s 查一次。Ctrl+C 結束。")
    try:
        run(args.idle_threshold, args.poll)
    except KeyboardInterrupt:
        print(f"\n[{now_str()}] 已停止監控。")


if __name__ == "__main__":
    main()
