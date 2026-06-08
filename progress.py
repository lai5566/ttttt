#!/usr/bin/env python3
"""
整體進度與 ETA 估算（給「要不要續費」決策用）。

解析兩個子排程器的 master.log：
  methods/01_mla_gan_ours/logs/master.log   (v3/kfold/gateA × IR × BS)
  methods/02_stylegan3/logs/master.log      (IR × BS)
計算：已完成 cell 數 + 正在跑 cell 的 epoch 進度 → 整體完成度 → 吞吐 → ETA → 預估完成時刻。

用法：
  python3 progress.py            # 印一次快照
  watch -n 60 python3 progress.py   # 每分鐘刷新
"""

import os
import re
import glob
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))

M1_LOG = os.path.join(ROOT, 'methods/01_mla_gan_ours/logs/master.log')
M2_LOG = os.path.join(ROOT, 'methods/02_stylegan3/logs/master.log')
M1_CELLS = os.path.join(ROOT, 'methods/01_mla_gan_ours/logs')
RUN_ALL = os.path.join(ROOT, 'logs/run_all.log')

RUN_FRAC_CAP = 0.9   # 訓練 100% 但還要 generate+eval，故進行中 cell 最多算 0.9，ETA 偏保守


def read(path):
    try:
        return open(path, encoding='utf-8', errors='replace').read()
    except OSError:
        return ''


def start_time():
    """從 run_all.log 取整體開始時間（完整日期）。"""
    m = re.search(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*★ 開始 run_all_and_pack', read(RUN_ALL))
    if m:
        return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    return None


def parse_master(log_text, is_m1):
    """回傳 (total_cells, done_tags:set, running_tags:set)。"""
    if not log_text:
        return 0, set(), set()
    # 矩陣行
    total = 0
    if is_m1:
        mm = re.search(r'METHODS=(.*?) IRS=(.*?) BSS=(.*)', log_text)
        if mm:
            methods = mm.group(1).split()
            irs = mm.group(2).split()
            bss = mm.group(3).split()
            total = len(methods) * len(irs) * len(bss)
    else:
        mm = re.search(r'IRS=(.*?) BSS=(.*)', log_text)
        if mm:
            total = len(mm.group(1).split()) * len(mm.group(2).split())
    done = set(re.findall(r'■ DONE (\S+)', log_text))
    started = set(re.findall(r'▶ (\S+)', log_text))
    running = started - done
    return total, done, running


def cell_epoch_frac(tag):
    """讀 method01 的 cell_<tag>.log，回傳最後 Epoch X/N 的 X/N；無則 0。"""
    txt = read(os.path.join(M1_CELLS, f'cell_{tag}.log'))
    if not txt:
        return 0.0
    eps = re.findall(r'Epoch (\d+)/(\d+)', txt)
    if not eps:
        return 0.0
    cur, tot = eps[-1]
    tot = int(tot) or 1
    return min(int(cur) / tot, 1.0)


def fmt_dur(sec):
    sec = max(int(sec), 0)
    h, rem = divmod(sec, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m:02d}m"


def bar(frac, n=24):
    f = max(0.0, min(frac, 1.0))
    k = int(round(n * f))
    return '█' * k + '░' * (n - k)


def main():
    start = start_time()
    now = datetime.now()

    # run_all 是否包含方法02
    run_all_txt = read(RUN_ALL)
    will_run_02 = bool(re.search(r'METHODS=.*\b02\b', run_all_txt)) or os.path.exists(M2_LOG)

    t1, done1, run1 = parse_master(read(M1_LOG), True)
    t2, done2, run2 = parse_master(read(M2_LOG), False)
    if t2 == 0 and will_run_02:
        t2 = 6   # 方法02 尚未開始：用預設矩陣 (3 IR × 2 BS)

    total = t1 + t2
    done_n = len(done1) + len(done2)

    # 進行中 cell 的部分完成度
    run_units = 0.0
    run_detail = []
    for tag in sorted(run1):
        fr = min(cell_epoch_frac(tag), RUN_FRAC_CAP)
        run_units += fr
        run_detail.append((f"01/{tag}", fr))
    for tag in sorted(run2):
        run_units += 0.5   # 方法02 step-based，粗估半完成
        run_detail.append((f"02/{tag}", 0.5))

    progress = len(done1) + len(done2) + run_units
    running_n = len(run1) + len(run2)
    pending_n = max(total - done_n - running_n, 0)

    print("═══════════════ 整體進度（MLA-GAN 全矩陣）═══════════════")
    if total == 0:
        print("（還沒看到 master.log，訓練可能剛起步或路徑不同）")
        return
    frac = progress / total
    print(f"[{bar(frac)}] {frac*100:5.1f}%   ({progress:.1f} / {total} cells)")
    print(f"已完成 {done_n} | 進行中 {running_n} | 待跑 {pending_n}")
    if run_detail:
        print("進行中明細：" + ", ".join(f"{t} {fr*100:.0f}%" for t, fr in run_detail))

    if start is None:
        print("\n（找不到 run_all.log 的開始時間，無法估 ETA）")
        return
    elapsed = (now - start).total_seconds()
    print(f"\n已耗時：{fmt_dur(elapsed)}  (開始於 {start:%Y-%m-%d %H:%M})")

    if progress <= 0.01:
        print("ETA：還沒有可估的進度，請過幾分鐘再看。")
        return
    rate = progress / elapsed            # cells/sec
    remaining = total - progress
    eta_sec = remaining / rate if rate > 0 else 0
    finish = now + timedelta(seconds=eta_sec)
    print(f"吞吐：{rate*3600:.2f} cells/hr")
    print(f"預估剩餘：約 {fmt_dur(eta_sec)}")
    print(f"預估完成：{finish:%Y-%m-%d %H:%M}（伺服器本地時間）")
    print("\n💡 續費建議：把租期確認到上面「預估完成」之後（含緩衝）。"
          "\n   註：進行中 cell 以訓練 epoch 估、且保守上限 90%，實際多半略早完成。")


if __name__ == '__main__':
    main()
