#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 (Screw Search) のジョブ投入スクリプト
指定されたディレクトリ内の step1_init_params.csv を読み込み、
指定された数（num_splits）に均等分割して、サブディレクトリを作成しジョブを分散投入します。

qstat -f でノードの空き状況をリアルタイムに確認し、
空いているノードに順次投入します。
"""

import os
import re
import time
import pandas as pd
import numpy as np
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# cmdell81 のキュー仕様
MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}
MAX_CONCURRENT = 6   # 同時実行ジョブ数の上限（全キュー合計）
POLL_INTERVAL  = 30  # 空き確認の間隔（秒）
USER = os.environ.get("USER", "miyoshi")


# ─── ノード空き確認 ────────────────────────────────────────────────────

def get_my_job_count() -> int:
    """自分が投入したジョブ数（実行中 + 待機中）を返す。"""
    try:
        r = subprocess.run(["qstat", "-u", USER],
                           capture_output=True, text=True, timeout=15)
        return sum(1 for ln in r.stdout.splitlines()
                   if re.match(r'^\s*\d+', ln))
    except Exception:
        return 0


def get_free_queue_instances() -> Dict[str, List[str]]:
    """qstat -f を解析して空き（used==0）のキューインスタンスを返す。"""
    try:
        r = subprocess.run(["qstat", "-f", "-u", "*"],
                           capture_output=True, text=True, timeout=15)
        output = r.stdout
    except Exception:
        return {"gr1.q": [], "gr2.q": []}

    free: Dict[str, List[str]] = {"gr1.q": [], "gr2.q": []}
    # 例: "gr1.q@cmdell01  BIP  0/0/40  0.00  lx-amd64"
    pat = re.compile(r'^((gr[12]\.q)@\S+)\s+\S+\s+\d+/(\d+)/\d+', re.MULTILINE)
    for m in pat.finditer(output):
        qi, qname, used = m.group(1), m.group(2), int(m.group(3))
        if used == 0 and qname in free:
            free[qname].append(qi)
    return free


def pick_free_instance(
    free_by_queue: Dict[str, List[str]],
    prefer_queue: Optional[str] = None,
) -> Optional[tuple]:
    """空きキューインスタンスを1つ選んで (qname, qi) を返す。"""
    all_queues = ["gr1.q", "gr2.q"]
    order = all_queues if not prefer_queue else \
            [prefer_queue] + [q for q in all_queues if q != prefer_queue]
    for qname in order:
        if free_by_queue.get(qname):
            return qname, free_by_queue[qname][0]
    return None


# ─── メイン処理 ────────────────────────────────────────────────────────

def init_process(args):
    auto_dir_root = Path(args.auto_dir).resolve()
    params_csv = auto_dir_root / 'step1_init_params.csv'

    if not params_csv.exists():
        raise FileNotFoundError(f"初期パラメータファイルが見つかりません: {params_csv}")

    df_init = pd.read_csv(params_csv)
    total_rows = len(df_init)

    if total_rows == 0:
        print("CSVが空です。処理を終了します。")
        return

    num_splits = args.num_splits
    print(f"Total rows: {total_rows}")
    print(f"Splitting into {num_splits} chunks...")

    df_chunks = [c for c in np.array_split(df_init, num_splits) if not c.empty]

    prefer_queue = None

    for i, df_chunk in enumerate(df_chunks):
        dir_name = f'split_{i}'
        subdir = auto_dir_root / dir_name
        subdir.mkdir(parents=True, exist_ok=True)

        # すでに結果ファイルがあるsplitはスキップ
        if (subdir / 'step1.csv').exists():
            print(f"  {dir_name}: step1.csv 既存 → スキップ")
            continue

        df_chunk.to_csv(subdir / 'step1_init_params.csv', index=False)

        # 自分のジョブ数が上限未満 かつ 空きノードがある まで待つ
        while True:
            if args.isTest:
                qname = "gr1.q" if i % 2 == 0 else "gr2.q"
                qi = f"{qname}@node{i+1:02d}"
                break

            n_jobs = get_my_job_count()
            if n_jobs >= MAX_CONCURRENT:
                print(f"  実行中ジョブ数={n_jobs}（上限{MAX_CONCURRENT}）。{POLL_INTERVAL}秒後に再確認...")
                time.sleep(POLL_INTERVAL)
                continue

            free_by_queue = get_free_queue_instances()
            pick = pick_free_instance(free_by_queue, prefer_queue)
            if pick:
                qname, qi = pick
                break

            print(f"  空きノードなし。{POLL_INTERVAL}秒後に再確認...")
            time.sleep(POLL_INTERVAL)

        nproc = MACHINE_SPEC[1 if qname == "gr1.q" else 2]["nproc"]
        # OSの余裕を2コア確保した上で並列数を決める
        actual_num_nodes = nproc - 2

        cmd = (
            'python -m auto_opt.stacking.driver_stacking_screw_phi '
            f'--auto-dir {str(subdir)} '
            f'--monomer-name {args.monomer_name} '
            f'--num-nodes {actual_num_nodes} '  # gr1: 38, gr2: 50
        )
        if args.isTest:
            cmd += ' --isTest'

        job_name = f"{args.monomer_name}_{i}"
        job_lines = [
            "#!/bin/sh\n",
            f"#$ -N {job_name}\n",
            "#$ -S /bin/sh\n",
            "#$ -cwd\n",
            "#$ -V\n",
            f"#$ -q {qi}\n",          # 空きノードを直接指定
            f"#$ -pe OpenMP {nproc}\n",
            f"#$ -o {subdir}/job.sh.o{i}\n",
            f"#$ -e {subdir}/job.sh.e{i}\n",
            "\n",
            "hostname\n",
            "\n",
            cmd + "\n",
            "\n",
        ]

        job_path = subdir / 'job.sh'
        with open(job_path, 'w') as f:
            f.writelines(job_lines)

        if not args.isTest:
            subprocess.run(['qsub', 'job.sh'], cwd=str(subdir))
            prefer_queue = "gr2.q" if qname == "gr1.q" else "gr1.q"
            time.sleep(5)  # qstatに反映されるまで少し待つ

        print(f"  {dir_name} ({len(df_chunk)}行) → {qi}  (num_nodes={actual_num_nodes})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 1 Screw Job Dispatcher (Split by rows)')
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, default='DNTT')
    parser.add_argument('--num-nodes', type=int, default=10)
    parser.add_argument('--num-splits', type=int, default=6,
                        help='CSVを何等分してジョブを投げるか（デフォルト: 6）')
    args = parser.parse_args()

    init_process(args)