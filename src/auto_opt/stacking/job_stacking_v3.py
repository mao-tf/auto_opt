#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 (Intralayer Search) のジョブ投入スクリプト
指定されたディレクトリ内の step1_init_params.csv を読み込み、
指定された数（num_splits）に均等分割して、サブディレクトリを作成しジョブを分散投入します。
"""

import os
import pandas as pd
import numpy as np
import argparse
import subprocess
from pathlib import Path

# cmdell81 のキュー仕様
MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}

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

    # ★修正ポイント: np.array_split を使って DataFrame を綺麗な等分リストに分割
    df_chunks = np.array_split(df_init, num_splits)

    for i, df_chunk in enumerate(df_chunks):
        if df_chunk.empty:
            continue
            
        # 交互にキュー（マシンタイプ）を振り分ける
        machine_type = 1 if i % 2 == 0 else 2
            
        spec = MACHINE_SPEC[machine_type]
        queue = spec["queue"]
        nproc = spec["nproc"]

        # ディレクトリ名は split_0, split_1 ... と連番にする
        dir_name = f'split_{i}'
        subdir = auto_dir_root / dir_name
        subdir.mkdir(parents=True, exist_ok=True)

        # 分割された数行〜数百行のデータを保存
        df_chunk.to_csv(subdir / 'step1_init_params.csv', index=False)
        auto_dir_for_driver = str(subdir)

        # 確保したコア数 (nproc) から、OSを動かす余裕(2コア)を引いた数を同時実行数にする
        actual_num_nodes = nproc - 2 

        # driver_stacking_v3 を呼び出すコマンド
        cmd = (
            'python -m auto_opt.stacking.driver_stacking_v3 '
            f'--auto-dir {auto_dir_for_driver} '
            f'--monomer-name {args.monomer_name} '
            f'--num-nodes {actual_num_nodes} ' # ← ノードに合わせて自動で 38 か 50 になる！
        )
        
        if args.isTest:
            cmd += ' --isTest'

        job_name = f"{args.monomer_name}_{i}"
        out_path = subdir / f"job.sh.o{i}" 
        err_path = subdir / f"job.sh.e{i}"

        job_lines = [
            "#!/bin/sh\n",
            f"#$ -N {job_name}\n",
            "#$ -S /bin/sh\n",
            "#$ -cwd\n",
            "#$ -V\n",
            f"#$ -q {queue}\n",
            f"#$ -pe OpenMP {nproc}\n",
            f"#$ -o {out_path}\n", 
            f"#$ -e {err_path}\n", 
            "\n",
            "hostname\n",
            "\n",
            cmd + "\n",
            "\n",
        ]

        job_path = subdir / 'job.sh'
        with open(job_path, 'w') as f:
            f.writelines(job_lines)

        print(f"Submitting job for {dir_name} (Rows: {len(df_chunk)}) into {queue}...")
        subprocess.run(['qsub', 'job.sh'], cwd=str(subdir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 1 Job Dispatcher (Split by rows)')
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, default='ANT')
    parser.add_argument('--num-nodes', type=int, default=10)
    # ★追加: 何分割するかをコマンドラインから指定できるようにしました（デフォルト6）
    parser.add_argument('--num-splits', type=int, default=6, help='CSVを何等分してジョブを投げるか')
    args = parser.parse_args()

    init_process(args)