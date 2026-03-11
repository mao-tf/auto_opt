#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 (Intralayer Search) のジョブ投入スクリプト
指定されたディレクトリ内の step1_init_params.csv を読み込み、
alpha1 (下層分子の傾き) ごとにサブディレクトリを作成して計算ジョブを分散投入します。

python -m auto_opt.stacking.job_stacking --auto-dir runs/ANT_stacking_test6 --monomer-name ANT --num-nodes 10
"""

import os
import pandas as pd
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

    # ★修正1: 'alpha' ではなく 'alpha1' をキーにする
    if 'alpha1' not in df_init.columns:
         raise ValueError(f"{params_csv} に 'alpha1' カラムが含まれていません。")
         
    alpha_list = sorted(df_init['alpha1'].unique())
    print(f"Detected alpha1 values: {alpha_list}")

    for i, alpha in enumerate(alpha_list):
        # 交互にキュー（マシンタイプ）を振り分ける
        machine_type = 1 if i % 2 == 0 else 2
            
        spec = MACHINE_SPEC[machine_type]
        queue = spec["queue"]
        nproc = spec["nproc"]

        # ディレクトリ名は alpha1 の値
        dir_name = f'alpha_{alpha}'
        subdir = auto_dir_root / dir_name
        subdir.mkdir(parents=True, exist_ok=True)

        # この alpha1 の行だけ抜き出して保存
        df_alpha = df_init[df_init['alpha1'] == alpha]
        
        if df_alpha.empty:
            continue

        df_alpha.to_csv(subdir / 'step1_init_params.csv', index=False)
        auto_dir_for_driver = str(subdir)

        # ★修正2: 呼び出す先を driver_stacking_v2 に変更
        cmd = (
            'python -m auto_opt.stacking.driver_stacking_v2 '
            f'--auto-dir {auto_dir_for_driver} '
            f'--monomer-name {args.monomer_name} '
            f'--num-nodes {args.num_nodes} '
            f'--max-2 3' # ← 必要に応じて引数を追加
        )
        if args.isTest:
            cmd += ' --isTest'

        job_name = f"{args.monomer_name}_{alpha}"
        out_path = subdir / f"job.sh.o{alpha}" 
        err_path = subdir / f"job.sh.e{alpha}"

        job_lines = [
            "#!/bin/sh\n",
            f"#$ -N {job_name}\n",
            "#$ -S /bin/sh\n",
            "#$ -cwd\n",
            "#$ -V\n",
            f"#$ -q {queue}\n",
            f"#$ -pe OpenMP {nproc}\n", # ← ここで52コアや40コアを確保！
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

        print(f"Submitting job for alpha1={alpha} into {queue}...")
        subprocess.run(['qsub', 'job.sh'], cwd=str(subdir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 1 Job Dispatcher')
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, default='ANT')
    parser.add_argument('--num-nodes', type=int, default=10)
    args = parser.parse_args()

    init_process(args)