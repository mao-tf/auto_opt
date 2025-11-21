#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python -m auto_opt.amber.job --auto-dir runs/PFA_test --monomer-name PFA --num-nodes 10 
"""
import os
os.environ['HOME'] = '/data/group1/z40145w'

import pandas as pd
import argparse
import subprocess
import numpy as np
from pathlib import Path

# cmdell81 のキュー仕様
MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}


def init_process(args):
    """
    args.auto_dir :
        step1_init_params.csv が置いてあるディレクトリ
        （絶対パスでも相対パスでもOK）
    """

    # ベースディレクトリ（絶対パスにそろえる）
    auto_dir_root = Path(args.auto_dir).resolve()

    # まとめてある init を読む
    df_init = pd.read_csv(auto_dir_root / 'step1_init_params.csv')

    # alpha ごとにサブディレクトリを作る
    phi_list = sorted(df_init['phi'].unique())
    # もしくは df_init["alpha"].unique() でもよい

    for i, phi in enumerate(phi_list):
        if i%2==0: 
            machine_type=1
        else: 
            machine_type=2
        spec=MACHINE_SPEC[machine_type]
        queue = spec["queue"]
        nproc = spec["nproc"]
        dir_name = f'{phi}'
        subdir = auto_dir_root / dir_name
        subdir.mkdir(parents=True, exist_ok=True)

        # この alpha の行だけ抜き出して保存
        df_phi = df_init[df_init['phi'] == phi]
        if df_phi.empty:
            # その alpha の初期点が無いならジョブを投げない
            continue

        df_phi.to_csv(subdir / 'step1_init_params.csv', index=False)

        # driver_gene に渡す auto-dir は「その alpha のディレクトリ」
        auto_dir_for_driver = str(subdir)

        # 実際に投げるコマンドを組み立てる
        cmd = (
            'python -m auto_opt.amber.driver_gene '
            f'--auto-dir {auto_dir_for_driver} '
            f'--monomer-name {args.monomer_name} '
            f'--num-nodes {args.num_nodes}'
        )
        if args.isTest:
            cmd += ' --isTest'

        job_lines = [
            "#!/bin/sh\n",
            "#$ -S /bin/sh\n",
            "#$ -cwd\n",
            "#$ -V\n",
            f"#$ -q {queue}\n",
            f"#$ -pe OpenMP {nproc}\n",
            "\n",
            "hostname\n",
            "\n",
            cmd + "\n",
            "\n",
            "#sleep 5\n",
        ]

        job_path = subdir / 'job.sh'
        with open(job_path, 'w') as f:
            f.writelines(job_lines)

        # ジョブ投げ
        subprocess.run(['qsub', str(job_path)])


def update_value_in_df(df, index, key, value):
    df.loc[index, key] = value
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--isTest', action='store_true',
        help='driver_gene に --isTest を付けてテスト実行にする',
    )
    parser.add_argument(
        '--auto-dir', type=str, required=True,
        help='step1_init_params.csv があるディレクトリ '
             '(例: /home/miyoshi/Working/auto_opt/runs/PFA_test)',
    )
    parser.add_argument(
        '--monomer-name', type=str, default='pentacene',
        help='driver_gene に渡す monomer 名',
    )
    parser.add_argument(
        '--num-nodes', type=int, default=10,
        help='driver_gene の --num-nodes',
    )
    
    args = parser.parse_args()

    print("----main process----")
    init_process(args)
    print("----finish process----")
