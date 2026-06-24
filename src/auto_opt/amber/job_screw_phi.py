#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python -m auto_opt.amber.job_screw_phi --auto-dir runs/DNTT_test --monomer-name DNTT --num-nodes 40
python -m auto_opt.amber.job_screw_phi --auto-dir runs/DNTT_test --monomer-name DNTT --num-nodes 40 --split-by phi
python -m auto_opt.amber.job_screw_phi --auto-dir runs/DNTT_test --monomer-name DNTT --num-nodes 40 --split-by alpha
"""
import os

import pandas as pd
import argparse
import subprocess
from pathlib import Path

MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}

VALID_SPLIT_KEYS = ['alpha', 'beta', 'z', 'phi']


def init_process(args):
    auto_dir_root = Path(args.auto_dir).resolve()
    split_by = args.split_by

    df_init = pd.read_csv(auto_dir_root / 'step1_init_params.csv')

    if split_by not in df_init.columns:
        raise ValueError(f"--split-by '{split_by}' が step1_init_params.csv に存在しません。"
                         f" 利用可能: {list(df_init.columns)}")

    split_vals = sorted(df_init[split_by].unique())
    print(f"Split by '{split_by}': {split_vals}")

    for i, val in enumerate(split_vals):
        machine_type = 1 if i % 2 == 0 else 2
        spec = MACHINE_SPEC[machine_type]
        queue = spec["queue"]
        nproc = spec["nproc"]

        subdir = auto_dir_root / f"{split_by}_{val}"
        subdir.mkdir(parents=True, exist_ok=True)

        df_sub = df_init[df_init[split_by] == val]
        if df_sub.empty:
            continue

        df_sub.to_csv(subdir / 'step1_init_params.csv', index=False)

        cmd = (
            'python -m auto_opt.amber.driver_screw_phi '
            f'--auto-dir {str(subdir)} '
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

        subprocess.run(['qsub', str(job_path)])
        print(f"  qsub: {job_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, default='DNTT')
    parser.add_argument('--num-nodes', type=int, default=10)
    parser.add_argument('--split-by', type=str, default='phi',
                        choices=VALID_SPLIT_KEYS,
                        help='並列化するパラメータ (default: phi)')
    args = parser.parse_args()

    print("----main process----")
    init_process(args)
    print("----finish process----")
