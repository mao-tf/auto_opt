#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python -m auto_opt.amber.job_screw_phi --auto-dir runs/DNTT_screw --monomer-name DNTT
python -m auto_opt.amber.job_screw_phi --auto-dir runs/DNTT_screw --monomer-name DNTT --split-by alpha

split-by で指定したパラメータごとにサブディレクトリを作成し、
空きノードを探して qsub で投入する。
キュー仕様は ~/.auto_opt.yaml から読み込む（なければ組み込みデフォルト）。
"""

import subprocess
import time
import pandas as pd
import argparse
from pathlib import Path

from auto_opt.cluster import wait_for_free_node, make_job_script

VALID_SPLIT_KEYS = ['alpha', 'beta', 'z', 'phi']


def init_process(args):
    auto_dir_root = Path(args.auto_dir).resolve()
    split_by = args.split_by

    df_init = pd.read_csv(auto_dir_root / 'step1_init_params.csv')

    if split_by not in df_init.columns:
        raise ValueError(
            f"--split-by '{split_by}' が step1_init_params.csv に存在しません。"
            f" 利用可能: {list(df_init.columns)}"
        )

    split_vals = sorted(df_init[split_by].unique())
    print(f"Split by '{split_by}': {split_vals}")

    prefer_queue = None

    for i, val in enumerate(split_vals):
        df_sub = df_init[df_init[split_by] == val]
        if df_sub.empty:
            continue

        subdir = auto_dir_root / f"{split_by}_{val}"
        subdir.mkdir(parents=True, exist_ok=True)
        df_sub.to_csv(subdir / 'step1_init_params.csv', index=False)

        qname, qi, num_nodes = wait_for_free_node(
            prefer_queue=prefer_queue,
            is_test=args.isTest,
            test_index=i,
        )

        cmd = (
            'python -m auto_opt.amber.driver_screw_phi '
            f'--auto-dir {subdir} '
            f'--monomer-name {args.monomer_name} '
            f'--num-nodes {num_nodes}'
        )
        if args.isTest:
            cmd += ' --isTest'

        script = make_job_script(cmd, queue=qname, queue_instance=qi,
                                 job_name=f"{args.monomer_name}_{split_by}{val}")
        job_path = subdir / 'job.sh'
        job_path.write_text(script)

        subprocess.run(['qsub', str(job_path)])
        print(f"  {split_by}={val} → {qi}  (num_nodes={num_nodes})")

        prefer_queue = [q for q in ['gr1.q', 'gr2.q'] if q != qname][0]
        if not args.isTest:
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, default='DNTT')
    parser.add_argument('--split-by', type=str, default='phi',
                        choices=VALID_SPLIT_KEYS,
                        help='並列化するパラメータ (default: phi)')
    args = parser.parse_args()

    print('----main process----')
    init_process(args)
    print('----finish process----')
