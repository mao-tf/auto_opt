#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 (Screw Search) のジョブ投入スクリプト

step1_init_params.csv を num_splits 等分し、空きノードを探して qsub で投入する。
キュー仕様は ~/.auto_opt.yaml から読み込む（なければ組み込みデフォルト）。
"""

import subprocess
import time
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from auto_opt.cluster import wait_for_free_node, make_job_script


def init_process(args):
    auto_dir_root = Path(args.auto_dir).resolve()
    params_csv = auto_dir_root / 'step1_init_params.csv'

    if not params_csv.exists():
        raise FileNotFoundError(f"初期パラメータファイルが見つかりません: {params_csv}")

    df_init = pd.read_csv(params_csv)
    if df_init.empty:
        print("CSV が空です。処理を終了します。")
        return

    num_splits = args.num_splits
    print(f"Total rows: {len(df_init)}")
    print(f"Splitting into {num_splits} chunks...")

    df_chunks = [c for c in np.array_split(df_init, num_splits) if not c.empty]

    prefer_queue = None

    for i, df_chunk in enumerate(df_chunks):
        dir_name = f'split_{i}'
        subdir = auto_dir_root / dir_name
        subdir.mkdir(parents=True, exist_ok=True)

        if (subdir / 'step1.csv').exists():
            print(f"  {dir_name}: step1.csv 既存 → スキップ")
            continue

        df_chunk.to_csv(subdir / 'step1_init_params.csv', index=False)

        qname, qi, num_nodes = wait_for_free_node(
            prefer_queue=prefer_queue,
            is_test=args.isTest,
            test_index=i,
        )

        cmd = (
            'python -m auto_opt.stacking.driver_stacking_screw '
            f'--auto-dir {subdir} '
            f'--monomer-name {args.monomer_name} '
            f'--num-nodes {num_nodes}'
        )
        if args.isTest:
            cmd += ' --isTest'

        script = make_job_script(
            cmd, queue=qname, queue_instance=qi,
            job_name=f"{args.monomer_name}_{i}",
            stdout=str(subdir / f'job.sh.o{i}'),
            stderr=str(subdir / f'job.sh.e{i}'),
        )
        job_path = subdir / 'job.sh'
        job_path.write_text(script)

        if not args.isTest:
            subprocess.run(['qsub', 'job.sh'], cwd=str(subdir))
            prefer_queue = [q for q in ['gr1.q', 'gr2.q'] if q != qname][0]
            time.sleep(5)

        print(f"  {dir_name} ({len(df_chunk)}行) → {qi}  (num_nodes={num_nodes})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 1 Screw Job Dispatcher (Split by rows)'
    )
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, default='ANT')
    parser.add_argument('--num-splits', type=int, default=6,
                        help='CSV を何等分してジョブを投げるか（デフォルト: 6）')
    args = parser.parse_args()

    init_process(args)
