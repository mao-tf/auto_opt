#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python -m auto_opt.amber.job_screw_phi --auto-dir runs/DNTT_screw --monomer-name DNTT

step1_init_params.csv を空きノード数で均等分割し、各ノードに qsub で投入する。
キュー仕様は ~/.auto_opt.yaml から読み込む（なければ組み込みデフォルト）。
"""

import subprocess
import time
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from auto_opt.cluster import (
    load_env, get_free_queue_instances, wait_for_free_node, make_job_script
)


def _n_splits(df: pd.DataFrame, is_test: bool) -> int:
    cfg = load_env()
    n_rows = len(df)

    if is_test:
        return min(3, n_rows)

    free = get_free_queue_instances()
    n_free = sum(len(v) for v in free.values())
    n_cap = cfg.get('max_concurrent_jobs', 6)
    n = n_free if n_free > 0 else n_cap
    return min(n, n_rows)


def init_process(args):
    auto_dir_root = Path(args.auto_dir).resolve()
    df_init = pd.read_csv(auto_dir_root / 'step1_init_params.csv')

    n = _n_splits(df_init, args.isTest)
    df_chunks = [c for c in np.array_split(df_init, n) if not c.empty]
    print(f"  {len(df_init)} 点を {len(df_chunks)} ノードに分割して投入します")

    prefer_queue = None

    for i, df_chunk in enumerate(df_chunks):
        subdir = auto_dir_root / f'split_{i}'
        subdir.mkdir(parents=True, exist_ok=True)

        if (subdir / 'job.sh').exists():
            print(f"  split_{i} → スキップ（既に投入済み）")
            continue

        df_chunk.to_csv(subdir / 'step1_init_params.csv', index=False)

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
                                 job_name=f"{args.monomer_name}_split{i}",
                                 stdout=str(subdir / 'job.sh.o'),
                                 stderr=str(subdir / 'job.sh.e'))
        job_path = subdir / 'job.sh'
        job_path.write_text(script)

        subprocess.run(['qsub', 'job.sh'], cwd=str(subdir))
        print(f"  split_{i} ({len(df_chunk)}点) → {qi}  (num_nodes={num_nodes})")

        prefer_queue = [q for q in ['gr1.q', 'gr2.q'] if q != qname][0]
        if not args.isTest:
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, default='DNTT')
    args = parser.parse_args()

    print('----main process----')
    init_process(args)
    print('----finish process----')
