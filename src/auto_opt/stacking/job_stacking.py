#!/usr/bin/env python3
"""
スタッキング計算ジョブ投入スクリプト (glide / screw 共通)

step1_init_params.csv を空きノード数に分割して qsub で投入する。
キュー設定は ~/.auto_opt.yaml から読み込む。

Usage:
  python -m auto_opt.stacking.job_stacking \
      --auto-dir runs/BTBT_screw_stacking \
      --monomer-name BTBT \
      --symmetry screw
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from auto_opt.cluster import (
    load_env, wait_for_free_node, make_job_script,
)


def init_process(args):
    auto_dir_root = Path(args.auto_dir).resolve()
    params_csv    = auto_dir_root / 'step1_init_params.csv'

    if not params_csv.exists():
        raise FileNotFoundError(f"初期パラメータファイルが見つかりません: {params_csv}")

    df_init = pd.read_csv(params_csv)
    if df_init.empty:
        print("CSV が空です。終了します。")
        return

    cfg        = load_env()
    n_cap      = cfg.get('max_concurrent_jobs', 6)
    n_splits   = args.num_splits if args.num_splits else min(n_cap, len(df_init))
    df_chunks  = [c for c in np.array_split(df_init, n_splits) if not c.empty]

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
            f'python -m auto_opt.stacking.driver_stacking '
            f'--auto-dir {subdir} '
            f'--monomer-name {args.monomer_name} '
            f'--symmetry {args.symmetry} '
            f'--num-nodes {num_nodes}'
        )
        if args.isTest:
            cmd += ' --isTest'

        script = make_job_script(
            cmd,
            queue=qname,
            queue_instance=qi,
            job_name=f"{args.monomer_name}_stk{i}",
            stdout=str(subdir / 'job.sh.o'),
            stderr=str(subdir / 'job.sh.e'),
        )
        job_path = subdir / 'job.sh'
        job_path.write_text(script)

        subprocess.run(['qsub', 'job.sh'], cwd=str(subdir))
        print(f"  split_{i} ({len(df_chunk)}点) → {qi}  (num_nodes={num_nodes})")

        prefer_queue = [q for q in ['gr1.q', 'gr2.q'] if q != qname][0]
        if not args.isTest:
            time.sleep(5)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--isTest',       action='store_true')
    ap.add_argument('--auto-dir',     required=True)
    ap.add_argument('--monomer-name', required=True)
    ap.add_argument('--symmetry',     choices=['glide', 'screw'], required=True)
    ap.add_argument('--num-splits',   type=int, default=None,
                    help='分割数（省略時は空きノード数に合わせる）')
    args = ap.parse_args()

    print('---- job_stacking start ----')
    init_process(args)
    print('---- job_stacking finish ----')
