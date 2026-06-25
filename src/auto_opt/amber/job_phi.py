#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python -m auto_opt.amber.job_phi --auto-dir runs/BTBT_glide --monomer-name BTBT

phi ごとにサブディレクトリを作成し、空きノードを探して qsub で投入する。
キュー仕様は ~/.auto_opt.yaml から読み込む（なければ組み込みデフォルト）。
"""

import subprocess
import time
import pandas as pd
import argparse
from pathlib import Path

from auto_opt.cluster import wait_for_free_node, make_job_script


def init_process(args):
    auto_dir_root = Path(args.auto_dir).resolve()
    df_init = pd.read_csv(auto_dir_root / 'step1_init_params.csv')
    phi_list = sorted(df_init['phi'].unique())

    prefer_queue = None

    for i, phi in enumerate(phi_list):
        df_phi = df_init[df_init['phi'] == phi]
        if df_phi.empty:
            continue

        subdir = auto_dir_root / str(phi)
        subdir.mkdir(parents=True, exist_ok=True)

        if (subdir / 'job.sh').exists():
            print(f"  phi={phi} → スキップ（既に投入済み）")
            continue

        df_phi.to_csv(subdir / 'step1_init_params.csv', index=False)

        qname, qi, num_nodes = wait_for_free_node(
            prefer_queue=prefer_queue,
            is_test=args.isTest,
            test_index=i,
        )

        cmd = (
            'python -m auto_opt.amber.driver_gene_phi '
            f'--auto-dir {subdir} '
            f'--monomer-name {args.monomer_name} '
            f'--num-nodes {num_nodes}'
        )
        if args.isTest:
            cmd += ' --isTest'

        script = make_job_script(cmd, queue=qname, queue_instance=qi,
                                 job_name=f"{args.monomer_name}_phi{phi}")
        job_path = subdir / 'job.sh'
        job_path.write_text(script)

        subprocess.run(['qsub', str(job_path)])
        print(f"  phi={phi} → {qi}  (num_nodes={num_nodes})")

        prefer_queue = [q for q in ['gr1.q', 'gr2.q'] if q != qname][0]
        if not args.isTest:
            time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, default='BTBT')
    args = parser.parse_args()

    print('----main process----')
    init_process(args)
    print('----finish process----')
