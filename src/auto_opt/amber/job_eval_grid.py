#!/usr/bin/env python3
"""
VdW グリッド評価ジョブを SGE に投入する。

step1_init_params.csv を空きノード数で分割し、各ノードに qsub で投入する。
全ジョブ完了後に eval_split_*/step1_init_params.csv をマージして
<auto_dir>/step1_init_params.csv に E カラムを追加する。

Usage:
  python -m auto_opt.amber.job_eval_grid \
      --auto-dir runs/BTBT_glide \
      --monomer-name BTBT \
      --symmetry glide
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from auto_opt.cluster import load_env, get_free_queue_instances, wait_for_free_node, make_job_script


def _n_splits(df: pd.DataFrame, is_test: bool) -> int:
    n_rows = len(df)
    if is_test:
        return min(2, n_rows)
    cfg   = load_env()
    free  = get_free_queue_instances()
    n_free = sum(len(v) for v in free.values())
    n_cap  = cfg.get('max_concurrent_jobs', 6)
    n = n_free if n_free > 0 else n_cap
    return min(n, n_rows)


def _submit_jobs(auto_dir: Path, monomer_name: str, symmetry: str,
                 is_test: bool, monomer_dir: str | None = None) -> list[Path]:
    df_init = pd.read_csv(auto_dir / 'step1_init_params.csv')
    n = _n_splits(df_init, is_test)
    chunks = [c for c in np.array_split(df_init, n) if not c.empty]
    print(f"  {len(df_init)} 点を {len(chunks)} ノードに分割して投入します")

    split_dirs = []
    prefer_queue = None

    for i, chunk in enumerate(chunks):
        split_dir = auto_dir / f'eval_split_{i}'
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs.append(split_dir)

        if (split_dir / 'job.sh').exists():
            print(f"  eval_split_{i} → スキップ（既に投入済み）")
            continue

        chunk.to_csv(split_dir / 'step1_init_params.csv', index=False)

        qname, qi, num_nodes = wait_for_free_node(prefer_queue=prefer_queue)
        prefer_queue = qname

        monomer_dir_opt = f" --monomer-dir {monomer_dir}" if monomer_dir else ""
        cmd = (
            f"python -m auto_opt.amber.eval_vdw_grid "
            f"--auto-dir {split_dir} "
            f"--monomer-name {monomer_name} "
            f"--symmetry {symmetry}"
            f"{monomer_dir_opt}"
        )
        job_sh = make_job_script(
            job_name=f'eval_{i}',
            queue=qname,
            queue_instance=qi,
            cmd=cmd,
        )
        job_path = split_dir / 'job.sh'
        job_path.write_text(job_sh)
        subprocess.run(['chmod', '+x', str(job_path)], check=False)
        subprocess.run(['qsub', str(job_path)], check=False)
        print(f"  eval_split_{i} → qsub 投入 ({qname})")

    return split_dirs


def _wait_and_merge(auto_dir: Path, split_dirs: list[Path],
                    poll: int = 30) -> None:
    """全スプリットが完了するまで待ち、結果をマージする。"""
    print("\nジョブ完了を待機中...")
    while True:
        done = [
            (d / 'step1_init_params.csv').exists() and
            'E' in pd.read_csv(d / 'step1_init_params.csv').columns
            for d in split_dirs
        ]
        print(f"  完了: {sum(done)}/{len(done)}")
        if all(done):
            break
        time.sleep(poll)

    dfs = [pd.read_csv(d / 'step1_init_params.csv') for d in split_dirs]
    merged = pd.concat(dfs, ignore_index=True)

    sort_cols = [c for c in ('alpha', 'beta', 'phi', 'z') if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    out = auto_dir / 'step1_init_params.csv'
    merged.to_csv(out, index=False)
    print(f"\nマージ完了: {out}  ({len(merged)} 行)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description='VdW グリッド全点を力場1点計算して step1_init_params.csv に E カラムを追加する'
    )
    ap.add_argument('--auto-dir',     required=True)
    ap.add_argument('--monomer-name', required=True)
    ap.add_argument('--symmetry',     default='glide', choices=['glide', 'screw'])
    ap.add_argument('--monomer-dir',  default=None,
                    help='モノマー CSV/XYZ ディレクトリ（省略時はパッケージ付属の data/monomer）')
    ap.add_argument('--isTest',       action='store_true',
                    help='ジョブスクリプトを生成するが qsub は実行しない')
    ap.add_argument('--poll',         type=int, default=30,
                    help='完了確認のポーリング間隔（秒）')
    args = ap.parse_args()

    auto_dir = Path(args.auto_dir).resolve()
    split_dirs = _submit_jobs(auto_dir, args.monomer_name, args.symmetry, args.isTest,
                              monomer_dir=args.monomer_dir)

    if not args.isTest:
        _wait_and_merge(auto_dir, split_dirs, poll=args.poll)


if __name__ == '__main__':
    main()
