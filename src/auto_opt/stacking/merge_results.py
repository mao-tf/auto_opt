#!/usr/bin/env python3
"""
スタッキング計算結果の収集スクリプト

各 split_* ディレクトリの step1.csv を結合して
<auto-dir>/stacking_results.csv として保存する。

Usage:
  python -m auto_opt.stacking.merge_results --auto-dir runs/BTBT_screw_stacking
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_csvs(auto_dir: str) -> None:
    base = Path(auto_dir).resolve()
    dfs  = []

    for split_dir in sorted(base.glob('split_*')):
        csv_path = split_dir / 'step1.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"  {split_dir.name}: {len(df)} 行")
        else:
            print(f"  {split_dir.name}: step1.csv なし（スキップ）")

    if not dfs:
        print("結合するファイルが見つかりませんでした。")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # cy → cz の順で並び替え（E 最小値が見やすくなる）
    sort_cols = [c for c in ('cy', 'cz', 'phi', 'z', 'beta') if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    out_path = base / 'stacking_results.csv'
    merged.to_csv(out_path, index=False)

    print(f"\n{len(dfs)} ファイルを結合しました。")
    print(f"合計 {len(merged)} 行 → {out_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--auto-dir', required=True,
                    help='split_* を含む親ディレクトリ')
    args = ap.parse_args()

    merge_csvs(args.auto_dir)
