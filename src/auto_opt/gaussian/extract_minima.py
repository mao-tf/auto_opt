#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_minima.py

Amber 最適化後の step1.csv から局所最小点を抽出し filtered_step1.csv を出力する。
driver_gene_phi.py / driver_screw_phi.py の完了後に自動呼び出しされる。

Usage:
  python -m auto_opt.gaussian.extract_minima \
      --auto-dir runs/DNTT \
      --symmetry glide         # または screw

  python -m auto_opt.gaussian.extract_minima \
      --step1-csv runs/DNTT/step1.csv \
      --out-csv   runs/DNTT/filtered_step1.csv \
      --symmetry  screw
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =========================================================
#   グリッド刻み定数
# =========================================================

STEP_A   = 0.1
STEP_B   = 0.1
STEP_BT1 = 0.1
STEP_BT2 = 0.1


# =========================================================
#   共通ユーティリティ
# =========================================================

def _assign_structure_type_from_init(
    df_result: pd.DataFrame,
    init_csv: Path,
    group_keys: list[str],
    opt_keys: list[str],
) -> pd.DataFrame:
    """
    step1_init_params.csv の structure_type を最近傍マッチングで引き継ぐ。
    同じ group_keys でグループ化し、opt_keys 空間で最近傍の初期点を探す。
    """
    df_init = pd.read_csv(init_csv)
    if 'structure_type' not in df_init.columns:
        return df_result

    result_types = []
    for _, row in df_result.iterrows():
        # 同じグループの初期点を抽出
        mask = pd.Series([True] * len(df_init), index=df_init.index)
        for k in group_keys:
            if k in df_init.columns:
                mask &= np.isclose(df_init[k], row[k], atol=1e-5)
        cands = df_init[mask]

        if cands.empty:
            result_types.append('unknown')
            continue

        # opt_keys 空間でユークリッド距離が最小の初期点を選ぶ
        dists = sum(
            (cands[k] - row[k]) ** 2
            for k in opt_keys if k in cands.columns and k in row.index
        )
        result_types.append(cands.loc[dists.idxmin(), 'structure_type'])

    df_result = df_result.copy()
    df_result['structure_type'] = result_types
    return df_result


def _read_step1_auto(step1_csv: Path, auto_dir: Optional[Path]) -> pd.DataFrame:
    """
    step1_csv が実在すればそれを読む。
    無ければ auto_dir/split_*/step1.csv を集めて結合する。
    """
    if step1_csv.is_file():
        return pd.read_csv(step1_csv)

    if auto_dir is None:
        raise FileNotFoundError(f"step1.csv が見つかりません: {step1_csv}")

    dfs = []
    for sub in sorted(auto_dir.iterdir()):
        if not sub.is_dir():
            continue
        p = sub / "step1.csv"
        if p.is_file():
            dfs.append(pd.read_csv(p))

    if not dfs:
        raise FileNotFoundError(
            f"step1.csv が見つかりません: {step1_csv} も {auto_dir}/*/step1.csv も無い"
        )

    return pd.concat(dfs, ignore_index=True)


# =========================================================
#   glide (映進対称): 2D グリッド (a, b)
# =========================================================

def _neighbors_2d(a: float, b: float) -> List[Tuple[float, float]]:
    offs = [(STEP_A, 0), (-STEP_A, 0), (0, STEP_B), (0, -STEP_B),
            (STEP_A, STEP_B), (STEP_A, -STEP_B), (-STEP_A, STEP_B), (-STEP_A, -STEP_B)]
    return [(round(a + da, 3), round(b + db, 3)) for da, db in offs]


def _is_local_min_2d(a: float, b: float, e: float,
                     grid: Dict[Tuple[float, float], float]) -> bool:
    nb = [grid.get(k) for k in _neighbors_2d(a, b)]
    nb = [v for v in nb if v is not None]
    if not nb:
        return False
    return all(e <= v for v in nb) and any(e < v for v in nb)


def extract_glide(step1_csv: Path, out_csv: Path,
                  auto_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    step1.csv から glide 局所最小を抽出。
    グループキー: (alpha, phi, z)
    最適化変数: (a, b) の 2D グリッド
    """
    df = _read_step1_auto(step1_csv, auto_dir)

    required = {'alpha', 'phi', 'a', 'b', 'z', 'E', 'status'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"step1.csv に必要な列がありません: {missing}")

    energy_cols = [c for c in ['E', 'E1', 'E2', 'E3'] if c in df.columns]

    df = df[df['status'].astype(str).str.lower() == 'done'].copy()
    if df.empty:
        print("[extract] Done 行が見つかりません。スキップします。")
        return pd.DataFrame()

    for c in ['a', 'b', 'z'] + energy_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['a', 'b', 'z', 'E'])

    df = df.sort_values('E').drop_duplicates(
        subset=['alpha', 'phi', 'a', 'b', 'z'], keep='first'
    )

    rows = []
    for (alpha, phi, z), g in df.groupby(['alpha', 'phi', 'z']):
        grid: Dict[Tuple[float, float], float] = {
            (round(r.a, 3), round(r.b, 3)): float(r.E)
            for r in g.itertuples(index=False)
        }
        for r in g.itertuples(index=False):
            a, b, e = float(r.a), float(r.b), float(r.E)
            if _is_local_min_2d(a, b, e, grid):
                rec: dict = {
                    'alpha': float(alpha),
                    'phi':   float(phi),
                    'a':   round(a, 1),
                    'b':   round(b, 1),
                    'z':   float(z),
                }
                for c in energy_cols:
                    rec[c] = float(getattr(r, c))
                rows.append(rec)

    if not rows:
        print("[extract] 局所最小点が見つかりませんでした。")
        return pd.DataFrame()

    df_new = pd.DataFrame(rows)

    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        cols_key  = ['alpha', 'phi', 'a', 'b', 'z']
        info_cols = [c for c in ['dft_status', 'log', 'inp', 'machine']
                     if c in existing.columns]
        df_new = pd.merge(df_new, existing[cols_key + info_cols], on=cols_key, how='left')
        if 'dft_status' not in df_new.columns:
            df_new['dft_status'] = float('nan')
        df_new['dft_status'] = df_new['dft_status'].fillna('NotYet')
        print(f"[extract] {len(df_new)} 点を抽出 (既存情報をマージ)")
    else:
        df_new['dft_status'] = 'NotYet'
        print(f"[extract] {len(df_new)} 点を新規作成")

    # step1_init_params.csv があれば structure_type を引き継ぐ、なければ a 値で分類
    init_csv = (out_csv.parent / 'step1_init_params.csv')
    if init_csv.exists():
        df_new = _assign_structure_type_from_init(
            df_new, init_csv,
            group_keys=['alpha', 'phi', 'z'],
            opt_keys=['a', 'b'],
        )
    else:
        df_new = _classify_structure_by_a_rank(df_new)

    out_cols = ['alpha', 'phi', 'a', 'b', 'z', 'structure_type', 'dft_status'] + energy_cols
    extra    = [c for c in df_new.columns if c not in out_cols]
    df_new   = (df_new[out_cols + extra]
                .drop_duplicates()
                .sort_values(['z', 'alpha', 'phi', 'a', 'b'])
                .reset_index(drop=True))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(out_csv, index=False)
    print(f"[extract] -> {out_csv} (n={len(df_new)})")
    return df_new


# =========================================================
#   screw (螺旋軸対称): 3D グリッド (a, bt1, bt2)
# =========================================================

def _neighbors_3d(a: float, bt1: float, bt2: float) -> List[Tuple[float, float, float]]:
    steps = [-STEP_A, 0.0, STEP_A]
    return [
        (round(a + da, 1), round(bt1 + dbt1, 1), round(bt2 + dbt2, 1))
        for da, dbt1, dbt2 in itertools.product(steps, steps, steps)
        if not (da == 0.0 and dbt1 == 0.0 and dbt2 == 0.0)
    ]


def _is_local_min_3d(a: float, bt1: float, bt2: float, e: float,
                     grid: Dict[Tuple[float, float, float], float]) -> bool:
    nb = [grid[k] for k in _neighbors_3d(a, bt1, bt2) if k in grid]
    if not nb:
        return False
    return all(e <= v for v in nb) and any(e < v for v in nb)


def _classify_structure_by_a_rank(df: pd.DataFrame) -> pd.DataFrame:
    """(alpha, beta, z, phi) グループ内で a の大小により structure_type を分類"""
    df = df.copy()
    if 'structure_type' not in df.columns:
        df['structure_type'] = 'unknown'
    for col in ('beta', 'phi'):
        if col not in df.columns:
            df[col] = 0.0

    CLUSTER_THR = 0.5
    for (alpha, beta, z, phi), group in df.groupby(['alpha', 'beta', 'z', 'phi']):
        a_vals = sorted(group['a'].unique())
        if not a_vals:
            continue

        clusters, curr = [], [a_vals[0]]
        for v in a_vals[1:]:
            if v - curr[-1] > CLUSTER_THR:
                clusters.append(curr)
                curr = [v]
            else:
                curr.append(v)
        clusters.append(curr)

        n = len(clusters)
        labels: dict = {}
        if n >= 3:
            labels[0] = 'a-stack'
            labels[n - 1] = 'b-stack'
            for i in range(1, n - 1):
                labels[i] = 'ch'
        elif n == 2:
            labels[0] = 'a-stack'
            labels[1] = 'b-stack'
        elif n == 1:
            labels[0] = 'unknown'

        val_to_label = {
            v: labels.get(i, 'unknown')
            for i, clust in enumerate(clusters)
            for v in clust
        }
        for idx, val in group['a'].items():
            df.at[idx, 'structure_type'] = val_to_label.get(val, 'unknown')

    return df


def extract_screw(step1_csv: Path, out_csv: Path,
                  auto_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    step1.csv から screw 局所最小を抽出。
    グループキー: (alpha, beta, phi, z)
    最適化変数: (a, bt1, bt2) の 3D グリッド
    """
    df = _read_step1_auto(step1_csv, auto_dir)

    required = {'alpha', 'a', 'b', 'bt1', 'bt2', 'z', 'E', 'status'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"step1.csv に必要な列がありません: {missing}")

    for col in ('beta', 'phi'):
        if col not in df.columns:
            df[col] = 0.0

    energy_cols = [c for c in ['E', 'E1', 'E2', 'E3', 'E4'] if c in df.columns]

    df = df[df['status'].astype(str).str.lower() == 'done'].copy()
    if df.empty:
        print("[extract] Done 行が見つかりません。スキップします。")
        return pd.DataFrame()

    for c in ['a', 'b', 'bt1', 'bt2', 'z', 'alpha', 'beta', 'phi'] + energy_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['a', 'b', 'bt1', 'bt2', 'z', 'E'])

    df = df.sort_values('E').drop_duplicates(
        subset=['alpha', 'beta', 'phi', 'a', 'bt1', 'bt2', 'z'], keep='first'
    )

    rows = []
    for (alpha, beta, z, phi), g in df.groupby(['alpha', 'beta', 'z', 'phi']):
        grid: Dict[Tuple[float, float, float], float] = {
            (round(r.a, 1), round(r.bt1, 1), round(r.bt2, 1)): float(r.E)
            for r in g.itertuples(index=False)
        }
        for r in g.itertuples(index=False):
            a, bt1, bt2, b, e = (
                float(r.a), float(r.bt1), float(r.bt2), float(r.b), float(r.E)
            )
            if _is_local_min_3d(a, bt1, bt2, e, grid):
                rec: dict = {
                    'alpha': float(alpha),
                    'beta':  float(beta),
                    'phi':   float(phi),
                    'a':   round(a,   1),
                    'b':   round(b,   1),
                    'bt1': round(bt1, 1),
                    'bt2': round(bt2, 1),
                    'z':   float(z),
                }
                for c in energy_cols:
                    rec[c] = float(getattr(r, c))
                rows.append(rec)

    if not rows:
        print("[extract] 局所最小点が見つかりませんでした。")
        return pd.DataFrame()

    df_new = pd.DataFrame(rows)

    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        for col in ('beta', 'phi'):
            if col not in existing.columns:
                existing[col] = 0.0
        cols_key  = ['alpha', 'beta', 'phi', 'a', 'b', 'bt1', 'bt2', 'z']
        info_cols = [c for c in ['dft_status', 'log', 'inp', 'machine']
                     if c in existing.columns]
        df_new = pd.merge(df_new, existing[cols_key + info_cols], on=cols_key, how='left')
        if 'dft_status' not in df_new.columns:
            df_new['dft_status'] = float('nan')
        df_new['dft_status'] = df_new['dft_status'].fillna('NotYet')
        print(f"[extract] {len(df_new)} 点を抽出 (既存情報をマージ)")
    else:
        df_new['dft_status'] = 'NotYet'
        print(f"[extract] {len(df_new)} 点を新規作成")

    init_csv = out_csv.parent / 'step1_init_params.csv'
    if init_csv.exists():
        df_new = _assign_structure_type_from_init(
            df_new, init_csv,
            group_keys=['alpha', 'beta', 'phi', 'z'],
            opt_keys=['a', 'b', 'bt1', 'bt2'],
        )
    else:
        df_new = _classify_structure_by_a_rank(df_new)

    out_cols = ['alpha', 'beta', 'phi', 'a', 'b', 'bt1', 'bt2', 'z',
                'structure_type', 'dft_status'] + energy_cols
    extra    = [c for c in df_new.columns if c not in out_cols]
    df_new   = df_new[out_cols + extra]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(out_csv, index=False)
    print(f"[extract] -> {out_csv} (n={len(df_new)})")
    return df_new


# =========================================================
#   統合エントリポイント
# =========================================================

def extract_minima(
    symmetry: str,
    auto_dir: Optional[str | Path] = None,
    step1_csv: Optional[str | Path] = None,
    out_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Args:
        symmetry : 'glide' または 'screw'
        auto_dir : 作業ディレクトリ（step1.csv の自動検索に使用）
        step1_csv: 明示的に指定する step1.csv パス（省略時 auto_dir/step1.csv）
        out_csv  : 出力パス（省略時 auto_dir/filtered_step1.csv）
    """
    if symmetry not in ('glide', 'screw'):
        raise ValueError(f"symmetry は 'glide' または 'screw' を指定してください: {symmetry!r}")

    auto_dir_path = Path(auto_dir) if auto_dir else None

    csv_path = (
        Path(step1_csv)
        if step1_csv
        else (auto_dir_path / 'step1.csv' if auto_dir_path else Path('step1.csv'))
    )

    out_path = (
        Path(out_csv)
        if out_csv
        else (auto_dir_path / 'filtered_step1.csv' if auto_dir_path else Path('filtered_step1.csv'))
    )

    if symmetry == 'glide':
        return extract_glide(csv_path, out_path, auto_dir=auto_dir_path)
    else:
        return extract_screw(csv_path, out_path, auto_dir=auto_dir_path)


# =========================================================
#   CLI
# =========================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Amber 最適化結果 step1.csv から局所最小点を抽出する'
    )
    p.add_argument('--symmetry',  required=True, choices=['glide', 'screw'],
                   help='結晶対称性: glide (映進) または screw (螺旋軸)')
    p.add_argument('--auto-dir',  default=None,
                   help='作業ディレクトリ (step1.csv の自動検索 / 出力先)')
    p.add_argument('--step1-csv', default=None,
                   help='入力ファイルパス (省略時: auto-dir/step1.csv)')
    p.add_argument('--out-csv',   default=None,
                   help='出力ファイルパス (省略時: auto-dir/filtered_step1.csv)')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    extract_minima(
        symmetry=args.symmetry,
        auto_dir=args.auto_dir,
        step1_csv=args.step1_csv,
        out_csv=args.out_csv,
    )
