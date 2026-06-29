#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_cluster_xyz.py  ―  9分子クラスター XYZ 生成

filtered_step1.csv の1行から、結晶対称性に基づいた9分子クラスターの
XYZ ファイルを生成する。

Usage:
  # glide
  python -m auto_opt.plot.make_cluster_xyz \\
      --csv runs/BTBT_glide/-5.0/filtered_step1.csv \\
      --monomer BTBT --alpha 25 --phi -5 --z -0.3 \\
      --out cluster.xyz

  # screw
  python -m auto_opt.plot.make_cluster_xyz \\
      --csv runs/DNTT_screw/filtered_step1.csv \\
      --monomer DNTT --alpha 65 --beta 0 --phi -10 --z 1.5 \\
      --out cluster.xyz
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from auto_opt.utils import place_monomer, R2atom

_MONOMER_DIR = Path(__file__).resolve().parents[3] / "data" / "monomer"


def _load_symbols(monomer_name: str, monomer_dir: str) -> list[str]:
    """モノマーの元素記号リストを返す。.xyz 優先、なければ CSV の R 列を使用。"""
    xyz_path = Path(monomer_dir) / f"{monomer_name}.xyz"
    csv_path = Path(monomer_dir) / f"{monomer_name}.csv"

    if xyz_path.exists():
        lines = xyz_path.read_text().splitlines()
        syms = [ln.split()[0] for ln in lines[2:] if ln.strip()]
        if syms:
            if csv_path.exists():
                n = len(pd.read_csv(csv_path))
                if len(syms) == n:
                    return syms
            else:
                return syms

    if not csv_path.exists():
        raise FileNotFoundError(
            f"{monomer_name}.csv が見つかりません: {csv_path}\n"
            "HPC からモノマーデータ (.xyz, .csv) をダウンロードしてください。"
        )

    df = pd.read_csv(csv_path)
    return [R2atom(r) for r in df['R'].to_numpy(dtype=float)]


def _xyz_block(syms: list[str], coords: np.ndarray) -> str:
    return "".join(
        f"{s:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n"
        for s, (x, y, z) in zip(syms, coords)
    )


def _cluster_glide(
    monomer_name: str,
    a: float, b: float, z: float,
    alpha: float, phi: float,
    monomer_dir: str,
) -> tuple[list[str], np.ndarray]:
    """
    glide 9分子クラスター。
    α-type (5分子): phi, +alpha
    t-type (4分子): phi, -alpha
    """
    configs = [
        # (tx,    ty,    tz,    alpha_sign)
        (0,      0,     0,     +1),
        (a,      0,     0,     +1),
        (-a,     0,     0,     +1),
        (0,      b,     2*z,   +1),
        (0,     -b,    -2*z,   +1),
        (a/2,    b/2,   z,    -1),
        (-a/2,   b/2,   z,    -1),
        (a/2,   -b/2,  -z,    -1),
        (-a/2,  -b/2,  -z,    -1),
    ]
    all_syms: list[str] = []
    all_xyz:  list[np.ndarray] = []
    syms = _load_symbols(monomer_name, monomer_dir)

    for tx, ty, tz, sign in configs:
        arr = place_monomer(monomer_name, tx, ty, tz, phi, sign * alpha,
                            monomer_dir=monomer_dir)
        all_syms.extend(syms)
        all_xyz.append(arr[:, :3])

    return all_syms, np.vstack(all_xyz)


def _cluster_screw(
    monomer_name: str,
    a: float, b: float, bt1: float, bt2: float, z: float,
    alpha: float, beta: float, phi: float,
    monomer_dir: str,
) -> tuple[list[str], np.ndarray]:
    """
    screw 9分子クラスター。
    α-type (5分子): phi, +alpha, beta
    t-type (4分子): phi, -alpha, beta
    """
    configs = [
        # (tx,    ty,    tz,   alpha_sign)
        (0,       0,     0,    +1),
        (a,       0,     0,    +1),
        (-a,      0,     0,    +1),
        (0,       b,     0,    +1),
        (0,      -b,     0,    +1),
        (a/2,    bt1,    z,   -1),
        (-a/2,   bt1,    z,   -1),
        (a/2,   -bt2,    z,   -1),
        (-a/2,  -bt2,    z,   -1),
    ]
    all_syms: list[str] = []
    all_xyz:  list[np.ndarray] = []
    syms = _load_symbols(monomer_name, monomer_dir)

    for tx, ty, tz, sign in configs:
        arr = place_monomer(monomer_name, tx, ty, tz, phi, sign * alpha, beta,
                            monomer_dir=monomer_dir)
        all_syms.extend(syms)
        all_xyz.append(arr[:, :3])

    return all_syms, np.vstack(all_xyz)


def make_cluster_xyz(
    row: pd.Series,
    monomer_name: str,
    symmetry: str,
    monomer_dir: str | None = None,
) -> str:
    """
    filtered_step1.csv の1行から9分子クラスターの XYZ 文字列を返す。

    Args:
        row       : filtered_step1.csv の1行 (pd.Series)
        monomer_name: モノマー名
        symmetry  : 'glide' または 'screw'
        monomer_dir: モノマーデータディレクトリ（省略時はパッケージ付属）
    """
    mdir = str(monomer_dir) if monomer_dir else str(_MONOMER_DIR)
    alpha = float(row['alpha'])
    phi   = float(row.get('phi', 0.0))
    z     = float(row['z'])
    a     = float(row['a'])
    b     = float(row['b'])

    e_str = f" | E={float(row['E']):.3f} kcal/mol" if 'E' in row.index else ""

    if symmetry == 'glide':
        syms, coords = _cluster_glide(monomer_name, a, b, z, alpha, phi, mdir)
        comment = (
            f"{monomer_name} glide cluster | "
            f"alpha={alpha:.1f} phi={phi:.1f} a={a:.3f} b={b:.3f} z={z:.3f}"
            f"{e_str}"
        )
    elif symmetry == 'screw':
        beta = float(row.get('beta', 0.0))
        bt1  = float(row.get('bt1', b / 2))
        bt2  = float(row.get('bt2', b / 2))
        syms, coords = _cluster_screw(
            monomer_name, a, b, bt1, bt2, z, alpha, beta, phi, mdir
        )
        comment = (
            f"{monomer_name} screw cluster | "
            f"alpha={alpha:.1f} beta={beta:.1f} phi={phi:.1f} "
            f"a={a:.3f} b={b:.3f} bt1={bt1:.3f} bt2={bt2:.3f} z={z:.3f}"
            f"{e_str}"
        )
    else:
        raise ValueError(f"symmetry は 'glide' または 'screw': {symmetry!r}")

    header = f"{len(syms)}\n{comment}\n"
    return header + _xyz_block(syms, coords)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="filtered_step1.csv から9分子クラスター XYZ を生成"
    )
    ap.add_argument('--csv',     required=True, help='filtered_step1.csv のパス')
    ap.add_argument('--monomer', required=True, help='モノマー名 (例: BTBT)')
    ap.add_argument('--alpha',   type=float, required=True)
    ap.add_argument('--phi',     type=float, default=0.0)
    ap.add_argument('--z',       type=float, required=True)
    ap.add_argument('--beta',    type=float, default=None, help='screw のみ')
    ap.add_argument('--monomer-dir', default=None)
    ap.add_argument('--out',     required=True, help='出力 XYZ ファイルパス')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    is_screw = 'beta' in df.columns

    targets = {'alpha': args.alpha, 'phi': args.phi, 'z': args.z}
    if is_screw and args.beta is not None:
        targets['beta'] = args.beta

    # 最近傍行を選択
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in targets.items():
        nearest = df[col].unique()[np.argmin(np.abs(df[col].unique() - val))]
        mask &= np.isclose(df[col], nearest, atol=1e-5)
    rows = df[mask]
    if rows.empty:
        raise ValueError(f"条件に合う行が見つかりません: {targets}")
    row = rows.loc[rows['E'].idxmin()] if len(rows) > 1 else rows.iloc[0]

    symmetry = 'screw' if is_screw else 'glide'
    xyz_str = make_cluster_xyz(row, args.monomer, symmetry, args.monomer_dir)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(xyz_str, encoding='utf-8')
    print(f"[make_cluster_xyz] {len(xyz_str.splitlines()) - 2} 原子 -> {args.out}")


if __name__ == '__main__':
    main()
