#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_xyz.py  ―  filtered_step1.csv から分子可視化用 XYZ ファイルを生成

filtered_step1.csv の中から指定パラメータに最も近い行を選び、
その (a, b, z, bt1, bt2) を使ってユニットセルを組み立てて .xyz で出力する。

対称性は CSV の列から自動判定:
  - 'beta' 列あり → screw
  - なし           → glide

--tiles na nb で a/b 方向にユニットセルを繰り返せる（デフォルト 2 2）。

Usage:
  # glide
  python -m auto_opt.plot.export_xyz \\
      --csv   runs/DNTT_glide/filtered_step1.csv \\
      --monomer DNTT \\
      --alpha 65 --phi -10 --z 1.5 \\
      --out   DNTT_phi-10.xyz

  # screw
  python -m auto_opt.plot.export_xyz \\
      --csv   runs/DNTT_screw/filtered_step1.csv \\
      --monomer DNTT \\
      --alpha 65 --beta 0 --phi -10 --z 1.5 \\
      --tiles 3 3 \\
      --out   DNTT_screw_phi-10.xyz
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from auto_opt.utils import Rod, vdw_radius, R2atom

MONOMER_DIR = str(Path(__file__).resolve().parents[3] / "data" / "monomer")


# ──────────────────────────────────────────────
#  モノマー読み込み
# ──────────────────────────────────────────────

def _load_monomer(monomer_name: str, monomer_dir: str) -> tuple[list[str], np.ndarray]:
    """
    元素記号リストと座標 (N×3) を返す。
    .xyz が存在すれば元素記号はそちらから、座標は .csv（重心補正済み）を使う。
    """
    csv_path = Path(monomer_dir) / f"{monomer_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"monomer CSV が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    coords = df[['X', 'Y', 'Z']].to_numpy(dtype=float)
    radii  = df['R'].to_numpy(dtype=float)

    # 元素記号: .xyz があればそちらを優先
    xyz_path = Path(monomer_dir) / f"{monomer_name}.xyz"
    if xyz_path.exists():
        lines = xyz_path.read_text().splitlines()
        symbols = [ln.split()[0] for ln in lines[2:] if ln.strip()]
        if len(symbols) != len(coords):
            print(f"[warn] .xyz の原子数 ({len(symbols)}) と .csv ({len(coords)}) が不一致。R2atom を使用。")
            symbols = [R2atom(r) for r in radii]
    else:
        symbols = [R2atom(r) for r in radii]

    return symbols, coords


# ──────────────────────────────────────────────
#  回転・配置
# ──────────────────────────────────────────────

def _rotate_glide(xyz: np.ndarray, alpha: float, phi: float) -> np.ndarray:
    """glide: phi(-x軸) → alpha(z軸)"""
    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])
    return xyz @ Rod(-ex, phi).T @ Rod(ez, alpha).T


def _rotate_screw(xyz: np.ndarray, alpha: float, beta: float, phi: float) -> np.ndarray:
    """screw: phi(-x軸) → alpha(z軸) → beta(-x軸)"""
    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])
    return xyz @ Rod(-ex, phi).T @ Rod(ez, alpha).T @ Rod(-ex, beta).T


def _build_unit_cell_glide(
    symbols: list[str], base_xyz: np.ndarray,
    alpha: float, phi: float,
    a: float, b: float, z: float,
) -> tuple[list[str], np.ndarray]:
    """
    glide ユニットセル（2分子）:
      分子1: phi, +alpha, 原点
      分子2: phi, -alpha, (a/2, b/2, z)
    """
    mol1 = _rotate_glide(base_xyz,  alpha, phi)
    mol2 = _rotate_glide(base_xyz, -alpha, phi) + np.array([a/2, b/2, z])
    return symbols * 2, np.vstack([mol1, mol2])


def _build_unit_cell_screw(
    symbols: list[str], base_xyz: np.ndarray,
    alpha: float, beta: float, phi: float,
    a: float, bt1: float, z: float,
) -> tuple[list[str], np.ndarray]:
    """
    screw ユニットセル（2分子）:
      分子1: phi, +alpha, beta, 原点
      分子2: phi, -alpha, beta, (a/2, bt1, z)
    """
    mol1 = _rotate_screw(base_xyz,  alpha, beta, phi)
    mol2 = _rotate_screw(base_xyz, -alpha, beta, phi) + np.array([a/2, bt1, z])
    return symbols * 2, np.vstack([mol1, mol2])


def _tile(
    symbols: list[str], coords: np.ndarray,
    a_vec: np.ndarray, b_vec: np.ndarray,
    na: int, nb: int,
) -> tuple[list[str], np.ndarray]:
    """ユニットセルを a/b 方向に na×nb 繰り返す。"""
    all_sym: list[str] = []
    all_xyz: list[np.ndarray] = []
    for ia in range(na):
        for ib in range(nb):
            shift = ia * a_vec + ib * b_vec
            all_sym.extend(symbols)
            all_xyz.append(coords + shift)
    return all_sym, np.vstack(all_xyz)


# ──────────────────────────────────────────────
#  CSV から最近傍行を検索
# ──────────────────────────────────────────────

def _find_row(df: pd.DataFrame, targets: dict[str, float]) -> pd.Series:
    """各パラメータについて最近傍値に丸め、一致する行を1つ返す。"""
    mask = pd.Series([True] * len(df), index=df.index)
    snapped: dict[str, float] = {}
    for col, val in targets.items():
        if col not in df.columns:
            raise ValueError(f"列 '{col}' が CSV に存在しません。")
        unique_vals = df[col].dropna().unique()
        nearest = unique_vals[np.argmin(np.abs(unique_vals - val))]
        if abs(nearest - val) > 0.5:
            print(f"[warn] {col}={val} → 最近傍 {nearest:.3f} を使用")
        snapped[col] = float(nearest)
        mask &= np.isclose(df[col], nearest, atol=1e-5)

    rows = df[mask]
    if rows.empty:
        raise ValueError(f"条件に合う行が見つかりません: {snapped}")
    if len(rows) > 1:
        print(f"[info] 複数行ヒット ({len(rows)}行)。E が最小の行を選択します。")
        rows = rows.loc[[rows['E'].idxmin()]]
    return rows.iloc[0]


# ──────────────────────────────────────────────
#  メイン処理
# ──────────────────────────────────────────────

def export_xyz(
    csv_path: str,
    monomer_name: str,
    targets: dict[str, float],
    out_path: str,
    na: int = 2,
    nb: int = 2,
    monomer_dir: str = MONOMER_DIR,
) -> None:
    df = pd.read_csv(csv_path)
    is_screw = 'beta' in df.columns

    row = _find_row(df, targets)
    symbols, base_xyz = _load_monomer(monomer_name, monomer_dir)

    a   = float(row['a'])
    b   = float(row['b'])
    alpha = float(row['alpha'])
    phi   = float(row['phi']) if 'phi' in row.index else 0.0

    if is_screw:
        beta = float(row['beta'])
        bt1  = float(row.get('bt1', b / 2))
        z    = float(row['z'])
        uc_sym, uc_xyz = _build_unit_cell_screw(
            symbols, base_xyz, alpha, beta, phi, a, bt1, z
        )
        a_vec = np.array([a, 0., 0.])
        b_vec = np.array([0., b, 0.])
        comment = (
            f"{monomer_name} screw | "
            f"alpha={alpha:.1f} beta={beta:.1f} phi={phi:.1f} "
            f"a={a:.3f} b={b:.3f} bt1={bt1:.3f} z={z:.3f} | "
            f"E={row['E']:.3f} kcal/mol | tiles={na}x{nb}"
        )
    else:
        z = float(row['z'])
        uc_sym, uc_xyz = _build_unit_cell_glide(
            symbols, base_xyz, alpha, phi, a, b, z
        )
        a_vec = np.array([a, 0., 0.])
        b_vec = np.array([0., b, 0.])
        comment = (
            f"{monomer_name} glide | "
            f"alpha={alpha:.1f} phi={phi:.1f} "
            f"a={a:.3f} b={b:.3f} z={z:.3f} | "
            f"E={row['E']:.3f} kcal/mol | tiles={na}x{nb}"
        )

    all_sym, all_xyz = _tile(uc_sym, uc_xyz, a_vec, b_vec, na, nb)

    # XYZ 書き出し
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{len(all_sym)}\n", f"{comment}\n"]
    for sym, (x, y, z_) in zip(all_sym, all_xyz):
        lines.append(f"{sym:2s}  {x:12.6f}  {y:12.6f}  {z_:12.6f}\n")
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    print(f"[export] {len(all_sym)} 原子 ({na}x{nb} タイル) -> {out_path}")


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="filtered_step1.csv から分子可視化用 XYZ ファイルを生成"
    )
    ap.add_argument('--csv',      required=True, help='filtered_step1.csv のパス')
    ap.add_argument('--monomer',  required=True, help='モノマー名 (例: DNTT)')
    ap.add_argument('--alpha',    type=float, required=True, help='alpha 値 (°)')
    ap.add_argument('--phi',      type=float, default=0.0,  help='phi 値 (°)')
    ap.add_argument('--z',        type=float, required=True, help='z 値 (Å)')
    ap.add_argument('--beta',     type=float, default=None,
                    help='beta 値 (°)  screw のみ')
    ap.add_argument('--tiles',    type=int, nargs=2, default=[2, 2],
                    metavar=('NA', 'NB'),
                    help='a/b 方向のユニットセル繰り返し数 (デフォルト: 2 2)')
    ap.add_argument('--monomer-dir', default=MONOMER_DIR,
                    help='monomer CSV/XYZ が入ったディレクトリ')
    ap.add_argument('--out',      required=True, help='出力 XYZ ファイルパス')
    args = ap.parse_args()

    targets: dict[str, float] = {'alpha': args.alpha, 'phi': args.phi, 'z': args.z}
    if args.beta is not None:
        targets['beta'] = args.beta

    export_xyz(
        csv_path=args.csv,
        monomer_name=args.monomer,
        targets=targets,
        out_path=args.out,
        na=args.tiles[0],
        nb=args.tiles[1],
        monomer_dir=args.monomer_dir,
    )


if __name__ == '__main__':
    main()
