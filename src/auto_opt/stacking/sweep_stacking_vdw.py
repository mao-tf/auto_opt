#!/usr/bin/env python3
"""
スタッキング VdW スキャン: stacking_candidates.csv → step1_init_params.csv

app.py でエクスポートした stacking_candidates.csv を読み込み、
各構造について cy を振って VdW 接触 cz を計算し、
Amber 最適化の初期値 CSV を生成する。

Usage:
  python -m auto_opt.stacking.sweep_stacking_vdw \
      --candidates runs/DNTT_stacking/stacking_candidates.csv \
      --monomer DNTT \
      --out-dir runs/DNTT_stacking \
      --symmetry screw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from auto_opt.utils import Rod

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
MONO = DATA / "monomer"


def _load_monomer_halves(monomer_name: str):
    """モノマー CSV を上半分 (Z > 0.1) と下半分 (Z < -0.1) に分けて返す。"""
    csv = MONO / f'{monomer_name}.csv'
    df = pd.read_csv(csv)
    df1 = df[df['Z'] >  0.1]
    df2 = df[df['Z'] < -0.1]
    return (
        df1[['X', 'Y', 'Z']].values, df1['R'].values.reshape(-1, 1),
        df2[['X', 'Y', 'Z']].values, df2['R'].values.reshape(-1, 1),
    )


def _rotate(xyz: np.ndarray, beta: float, alpha: float, phi: float) -> np.ndarray:
    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])
    r = np.matmul(xyz, Rod(-ex, phi).T)
    r = np.matmul(r,   Rod(ez,  alpha).T)
    r = np.matmul(r,   Rod(-ex, beta).T)
    return r


def _vdw_min_cz(xyz1: np.ndarray, R1: np.ndarray,
                xyz2: np.ndarray, R2: np.ndarray,
                v1: np.ndarray,   v2: np.ndarray) -> float:
    """v1 平行移動した上半分と v2 平行移動した下半分の VdW 接触 cz を返す。"""
    c1 = xyz1 + v1
    c2 = xyz2 + v2
    dx   = c2[:, 0] - c1[:, 0].reshape(-1, 1)
    dy   = c2[:, 1] - c1[:, 1].reshape(-1, 1)
    dz   = c1[:, 2].reshape(-1, 1) - c2[:, 2]
    dxy2 = dx**2 + dy**2
    sr2  = (R1 + R2.reshape(1, -1))**2
    mask = dxy2 < sr2
    if not np.any(mask):
        return 0.0
    return float(np.max(dz[mask] + np.sqrt(sr2[mask] - dxy2[mask])))


def _v_list_p(a, b):
    return [
        [0, 0, 0], [0, b, 0], [0, -b, 0],
        [a, 0, 0], [-a, 0, 0],
        [a, b, 0], [-a, b, 0], [a, -b, 0], [-a, -b, 0],
    ]


def _v_list_t_screw(a, bt1, bt2, z):
    return [
        [a/2,  bt1,  z], [a/2,  -bt2,  z],
        [-a/2, -bt2, z], [-a/2,  bt1,  z],
    ]


def _compute_cz(row: pd.Series,
                xyz1_base: np.ndarray, R1: np.ndarray,
                xyz2_base: np.ndarray, R2: np.ndarray,
                cy: float, symmetry: str) -> float:
    a    = float(row['a'])
    bt1  = float(row.get('bt1', 0.0))
    bt2  = float(row.get('bt2', 0.0))
    b    = bt1 + bt2 if 'bt1' in row.index else float(row['b'])
    z    = float(row['z'])
    beta = float(row.get('beta', 0.0))
    phi  = float(row.get('phi', 0.0))
    alpha = float(row.get('alpha1', row.get('alpha', 0.0)))

    rot1_p = _rotate(xyz1_base,  beta,  alpha, phi)
    rot2_p = _rotate(xyz2_base,  beta,  alpha, phi)
    rot1_m = _rotate(xyz1_base,  beta, -alpha, phi)
    rot2_m = _rotate(xyz2_base,  beta, -alpha, phi)

    v2 = np.array([0., cy, 0.])
    d_list = []

    for v1 in _v_list_p(a, b):
        v1a = np.array(v1)
        d_list.append(_vdw_min_cz(rot1_p, R1, rot2_p, R2, v1a,  v2))
        d_list.append(_vdw_min_cz(rot1_m, R1, rot2_m, R2, v1a,  v2))

    if symmetry == 'screw':
        for v1 in _v_list_t_screw(a, bt1, bt2, -z):
            v1a = np.array(v1)
            d_list.append(_vdw_min_cz(rot1_m, R1, rot2_p, R2, v1a,  v2))
            d_list.append(_vdw_min_cz(rot1_p, R1, rot2_m, R2, v1a, -v2))

    return float(max(d_list)) if d_list else 0.0


def sweep(candidates_csv: Path, monomer: str, out_dir: Path, symmetry: str) -> Path:
    df_cands = pd.read_csv(candidates_csv)
    xyz1, R1, xyz2, R2 = _load_monomer_halves(monomer)

    rows_out = []
    for idx, row in df_cands.iterrows():
        bt1 = float(row.get('bt1', 0.0))
        bt2 = float(row.get('bt2', 0.0))
        b   = bt1 + bt2 if 'bt1' in row.index else float(row['b'])
        alpha = float(row.get('alpha1', row.get('alpha', 0.0)))

        Ny = int(np.round(b / 0.1)) + 1
        for j in range(Ny):
            cy = round(j * 0.1 - b / 2.0, 1)
            cz = round(_compute_cz(row, xyz1, R1, xyz2, R2, cy, symmetry), 1)
            rows_out.append({
                'cx':    0.0,
                'cy':    cy,
                'cz':    cz,
                'a':     row['a'],
                'bt1':   bt1,
                'bt2':   bt2,
                'b':     b,
                'z':     row['z'],
                'beta':  row.get('beta', 0.0),
                'phi':   row.get('phi', 0.0),
                'alpha1': alpha,
                'alpha2': alpha,
                'status': 'NotYet',
            })
        print(f"[sweep] [{idx+1}/{len(df_cands)}] cy scan done  "
              f"(phi={row.get('phi', '?')}, z={row['z']})")

    df_out = pd.DataFrame(rows_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'step1_init_params.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"[sweep_stacking_vdw] → {out_csv}  (n={len(df_out)})")
    return out_csv


def main():
    ap = argparse.ArgumentParser(
        description='stacking_candidates.csv から VdW スキャンで step1_init_params.csv を生成する'
    )
    ap.add_argument('--candidates', required=True, help='stacking_candidates.csv のパス')
    ap.add_argument('--monomer',    required=True, help='モノマー名')
    ap.add_argument('--out-dir',    required=True, help='出力ディレクトリ')
    ap.add_argument('--symmetry',   choices=['glide', 'screw'], default='screw')
    args = ap.parse_args()

    sweep(
        candidates_csv=Path(args.candidates),
        monomer=args.monomer,
        out_dir=Path(args.out_dir),
        symmetry=args.symmetry,
    )


if __name__ == '__main__':
    main()
