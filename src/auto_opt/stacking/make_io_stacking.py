#!/usr/bin/env python3
"""
Glide スタッキング: ダイマーペア座標生成 (14 ペア)

ペア構成:
  [0-8]  : stacking 分子 (+alpha, at cx/cy/cz) vs 同層 9 近接分子
  [9-13] : stacking 分子 (-alpha, at cx/cy/cz) vs 同層 5 平行近接分子
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from auto_opt.utils import Rod, R2atom
from auto_opt.amber.make_io_gene_phi import _load_mol2_params, _guess_mol2_path

ROOT = Path(__file__).resolve().parents[3]
MONO = ROOT / "data" / "monomer"

N_PAIRS = 14


def _place(monomer_name: str,
           Ta: float, Tb: float, Tc: float,
           phi: float, alpha: float) -> np.ndarray:
    """回転順: phi(-ex) → alpha(ez) → 平行移動。戻り値: (N, 4) [x, y, z, R]"""
    df = pd.read_csv(MONO / f'{monomer_name}.csv')
    xyz = df[['X', 'Y', 'Z']].values.astype(float)
    R   = df['R'].values.reshape(-1, 1).astype(float)
    ex  = np.array([1., 0., 0.])
    ez  = np.array([0., 0., 1.])
    xyz = np.matmul(xyz, Rod(-ex, phi).T)
    xyz = np.matmul(xyz, Rod(ez,  alpha).T)
    xyz += np.array([Ta, Tb, Tc])
    return np.concatenate([xyz, R], axis=1)


def get_pairs_xyzR(monomer_name: str, params: dict) -> List[np.ndarray]:
    """14 ダイマーペアの座標配列を返す。"""
    a      = float(params.get('a',      0.0))
    b      = float(params.get('b',      0.0))
    z      = float(params.get('z',      0.0))
    phi    = float(params.get('phi',    0.0))
    alpha1 = float(params.get('alpha1', 0.0))
    alpha2 = float(params.get('alpha2', 0.0))
    cx     = float(params.get('cx',     0.0))
    cy     = float(params.get('cy',     0.0))
    cz     = float(params.get('cz',     0.0))

    def p(Ta, Tb, Tc, al):
        return _place(monomer_name, Ta, Tb, Tc, phi, al)

    c   = p(cx,    cy,     cz,      alpha2)
    c_  = p(cx,    cy,     cz,     -alpha2)
    i   = p(0,     0,      0,       alpha1)
    i_  = p(0,     0,      0,      -alpha1)
    p1  = p(0,     b,      2*z,     alpha1);  p1_ = p(0,    b,   2*z,  -alpha1)
    p2  = p(0,    -b,     -2*z,    alpha1);  p2_ = p(0,   -b,  -2*z,  -alpha1)
    p3  = p(a,     0,      0,       alpha1);  p3_ = p(a,    0,   0,    -alpha1)
    p4  = p(-a,    0,      0,       alpha1);  p4_ = p(-a,   0,   0,    -alpha1)
    t1  = p(a/2,   b/2,    z,      -alpha1)
    t2  = p(-a/2,  b/2,    z,      -alpha1)
    t3  = p(a/2,  -b/2,   -z,      -alpha1)
    t4  = p(-a/2, -b/2,   -z,      -alpha1)

    def cat(a, b):
        return np.concatenate([a, b], axis=0)

    return [
        cat(c,  i),   cat(c,  p1),  cat(c,  p2),  cat(c,  p3),  cat(c,  p4),
        cat(c,  t1),  cat(c,  t2),  cat(c,  t3),  cat(c,  t4),
        cat(c_, i_),  cat(c_, p1_), cat(c_, p2_), cat(c_, p3_), cat(c_, p4_),
    ]


def calc_E_total(E_list: List[float]) -> float:
    """14 ペアのエネルギーからスタッキングエネルギーを計算。
    平行ペア [0:5][9:14] は半重み（glide 対称で共有）、T字ペア [5:9] は全重み。"""
    return float((sum(E_list[0:5]) + sum(E_list[9:14])) / 2.0 + sum(E_list[5:9]))


def get_mol2_lines(xyzr: np.ndarray, monomer_name: str) -> List[str]:
    """ダイマー mol2 テキストを返す。"""
    types_charges, bonds = _load_mol2_params(monomer_name)
    n_mono  = len(types_charges)
    n_total = xyzr.shape[0]
    n_bonds = 2 * len(bonds)

    lines: List[str] = [
        "@<TRIPOS>MOLECULE\n",
        f"{monomer_name}_dimer\n",
        f"{n_total:6d}{n_bonds:7d}{2:6d}{0:6d}{0:6d}\n",
        "SMALL\n", "bcc\n", "\n",
        "@<TRIPOS>ATOM\n",
    ]
    for i in range(n_total):
        x, y, z, r = xyzr[i]
        atype, charge = types_charges[i % n_mono]
        frag = 1 if i < n_mono else 2
        resname = 'RES1' if frag == 1 else 'RES2'
        lines.append(
            f"{i+1:6d} {R2atom(r):<2s} {x: .6f} {y: .6f} {z: .6f} "
            f"{atype} {frag:3d} {resname:<4s} {charge: .6f}\n"
        )
    lines.append("@<TRIPOS>BOND\n")
    bid = 1
    for a, b, btype in bonds:
        lines.append(f"{bid:6d}{a:6d}{b:6d} {btype}\n"); bid += 1
    off = n_mono
    for a, b, btype in bonds:
        lines.append(f"{bid:6d}{a+off:6d}{b+off:6d} {btype}\n"); bid += 1
    lines += [
        "@<TRIPOS>SUBSTRUCTURE\n",
        f"{1:6d} RES1{1:10d} GROUP             0 **** **** 0\n",
        f"{2:6d} RES2{n_mono+1:10d} GROUP             0 **** **** 0\n",
        "\n",
    ]
    return lines
