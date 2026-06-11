#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QEのrelaxation outputファイルからGaussian用XYZファイルを生成する。

各zについて以下の3種類のXYZファイルを作成する:

  intra.xyz  : 5ユニットセル×2分子=10分子
               配置: (0,0,0)(0,0,-1)(-1,0,-1)(0,-1,0)(1,0,0)
               中心=分子1, 周辺=分子4,6,7,9

  inter1.xyz : 10ユニットセル×2分子=20分子
               配置: (0,0,1)(0,0,0)(0,-1,0)(0,1,0)(1,0,0)(-1,0,0)(0,0,-1)(0,1,-1)(-1,0,-1)(-1,1,-1)
               中心=分子1(上層), 周辺=分子3,5,7,9,11,14,16,18,20(下層)

  inter2.xyz : 4ユニットセル×2分子=8分子
               配置: (0,0,0)(1,0,0)(1,-1,0)(0,-1,0)
               中心=分子2(上層), 周辺=分子1,3,5,7(下層)

使い方:
  python -m auto_opt.gaussian.make_qe_xyz \\
      --qe-dir runs/DNTT_qe_gaussian \\
      --xyz-dir runs/DNTT_qe_gaussian/xyz \\
      --monomer DNTT
"""

import re
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

BOHR_TO_ANG = 0.529177210903
N_ATOMS_PER_MOL = 36  # DNTT

# ─── 各XYZタイプの構成定義 ────────────────────────────────────────────
#
# unit_cells: ユニットセルの格子ベクトル係数 (i,j,k) のリスト
#   分子番号 n（1始まり）= cell index (n-1)//2 の (n%2==1 → A分子, n%2==0 → B分子)
#
# center_idx: 全分子リスト中の中心分子インデックス（0始まり）
# surround_idx: 周辺分子のインデックスリスト（0始まり）
#
# XYZファイルの分子順: [center, surround[0], surround[1], ...]

XYZ_CONFIGS = {
    "intra": {
        "unit_cells": [(0,0,0), (0,0,-1), (-1,0,-1), (0,-1,0), (1,0,0)],
        # 分子1(A@(0,0,0)) が中心、分子4(B@(0,0,-1)),6(B@(-1,0,-1)),7(A@(0,-1,0)),9(A@(1,0,0)) が周辺
        "center_idx"  : 0,
        "surround_idx": [3, 5, 6, 8],
    },
    "inter1": {
        "unit_cells": [
            (0,0,1),(0,0,0),(0,-1,0),(0,1,0),(1,0,0),
            (-1,0,0),(0,0,-1),(0,1,-1),(-1,0,-1),(-1,1,-1)
        ],
        # 分子1(A@(0,0,1)) が上層中心
        # 分子3,5,7,9,11,14,16,18,20 が下層周辺
        "center_idx"  : 0,
        "surround_idx": [2, 4, 6, 8, 10, 13, 15, 17, 19],
    },
    "inter2": {
        "unit_cells": [(0,0,0), (1,0,0), (1,-1,0), (0,-1,0)],
        # 分子2(B@(0,0,0)) が上層中心
        # 分子1,3,5,7 が下層周辺
        "center_idx"  : 1,
        "surround_idx": [0, 2, 4, 6],
    },
}


# ─────────────────────────────────────────────────────────────────────
#  QE output パーサ
# ─────────────────────────────────────────────────────────────────────

def parse_qe_output(out_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    QE の relax output ファイルから以下を読み取る:
      - 格子ベクトル (3×3 ndarray, Angstrom単位)
      - 最終最適化座標 (N_atoms×3 ndarray, Angstrom単位)
      - 元素記号リスト

    Returns:
        lat_vecs : (3,3) ndarray  各行が格子ベクトル [a_vec, b_vec, c_vec] (Å)
        positions: (N,3) ndarray  原子座標 (Å)
        elements : list of str
    """
    text = Path(out_path).read_text(encoding="utf-8", errors="ignore")

    # ── 格子定数 ──────────────────────────────────────────────────────
    # celldm(1) = alat (Bohr)
    m = re.search(r'celldm\(1\)=\s*([\d.]+)', text)
    if not m:
        raise ValueError(f"celldm(1) が見つかりません: {out_path}")
    alat_bohr = float(m.group(1))
    alat_ang  = alat_bohr * BOHR_TO_ANG

    # crystal axes: a(1), a(2), a(3) in units of alat
    axes_pat = re.compile(
        r'a\([123]\)\s*=\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)'
    )
    axes = axes_pat.findall(text)
    if len(axes) < 3:
        raise ValueError(f"格子ベクトルが3本見つかりません: {out_path}")
    lat_vecs = np.array([[float(x) for x in row] for row in axes[:3]]) * alat_ang

    # ── 最終座標（"Begin final coordinates" 〜 "End final coordinates"）──
    m_final = re.search(
        r'Begin final coordinates.*?ATOMIC_POSITIONS \(angstrom\)\n(.*?)End final coordinates',
        text, re.DOTALL
    )
    if not m_final:
        raise ValueError(f"'Begin final coordinates' が見つかりません: {out_path}")

    coord_block = m_final.group(1).strip()
    elements = []
    positions = []
    for line in coord_block.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            elem = parts[0]
            xyz  = [float(parts[1]), float(parts[2]), float(parts[3])]
            elements.append(elem)
            positions.append(xyz)
        except ValueError:
            continue

    if not positions:
        raise ValueError(f"最終座標が読み取れませんでした: {out_path}")

    return lat_vecs, np.array(positions), elements


# ─────────────────────────────────────────────────────────────────────
#  結晶構造の構築
# ─────────────────────────────────────────────────────────────────────

def build_all_molecules(
    lat_vecs  : np.ndarray,      # (3,3): [a_vec, b_vec, c_vec]
    positions : np.ndarray,      # (2*N_MOL, 3) 最終座標
    elements  : List[str],
    unit_cells: List[Tuple[int,int,int]],
) -> Tuple[List[np.ndarray], List[List[str]]]:
    """
    指定ユニットセル位置に分子を配置して全分子の座標リストを返す。

    Returns:
        mol_coords : [n_mols_total] × (N_ATOMS_PER_MOL, 3)
        mol_elems  : [n_mols_total] × N_ATOMS_PER_MOL
    """
    a_vec, b_vec, c_vec = lat_vecs[0], lat_vecs[1], lat_vecs[2]

    # ユニットセル内の2分子の基底座標
    pos_A = positions[:N_ATOMS_PER_MOL]    # 分子A（原子0〜35）
    pos_B = positions[N_ATOMS_PER_MOL: 2*N_ATOMS_PER_MOL]  # 分子B（原子36〜71）
    elem_A = elements[:N_ATOMS_PER_MOL]
    elem_B = elements[N_ATOMS_PER_MOL: 2*N_ATOMS_PER_MOL]

    mol_coords = []
    mol_elems  = []

    for (i, j, k) in unit_cells:
        shift = i*a_vec + j*b_vec + k*c_vec
        mol_coords.append(pos_A + shift)
        mol_elems.append(elem_A)
        mol_coords.append(pos_B + shift)
        mol_elems.append(elem_B)

    return mol_coords, mol_elems


# ─────────────────────────────────────────────────────────────────────
#  XYZ 書き出し
# ─────────────────────────────────────────────────────────────────────

def write_xyz(
    out_path  : str,
    mol_order : List[int],       # 先頭が中心分子
    mol_coords: List[np.ndarray],
    mol_elems : List[List[str]],
    comment   : str,
) -> None:
    """指定した分子順でXYZファイルを書き出す。先頭が中心分子。"""
    lines = []
    for idx in mol_order:
        for elem, (x, y, z) in zip(mol_elems[idx], mol_coords[idx]):
            lines.append(f"{elem} {x:.6f} {y:.6f} {z:.6f}")

    n_atoms = len(lines)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{n_atoms}\n{comment}\n")
        for ln in lines:
            f.write(ln + "\n")


# ─────────────────────────────────────────────────────────────────────
#  メイン処理
# ─────────────────────────────────────────────────────────────────────

def process_one(
    qe_out  : str,
    xyz_dir : str,
    z_label : str,
    monomer : str,
) -> None:
    """1つのQEoutputから intra/inter1/inter2 のXYZを生成する。"""
    print(f"\n  解析中: {Path(qe_out).name}")
    try:
        lat_vecs, positions, elements = parse_qe_output(qe_out)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return

    n_atoms = len(positions)
    if n_atoms != 2 * N_ATOMS_PER_MOL:
        print(f"  [WARNING] 原子数={n_atoms}, 期待値={2*N_ATOMS_PER_MOL}. スキップします。")
        return

    print(f"  格子ベクトル (Å):")
    print(f"    a = {lat_vecs[0]}")
    print(f"    b = {lat_vecs[1]}")
    print(f"    c = {lat_vecs[2]}")

    for calc_type, cfg in XYZ_CONFIGS.items():
        mol_coords, mol_elems = build_all_molecules(
            lat_vecs, positions, elements, cfg["unit_cells"]
        )

        center   = cfg["center_idx"]
        surround = cfg["surround_idx"]
        mol_order = [center] + surround

        out_name = f"{monomer}_z_{z_label}_{calc_type}.xyz"
        out_path = str(Path(xyz_dir) / out_name)

        comment = (
            f"{monomer} QE-opt z={z_label} {calc_type}: "
            f"center=mol{center+1}, "
            f"surround=mol{[s+1 for s in surround]}"
        )
        write_xyz(out_path, mol_order, mol_coords, mol_elems, comment)
        n_mols = len(mol_order)
        print(f"  {calc_type}: {n_mols}分子 → {out_path}")


def run(qe_dir: str, xyz_dir: str, monomer: str) -> None:
    qe_root = Path(qe_dir)
    # "dntt_relax_{z}.out" パターンを検索
    pat = re.compile(r'dntt_relax_(-?[\d.]+)\.out$')
    out_files = sorted(
        qe_root.glob("dntt_relax_*.out"),
        key=lambda f: float(pat.match(f.name).group(1))
    )

    if not out_files:
        print(f"ERROR: {qe_dir} に dntt_relax_*.out が見つかりません")
        sys.exit(1)

    print(f"対象ファイル数: {len(out_files)}")

    for out_file in out_files:
        m = pat.match(out_file.name)
        z_label = m.group(1)
        process_one(str(out_file), xyz_dir, z_label, monomer)

    print(f"\n完了。XYZファイル出力先: {xyz_dir}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qe-dir",  required=True,
                        help="QE output (.out) ファイルが入ったディレクトリ")
    parser.add_argument("--xyz-dir", required=True,
                        help="XYZファイルの出力先ディレクトリ")
    parser.add_argument("--monomer", default="DNTT",
                        help="分子名（デフォルト: DNTT）")
    args = parser.parse_args()

    run(args.qe_dir, args.xyz_dir, args.monomer)


if __name__ == "__main__":
    main()
