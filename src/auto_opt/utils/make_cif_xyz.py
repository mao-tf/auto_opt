#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFの分子リストCSVから intra/inter1/inter2 のXYZファイルを生成し、
DFT計算用 .inp ファイルまで一括作成する。

分子の構成（中心分子を必ず先頭に置く）:
  intra : 中心=分子10, 周辺=4,8,12,16,20,22,24,26  (9分子, 324原子)
  inter1: 中心=分子9,  周辺=4,8,10,12,16,20,22,24,26 (10分子, 360原子)
  inter2: 中心=分子25, 周辺=18,12,16,10             (5分子, 180原子)

使い方:
  python -m auto_opt.utils.make_cif_xyz \\
      --csv /path/to/DNTT-Atoms.csv \\
      --xyz-dir /path/to/out/xyz \\
      --inp-dir /path/to/out/inp
"""

import re
import csv
import math
import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# ─── 格子定数 ─────────────────────────────────────────────────────────
CELL = {
    "a": 6.187, "b": 7.662, "c": 16.21,
    "alpha": 90.0, "beta": 92.49, "gamma": 90.0,
}

# ─── 分子構成（中心を先頭に、その後に周辺分子） ─────────────────────
MOL_CONFIGS = {
    "intra" : {"center": 10, "surround": [4, 8, 12, 16, 20, 22, 24, 26]},
    "inter1": {"center":  9, "surround": [4, 8, 10, 12, 16, 20, 22, 24, 26]},
    "inter2": {"center": 25, "surround": [18, 12, 16, 10]},
}

N_PER_MOL = 36  # DNTT: 1分子あたりの原子数

# ─── Gaussian 設定 ─────────────────────────────────────────────────────
MACHINE_SPEC = {
    "gr1": {"nproc": 40, "mem": "15GB"},
    "gr2": {"nproc": 52, "mem": "15GB"},
}
METHOD = "PBEPBE/cc-pVTZ EmpiricalDispersion=GD3BJ Counterpoise=2"
CHARGE_MULT = "0 1 0 1 0 1"


# ─────────────────────────────────────────────────────────────────────
#  変換ユーティリティ
# ─────────────────────────────────────────────────────────────────────

def make_transform_matrix(cell: dict) -> List[List[float]]:
    a, b, c = cell["a"], cell["b"], cell["c"]
    cos_a = math.cos(math.radians(cell["alpha"]))
    cos_b = math.cos(math.radians(cell["beta"]))
    cos_g = math.cos(math.radians(cell["gamma"]))
    sin_g = math.sin(math.radians(cell["gamma"]))
    vol_f = math.sqrt(1 - cos_a**2 - cos_b**2 - cos_g**2 + 2*cos_a*cos_b*cos_g)
    return [
        [a,   b*cos_g,  c*cos_b],
        [0.0, b*sin_g,  c*(cos_a - cos_b*cos_g)/sin_g],
        [0.0, 0.0,      c*vol_f/sin_g],
    ]


def frac_to_cart(xf, yf, zf, M):
    x = M[0][0]*xf + M[0][1]*yf + M[0][2]*zf
    y = M[1][0]*xf + M[1][1]*yf + M[1][2]*zf
    z = M[2][0]*xf + M[2][1]*yf + M[2][2]*zf
    return x, y, z


def parse_coord(s: str) -> float:
    return float(re.sub(r'\(.*?\)', '', s.strip()))


def label_to_element(label: str) -> str:
    m = re.match(r'^([A-Za-z]+)', label)
    return m.group(1) if m else label


# ─────────────────────────────────────────────────────────────────────
#  CSV 読み込み
# ─────────────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> List[Dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_mol_atoms(all_rows: List[Dict], mol_num: int, n_per_mol: int) -> List[Dict]:
    """mol_num番目の分子（1始まり）の原子行リストを返す。"""
    start = (mol_num - 1) * n_per_mol
    return all_rows[start: start + n_per_mol]


# ─────────────────────────────────────────────────────────────────────
#  XYZ ファイル生成
# ─────────────────────────────────────────────────────────────────────

def make_xyz(
    all_rows: List[Dict],
    mol_order: List[int],      # 先頭が中心分子
    out_path: str,
    label: str,
    M: List[List[float]],
    n_per_mol: int = N_PER_MOL,
) -> None:
    """
    指定した分子順でXYZファイルを生成する。
    mol_order[0] が中心分子（Fragment=1）になる。
    """
    atom_lines = []
    for mol_num in mol_order:
        rows = get_mol_atoms(all_rows, mol_num, n_per_mol)
        for row in rows:
            elem = label_to_element(row.get("Label", ""))
            xf = parse_coord(row.get("Xfrac + ESD", "0"))
            yf = parse_coord(row.get("Yfrac + ESD", "0"))
            zf = parse_coord(row.get("Zfrac + ESD", "0"))
            x, y, z = frac_to_cart(xf, yf, zf, M)
            atom_lines.append(f"{elem} {x:.6f} {y:.6f} {z:.6f}")

    n_atoms = len(atom_lines)
    center = mol_order[0]
    surround = mol_order[1:]
    comment = f"{label}: center=mol{center}, surround={surround}"

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{n_atoms}\n{comment}\n")
        for ln in atom_lines:
            f.write(ln + "\n")

    print(f"  → {out_path}  ({n_atoms}原子, {len(mol_order)}分子)")


# ─────────────────────────────────────────────────────────────────────
#  inp ファイル生成
# ─────────────────────────────────────────────────────────────────────

def make_inp_from_xyz_file(
    xyz_path: str,
    inp_path: str,
    description: str,
    machine: str = "gr1",
    n_per_mol: int = N_PER_MOL,
) -> int:
    """XYZファイルを読んで Gaussian .inp ファイルを生成する。"""
    spec = MACHINE_SPEC[machine]

    with open(xyz_path) as f:
        lines = f.readlines()
    n_total = int(lines[0].strip())
    elements, coords = [], []
    for ln in lines[2: 2 + n_total]:
        p = ln.split()
        if len(p) < 4:
            continue
        elements.append(p[0])
        coords.append((float(p[1]), float(p[2]), float(p[3])))

    n_mols = n_total // n_per_mol
    center_elems  = elements[:n_per_mol]
    center_coords = coords[:n_per_mol]

    all_lines = []
    for mol_idx in range(1, n_mols):
        peri_elems  = elements[mol_idx*n_per_mol: (mol_idx+1)*n_per_mol]
        peri_coords = coords [mol_idx*n_per_mol: (mol_idx+1)*n_per_mol]
        desc = f"{description}_Pair_{mol_idx}"

        if mol_idx > 1:
            all_lines.append("--Link1--\n")

        all_lines += [
            f"%mem={spec['mem']}\n",
            f"%nproc={spec['nproc']}\n",
            f"#P {METHOD}\n",
            "\n",
            f"{desc}\n",
            "\n",
            f"{CHARGE_MULT}\n",
        ]
        for sym, (x, y, z) in zip(center_elems, center_coords):
            all_lines.append(f"{sym}(Fragment=1) {x:.6f} {y:.6f} {z:.6f}\n")
        for sym, (x, y, z) in zip(peri_elems, peri_coords):
            all_lines.append(f"{sym}(Fragment=2) {x:.6f} {y:.6f} {z:.6f}\n")
        all_lines.append("\n")

    all_lines += ["\n", "\n", "\n"]

    Path(inp_path).parent.mkdir(parents=True, exist_ok=True)
    with open(inp_path, "w") as f:
        f.writelines(all_lines)

    return n_mols - 1


# ─────────────────────────────────────────────────────────────────────
#  メイン処理
# ─────────────────────────────────────────────────────────────────────

def run(csv_path: str, xyz_dir: str, inp_dir: str, monomer: str = "DNTT") -> None:
    M = make_transform_matrix(CELL)
    all_rows = load_csv(csv_path)
    n_total = len(all_rows)
    n_mols  = n_total // N_PER_MOL
    print(f"CSV読み込み: {n_total}原子 / {n_mols}分子")

    for calc_type, cfg in MOL_CONFIGS.items():
        center   = cfg["center"]
        surround = cfg["surround"]
        mol_order = [center] + surround

        print(f"\n[{calc_type}] 中心=分子{center}, 周辺={surround}")

        # XYZ生成
        xyz_name = f"{monomer}_cif_{calc_type}.xyz"
        xyz_path = str(Path(xyz_dir) / xyz_name)
        make_xyz(all_rows, mol_order, xyz_path, f"{monomer}_cif_{calc_type}", M)

        # inp生成
        inp_name = f"{monomer}_cif_{calc_type}.inp"
        inp_path = str(Path(inp_dir) / calc_type / inp_name)
        machine  = "gr1" if calc_type in ("intra", "inter2") else "gr2"
        n_pairs  = make_inp_from_xyz_file(
            xyz_path, inp_path,
            description=f"{monomer}_cif_{calc_type}",
            machine=machine,
        )
        print(f"  → {inp_path}  ({n_pairs}ペア, {machine})")

    print(f"\n完了。\n  XYZ: {xyz_dir}\n  inp: {inp_dir}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv",     required=True, help="入力CSVファイル（分数座標）")
    parser.add_argument("--xyz-dir", required=True, help="XYZファイルの出力先ディレクトリ")
    parser.add_argument("--inp-dir", required=True, help="DFT inpファイルの出力先ディレクトリ")
    parser.add_argument("--monomer", default="DNTT", help="分子名（デフォルト: DNTT）")
    args = parser.parse_args()

    run(args.csv, args.xyz_dir, args.inp_dir, args.monomer)


if __name__ == "__main__":
    main()
