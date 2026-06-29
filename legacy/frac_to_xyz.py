#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIF由来の分子リスト（分数座標）から指定分子を抽出し、
格子定数を使ってデカルト座標のXYZファイルに変換する。

使い方:
  python frac_to_xyz.py --csv molecules.csv --out output.xyz

格子定数はスクリプト内の CELL 辞書で指定する。
"""

import re
import csv
import math
import argparse
import sys
from pathlib import Path
from typing import List, Tuple


# ─── 格子定数（単位: Å, 度） ─────────────────────────────────────────
CELL = {
    "a"     : 6.187,
    "b"     : 7.662,
    "c"     : 16.21,
    "alpha" : 90.0,
    "beta"  : 92.49,
    "gamma" : 90.0,
}

# ─── 抽出する分子番号（1始まり） ─────────────────────────────────────
TARGET_MOLS = [4, 8, 9, 10, 12, 16, 18, 20, 22, 24, 25, 26]

N_ATOMS_PER_MOL = 36


# ─────────────────────────────────────────────────────────────────────
#  格子定数 → 変換行列
# ─────────────────────────────────────────────────────────────────────

def make_transform_matrix(cell: dict) -> List[List[float]]:
    """
    分数座標 → デカルト座標の変換行列を作る（単斜晶系対応）。

    単斜晶系 (α=γ=90°, β≠90°) の場合:
        x_cart = a*xf + c*cos(β)*zf
        y_cart = b*yf
        z_cart = c*sin(β)*zf
    """
    a = cell["a"]
    b = cell["b"]
    c = cell["c"]
    beta  = math.radians(cell["beta"])
    alpha = math.radians(cell["alpha"])
    gamma = math.radians(cell["gamma"])

    # 一般式（三斜晶系でも使える）
    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)

    # 体積因子
    vol_factor = math.sqrt(
        1 - cos_a**2 - cos_b**2 - cos_g**2
        + 2 * cos_a * cos_b * cos_g
    )

    # 変換行列 M（列ベクトル形式: r_cart = M @ r_frac）
    M = [
        [a,          b * cos_g,   c * cos_b                          ],
        [0.0,        b * sin_g,   c * (cos_a - cos_b * cos_g) / sin_g],
        [0.0,        0.0,         c * vol_factor / sin_g             ],
    ]
    return M


def frac_to_cart(xf: float, yf: float, zf: float, M: List[List[float]]) -> Tuple[float, float, float]:
    """分数座標 → デカルト座標"""
    x = M[0][0]*xf + M[0][1]*yf + M[0][2]*zf
    y = M[1][0]*xf + M[1][1]*yf + M[1][2]*zf
    z = M[2][0]*xf + M[2][1]*yf + M[2][2]*zf
    return x, y, z


# ─────────────────────────────────────────────────────────────────────
#  座標値のパース（ESDつき文字列 "-0.4950(1)" → -0.4950）
# ─────────────────────────────────────────────────────────────────────

def parse_coord(s: str) -> float:
    """"-0.4950(1)" や "0.5050" などをfloatに変換する。"""
    return float(re.sub(r'\(.*?\)', '', s.strip()))


def label_to_element(label: str) -> str:
    """
    "S(1)" → "S", "C(3)" → "C", "H(1)" → "H" に変換する。
    """
    m = re.match(r'^([A-Za-z]+)', label)
    return m.group(1) if m else label


# ─────────────────────────────────────────────────────────────────────
#  メイン処理
# ─────────────────────────────────────────────────────────────────────

def extract_and_convert(
    csv_path: str,
    out_path: str,
    target_mols: List[int] = TARGET_MOLS,
    n_per_mol: int = N_ATOMS_PER_MOL,
    cell: dict = CELL,
) -> None:
    """
    CSVから指定分子を抽出し、デカルト座標のXYZファイルに書き出す。

    Args:
        csv_path   : 入力CSVファイル（Number,Label,...,Xfrac,...）
        out_path   : 出力XYZファイル
        target_mols: 抽出する分子番号リスト（1始まり）
        n_per_mol  : 1分子あたりの原子数
        cell       : 格子定数辞書
    """
    # 変換行列を計算
    M = make_transform_matrix(cell)

    # CSVを全行読み込み
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    n_total = len(all_rows)
    n_mols  = n_total // n_per_mol
    print(f"総原子数: {n_total}, 分子数: {n_mols}, 1分子あたり: {n_per_mol}原子")
    print(f"抽出対象: {target_mols} ({len(target_mols)}分子 × {n_per_mol}原子 = {len(target_mols)*n_per_mol}原子)")

    # 指定分子の行を抽出
    extracted = []
    for mol_num in target_mols:
        if mol_num < 1 or mol_num > n_mols:
            print(f"  [WARNING] 分子{mol_num}は範囲外です（総分子数={n_mols}）。スキップします。")
            continue
        start = (mol_num - 1) * n_per_mol
        end   = mol_num * n_per_mol
        mol_rows = all_rows[start:end]
        extracted.extend(mol_rows)
        # 確認用ログ
        symm = mol_rows[0].get("Symm. op.", "")
        print(f"  分子{mol_num:2d}: 行{start+1:4d}～{end:4d}  対称操作: {symm}")

    # デカルト座標に変換してXYZに書き出す
    n_out = len(extracted)
    lines = [f"{n_out}\n", f"Extracted molecules {target_mols} from {Path(csv_path).name}\n"]

    for row in extracted:
        label = row.get("Label", "")
        elem  = label_to_element(label)

        # 座標列名（ヘッダ名に応じて調整）
        xf = parse_coord(row.get("Xfrac + ESD", row.get("Xfrac", "0")))
        yf = parse_coord(row.get("Yfrac + ESD", row.get("Yfrac", "0")))
        zf = parse_coord(row.get("Zfrac + ESD", row.get("Zfrac", "0")))

        x, y, z = frac_to_cart(xf, yf, zf, M)
        lines.append(f"{elem} {x:.6f} {y:.6f} {z:.6f}\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("".join(lines), encoding="utf-8")
    print(f"\nXYZファイルを保存しました: {out_path}  ({n_out}原子)")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True,
                        help="入力CSVファイル（分数座標）")
    parser.add_argument("--out", default="extracted.xyz",
                        help="出力XYZファイル（デフォルト: extracted.xyz）")
    parser.add_argument("--mols", type=int, nargs="+", default=TARGET_MOLS,
                        help=f"抽出する分子番号（デフォルト: {TARGET_MOLS}）")
    parser.add_argument("--n-per-mol", type=int, default=N_ATOMS_PER_MOL,
                        help=f"1分子あたりの原子数（デフォルト: {N_ATOMS_PER_MOL}）")
    args = parser.parse_args()

    extract_and_convert(
        csv_path   = args.csv,
        out_path   = args.out,
        target_mols= args.mols,
        n_per_mol  = args.n_per_mol,
    )


if __name__ == "__main__":
    main()
