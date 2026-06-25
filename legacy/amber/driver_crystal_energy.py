#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zスキャンのXYZファイル群から結晶相互作用エネルギーをAMBER(GAFF2)で一括計算する。

=========================================================
入力ファイルの命名規則（xyz_dir 以下に置く）
=========================================================

  DNTT_z_{z}_{...}_intra.xyz    層内相互作用（中心1分子 vs 周辺4分子 → 4ペア）
  DNTT_z_{z}_{...}_inter1.xyz   層間相互作用1（中心1分子 vs 周辺9分子 → 9ペア）
  DNTT_z_{z}_{...}_inter2.xyz   層間相互作用2（中心1分子 vs 周辺4分子 → 4ペア）

=========================================================
エネルギー計算式
=========================================================

  intra  → E1, E2, E3, E4          (4ペア)
  inter1 → E5, E6, E7, E8, E9,     (先5ペア: 全カウント)
            E10, E11, E12, E13      (後4ペア: 1/2カウント)
  inter2 → E14, E15, E16, E17      (4ペア: 1/2カウント)

  E_total = 2*(E1+E2+E3+E4
              +E5+E6+E7+E8+E9
              +(E10+E11+E12+E13+E14+E15+E16+E17)/2)

=========================================================
使い方
=========================================================

  # 本番実行（AMBERが必要）:
  python -m auto_opt.amber.driver_crystal_energy \\
      --xyz-dir runs/DNTT_z_scan/xyz \\
      --monomer DNTT \\
      --out-dir runs/DNTT_z_scan/amber

  # テストモード（mol2/tleap.in だけ生成、sander 未実行）:
  python -m auto_opt.amber.driver_crystal_energy \\
      --xyz-dir runs/DNTT_z_scan/xyz \\
      --monomer DNTT \\
      --out-dir runs/DNTT_z_scan/amber \\
      --test
"""

import os
import re
import sys
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# calc_pentamer_interaction から主要関数を流用
from auto_opt.amber.calc_pentamer_interaction import calc_interaction_energies


# ─────────────────────────────────────────────────────────────────────
#  ファイル名解析
# ─────────────────────────────────────────────────────────────────────

# 例: DNTT_z_0.0_7.7_3.0_3.0_6.0_-10.0_25_intra.xyz
_PAT = re.compile(r'.+_z_(-?[\d.]+)_.+_(intra|inter1|inter2)\.xyz$')


def parse_xyz_name(fname: str) -> Tuple[Optional[float], Optional[str]]:
    """ファイル名から (z値, 種別) を返す。マッチしなければ (None, None)。"""
    m = _PAT.match(fname)
    if m:
        return float(m.group(1)), m.group(2)
    return None, None


def collect_xyz_files(
    xyz_dir: str,
    monomer: str,
) -> Dict[float, Dict[str, str]]:
    """
    xyz_dir 内のXYZファイルを z値ごとに {z: {type: path}} に整理する。
    monomer 名で始まるファイルのみ対象。
    """
    groups: Dict[float, Dict[str, str]] = {}
    for f in sorted(Path(xyz_dir).glob(f"{monomer}_z_*.xyz")):
        z, ftype = parse_xyz_name(f.name)
        if z is None:
            continue
        if z not in groups:
            groups[z] = {}
        groups[z][ftype] = str(f)
    return groups


# ─────────────────────────────────────────────────────────────────────
#  エネルギー式の適用
# ─────────────────────────────────────────────────────────────────────

#  intra  → 4ペア (E1-E4)
#  inter1 → 9ペア (E5-E9: 全カウント, E10-E13: 1/2カウント)
#  inter2 → 4ペア (E14-E17: 1/2カウント)
N_INTRA  = 4
N_INTER1_FULL = 5   # inter1 の先頭5ペア（全カウント分）
N_INTER1_HALF = 4   # inter1 の後ろ4ペア（1/2カウント分）
N_INTER2 = 4


def calc_total_energy(
    E_intra : List[Optional[float]],   # 4ペア
    E_inter1: List[Optional[float]],   # 9ペア（順序: 先5=全, 後4=半）
    E_inter2: List[Optional[float]],   # 4ペア（半カウント）
) -> Tuple[Optional[float], Dict[str, float]]:
    """
    E_total = 2*(E_intra_sum + E_inter1_full_sum
                 + (E_inter1_half_sum + E_inter2_sum) / 2)

    None が含まれる場合は total=None を返す。
    """
    # None チェック
    all_vals = E_intra + E_inter1 + E_inter2
    if any(v is None for v in all_vals):
        return None, {}

    intra_sum       = sum(E_intra)                          # E1-E4
    inter1_full_sum = sum(E_inter1[:N_INTER1_FULL])         # E5-E9
    inter1_half_sum = sum(E_inter1[N_INTER1_FULL:])         # E10-E13
    inter2_sum      = sum(E_inter2)                         # E14-E17

    total = 2.0 * (
        intra_sum
        + inter1_full_sum
        + (inter1_half_sum + inter2_sum) / 2.0
    )

    breakdown = {
        "E_intra_sum"       : intra_sum,
        "E_inter1_full_sum" : inter1_full_sum,
        "E_inter1_half_sum" : inter1_half_sum,
        "E_inter2_sum"      : inter2_sum,
        "E_total"           : total,
    }
    return total, breakdown


# ─────────────────────────────────────────────────────────────────────
#  1つの z に対する計算
# ─────────────────────────────────────────────────────────────────────

def run_one_z(
    z          : float,
    xyz_intra  : str,
    xyz_inter1 : str,
    xyz_inter2 : str,
    monomer    : str,
    out_dir_z  : str,
    isTest     : bool,
) -> Dict:
    """
    z 一点分の intra/inter1/inter2 を計算し、ペアエネルギーリストを返す。
    """
    os.makedirs(out_dir_z, exist_ok=True)

    def run(xyz_path, subdir, n_expected_pairs):
        out = os.path.join(out_dir_z, subdir)
        result = calc_interaction_energies(
            xyz_path     = xyz_path,
            monomer_name = monomer,
            out_dir      = out,
            center_index = 0,
            isTest       = isTest,
        )
        pairs_sorted = sorted(result["interaction_energies"].keys())
        energies = [result["interaction_energies"][p] for p in pairs_sorted]
        if len(energies) != n_expected_pairs:
            print(
                f"  [WARNING] z={z}: {subdir} のペア数が期待値({n_expected_pairs})と"
                f"異なります (実際: {len(energies)})"
            )
        return energies

    print(f"\n{'='*60}")
    print(f"  z = {z}")
    print(f"{'='*60}")

    print(f"\n--- intra ({N_INTRA}ペア) ---")
    E_intra = run(xyz_intra, "intra", N_INTRA)

    print(f"\n--- inter1 ({N_INTER1_FULL + N_INTER1_HALF}ペア) ---")
    E_inter1 = run(xyz_inter1, "inter1", N_INTER1_FULL + N_INTER1_HALF)

    print(f"\n--- inter2 ({N_INTER2}ペア) ---")
    E_inter2 = run(xyz_inter2, "inter2", N_INTER2)

    total, breakdown = calc_total_energy(E_intra, E_inter1, E_inter2)

    if total is not None:
        print(f"\n  E_intra  (E1-E4,  ×2)  = {breakdown['E_intra_sum']:10.4f} kcal/mol")
        print(f"  E_inter1_full (E5-E9, ×2)  = {breakdown['E_inter1_full_sum']:10.4f} kcal/mol")
        print(f"  E_inter1_half (E10-E13,×1) = {breakdown['E_inter1_half_sum']:10.4f} kcal/mol")
        print(f"  E_inter2 (E14-E17, ×1) = {breakdown['E_inter2_sum']:10.4f} kcal/mol")
        print(f"  E_total                 = {total:10.4f} kcal/mol")
    else:
        print(f"\n  [WARNING] z={z}: 計算失敗ペアがあるため E_total を計算できません")

    return {
        "z"        : z,
        "E_intra"  : E_intra,
        "E_inter1" : E_inter1,
        "E_inter2" : E_inter2,
        "breakdown": breakdown,
        "E_total"  : total,
    }


# ─────────────────────────────────────────────────────────────────────
#  メイン（全z一括実行）
# ─────────────────────────────────────────────────────────────────────

def run_all(
    xyz_dir : str,
    monomer : str,
    out_dir : str,
    isTest  : bool = False,
) -> None:
    groups = collect_xyz_files(xyz_dir, monomer)
    if not groups:
        print(f"ERROR: {xyz_dir} に {monomer}_z_*.xyz が見つかりません", file=sys.stderr)
        sys.exit(1)

    z_values = sorted(groups.keys())
    print(f"対象 z 値: {z_values}")

    missing = []
    for z in z_values:
        g = groups[z]
        for ftype in ("intra", "inter1", "inter2"):
            if ftype not in g:
                missing.append(f"z={z}: {ftype} が見つかりません")
    if missing:
        print("ERROR: 以下のファイルが不足しています:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    all_results = []
    for z in z_values:
        g = groups[z]
        z_label = f"{z:.1f}".replace(".", "p")
        out_dir_z = os.path.join(out_dir, f"z_{z_label}")
        result = run_one_z(
            z          = z,
            xyz_intra  = g["intra"],
            xyz_inter1 = g["inter1"],
            xyz_inter2 = g["inter2"],
            monomer    = monomer,
            out_dir_z  = out_dir_z,
            isTest     = isTest,
        )
        all_results.append(result)

    # ── 相対エネルギーの計算（z=0 基準）───────────────────────────────
    valid = [r for r in all_results if r["E_total"] is not None]
    E_ref = valid[0]["E_total"] if valid else None   # 最小z値のE_totalを基準

    # ── サマリー表示 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  サマリー")
    print(f"{'='*60}")
    ref_label = f"z={valid[0]['z']}" if valid else "N/A"
    print(f"基準: {ref_label} の E_total = {E_ref:.4f} kcal/mol" if E_ref else "基準: N/A")
    print()
    print(f"{'z':>6}  {'E_intra':>10}  {'E_inter1_full':>13}  "
          f"{'E_inter1_half':>13}  {'E_inter2':>10}  {'E_total':>12}  {'ΔE_total':>10}")
    print("-" * 85)
    min_dE = min((r["E_total"] - E_ref for r in valid), default=None)
    for r in all_results:
        bd = r["breakdown"]
        if r["E_total"] is not None:
            dE   = r["E_total"] - E_ref
            mark = " ←最小" if abs(dE - min_dE) < 1e-6 else ""
            print(
                f"{r['z']:>6.1f}  "
                f"{bd['E_intra_sum']:>10.4f}  "
                f"{bd['E_inter1_full_sum']:>13.4f}  "
                f"{bd['E_inter1_half_sum']:>13.4f}  "
                f"{bd['E_inter2_sum']:>10.4f}  "
                f"{r['E_total']:>12.4f}  "
                f"{dE:>+10.4f}{mark}"
            )
        else:
            print(f"{r['z']:>6.1f}  {'N/A':>10}  {'N/A':>13}  "
                  f"{'N/A':>13}  {'N/A':>10}  {'N/A':>12}  {'N/A':>10}")

    # ── CSV 出力 ──────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "summary.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # ヘッダ
        intra_headers  = [f"E{i+1}_kcalmol"  for i in range(N_INTRA)]
        inter1_headers = [f"E{i+5}_kcalmol"  for i in range(N_INTER1_FULL + N_INTER1_HALF)]
        inter2_headers = [f"E{i+14}_kcalmol" for i in range(N_INTER2)]
        writer.writerow(
            ["z"]
            + intra_headers
            + inter1_headers
            + inter2_headers
            + ["E_intra_sum", "E_inter1_full_sum", "E_inter1_half_sum",
               "E_inter2_sum", "E_total_kcalmol", "dE_total_kcalmol"]
        )
        for r in all_results:
            bd = r["breakdown"]

            def fmt(v):
                return f"{v:.4f}" if v is not None else ""

            dE = (r["E_total"] - E_ref) if (r["E_total"] is not None and E_ref is not None) else None
            row = (
                [r["z"]]
                + [fmt(v) for v in r["E_intra"]]
                + [fmt(v) for v in r["E_inter1"]]
                + [fmt(v) for v in r["E_inter2"]]
                + [
                    fmt(bd.get("E_intra_sum")),
                    fmt(bd.get("E_inter1_full_sum")),
                    fmt(bd.get("E_inter1_half_sum")),
                    fmt(bd.get("E_inter2_sum")),
                    fmt(r["E_total"]),
                    fmt(dE),
                ]
            )
            writer.writerow(row)

    print(f"\nサマリーCSV: {csv_path}")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--xyz-dir", required=True,
        help="XYZファイルが入っているディレクトリ",
    )
    parser.add_argument(
        "--monomer", default="DNTT",
        help="モノマー名（デフォルト: DNTT）",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="計算ファイルの出力ルートディレクトリ",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="テストモード: mol2/tleap.inを生成するが sander は実行しない",
    )
    args = parser.parse_args()

    run_all(
        xyz_dir = args.xyz_dir,
        monomer = args.monomer,
        out_dir = args.out_dir,
        isTest  = args.test,
    )


if __name__ == "__main__":
    main()
