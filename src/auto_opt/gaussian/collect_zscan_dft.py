#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zスキャンのDFT計算結果（Gaussian counterpoise=2）を集計してCSVに出力する。

ディレクトリ構成:
  inp_dir/
    z_{z_label}/
      intra/  DNTT_z_{z}_intra.log   → E1-E4  (4ペア)
      inter1/ DNTT_z_{z}_inter1.log  → E5-E13 (9ペア)
      inter2/ DNTT_z_{z}_inter2.log  → E14-E17 (4ペア)

エネルギー計算式:
  E_total = 2*(E1+E2+E3+E4
              +E5+E6+E7+E8+E9
              +(E10+E11+E12+E13+E14+E15+E16+E17)/2)

使い方:
  python -m auto_opt.gaussian.collect_zscan_dft \\
      --inp-dir runs/DNTT_z_scan_2/inp \\
      --monomer DNTT
"""

import re
import csv
import argparse
import sys
from pathlib import Path
from typing import List, Optional


# ─── エネルギー読み取り ───────────────────────────────────────────────

def dft_get_E(log_path: str) -> List[float]:
    """Gaussian counterpoise=2 のlogから相互作用エネルギーリストを返す。"""
    lines_E = []
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "E(R" in line and len(line.split()) > 5:
                    lines_E.append(float(line.split()[4]) * 627.510)
    except FileNotFoundError:
        return []
    return [lines_E[5*i] - lines_E[5*i+1] - lines_E[5*i+2]
            for i in range(len(lines_E) // 5)]


def check_normal_termination(log_path: str) -> bool:
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            return "Normal termination" in f.read()
    except FileNotFoundError:
        return False


# ─── 式の適用 ────────────────────────────────────────────────────────

def calc_E_total(E_intra, E_inter1, E_inter2) -> Optional[float]:
    """
    E_total = 2*(E_intra_sum + E_inter1_full + (E_inter1_half + E_inter2_sum)/2)
    """
    if len(E_intra) != 4 or len(E_inter1) != 9 or len(E_inter2) != 4:
        return None
    intra_sum       = sum(E_intra)
    inter1_full_sum = sum(E_inter1[:5])
    inter1_half_sum = sum(E_inter1[5:])
    inter2_sum      = sum(E_inter2)
    return 2.0 * (intra_sum + inter1_full_sum + (inter1_half_sum + inter2_sum) / 2.0)


# ─── メイン処理 ──────────────────────────────────────────────────────

def collect(inp_dir: str, monomer: str) -> None:
    inp_root = Path(inp_dir)

    # z ディレクトリを収集してz値でソート
    z_dirs = sorted(
        [d for d in inp_root.iterdir() if d.is_dir() and d.name.startswith("z_")],
        key=lambda d: float(d.name[2:].replace("p", "."))
    )

    if not z_dirs:
        print(f"ERROR: {inp_dir} に z_* ディレクトリが見つかりません")
        sys.exit(1)

    results = []

    for z_dir in z_dirs:
        z_val = float(z_dir.name[2:].replace("p", "."))

        # ログファイルのパスを構築
        def find_log(calc_type):
            # ファイル名パターン: DNTT_z_{z}_{type}.log
            logs = list((z_dir / calc_type).glob(f"{monomer}_z_*_{calc_type}.log"))
            return str(logs[0]) if logs else None

        log_intra  = find_log("intra")
        log_inter1 = find_log("inter1")
        log_inter2 = find_log("inter2")

        # 計算状況の確認
        ok_intra  = check_normal_termination(log_intra)  if log_intra  else False
        ok_inter1 = check_normal_termination(log_inter1) if log_inter1 else False
        ok_inter2 = check_normal_termination(log_inter2) if log_inter2 else False

        status = "OK" if (ok_intra and ok_inter1 and ok_inter2) else "INCOMPLETE"

        # エネルギー読み取り
        E_intra  = dft_get_E(log_intra)  if log_intra  else []
        E_inter1 = dft_get_E(log_inter1) if log_inter1 else []
        E_inter2 = dft_get_E(log_inter2) if log_inter2 else []

        E_total = calc_E_total(E_intra, E_inter1, E_inter2)

        results.append({
            "z"        : z_val,
            "status"   : status,
            "E_intra"  : E_intra,
            "E_inter1" : E_inter1,
            "E_inter2" : E_inter2,
            "E_total"  : E_total,
        })

        n_i  = len(E_intra)
        n_i1 = len(E_inter1)
        n_i2 = len(E_inter2)
        E_str = f"{E_total:.4f}" if E_total is not None else "N/A"
        print(f"  z={z_val:+.1f}  [{status}]  "
              f"intra:{n_i}/4  inter1:{n_i1}/9  inter2:{n_i2}/4  "
              f"E_total={E_str} kcal/mol")

    # ── 相対エネルギーの計算（最小zを基準）──────────────────────────
    valid = [r for r in results if r["E_total"] is not None]
    E_ref = valid[0]["E_total"] if valid else None

    # ── サマリー表示 ──────────────────────────────────────────────────
    print(f"\n{'z':>6}  {'E_total':>12}  {'ΔE_total':>12}")
    print("-" * 35)
    min_dE = min((r["E_total"] - E_ref for r in valid), default=None)
    for r in results:
        if r["E_total"] is not None:
            dE   = r["E_total"] - E_ref
            mark = "  ←最小" if abs(dE - min_dE) < 1e-6 else ""
            print(f"{r['z']:>+6.1f}  {r['E_total']:>12.4f}  {dE:>+12.4f}{mark}")
        else:
            print(f"{r['z']:>+6.1f}  {'N/A':>12}  {'N/A':>12}  [{r['status']}]")

    # ── CSV 出力 ──────────────────────────────────────────────────────
    csv_path = inp_root / "dft_summary.csv"
    intra_headers  = [f"E{i}_kcalmol"  for i in range(1,  5)]
    inter1_headers = [f"E{i}_kcalmol"  for i in range(5, 14)]
    inter2_headers = [f"E{i}_kcalmol"  for i in range(14, 18)]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["z", "status"]
            + intra_headers + inter1_headers + inter2_headers
            + ["E_total_kcalmol", "dE_total_kcalmol"]
        )
        for r in results:
            def fmt(v): return f"{v:.4f}" if v is not None else ""
            dE = (r["E_total"] - E_ref) if r["E_total"] is not None and E_ref else None
            all_E = r["E_intra"] + r["E_inter1"] + r["E_inter2"]
            # 不足分をNoneで埋める
            all_E += [None] * (17 - len(all_E))
            w.writerow(
                [r["z"], r["status"]]
                + [fmt(e) for e in all_E]
                + [fmt(r["E_total"]), fmt(dE)]
            )

    print(f"\nCSV保存: {csv_path}")


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inp-dir", required=True,
                        help="z_* サブディレクトリを含む inp ディレクトリ")
    parser.add_argument("--monomer", default="DNTT",
                        help="モノマー名（デフォルト: DNTT）")
    args = parser.parse_args()
    collect(args.inp_dir, args.monomer)


if __name__ == "__main__":
    main()
