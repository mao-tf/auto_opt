#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIF構造のDFT計算（Gaussian counterpoise=2）のlogファイルからエネルギーを回収する。

Gaussianのcounterpoise=2計算では各ペアの計算後に以下の行が出力される:
  Counterpoise: corrected interaction energy =  -0.01234 a.u.  -7.748 kcal/mol

この行からkcal/molの値を直接読み取る。

使い方:
  python -m auto_opt.gaussian.collect_cif \\
      --log-dir /Users/miyoshimao/Working/DNTT/inp/cif
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────
#  ログファイルからエネルギーを読み取る
# ─────────────────────────────────────────────────────────────────────

def read_cp_energies(log_path: str) -> List[float]:
    """
    Gaussian counterpoise=2 のlogファイルから
    各ペアの相互作用エネルギー [kcal/mol] のリストを返す。

    読み取る行の例:
      Counterpoise: corrected interaction energy =  -0.012345 a.u.  -7.748 kcal/mol
    """
    energies = []
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Counterpoise: corrected interaction energy" in line:
                    # "... = -0.012345 a.u.  -7.748 kcal/mol"
                    # kcal/mol の値を取得
                    if "kcal/mol" in line:
                        val = float(line.split("kcal/mol")[0].split()[-1])
                    else:
                        # kcal/mol 表記がない場合は a.u. から変換
                        val = float(line.split("=")[1].split()[0]) * 627.509
                    energies.append(val)
    except FileNotFoundError:
        pass
    return energies


def check_normal_termination(log_path: str) -> bool:
    """ログファイルが Normal termination で終わっているか確認する。"""
    try:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return "Normal termination" in content
    except FileNotFoundError:
        return False


# ─────────────────────────────────────────────────────────────────────
#  メイン処理
# ─────────────────────────────────────────────────────────────────────

def collect_cif_energies(log_dir: str) -> None:
    """
    log_dir 以下の intra/inter1/inter2 サブディレクトリから
    ログを読み取ってエネルギーを表示する。
    """
    log_dir = Path(log_dir)

    # 期待するファイルと構成
    configs = {
        "intra" : {"log": log_dir / "intra"  / "DNTT_cif_intra.log",
                   "n_pairs": 8,
                   "pair_label": "E1-E8"},
        "inter1": {"log": log_dir / "inter1" / "DNTT_cif_inter1.log",
                   "n_pairs": 9,
                   "pair_label": "E9-E17"},
        "inter2": {"log": log_dir / "inter2" / "DNTT_cif_inter2.log",
                   "n_pairs": 4,
                   "pair_label": "E18-E21"},
    }

    all_energies = {}

    for calc_type, cfg in configs.items():
        log_path = str(cfg["log"])
        print(f"\n[{calc_type}] {log_path}")

        if not Path(log_path).exists():
            print(f"  ログファイルが見つかりません")
            all_energies[calc_type] = []
            continue

        ok = check_normal_termination(log_path)
        print(f"  Normal termination: {'✓' if ok else '✗ 計算未完了または失敗'}")

        energies = read_cp_energies(log_path)
        all_energies[calc_type] = energies

        if not energies:
            print(f"  エネルギーが読み取れませんでした")
            continue

        print(f"  ペア数: {len(energies)} / {cfg['n_pairs']} ({cfg['pair_label']})")
        for i, E in enumerate(energies, 1):
            print(f"    Pair {i:2d}: {E:10.4f} kcal/mol")

    # 全エネルギーが揃っているか確認
    E_intra  = all_energies.get("intra",  [])
    E_inter1 = all_energies.get("inter1", [])
    E_inter2 = all_energies.get("inter2", [])

    if len(E_intra) == 8 and len(E_inter1) == 9 and len(E_inter2) == 4:
        _print_summary(E_intra, E_inter1, E_inter2)
    else:
        print(f"\n[!] 全ペアが揃っていないため合計は計算できません")
        print(f"    intra: {len(E_intra)}/8, inter1: {len(E_inter1)}/9, inter2: {len(E_inter2)}/4")


def _print_summary(E_intra, E_inter1, E_inter2):
    """
    E_total の計算式を適用して表示する。

    CIF構造の場合:
      intra  : E1-E8  (8ペア)
      inter1 : E9-E17 (先5ペア=全カウント, 後4ペア=1/2カウント)
      inter2 : E18-E21 (1/2カウント)

    E_total = 2*(intra_sum + inter1_full + (inter1_half + inter2_sum)/2)
    """
    intra_sum       = sum(E_intra)           # E1-E8 (8ペア全部)
    inter1_full_sum = sum(E_inter1[:5])      # E9-E13
    inter1_half_sum = sum(E_inter1[5:])      # E14-E17
    inter2_sum      = sum(E_inter2)          # E18-E21

    E_total = 2 * (intra_sum + inter1_full_sum + (inter1_half_sum + inter2_sum) / 2)

    print(f"\n{'='*50}")
    print(f"  エネルギー集計")
    print(f"{'='*50}")
    print(f"  E_intra  (E1-E8,  ×2): {intra_sum:10.4f} kcal/mol")
    print(f"  E_inter1_full (E9-E13, ×2): {inter1_full_sum:10.4f} kcal/mol")
    print(f"  E_inter1_half (E14-E17,×1): {inter1_half_sum:10.4f} kcal/mol")
    print(f"  E_inter2 (E18-E21, ×1): {inter2_sum:10.4f} kcal/mol")
    print(f"  {'─'*40}")
    print(f"  E_total                 = {E_total:10.4f} kcal/mol")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", required=True,
                        help="intra/inter1/inter2 サブディレクトリを含む親ディレクトリ")
    args = parser.parse_args()
    collect_cif_energies(args.log_dir)


if __name__ == "__main__":
    main()
