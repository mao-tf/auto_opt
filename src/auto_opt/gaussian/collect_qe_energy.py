#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QEのrelaxation outputファイルから最終全エネルギーを収集してCSVにまとめる。

エネルギーはRy → kcal/mol に変換し、さらに1/2倍（2分子/ユニットセル）する。

使い方:
  python -m auto_opt.gaussian.collect_qe_energy \\
      --qe-dir runs/DNTT_qe_gaussian \\
      --output runs/DNTT_qe_gaussian/energy.csv
"""

import re
import sys
import csv
import argparse
from pathlib import Path
from typing import Optional

RY_TO_KCAL_PER_MOL = 313.7545  # 1 Ry = 13.6057 eV, 1 eV = 23.0605 kcal/mol


def extract_final_energy(out_path: str) -> Optional[float]:
    """QE outputファイルから最終全エネルギー (Ry) を返す。最後の出現を使う。"""
    text = Path(out_path).read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r'!\s+total energy\s+=\s+([-\d.]+)\s+Ry', text)
    if not matches:
        return None
    return float(matches[-1])


def run(qe_dir: str, output: str) -> None:
    qe_root = Path(qe_dir)
    pat = re.compile(r'dntt_relax_(-?[\d.]+)(?:_\w+)?\.out$')

    out_files = [f for f in qe_root.glob("dntt_relax_*.out") if pat.match(f.name)]
    if not out_files:
        print(f"ERROR: {qe_dir} に dntt_relax_*.out が見つかりません")
        sys.exit(1)
    out_files = sorted(out_files, key=lambda f: float(pat.match(f.name).group(1)))

    rows = []
    for out_file in out_files:
        z = float(pat.match(out_file.name).group(1))
        energy_ry = extract_final_energy(str(out_file))
        if energy_ry is None:
            print(f"  [SKIP] {out_file.name}: 最終エネルギーが見つかりません")
            continue
        energy_kcal = energy_ry * RY_TO_KCAL_PER_MOL * 0.5
        rows.append({"z": z, "E_Ry": energy_ry, "E_kcal_per_mol": energy_kcal})
        print(f"  z={z}: {energy_ry:.8f} Ry → {energy_kcal:.4f} kcal/mol")

    ref = next((r["E_kcal_per_mol"] for r in rows if r["z"] == 0.0), None)
    if ref is None:
        print("  [WARNING] z=0 のデータがないため相対エネルギーは計算できません")
    for r in rows:
        r["dE_kcal_per_mol"] = (r["E_kcal_per_mol"] - ref) if ref is not None else ""

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["z", "E_Ry", "E_kcal_per_mol", "dE_kcal_per_mol"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n完了。出力: {output}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--qe-dir", required=True,
                        help="QE output (.out) ファイルが入ったディレクトリ")
    parser.add_argument("--output", required=True,
                        help="出力CSVファイルのパス")
    args = parser.parse_args()
    run(args.qe_dir, args.output)


if __name__ == "__main__":
    main()
