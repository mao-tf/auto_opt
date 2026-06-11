#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMBER計算と同じXYZファイルから Gaussian DFT 用 .inp ファイルを生成する。

=========================================================
AMBER と DFT で「同じ相互作用」を計算できているか
=========================================================

  AMBER: E_int(i,j) = E_dimer(center,j) - 2 × E_mono_ref
    → XYZ の分子[0] を中心として、分子[1..N-1] との各ペアを計算

  DFT: Counterpoise=2 (B3LYP/6-311G** GD3BJ)
    → 同じXYZの分子[0] を Fragment=1
       分子[j] を Fragment=2 として CP 補正相互作用エネルギーを計算

  同じXYZファイルを使う → 同じ幾何構造 → 同じペアの相互作用を計算

=========================================================
生成される .inp ファイルの構造
=========================================================

  1ファイルに全ペアを --Link1-- で連結:

    %mem=15GB
    %nproc=40
    #P B3LYP/6-311G** EmpiricalDispersion=GD3BJ Counterpoise=2

    description_Pair_1

    0 1 0 1 0 1
    S(Fragment=1) x y z   ← 中心分子
    ...
    S(Fragment=2) x y z   ← 周辺分子1
    ...

    --Link1--
    ...                   ← 周辺分子2 との同様のブロック

=========================================================
使い方
=========================================================

  python -m auto_opt.gaussian.make_inp_from_xyz \\
      --xyz-dir runs/DNTT_z_scan/xyz \\
      --inp-dir runs/DNTT_z_scan/inp \\
      --monomer DNTT

  # テストモード（生成内容を確認するだけ）
  python -m auto_opt.gaussian.make_inp_from_xyz \\
      --xyz-dir runs/DNTT_z_scan/xyz \\
      --inp-dir runs/DNTT_z_scan/inp \\
      --monomer DNTT --test
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

# ─── Gaussian 計算設定 ────────────────────────────────────────────────
MACHINE_SPEC = {
    "gr1": {"nproc": 40, "mem": "15GB", "queue": "gr1.q"},
    "gr2": {"nproc": 52, "mem": "15GB", "queue": "gr2.q"},
}
METHOD = "PBEPBE/cc-pVTZ EmpiricalDispersion=GD3BJ Counterpoise=2"
CHARGE_MULT = "0 1 0 1 0 1"   # 全体: 0,1  Fragment1: 0,1  Fragment2: 0,1


# ─────────────────────────────────────────────────────────────────────
#  XYZ 読み込み・分割
# ─────────────────────────────────────────────────────────────────────

def parse_xyz(xyz_path: str) -> Tuple[List[str], np.ndarray]:
    """XYZファイルを読んで (elements, coords) を返す。"""
    with open(xyz_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    elements, coords = [], []
    for ln in lines[2: 2 + n]:
        p = ln.split()
        if len(p) < 4:
            continue
        elements.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return elements, np.array(coords)


def split_monomers(
    elements: List[str],
    coords: np.ndarray,
    n_mono: int,
) -> List[Tuple[List[str], np.ndarray]]:
    """全原子を n_mono 原子ずつに分割してモノマーリストを返す。"""
    n_total = len(elements)
    if n_total % n_mono != 0:
        raise ValueError(
            f"全原子数({n_total}) がモノマー原子数({n_mono}) で割り切れません"
        )
    n_mols = n_total // n_mono
    return [
        (elements[i * n_mono:(i + 1) * n_mono],
         coords  [i * n_mono:(i + 1) * n_mono])
        for i in range(n_mols)
    ]


def detect_n_mono(monomer_name: str) -> int:
    """
    data/monomer/{monomer}.csv から1モノマーの原子数を取得する。
    CSV がなければ 36（DNTT デフォルト）を返す。
    """
    root = Path(__file__).resolve().parents[3]
    csv_path = root / "data" / "monomer" / f"{monomer_name}.csv"
    if csv_path.exists():
        import pandas as pd
        return len(pd.read_csv(csv_path))
    return 36  # DNTT デフォルト


# ─────────────────────────────────────────────────────────────────────
#  Gaussian inp 生成
# ─────────────────────────────────────────────────────────────────────

def _pair_block(
    center_elems  : List[str],
    center_coords : np.ndarray,
    peri_elems    : List[str],
    peri_coords   : np.ndarray,
    description   : str,
    nproc         : int,
    mem           : str,
) -> List[str]:
    """1ペア分の Gaussian ジョブブロックを返す。"""
    lines = [
        f"%mem={mem}\n",
        f"%nproc={nproc}\n",
        f"#P {METHOD}\n",
        "\n",
        f"{description}\n",
        "\n",
        f"{CHARGE_MULT}\n",
    ]
    # Fragment=1（中心分子）
    for sym, (x, y, z) in zip(center_elems, center_coords):
        lines.append(f"{sym}(Fragment=1) {x:.6f} {y:.6f} {z:.6f}\n")
    # Fragment=2（周辺分子）
    for sym, (x, y, z) in zip(peri_elems, peri_coords):
        lines.append(f"{sym}(Fragment=2) {x:.6f} {y:.6f} {z:.6f}\n")
    lines.append("\n")
    return lines


def make_inp_from_xyz(
    xyz_path    : str,
    inp_path    : str,
    description : str,
    machine     : str = "gr1",
    n_mono      : int = 36,
) -> int:
    """
    XYZファイルを読んで Gaussian .inp ファイルを生成する。

    - 分子[0] を中心（Fragment=1）として扱う
    - 分子[1..N-1] それぞれと --Link1-- で連結したペアブロックを生成
    - AMBER の --center 0 と完全に同じペアを計算

    Returns:
        生成したペア数（= 分子数 - 1）
    """
    spec = MACHINE_SPEC[machine]
    elements, coords = parse_xyz(xyz_path)
    monomers = split_monomers(elements, coords, n_mono)
    n_mols = len(monomers)

    center_elems, center_coords = monomers[0]

    all_lines: List[str] = []
    for i in range(1, n_mols):
        peri_elems, peri_coords = monomers[i]
        desc = f"{description}_Pair_{i}"

        if i > 1:
            all_lines.append("--Link1--\n")

        all_lines += _pair_block(
            center_elems, center_coords,
            peri_elems, peri_coords,
            desc,
            spec["nproc"], spec["mem"],
        )

    # 末尾に空行3行（Gaussian の慣例）
    all_lines += ["\n", "\n", "\n"]

    Path(inp_path).parent.mkdir(parents=True, exist_ok=True)
    Path(inp_path).write_text("".join(all_lines), encoding="utf-8")
    return n_mols - 1


# ─────────────────────────────────────────────────────────────────────
#  XYZ ディレクトリの一括処理
# ─────────────────────────────────────────────────────────────────────

_PAT = re.compile(r'.+_z_(-?[\d.]+)_(intra|inter1|inter2)\.xyz$')

def parse_xyz_name(fname: str) -> Tuple[Optional[float], Optional[str]]:
    m = _PAT.match(fname)
    if m:
        return float(m.group(1)), m.group(2)
    return None, None


def make_all_inp(
    xyz_dir     : str,
    inp_dir     : str,
    monomer_name: str,
    isTest      : bool = False,
) -> None:
    """
    xyz_dir 内の全 XYZ ファイルから inp ファイルを一括生成する。

    出力構造:
      inp_dir/
        z_{z}/
          intra/   {monomer}_z_{z}_intra.inp
          inter1/  {monomer}_z_{z}_inter1.inp
          inter2/  {monomer}_z_{z}_inter2.inp
    """
    n_mono = detect_n_mono(monomer_name)
    print(f"モノマー原子数: {n_mono}")

    # XYZ ファイルを z 値・種別でグループ化
    groups: Dict[float, Dict[str, str]] = {}
    for f in sorted(Path(xyz_dir).glob(f"{monomer_name}_z_*.xyz")):
        z, ftype = parse_xyz_name(f.name)
        if z is None:
            continue
        groups.setdefault(z, {})[ftype] = str(f)

    if not groups:
        print(f"ERROR: {xyz_dir} に {monomer_name}_z_*.xyz が見つかりません")
        sys.exit(1)

    # 期待するペア数の定義
    expected_pairs = {"intra": 4, "inter1": 9, "inter2": 4}

    total_inp = 0
    for z in sorted(groups.keys()):
        z_label = f"{z:.1f}".replace(".", "p")
        print(f"\nz = {z}")

        for ftype in ("intra", "inter1", "inter2"):
            if ftype not in groups[z]:
                print(f"  [SKIP] {ftype}: ファイルなし")
                continue

            xyz_path = groups[z][ftype]

            # gr1/gr2 の交互割り当て（z × 3種の通し番号で判定）
            machine = "gr1" if total_inp % 2 == 0 else "gr2"

            inp_name  = f"{monomer_name}_z_{z}_{ftype}.inp"
            inp_path  = os.path.join(inp_dir, f"z_{z_label}", ftype, inp_name)
            desc      = f"{monomer_name}_z_{z}_{ftype}"

            if isTest:
                elems, coords = parse_xyz(xyz_path)
                n_mols = len(elems) // n_mono
                n_pairs = n_mols - 1
                print(f"  [TEST] {ftype}: {n_mols}分子 → {n_pairs}ペア"
                      f"  ({machine}, {MACHINE_SPEC[machine]['nproc']}procs)")
                print(f"         inp → {inp_path}")
                if n_pairs != expected_pairs.get(ftype, n_pairs):
                    print(f"  [WARNING] 期待ペア数={expected_pairs[ftype]}, 実際={n_pairs}")
            else:
                n_pairs = make_inp_from_xyz(
                    xyz_path, inp_path, desc,
                    machine=machine, n_mono=n_mono,
                )
                print(f"  {ftype}: {n_pairs}ペア → {inp_path}  ({machine})")
                if n_pairs != expected_pairs.get(ftype, n_pairs):
                    print(f"  [WARNING] 期待ペア数={expected_pairs[ftype]}, 実際={n_pairs}")

            total_inp += 1

    if not isTest:
        print(f"\n合計 {total_inp} ファイル生成完了: {inp_dir}")
    else:
        print(f"\n[TEST] 合計 {total_inp} ファイルを生成予定")


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
        help="XYZファイルが入ったディレクトリ（driver_crystal_energy と同じもの）",
    )
    parser.add_argument(
        "--inp-dir", required=True,
        help="生成した .inp ファイルの出力先ディレクトリ",
    )
    parser.add_argument(
        "--monomer", default="DNTT",
        help="モノマー名（デフォルト: DNTT）",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="テストモード: 生成内容を確認するだけ（ファイルは作らない）",
    )
    args = parser.parse_args()

    make_all_inp(
        xyz_dir      = args.xyz_dir,
        inp_dir      = args.inp_dir,
        monomer_name = args.monomer,
        isTest       = args.test,
    )


if __name__ == "__main__":
    main()
