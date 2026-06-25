#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XYZファイル中の複数DNTT分子の全ペア相互作用エネルギーをAMBER(GAFF2)で計算する。

=========================================================
必要なファイルの役割（整理）
=========================================================

  data/monomer/{MONOMER}_HF_esp.mol2
      GAFF2アトムタイプ + RESP部分電荷が入ったモノマーmol2
      → ダイマーmol2のアトムタイプ・電荷・結合情報のテンプレート
      → parmchk2 の入力（frcmod生成）

  data/amber_ref/{MONOMER}_HF_esp_gaff2.out
      モノマー1分子のsander計算結果（HPCで計算済み）
      → モノマーのエネルギー E_mono をここから読み取る
      → ダイマー計算のたびにモノマーを再計算する必要はない

  このスクリプトが自動生成するもの（手動作業不要）:
      {out_dir}/{MONOMER}_gaff2.frcmod    parmchk2 で生成
      {out_dir}/dimer_{i}_{j}.mol2        ダイマー座標 + テンプレートの型/電荷
      {out_dir}/dimer_{i}_{j}_tleap.in   tleap 入力
      {out_dir}/dimer_{i}_{j}.prmtop/.inpcrd/.out   AMBER 出力

=========================================================
相互作用エネルギーの計算式
=========================================================

  E_int(i,j) = E_dimer(i,j) - E_mono_ref - E_mono_ref
             = E_dimer(i,j) - 2 × E_mono_ref

  E_total = Σ_{i<j} E_int(i,j)

=========================================================
使い方
=========================================================

  # 本番実行（AMBERが必要）:
  python -m auto_opt.amber.calc_pentamer_interaction \\
      --xyz pentamer.xyz --monomer DNTT --out-dir ./pentamer_calc

  # テストモード（mol2/tleap.in だけ生成、sander未実行）:
  python -m auto_opt.amber.calc_pentamer_interaction \\
      --xyz pentamer.xyz --monomer DNTT --out-dir ./pentamer_calc --test
"""

import os
import sys
import shutil
import argparse
import subprocess
import itertools
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

import numpy as np

# ─── パス設定 ────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[3]
DATA     = ROOT / "data"
MONO_DIR = DATA / "monomer"
REF_DIR  = DATA / "amber_ref"
FF_CALC_IN = Path(__file__).parent / "resources" / "FF_calc.in"

AMBER_ENV = "source ~/anaconda3/etc/profile.d/conda.sh && conda activate amber"


# ─────────────────────────────────────────────────────────────────────
#  mol2 解析ユーティリティ
# ─────────────────────────────────────────────────────────────────────

def _find_section(lines: List[str], tag: str) -> int:
    tag_u = tag.strip().upper()
    for i, ln in enumerate(lines):
        if ln.strip().upper() == tag_u:
            return i
    return -1


def _next_section(lines: List[str], start: int) -> int:
    j = start + 1
    while j < len(lines) and not lines[j].startswith("@<TRIPOS>"):
        j += 1
    return j


def find_template_mol2(monomer_name: str) -> Path:
    """
    data/monomer/ から {monomer}_HF_esp.mol2 を優先し、なければ {monomer}.mol2 を探す。
    見つからない場合は FileNotFoundError を raise する。
    """
    for cand in [
        MONO_DIR / f"{monomer_name}_HF_esp.mol2",
        MONO_DIR / f"{monomer_name}.mol2",
    ]:
        if cand.exists():
            return cand
    # ワイルドカード検索
    matches = sorted(MONO_DIR.glob(f"{monomer_name}*.mol2"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"モノマーmol2が見つかりません: data/monomer/{monomer_name}_HF_esp.mol2\n"
        f"HPCからダウンロードするか、antechamber で作成してください。"
    )


def find_ref_energy_file(monomer_name: str) -> Optional[Path]:
    """
    data/amber_ref/ または data/monomer/ から モノマーsander出力ファイルを探す。
    """
    for cand in [
        REF_DIR  / f"{monomer_name}_HF_esp_gaff2.out",
        MONO_DIR / f"{monomer_name}_HF_esp_gaff2.out",
    ]:
        if cand.exists():
            return cand
    return None


@lru_cache(maxsize=4)
def load_mol2_params(
    mol2_path: str,
) -> Tuple[List[Tuple[str, float]], List[Tuple[int, int, str]]]:
    """
    mol2 ファイルからアトムタイプ・部分電荷・結合情報を抽出する。

    Returns:
        types_charges : [(atom_type, charge), ...]   (原子順, 1-index)
        bonds         : [(a, b, bond_type), ...]     (1-index)
    """
    lines = Path(mol2_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    # ATOM セクション
    i_a = _find_section(lines, "@<TRIPOS>ATOM")
    if i_a < 0:
        raise ValueError(f"@<TRIPOS>ATOM が見つかりません: {mol2_path}")
    j_a = _next_section(lines, i_a)

    types_charges: List[Tuple[str, float]] = []
    for ln in lines[i_a + 1 : j_a]:
        p = ln.split()
        if len(p) < 9:
            continue
        try:
            int(p[0])                   # atom_id（数値チェック）
            types_charges.append((p[5], float(p[8])))   # (type, charge)
        except (ValueError, IndexError):
            continue

    if not types_charges:
        raise ValueError(f"ATOMレコードが解析できませんでした: {mol2_path}")

    # BOND セクション
    i_b = _find_section(lines, "@<TRIPOS>BOND")
    bonds: List[Tuple[int, int, str]] = []
    if i_b >= 0:
        j_b = _next_section(lines, i_b)
        for ln in lines[i_b + 1 : j_b]:
            p = ln.split()
            if len(p) < 4:
                continue
            try:
                bonds.append((int(p[1]), int(p[2]), p[3]))
            except (ValueError, IndexError):
                pass

    return types_charges, bonds


# ─────────────────────────────────────────────────────────────────────
#  XYZ 解析
# ─────────────────────────────────────────────────────────────────────

def parse_xyz(
    xyz_path: str,
) -> Tuple[List[str], np.ndarray]:
    """
    XYZファイルを読み込み (elements, coords) を返す。
    """
    with open(xyz_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    elements, coords = [], []
    for ln in lines[2 : 2 + n]:
        p = ln.split()
        if len(p) < 4:
            continue
        elements.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    if len(coords) != n:
        raise ValueError(f"XYZ: ヘッダの原子数({n}) と実際の行数({len(coords)}) が不一致")
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
            f"全原子数({n_total}) がモノマー原子数({n_mono}) で割り切れません。\n"
            f"--n-atoms で正しい値を指定してください。"
        )
    n_mols = n_total // n_mono
    return [
        (elements[i * n_mono : (i + 1) * n_mono],
         coords  [i * n_mono : (i + 1) * n_mono])
        for i in range(n_mols)
    ]


# ─────────────────────────────────────────────────────────────────────
#  mol2 書き出し
# ─────────────────────────────────────────────────────────────────────

def write_dimer_mol2(
    out_path: str,
    xyz_i: np.ndarray,
    xyz_j: np.ndarray,
    elems_i: List[str],
    elems_j: List[str],
    template_mol2: str,
) -> None:
    """
    ダイマーmol2を生成する。
      - 座標      : xyz_i (RES1) + xyz_j (RES2)
      - 型/電荷   : template_mol2 から複製
      - 結合      : template の結合を両残基分それぞれ記述
    """
    types_charges, bonds = load_mol2_params(template_mol2)
    n_mono = len(types_charges)

    if xyz_i.shape[0] != n_mono or xyz_j.shape[0] != n_mono:
        raise ValueError(
            f"座標の原子数({xyz_i.shape[0]}, {xyz_j.shape[0]}) と "
            f"テンプレートmol2の原子数({n_mono}) が一致しません"
        )

    n_total = 2 * n_mono
    n_bonds  = 2 * len(bonds)
    all_xyz   = np.vstack([xyz_i, xyz_j])
    all_elems = list(elems_i) + list(elems_j)

    lines = [
        "@<TRIPOS>MOLECULE\n",
        "DNTT_dimer\n",
        f"{n_total:6d}{n_bonds:7d}{2:6d}{0:6d}{0:6d}\n",
        "SMALL\n",
        "bcc\n",
        "\n",
        "@<TRIPOS>ATOM\n",
    ]
    for i in range(n_total):
        x, y, z        = all_xyz[i]
        atype, charge  = types_charges[i % n_mono]
        frag           = 1 if i < n_mono else 2
        res            = "RES1" if frag == 1 else "RES2"
        sym            = all_elems[i]
        lines.append(
            f"{i+1:6d} {sym:<2s} {x: .6f} {y: .6f} {z: .6f} "
            f"{atype} {frag:3d} {res:<4s} {charge: .6f}\n"
        )

    lines.append("@<TRIPOS>BOND\n")
    bid = 1
    for a, b, bt in bonds:                       # RES1
        lines.append(f"{bid:6d}{a:6d}{b:6d} {bt}\n"); bid += 1
    for a, b, bt in bonds:                       # RES2（インデックスオフセット）
        lines.append(f"{bid:6d}{a+n_mono:6d}{b+n_mono:6d} {bt}\n"); bid += 1

    lines += [
        "@<TRIPOS>SUBSTRUCTURE\n",
        f"{1:6d} RES1{1:10d} GROUP             0 ****  ****    0\n",
        f"{2:6d} RES2{n_mono+1:10d} GROUP             0 ****  ****    0\n",
        "\n",
    ]
    Path(out_path).write_text("".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
#  AMBER 実行 / エネルギー読み取り
# ─────────────────────────────────────────────────────────────────────

def read_amber_energy(out_path: str) -> Optional[float]:
    """
    sander の .out ファイルから全ポテンシャルエネルギー [kcal/mol] を読み取る。
    "NSTEP  ENERGY  RMS ..." 行の直後の行の 2列目が エネルギー。
    """
    try:
        with open(out_path) as f:
            hit = False
            for ln in f:
                if hit:
                    parts = ln.split()
                    if len(parts) >= 2:
                        return float(parts[1])
                    break
                if " RMS " in ln:
                    hit = True
    except Exception:
        pass
    return None


def run_dimer(
    work_dir: str,
    mol2_file: str,
    frcmod_name: str,
    ff_calc_in: str = "FF_calc.in",
    isTest: bool   = False,
) -> Optional[float]:
    """
    tleap → sander を実行してダイマーエネルギーを返す。
    work_dir 内のファイルをすべて work_dir 相対で処理する。
    """
    base       = os.path.splitext(mol2_file)[0]
    tleap_in   = f"{base}_tleap.in"
    tleap_out  = f"{base}_tleap.out"
    prmtop     = f"{base}.prmtop"
    inpcrd     = f"{base}.inpcrd"
    rst7       = f"{base}.rst7"
    out_file   = f"{base}.out"

    # tleap 入力を書く
    tleap_lines = [
        "source leaprc.gaff2\n",
        f"loadamberparams {frcmod_name}\n",
        f"SYS = loadmol2 {mol2_file}\n",
        f"saveamberparm SYS {prmtop} {inpcrd}\n",
        "quit\n",
    ]
    with open(os.path.join(work_dir, tleap_in), "w") as f:
        f.writelines(tleap_lines)

    cmd = (
        f"cd {work_dir} && {AMBER_ENV} && "
        f"tleap -f {tleap_in} > {tleap_out} 2>&1 && "
        f"sander -O -i {ff_calc_in} -o {out_file} "
        f"-p {prmtop} -c {inpcrd} -r {rst7} 2>&1"
    )

    if isTest:
        print(f"    [TEST] {mol2_file}")
        return None

    result = subprocess.run(
        cmd, shell=True, executable="/bin/bash",
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        # tleap ログを表示して原因を把握しやすくする
        tlog = os.path.join(work_dir, tleap_out)
        if os.path.exists(tlog):
            with open(tlog) as f:
                tail = f.read()[-600:]
            print(f"    [WARNING] tleap エラー ({mol2_file}):\n{tail}")
        else:
            print(f"    [WARNING] AMBER失敗 ({mol2_file})")
        return None

    return read_amber_energy(os.path.join(work_dir, out_file))


# ─────────────────────────────────────────────────────────────────────
#  メイン計算ルーティン
# ─────────────────────────────────────────────────────────────────────

def calc_interaction_energies(
    xyz_path    : str,
    monomer_name: str,
    out_dir     : str,
    center_index: Optional[int] = None,
    isTest      : bool = False,
) -> Dict:
    """
    XYZ内の分子の相互作用エネルギーを計算する。

    center_index=None  → 全ユニークペア C(N,2) を計算
    center_index=k     → 分子k を中心として (k,j) のペアのみ計算（N-1 ペア）

    E_int(i,j) = E_dimer(i,j) - 2 × E_mono_ref

    Returns:
        {
          "n_molecules": int,
          "E_mono_ref": float,                       # kcal/mol
          "dimer_energies":       {(i,j): float},
          "interaction_energies": {(i,j): float},
          "total_interaction":    float,
        }
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── (A) テンプレートmol2 ──────────────────────────────────────────
    template = str(find_template_mol2(monomer_name))
    print(f"テンプレートmol2 : {template}")

    types_charges, _ = load_mol2_params(template)
    n_mono = len(types_charges)

    # ── (B) モノマー参照エネルギー ────────────────────────────────────
    ref_file = find_ref_energy_file(monomer_name)
    if ref_file is None:
        raise FileNotFoundError(
            f"モノマー参照エネルギーファイルが見つかりません。\n"
            f"期待するパス: data/amber_ref/{monomer_name}_HF_esp_gaff2.out"
        )
    e_mono_list = read_amber_energy(str(ref_file))
    if e_mono_list is None:
        raise ValueError(f"参照ファイルからエネルギーを読み取れませんでした: {ref_file}")
    E_mono_ref: float = e_mono_list  # type: ignore
    print(f"参照エネルギーファイル: {ref_file}")
    print(f"モノマー参照エネルギー: E_mono = {E_mono_ref:.4f} kcal/mol")

    # ── (C) XYZ 解析・分割 ───────────────────────────────────────────
    elements, coords = parse_xyz(xyz_path)
    monomers = split_monomers(elements, coords, n_mono)
    n_mols = len(monomers)

    # ── ペアの決定 ───────────────────────────────────────────────────
    if center_index is None:
        # 全ユニークペア
        pairs = list(itertools.combinations(range(n_mols), 2))
        mode_label = f"全ペア C({n_mols},2) = {len(pairs)} ペア"
    else:
        if not (0 <= center_index < n_mols):
            raise ValueError(
                f"--center {center_index} が範囲外です（分子数: {n_mols}）"
            )
        # 中心分子 center_index vs それ以外
        pairs = [
            (min(center_index, j), max(center_index, j))
            for j in range(n_mols) if j != center_index
        ]
        mode_label = (
            f"中心分子[{center_index}] vs 周辺 {n_mols - 1} 分子 = {len(pairs)} ペア"
        )

    print(f"分子数: {n_mols}  →  {mode_label}\n")

    # ── (D) FF_calc.in をコピー ──────────────────────────────────────
    ff_dest = os.path.join(out_dir, "FF_calc.in")
    if not os.path.exists(ff_dest):
        shutil.copy(str(FF_CALC_IN), ff_dest)

    # ── (E) frcmod 生成（1回だけ） ───────────────────────────────────
    frcmod_name = f"{monomer_name}_gaff2.frcmod"
    frcmod_path = os.path.join(out_dir, frcmod_name)
    if not os.path.exists(frcmod_path):
        if isTest:
            print(f"[TEST] parmchk2 → {frcmod_name}\n")
        else:
            print(f"parmchk2 実行中: {os.path.basename(template)} → {frcmod_name} ...")
            cmd = (
                f"cd {out_dir} && {AMBER_ENV} && "
                f'parmchk2 -s gaff2 -i "{template}" -f mol2 -o "{frcmod_name}"'
            )
            subprocess.run(cmd, shell=True, executable="/bin/bash")
            if not os.path.exists(frcmod_path):
                raise RuntimeError(
                    f"parmchk2 が {frcmod_name} を生成しませんでした。\n"
                    f"  conda activate amber が正常に動作しているか確認してください。"
                )
            print(f"  → {frcmod_path}\n")

    # ── (F) ダイマー計算 ─────────────────────────────────────────────
    print("[1/2] ダイマーmol2生成 + AMBER実行 ...")
    dimer_energies: Dict[Tuple[int, int], Optional[float]] = {}

    for i, j in pairs:
        elems_i, xyz_i = monomers[i]
        elems_j, xyz_j = monomers[j]
        mol2_file = f"dimer_{i}_{j}.mol2"

        write_dimer_mol2(
            os.path.join(out_dir, mol2_file),
            xyz_i, xyz_j, elems_i, elems_j, template,
        )
        E = run_dimer(out_dir, mol2_file, frcmod_name, isTest=isTest)
        status = f"{E:.4f} kcal/mol" if E is not None else "FAILED"
        print(f"  dimer({i},{j}): {status}")
        dimer_energies[(i, j)] = E

    # ── (G) 相互作用エネルギー ───────────────────────────────────────
    print(f"\n[2/2] 相互作用エネルギー計算  E_int = E_dimer - 2 × {E_mono_ref:.4f}")
    header = f"{'Pair':>8}  {'E_dimer':>14}  {'E_int':>14}"
    print(header)
    print("-" * len(header))

    interaction_energies: Dict[Tuple[int, int], Optional[float]] = {}
    total = 0.0
    for i, j in pairs:
        E_d = dimer_energies[(i, j)]
        if E_d is not None:
            dE = E_d - 2.0 * E_mono_ref
            interaction_energies[(i, j)] = dE
            total += dE
            print(f"  ({i},{j}):  {E_d:>14.4f}  {dE:>14.4f}")
        else:
            interaction_energies[(i, j)] = None
            print(f"  ({i},{j}):  {'N/A':>14}  {'N/A':>14}")

    print("-" * len(header))
    print(f"  {'合計':>6}   {'':>14}  {total:>14.4f} kcal/mol\n")

    # ── (H) CSV 出力 ─────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "interaction_energies.csv")
    with open(csv_path, "w") as f:
        f.write("pair_i,pair_j,E_dimer_kcalmol,E_mono_ref_kcalmol,E_int_kcalmol\n")
        for i, j in pairs:
            E_d = dimer_energies.get((i, j))
            dE  = interaction_energies.get((i, j))
            f.write(
                f"{i},{j},"
                f"{E_d  if E_d  is not None else ''},"
                f"{E_mono_ref:.4f},"
                f"{dE   if dE   is not None else ''}\n"
            )
    print(f"結果CSV: {csv_path}")

    return {
        "n_molecules"          : n_mols,
        "E_mono_ref"           : E_mono_ref,
        "dimer_energies"       : dimer_energies,
        "interaction_energies" : interaction_energies,
        "total_interaction"    : total,
    }


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--xyz", required=True,
        help="入力XYZファイルのパス（複数分子）",
    )
    parser.add_argument(
        "--monomer", default="DNTT",
        help="モノマー名（デフォルト: DNTT）。"
             "data/monomer/{name}_HF_esp.mol2 と data/amber_ref/{name}_HF_esp_gaff2.out を使用する。",
    )
    parser.add_argument(
        "--center", type=int, default=None, metavar="INDEX",
        help=(
            "中心分子のインデックス（0始まり）。"
            "指定すると その分子 vs 残り全分子 のペアだけを計算する。"
            "省略すると全ユニークペア C(N,2) を計算する。"
            "例: --center 0 → 分子0 vs 分子1,2,...,N-1"
        ),
    )
    parser.add_argument(
        "--out-dir", default="./pentamer_calc",
        help="計算ファイルの出力ディレクトリ（デフォルト: ./pentamer_calc）",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="テストモード: mol2/tleap.inを生成するが sander は実行しない",
    )
    args = parser.parse_args()

    calc_interaction_energies(
        xyz_path     = args.xyz,
        monomer_name = args.monomer,
        out_dir      = args.out_dir,
        center_index = args.center,
        isTest       = args.test,
    )


if __name__ == "__main__":
    main()
