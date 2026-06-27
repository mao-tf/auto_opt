#!/usr/bin/env python3
"""
VdW グリッド全点の力場1点エネルギー計算。

step1_init_params.csv の各行についてダイマー mol2 を生成し
sander maxcyc=0 を実行する。
  glide: E = 2*E1 + 2*E2 + 4*E3
  screw: E = 2*E1 + 2*E2 + 2*E3 + 2*E4

結果は step1_init_params.csv に E カラムとして追記される。

Usage (単独/分割後の split dir で実行):
  python -m auto_opt.amber.eval_vdw_grid \
      --auto-dir runs/BTBT_glide/eval_split_0 \
      --monomer-name BTBT \
      --symmetry glide
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from auto_opt.utils import amber_get_E

_RES      = Path(__file__).resolve().parent / "resources"
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_AMBER_REF = _DATA_DIR / "amber_ref"

# structure_type → 重み（E = Σ weight * E_dimer）
_GLIDE_WEIGHTS = {1: 2, 2: 2, 3: 4}
_SCREW_WEIGHTS = {1: 2, 2: 2, 3: 2, 4: 2}


def _get_e_mono(monomer_name: str) -> float:
    cand = _AMBER_REF / f'{monomer_name}_gaff2.out'
    if not cand.exists():
        cands = sorted(_AMBER_REF.glob(f'{monomer_name}*.out'))
        if not cands:
            raise FileNotFoundError(f"amber_ref not found for {monomer_name}")
        cand = cands[0]
    return amber_get_E(str(cand))[0]


def _prepare_resources(auto_dir: Path) -> None:
    amber_dir = auto_dir / 'amber'
    amber_dir.mkdir(parents=True, exist_ok=True)
    src = _RES / 'FF_calc.in'
    if src.exists():
        shutil.copy2(src, amber_dir / 'FF_calc.in')


def _eval_row(auto_dir: str, monomer_name: str, params: dict,
              e_mono: float, weights: dict, exec_gjf) -> float | None:
    amber_dir = Path(auto_dir) / 'amber'
    total = 0.0
    for stype, w in weights.items():
        out_name = exec_gjf(auto_dir, monomer_name, params, stype, isTest=False)
        e_list = amber_get_E(str(amber_dir / out_name))
        if not e_list:
            return None
        total += w * (e_list[0] - 2 * e_mono)
    return total


def run_eval(auto_dir: str, monomer_name: str, symmetry: str) -> None:
    base = Path(auto_dir).resolve()
    _prepare_resources(base)

    csv_path = base / 'step1_init_params.csv'
    df = pd.read_csv(csv_path)

    e_mono = _get_e_mono(monomer_name)
    print(f"E_mono = {e_mono:.4f} kcal/mol")
    print(f"{len(df)} 点を評価します ({symmetry})...")

    if symmetry == 'glide':
        from auto_opt.amber.make_io_gene_phi import exec_gjf
        weights = _GLIDE_WEIGHTS
    else:
        from auto_opt.amber.make_io_gene_screw_phi import exec_gjf
        weights = _SCREW_WEIGHTS

    results = []
    for idx, row in df.iterrows():
        E = _eval_row(str(base), monomer_name, row.to_dict(), e_mono, weights, exec_gjf)
        results.append(round(E, 4) if E is not None else None)
        status = f"{E:.3f}" if E is not None else "FAILED"
        print(f"  [{idx+1}/{len(df)}] E = {status}")

    df['E'] = results
    df.to_csv(csv_path, index=False)
    print(f"\n完了: {csv_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='VdW グリッド全点の力場1点エネルギーを計算して E カラムを追加する'
    )
    ap.add_argument('--auto-dir',     required=True)
    ap.add_argument('--monomer-name', required=True)
    ap.add_argument('--symmetry',     default='glide', choices=['glide', 'screw'])
    args = ap.parse_args()
    run_eval(args.auto_dir, args.monomer_name, args.symmetry)
