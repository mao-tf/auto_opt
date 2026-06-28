#!/usr/bin/env python3
"""
スタッキング計算結果の収集スクリプト

各 split_* ディレクトリの step1.csv を結合し、モノマーエネルギーを引いた
相互作用エネルギー E_int を計算して <auto-dir>/stacking_results.csv に保存する。

Usage:
  python -m auto_opt.stacking.merge_results \
      --auto-dir runs/DNTT_screw_stacking \
      --monomer-name DNTT
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from auto_opt.utils import amber_get_E

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_N_MONO_EFF = 18  # glide/screw ともにダイマーペアの重み付き合計で18モノマー分


def _find_amber_ref(monomer_name: str, data_dir: Path) -> Path:
    """amber_ref/{MON}_gaff2.out を探す。旧命名 ({MON}_HF_esp_gaff2.out) にも対応。"""
    cand = data_dir / 'amber_ref' / f'{monomer_name}_gaff2.out'
    if cand.exists():
        return cand
    cands = sorted((data_dir / 'amber_ref').glob(f'{monomer_name}*.out'))
    if cands:
        return cands[0]
    raise FileNotFoundError(
        f"amber_ref が見つかりません: {data_dir}/amber_ref/{monomer_name}*.out"
    )


_LAYER_JOIN_KEYS = ['alpha1', 'beta', 'phi', 'z']


def _join_layer_energy(merged: pd.DataFrame, candidates_csv: Path) -> pd.DataFrame:
    """stacking_candidates.csv の E（層内エネルギー）を E_layer として結合する。"""
    df_cands = pd.read_csv(candidates_csv)

    # candidates の alpha → alpha1 に合わせる
    if 'alpha' in df_cands.columns and 'alpha1' not in df_cands.columns:
        df_cands = df_cands.rename(columns={'alpha': 'alpha1'})

    join_keys = [k for k in _LAYER_JOIN_KEYS if k in merged.columns and k in df_cands.columns]
    if not join_keys:
        print("[警告] 結合キーが見つかりません。E_layer は追加しません。")
        return merged

    df_layer = df_cands[join_keys + ['E']].rename(columns={'E': 'E_layer'})
    df_layer = df_layer.drop_duplicates(subset=join_keys)

    merged = merged.merge(df_layer, on=join_keys, how='left')
    n_matched = merged['E_layer'].notna().sum()
    print(f"E_layer を結合しました ({n_matched}/{len(merged)} 行マッチ, キー: {join_keys})")
    return merged


def merge_csvs(auto_dir: str, monomer_name: str | None = None,
               data_dir: Path | None = None,
               candidates_csv: Path | None = None) -> None:
    base     = Path(auto_dir).resolve()
    data_dir = data_dir or _DATA_DIR
    dfs      = []

    for split_dir in sorted(base.glob('split_*')):
        csv_path = split_dir / 'step1.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"  {split_dir.name}: {len(df)} 行")
        else:
            print(f"  {split_dir.name}: step1.csv なし（スキップ）")

    if not dfs:
        print("結合するファイルが見つかりませんでした。")
        return

    merged = pd.concat(dfs, ignore_index=True)

    # モノマーエネルギーを引いて E_stack を計算
    if monomer_name and 'E' in merged.columns:
        ref_path = _find_amber_ref(monomer_name, data_dir)
        e_mono_list = amber_get_E(str(ref_path))
        if e_mono_list:
            e_mono = e_mono_list[0]
            merged['E_int'] = merged['E'] - _N_MONO_EFF * e_mono
            merged['E_stack'] = merged['E_int']
            print(f"\nモノマーエネルギー: {e_mono:.3f} kcal/mol ({ref_path.name})")
            print(f"E_stack = E - {_N_MONO_EFF} × {e_mono:.3f}")
        else:
            print(f"[警告] {ref_path} からエネルギーを読み取れませんでした。E_stack は計算しません。")

    # 候補 CSV から E_layer を結合して E_total を計算
    if candidates_csv is None:
        default_cands = base / 'stacking_candidates.csv'
        if default_cands.exists():
            candidates_csv = default_cands
    if candidates_csv is not None and candidates_csv.exists():
        merged = _join_layer_energy(merged, candidates_csv)
        if 'E_layer' in merged.columns and 'E_stack' in merged.columns:
            merged['E_total'] = merged['E_layer'] + 2 * merged['E_stack']

    # cy → cz の順で並び替え
    sort_cols = [c for c in ('cy', 'cz', 'phi', 'z', 'beta') if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    out_path = base / 'stacking_results.csv'
    merged.to_csv(out_path, index=False)

    print(f"\n{len(dfs)} ファイルを結合しました。")
    print(f"合計 {len(merged)} 行 → {out_path}")
    if 'E_stack' in merged.columns:
        best = merged.loc[merged['E_stack'].idxmin()]
        print(f"最安定: E_stack = {best['E_stack']:.3f} kcal/mol  (cy={best.get('cy', '?')}, cz={best.get('cz', '?')})")
    if 'E_total' in merged.columns:
        best_t = merged.loc[merged['E_total'].idxmin()]
        print(f"最安定(合計): E_total = {best_t['E_total']:.3f} kcal/mol  (cy={best_t.get('cy', '?')}, cz={best_t.get('cz', '?')})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='split_*/step1.csv を結合してスタッキング結果を出力する'
    )
    ap.add_argument('--auto-dir',     required=True, help='split_* を含む親ディレクトリ')
    ap.add_argument('--monomer-name', default=None,  help='モノマー名（E_stack 計算に使用）')
    ap.add_argument('--data-dir',     default=None,  help='data/ ディレクトリのパス（省略時はパッケージ付属）')
    ap.add_argument('--candidates',   default=None,  help='stacking_candidates.csv のパス（E_layer 結合用）')
    args = ap.parse_args()

    data_dir      = Path(args.data_dir)   if args.data_dir   else None
    candidates_csv = Path(args.candidates) if args.candidates else None
    merge_csvs(args.auto_dir, monomer_name=args.monomer_name,
               data_dir=data_dir, candidates_csv=candidates_csv)
