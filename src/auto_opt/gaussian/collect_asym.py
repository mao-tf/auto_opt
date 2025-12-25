#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python -m auto_opt.gaussian.collect --auto-dir runs/ANT_test --monomer ANT 

- filtered_step1.csv（列: alpha,a,b,z …）を読み、
  auto-dir/gaussian/内の .log を utils.dft_get_E() で解析。
- 列: alpha,a,b,z,E,E1,E2,E3,status,structure_type を dft_results.csv に出力。

structure_type の付け方（同じ (alpha,z) 内で判定）:
- 行が2つ:   a の小さい方 → 'a-stack', 大きい方 → 'b-stack'
- 行が3つ以上: a 昇順に 'a-stack', 'ch', 'b-stack'、4件目以降は 'other'
  ※まず E が有限(Done)の行だけで 2 or 3 件あるかを優先的に見て、該当しなければその (alpha,z) の全行で判定
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from auto_opt.utils import dft_get_E  # ログから [E1,E2,E3] を返す関数を想定

# ----------------- ヘルパ -----------------

def _round_for_name(alpha: float, phi: float, a: float, b: float, z: float):
    return int(round(alpha)), int(round(phi)), round(float(a), 1), round(float(b), 1), round(float(z), 1)

def _log_path(auto_dir: str, monomer: str, alpha: float, phi: float, a: float, b: float, z: float) -> Path:
    ai, pi, ar, br, zr = _round_for_name(alpha, phi, a, b, z)
    base = f"{monomer}_alpha={ai}_phi={pi}_a={ar}_b={br}_z={zr}"
    return Path(auto_dir) / "gaussian" / f"{base}.log"

def _combine_total(E1: float, E2: float, E3: float, E4: float) -> float:
    return 2*(E1 + E2 + E3 + E4)

def _assign_structure_types(df: pd.DataFrame) -> pd.DataFrame:
    """(alpha,z) ごとに a の昇順で a-stack/ch/b-stack を付与。"""
    out = df.copy()
    out['structure_type'] = ''
    if out.empty:
        return out

    for (alpha, phi, z), g_all in out.groupby(['alpha', 'phi', 'z'], sort=False):
        g_all = g_all.sort_values('a')
        order = list(g_all.index)

        if len(order) == 2:
            out.loc[order[0], 'structure_type'] = 'a-stack'
            out.loc[order[1], 'structure_type'] = 'b-stack'
        elif len(order) >= 3:
            out.loc[order[0], 'structure_type'] = 'a-stack'
            out.loc[order[1], 'structure_type'] = 'ch'
            out.loc[order[2], 'structure_type'] = 'b-stack'
            if len(order) > 3:
                out.loc[order[3:], 'structure_type'] = 'other'
        elif len(order) == 1:
            out.loc[order[0], 'structure_type'] = 'single'  # 念のため

    return out

# ----------------- 本体 -----------------

def collect_results(auto_dir: str, monomer: str, cand_csv: str, out_csv: str) -> pd.DataFrame:
    """
    cand_csv（通常: filtered_step1.csv）を読んで対応する .log を解析。
    出力: alpha,a,b,z,E,E1,E2,E3,status,structure_type を out_csv に保存。
    """
    df = pd.read_csv(cand_csv)
    
    need = {'alpha','phi','a','b','z'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{cand_csv} に必須列が不足しています: {sorted(missing)}")

    rows = []
    for r in df.itertuples(index=False):
        alpha, phi, a, b, z = float(r.alpha), float(r.phi), float(r.a), float(r.b), float(r.z)
        logp = _log_path(auto_dir, monomer, alpha, phi, a, b, z)

        rec = {'alpha': alpha, 'phi': phi, 'a': a, 'b': b, 'z': z,
               'E': np.nan, 'E1': np.nan, 'E2': np.nan, 'E3': np.nan, 'E4': np.nan, 'status': ''}

        if not logp.exists():
            rec['status'] = 'Missing'
            rows.append(rec); continue

        try:
            E_list = dft_get_E(str(logp))  # ログ → [E1,E2,E3]
        except Exception as e:
            rec['status'] = f'ParseError: {e}'
            rows.append(rec); continue

        if not E_list:
            rec['status'] = 'Empty'
            rows.append(rec); continue

        try:
            if len(E_list) >= 1: rec['E1'] = float(E_list[0])
            if len(E_list) >= 2: rec['E2'] = float(E_list[1])
            if len(E_list) >= 3: rec['E3'] = float(E_list[2])
            if len(E_list) >= 4: rec['E4'] = float(E_list[3])
        except Exception:
            rec['status'] = 'BadValues'
            rows.append(rec); continue

        if np.isfinite(rec['E1']) and np.isfinite(rec['E2']) and np.isfinite(rec['E3']) and np.isfinite(rec['E4']):
            rec['E'] = _combine_total(rec['E1'], rec['E2'], rec['E3'], rec['E4'])
            rec['status'] = 'Done'
        else:
            rec['status'] = 'Partial'

        rows.append(rec)

    out_df = pd.DataFrame(rows, columns=['alpha','phi', 'a','b','z','E','E1','E2','E3','E4','status'])

    # ★ ここで構造タイプを付ける
    out_df = _assign_structure_types(out_df)

    out_df.to_csv(out_csv, index=False)
    print(f"[collect] wrote {out_csv} (n={len(out_df)})")
    return out_df

def main():
    ap = argparse.ArgumentParser(description="DFT ログ(.log)集計 + structure_type 付与")
    ap.add_argument("--auto-dir", required=True, help="作業ディレクトリ（gaussian/ がある場所）")
    ap.add_argument("--monomer", required=True, help="モノマー名（例: PFA）")
    ap.add_argument("--candidates-csv", default=None, help="既定: <auto-dir>/filtered_step1.csv")
    ap.add_argument("--out-csv", default=None, help="既定: <auto-dir>/dft_results.csv")
    args = ap.parse_args()

    auto_dir = args.auto_dir
    cand_csv = args.candidates_csv or str(Path(auto_dir) / "filtered_step1.csv")
    out_csv  = args.out_csv or str(Path(auto_dir) / "dft_results.csv")

    collect_results(auto_dir, args.monomer, cand_csv, out_csv)

if __name__ == "__main__":
    main()
