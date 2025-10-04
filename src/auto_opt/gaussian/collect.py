#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dft_collect.py

- filtered_step1.csv（列: alpha,a,b,z …）を読み、
  auto-dir/gaussian/内の .log を utils.dft_get_E() で解析。
- 列: alpha,a,b,z,E,E1,E2,E3,status を dft_results.csv に出力する。

前提:
- 3ダイマー (a1, b1, t1) を1つのGaussian入力にLink1でまとめている
- .log の命名規則:  monomer_alpha=<int>_a=<0.1>_b=<0.1>_z=<0.1>.log
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from auto_opt.utils import dft_get_E  # ← 分岐なしで直接インポート

def _ensure_alpha_col(df: pd.DataFrame) -> pd.DataFrame:
    """alpha 列が無くて theta があるなら alpha に寄せる"""
    if 'alpha' in df.columns:
        return df
    if 'theta' in df.columns:
        return df.rename(columns={'theta': 'alpha'})
    raise ValueError("候補CSVに alpha / theta 列が見つかりませんでした。")

def _round_for_name(alpha: float, a: float, b: float, z: float):
    """ファイル名で使う丸め（投入側と一致させる）"""
    return int(round(alpha)), round(float(a), 1), round(float(b), 1), round(float(z), 1)

def _log_path(auto_dir: str, monomer: str, alpha: float, a: float, b: float, z: float) -> Path:
    """生成側と同じ命名規則で .log のパスを作る"""
    ai, ar, br, zr = _round_for_name(alpha, a, b, z)
    base = f"{monomer}_alpha={ai}_a={ar}_b={br}_z={zr}"
    return Path(auto_dir) / "gaussian" / f"{base}.log"

def _combine_total(E1: float, E2: float, E3: float, w1: int, w2: int, w3: int) -> float:
    return w1*E1 + w2*E2 + w3*E3

def collect_results(auto_dir: str, monomer: str, cand_csv: str, out_csv: str,
                    weights: str = "2,2,4") -> pd.DataFrame:
    """
    cand_csv（通常: filtered_step1.csv）を読んで対応する .log を解析。
    出力: alpha,a,b,z,E,E1,E2,E3,status を out_csv に保存。
    """
    df = pd.read_csv(cand_csv)
    df = _ensure_alpha_col(df)

    need = {'alpha','a','b','z'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{cand_csv} に必須列が不足しています: {sorted(missing)}")

    # 型整形
    for c in ['alpha','a','b','z']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['alpha','a','b','z'])
    df = df.drop_duplicates(subset=['alpha','a','b','z']).reset_index(drop=True)

    try:
        w1, w2, w3 = [int(x) for x in weights.split(",")]
    except Exception:
        raise SystemExit("--weights は '2,2,4' のようにカンマ区切りの整数で指定してください。")

    rows = []
    for r in df.itertuples(index=False):
        alpha, a, b, z = float(r.alpha), float(r.a), float(r.b), float(r.z)
        logp = _log_path(auto_dir, monomer, alpha, a, b, z)

        rec = {'alpha': alpha, 'a': a, 'b': b, 'z': z,
               'E': np.nan, 'E1': np.nan, 'E2': np.nan, 'E3': np.nan, 'status': ''}

        if not logp.exists():
            rec['status'] = 'Missing'
            rows.append(rec); continue

        try:
            E_list = dft_get_E(str(logp))  # ← あなたの utils の関数
        except Exception as e:
            rec['status'] = f'ParseError: {e}'
            rows.append(rec); continue

        if not E_list:
            rec['status'] = 'Empty'
            rows.append(rec); continue

        # 期待: a1,b1,t1 の 3値。足りなければある分だけ入れて Partial に。
        try:
            if len(E_list) >= 1: rec['E1'] = float(E_list[0])
            if len(E_list) >= 2: rec['E2'] = float(E_list[1])
            if len(E_list) >= 3: rec['E3'] = float(E_list[2])
        except Exception:
            rec['status'] = 'BadValues'
            rows.append(rec); continue

        if np.isfinite(rec['E1']) and np.isfinite(rec['E2']) and np.isfinite(rec['E3']):
            rec['E'] = _combine_total(rec['E1'], rec['E2'], rec['E3'], w1, w2, w3)
            rec['status'] = 'Done'
        else:
            rec['status'] = 'Partial'

        rows.append(rec)

    out_df = pd.DataFrame(rows, columns=['alpha','a','b','z','E','E1','E2','E3','status'])
    out_df.to_csv(out_csv, index=False)
    print(f"[collect] wrote {out_csv} (n={len(out_df)})")
    return out_df

def main():
    ap = argparse.ArgumentParser(description="DFT ログ(.log)集計: alpha,a,b,z,E,E1,E2,E3 を出力")
    ap.add_argument("--auto-dir", required=True, help="作業ディレクトリ（gaussian/ がある場所）")
    ap.add_argument("--monomer", required=True, help="モノマー名（ファイル名に使う。例: PFA）")
    ap.add_argument("--candidates-csv", default=None, help="既定: <auto-dir>/filtered_step1.csv")
    ap.add_argument("--out-csv", default=None, help="既定: <auto-dir>/dft_results.csv")
    ap.add_argument("--weights", default="2,2,4", help="E 合成重み（例 '2,2,4'）")
    args = ap.parse_args()

    auto_dir = args.auto_dir
    cand_csv = args.candidates_csv or str(Path(auto_dir) / "filtered_step1.csv")
    out_csv  = args.out_csv or str(Path(auto_dir) / "dft_results.csv")

    collect_results(auto_dir, args.monomer, cand_csv, out_csv, weights=args.weights)

if __name__ == "__main__":
    main()
