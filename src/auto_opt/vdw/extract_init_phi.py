#!/usr/bin/env python3
# extract_init_from_vdw_minima.py
"""
端点（a-stack, b-stack）のみ（デフォルト）
python -m auto_opt.vdw.extract_init_phi --vdw-csv ... --out ...

端点＋局所最小（--minima 指定時）
python -m auto_opt.vdw.extract_init_phi --vdw-csv ... --out ... --minima
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

TRUE_LIKE = {"true", "1", "t", "y", "yes"}

def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(TRUE_LIKE)

def _compute_geo(df: pd.DataFrame) -> pd.DataFrame:
    """a,b,value(=a*b) を追加（beta は度数前提）"""
    beta_rad = np.deg2rad(pd.to_numeric(df["beta"], errors="coerce"))
    R = pd.to_numeric(df["R_clps"], errors="coerce")
    a = 2.0 * R * np.cos(beta_rad)
    b = 2.0 * R * np.sin(beta_rad)
    value = a * b
    return df.assign(a=a, b=b, value=value)

def _local_minima_indices(values: np.ndarray) -> list[int]:
    """離散列の局所極小インデックス（端点は含めない）"""
    idx = []
    n = len(values)
    if n < 3:
        return idx
    v = values
    # 端点を含まない内部の極小のみ判定
    for i in range(1, n - 1):
        if (v[i] <= v[i - 1]) and (v[i] <= v[i + 1]) and ((v[i] < v[i - 1]) or (v[i] < v[i + 1])):
            idx.append(i)
    return idx

def _true_runs(mask: np.ndarray) -> list[list[int]]:
    """bool配列から True の連結区間（元の配列インデックス）を抽出"""
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        runs.append(list(range(i, j + 1)))
        i = j + 1
    return runs

def extract_init(vdw_csv: str, out_csv: str, round_ab: int = 1, minima: bool = False) -> None:
    """
    入力: vdw_r_contact_<monomer>.csv
    出力: step1_init_params.csv (列: alpha, phi, a, b, z, status, structure_type)
    """
    df = pd.read_csv(vdw_csv)
    need = {"alpha", "phi", "beta", "z", "R_clps", "TorF"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"必要列が不足: {miss} in {vdw_csv}")

    df = _compute_geo(df)
    rows = []

    # グループは (z, alpha, phi) ごとに評価
    for (z, alpha, phi), g in df.groupby(["z", "alpha", "phi"], sort=False):
        g = g.sort_values("beta").reset_index(drop=True)
        mask = _as_bool_series(g["TorF"]).to_numpy()
        if not mask.any():
            continue

        for run in _true_runs(mask):
            # 採用する候補点を {index: structure_type} で管理
            candidates = {}

            # 1. 端点 (Endpoints) -> Stack構造として分類
            # 区間の開始点 (beta最小側 -> bが最小 -> b-stack)
            candidates[run[0]] = "b-stack"
            
            # 区間の終了点 (beta最大側 -> aが最小 -> a-stack)
            # ※ run長が1の場合上書きされますが、物理的には両方の性質を持つ極限状態です
            candidates[run[-1]] = "a-stack"

            # 2. 局所最小 (Local Minima) - オプション
            if minima:
                vals = g.loc[run, "value"].to_numpy()
                mins_local = _local_minima_indices(vals)
                for k in mins_local:
                    idx = run[k]
                    candidates[idx] = "ch"
            
            # 抽出処理
            for i in sorted(candidates.keys()):
                st_type = candidates[i]
                
                a = float(g.loc[i, "a"])
                b = float(g.loc[i, "b"])
                a_r = np.round(a, round_ab)
                b_r = np.round(b, round_ab)
                
                rows.append([
                    float(alpha), 
                    float(phi), 
                    a_r, 
                    b_r, 
                    float(z), 
                    "NotYet", 
                    st_type  # a-stack, b-stack, local_min
                ])

    if not rows:
        raise ValueError("抽出結果が空です。TorF条件や入力CSVを確認してください。")

    out_cols = ["alpha", "phi", "a", "b", "z", "status", "structure_type"]
    out_df = pd.DataFrame(rows, columns=out_cols)
    
    # 重複排除
    out_df = out_df.drop_duplicates().sort_values(["z", "alpha", "phi", "a", "b"]).reset_index(drop=True)
    
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (n={len(out_df)})")

def main():
    ap = argparse.ArgumentParser(
        description="vdW単一CSVからTrue区間の端点(a-stack, b-stack)＋局所極小を抽出"
    )
    ap.add_argument("--vdw-csv", required=True, help="vdW_r_contact_<monomer>.csv")
    ap.add_argument("--out", required=True, help="出力ファイル（step1_init_params.csv）")
    ap.add_argument("--round-ab", type=int, default=1, help="a,b の小数丸め桁（既定 1）")
    ap.add_argument("--minima", action="store_true", help="局所最小（local_min）も抽出")
    args = ap.parse_args()
    extract_init(args.vdw_csv, args.out, round_ab=args.round_ab, minima=args.minima)

if __name__ == "__main__":
    main()