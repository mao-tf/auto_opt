#!/usr/bin/env python3
# extract_init_from_vdw_minima.py
"""
端点のみ（デフォルト）
python -m auto_opt.vdw.extract_init --vdw-csv .../vdW_r_contact_PFA.csv --out .../step1_init_params.csv

端点＋局所最小（--minima 指定時）
python -m auto_opt.vdw.extract_init.py --vdw-csv .../vdW_r_contact_PFA.csv --out .../step1_init_params.csv --minima
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
    入力: vdw_r_contact_<monomer>.csv （単一CSV, 列: alpha,beta,z,R_clps,TorF）
    出力: step1_init_params.csv （列: alpha,a,b,z,status）※zで分割しない
    """
    df = pd.read_csv(vdw_csv)
    need = {"alpha", "phi", "beta", "z", "R_clps", "TorF"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"必要列が不足: {miss} in {vdw_csv}")

    df = _compute_geo(df)
    rows = []

    # グループは (z, alpha) ごとに評価（最終出力は1本に集約）
    for (z, alpha, phi), g in df.groupby(["z", "alpha", "phi"], sort=False):
        g = g.sort_values("beta").reset_index(drop=True)
        mask = _as_bool_series(g["TorF"]).to_numpy()
        if not mask.any():
            continue

        for run in _true_runs(mask):
            # 端点は常に採用
            pick = {run[0], run[-1]}

            # --minima 指定時のみ、局所最小も採用
            if minima:
                vals = g.loc[run, "value"].to_numpy()
                mins_local = _local_minima_indices(vals)
                for k in mins_local:
                    pick.add(run[k])

            for i in sorted(pick):
                a = float(g.loc[i, "a"])
                b = float(g.loc[i, "b"])
                a_r = np.round(a, round_ab)
                b_r = np.round(b, round_ab)
                rows.append([float(alpha), a_r, b_r, float(z), "NotYet"])

    if not rows:
        raise ValueError("抽出結果が空です。TorF条件や入力CSVを確認してください。")

    out_df = pd.DataFrame(rows, columns=["alpha", "a", "b", "z", "status"])
    out_df = out_df.drop_duplicates().sort_values(["z", "alpha", "phi", "a", "b"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (n={len(out_df)})")

def main():
    ap = argparse.ArgumentParser(
        description="vdW単一CSVから True区間の端点＋（オプションで局所極小）を抽出し、step1_init_params.csv を作成"
    )
    ap.add_argument("--vdw-csv", required=True, help="vdW_r_contact_<monomer>.csv（z列あり）")
    ap.add_argument("--out", required=True, help="出力ファイル（step1_init_params.csv）")
    ap.add_argument("--round-ab", type=int, default=1, help="a,b の小数丸め桁（既定 1 → 0.1刻み）")
    ap.add_argument("--minima", action="store_true", help="True区間の局所最小（valueの離散極小）も抽出（端点＋極小）")
    args = ap.parse_args()
    extract_init(args.vdw_csv, args.out, round_ab=args.round_ab, minima=args.minima)

if __name__ == "__main__":
    main()
