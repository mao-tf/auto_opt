#!/usr/bin/env python3
# extract_init_from_vdw_minima.py
"""
使用例:
1. 全て出力（デフォルト）
   python -m auto_opt.vdw.extract_init_phi --vdw-csv ... --out ... --minima

2. a-stack のみ出力
   python -m auto_opt.vdw.extract_init_phi --vdw-csv ... --out ... --select a-stack

3. b-stack と local_min のみ出力
   python -m auto_opt.vdw.extract_init_phi --vdw-csv ... --out ... --minima --select b-stack local_min
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

def extract_init(vdw_csv: str, out_csv: str, round_ab: int = 1, minima: bool = False, select_types: list = None) -> None:
    """
    入力: vdw_r_contact_<monomer>.csv
    出力: step1_init_params.csv
    select_types: 出力対象のリスト (例: ["a-stack", "local_min"])。None または "all" を含む場合は全出力。
    """
    if select_types is None:
        select_types = ["all"]
    
    # フィルタリング設定
    target_set = set(select_types)
    accept_all = "all" in target_set

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
            candidates = {}

            # 1. 端点判定
            candidates[run[0]] = "b-stack"   # beta最小 -> b最小
            candidates[run[-1]] = "a-stack"  # beta最大 -> a最小

            # 2. 局所最小判定
            if minima:
                vals = g.loc[run, "value"].to_numpy()
                mins_local = _local_minima_indices(vals)
                for k in mins_local:
                    idx = run[k]
                    candidates[idx] = "local_min"
            
            # 3. フィルタリングと抽出
            for i in sorted(candidates.keys()):
                st_type = candidates[i]

                # 指定されたタイプ以外はスキップ
                if not accept_all and st_type not in target_set:
                    continue
                
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
                    st_type
                ])

    if not rows:
        print(f"Warning: 条件に合致するデータが見つかりませんでした (select={select_types})")
        # 空でもヘッダーだけ持つCSVを出力しておくと後続のエラーを防げる場合があります
        out_df = pd.DataFrame(columns=["alpha", "phi", "a", "b", "z", "status", "structure_type"])
    else:
        out_cols = ["alpha", "phi", "a", "b", "z", "status", "structure_type"]
        out_df = pd.DataFrame(rows, columns=out_cols)
        out_df = out_df.drop_duplicates().sort_values(["z", "alpha", "phi", "a", "b"]).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (n={len(out_df)})")

def main():
    ap = argparse.ArgumentParser(
        description="vdW単一CSVから特定の構造タイプ(a-stack, b-stack, local_min)を抽出"
    )
    ap.add_argument("--vdw-csv", required=True, help="vdW_r_contact_<monomer>.csv")
    ap.add_argument("--out", required=True, help="出力ファイル")
    ap.add_argument("--round-ab", type=int, default=1, help="a,b の小数丸め桁")
    ap.add_argument("--minima", action="store_true", help="局所最小（local_min）も候補に含める")
    
    # フィルタリング用の引数を追加
    ap.add_argument("--select", nargs="+", default=["all"],
                    choices=["all", "a-stack", "b-stack", "local_min"],
                    help="出力する構造タイプを指定 (スペース区切りで複数可)。例: --select a-stack b-stack")
    
    args = ap.parse_args()
    extract_init(args.vdw_csv, args.out, round_ab=args.round_ab, minima=args.minima, select_types=args.select)

if __name__ == "__main__":
    main()