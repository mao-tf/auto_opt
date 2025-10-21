#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dft_results.csv から (alpha, z) 2D の局所最小を抽出（a-stack / b-stack を個別に）

- 入力 CSV は、各 (alpha, z) について structure_type ∈ {a-stack, b-stack, ch} の E を持つ想定
- ch は無視。a-stack と b-stack をそれぞれ別ファイルに書き出す
- α,z のグリッドは浮動小数の揺れを丸めで吸収し、8近傍(上下左右+斜め)で局所最小を定義
- 端点（近傍が取れない点）は局所最小にしない
- 近接間引き（ユークリッド距離 in (alpha, z)）や top-k も指定可

使い方例:
  python -m auto_opt.gaussian.select_minima \
    --dft-csv runs/PFA_test/gaussian/dft_results.csv \
    --out     runs/PFA_test/gaussian/min_az_local.csv \
    --round-alpha 2 --round-z 3 --min-az-sep 0.0 --top-k 0

出力:
  --out が *.csv のとき:
    <out_basename>.a_stack.csv
    <out_basename>.b_stack.csv
  それ以外のとき:
    <out>.a_stack.csv, <out>.b_stack.csv
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

PARAM_COLS = ["alpha", "z"]
ENERGY_COLS_CAND = ["E", "E1", "E2", "E3"]
VALID_STACKS = ("a-stack", "b-stack")

# ----- utils -----

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _filter_done(df: pd.DataFrame, include_incomplete: bool) -> pd.DataFrame:
    if include_incomplete or ("status" not in df.columns):
        return df
    return df[df["status"].astype(str).str.lower().eq("done")]

def _estimate_step(values: np.ndarray, default: float = 0.1) -> float:
    u = np.unique(np.round(values.astype(float), 6))
    if len(u) < 2:
        return default
    diffs = np.diff(u)
    pos = diffs[diffs > 1e-8]
    return float(np.round(pos.min(), 6)) if len(pos) else default

def _neighbors_2d(ax: float, zx: float, da: float, dz: float) -> List[Tuple[float, float]]:
    offs = [( da,0),(-da,0),(0,dz),(0,-dz),(da,dz),(da,-dz),(-da,dz),(-da,-dz)]
    return [(round(ax+x,6), round(zx+y,6)) for x,y in offs]

def _is_local_min_2d(a: float, z: float, e: float, grid: dict) -> bool:
    nb = [grid.get((aa, zz)) for (aa, zz) in _neighbors_2d(a, z, grid["_da"], grid["_dz"])]
    nb = [v for v in nb if v is not None]
    if not nb:  # 端点は局所最小にしない
        return False
    return (all(e <= v for v in nb)) and any(e < v for v in nb)

def _greedy_by_distance(df: pd.DataFrame, top_k: int, min_sep: float,
                        coords: List[str]) -> pd.DataFrame:
    """E昇順で選別。coords（例: ["alpha","z"]）距離が min_sep 以上になるよう貪欲に採用。"""
    if top_k <= 0:
        return df.iloc[0:0].copy()
    picked_idx = []
    picked_pts: List[np.ndarray] = []
    for i, r in df.iterrows():
        pt = np.array([float(r[c]) for c in coords], float)
        if not picked_pts:
            picked_pts.append(pt); picked_idx.append(i)
        else:
            d = np.linalg.norm(np.array(picked_pts) - pt, axis=1)
            if (min_sep <= 0.0) or np.all(d >= min_sep):
                picked_pts.append(pt); picked_idx.append(i)
        if len(picked_idx) >= top_k:
            break
    return df.loc[picked_idx]

def _pick_energy_col(df: pd.DataFrame) -> str:
    for c in ENERGY_COLS_CAND:
        if c in df.columns:
            return c
    raise ValueError(f"Energy column not found. Tried: {ENERGY_COLS_CAND}")

# ----- core -----

def _az_local_for_stack(df: pd.DataFrame,
                        stack: str,
                        round_alpha: int,
                        round_z: int,
                        min_az_sep: float,
                        top_k: Optional[int]) -> pd.DataFrame:
    if "structure_type" not in df.columns:
        raise ValueError("必要列 'structure_type' が見つからない。a-stack/b-stack 判定に必須。")
    if stack not in VALID_STACKS:
        raise ValueError(f"未知の stack: {stack}. 有効: {VALID_STACKS}")

    energy_col = _pick_energy_col(df)

    gg = df[df["structure_type"].astype(str) == stack].copy()
    if gg.empty:
        return gg.iloc[0:0].copy()

    # 丸めで同一格子点(α,z)を代表化（E最小の行を残す）
    gg["alpha_r"] = gg["alpha"].round(round_alpha)
    gg["z_r"]     = gg["z"].round(round_z)
    gg = gg.sort_values(energy_col).drop_duplicates(subset=["alpha_r","z_r"], keep="first")

    # 近傍ステップ推定
    da = _estimate_step(gg["alpha_r"].to_numpy(), default=0.1)
    dz = _estimate_step(gg["z_r"].to_numpy(),     default=0.1)

    # グリッド辞書 (α,z) -> E
    grid = {(float(a), float(z)): float(e)
            for a, z, e in gg[["alpha_r","z_r", energy_col]].itertuples(index=False, name=None)}
    grid["_da"], grid["_dz"] = da, dz

    # 2D 8近傍の局所最小を抽出
    mins = []
    for r in gg.itertuples(index=False):
        a, z, e = float(r.alpha_r), float(r.z_r), float(getattr(r, energy_col))
        if _is_local_min_2d(a, z, e, grid):
            # 丸め代表の元行を1つだけ選ぶ
            mins.append(gg.loc[(gg["alpha_r"] == a) & (gg["z_r"] == z)].iloc[0])

    if not mins:
        return gg.iloc[0:0].copy()

    out = pd.DataFrame(mins).drop(columns=["alpha_r","z_r"], errors="ignore")
    out = out.sort_values([energy_col, "z", "alpha"]).reset_index(drop=True)

    # 近接間引き（(alpha,z) 距離）
    if min_az_sep > 0:
        out = _greedy_by_distance(
            out.sort_values(energy_col),
            top_k=top_k or len(out),
            min_sep=min_az_sep,
            coords=["alpha","z"]
        ).sort_values([energy_col, "z", "alpha"]).reset_index(drop=True)

    # 全体 top-k（エネルギー昇順）
    if top_k:
        out = out.sort_values(energy_col).head(top_k).reset_index(drop=True)

    return out

def select_minima_az_by_stack(
    dft_csv: str,
    out_path: str,
    round_alpha: int = 3,
    round_z: int = 3,
    min_az_sep: float = 0.0,
    top_k: int = 0,
    include_incomplete: bool = False,
    stacks: Tuple[str, ...] = VALID_STACKS,
) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(dft_csv)

    # α 列名の正規化
    if "alpha" not in df.columns and "theta" in df.columns:
        df = df.rename(columns={"theta":"alpha"})

    # 型整形 & フィルタ
    df = _coerce_numeric(df, ["alpha","z"] + ENERGY_COLS_CAND)
    df = _filter_done(df, include_incomplete)
    df = df.dropna(subset=["alpha","z"]).copy()

    energy_col = _pick_energy_col(df)

    # 出力列の選定
    base_cols   = [c for c in ["structure_type","alpha","z"] if c in df.columns]
    energy_cols = [c for c in ENERGY_COLS_CAND if c in df.columns]
    extras      = [c for c in ["status","log","inp","a","b"] if c in df.columns]  # a,b は参考に残す
    out_cols    = base_cols + energy_cols + extras

    # 出力ファイル名
    outp = Path(out_path)
    if outp.suffix.lower() == ".csv":
        base = outp.with_suffix("")
    else:
        base = outp

    results: Dict[str, pd.DataFrame] = {}
    for st in stacks:
        picked = _az_local_for_stack(
            df=df.copy(),
            stack=st,
            round_alpha=round_alpha,
            round_z=round_z,
            min_az_sep=min_az_sep,
            top_k=(top_k or None),
        )
        out_df = picked[out_cols].reset_index(drop=True)
        results[st] = out_df

        out_file = base.parent / f"{base.name}.{st.replace('-','_')}.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_file, index=False)
        print(f"[select_minima_az] {st}: wrote {out_file} (n={len(out_df)}) "
              f"(energy={energy_col}, round=({round_alpha},{round_z}), sep={min_az_sep})")

    return results

# ----- CLI -----

def main():
    ap = argparse.ArgumentParser(description="(alpha, z) 2D の局所最小抽出（a-stack / b-stack 別）")
    ap.add_argument("--dft-csv", required=True)
    ap.add_argument("--out", required=True,
                    help="出力ファイルのベース名。*.csv なら <name>.a_stack.csv / <name>.b_stack.csv を生成")
    ap.add_argument("--round-alpha", type=int, default=3, help="alpha の丸め桁")
    ap.add_argument("--round-z",     type=int, default=3, help="z の丸め桁")
    ap.add_argument("--min-az-sep",  type=float, default=0.0,
                    help="(alpha,z) の最小距離（近接候補の間引き）")
    ap.add_argument("--top-k", type=int, default=0,
                    help="全体の上限件数（0で無制限）")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="status!=Done も含める")
    ap.add_argument("--stacks", nargs="+", default=list(VALID_STACKS),
                    help="対象の structure_type。デフォルト: a-stack b-stack")
    args = ap.parse_args()

    select_minima_az_by_stack(
        dft_csv=args.dft_csv,
        out_path=args.out,
        round_alpha=args.round_alpha,
        round_z=args.round_z,
        min_az_sep=args.min_az_sep,
        top_k=args.top_k,
        include_incomplete=args.include_incomplete,
        stacks=tuple(args.stacks),
    )

if __name__ == "__main__":
    main()
