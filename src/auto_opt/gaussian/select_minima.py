#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dft_results.csv から (alpha, z) 2D の局所最小を抽出

- structure_type ∈ {a-stack, b-stack, ch} を想定
- a-stack / b-stack: α=0/90°は反射境界（-5→5, 95→85）で近傍を作り、局所最小を抽出
- ch: 局所最小は取らず、フィルタ後にそのまま並べ替えて出力（top-k 指定があれば上位のみ）
- α,z のグリッドは丸めで吸収（外れ値の微小ズレ対策）
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

ENERGY_COLS_CAND = ["E", "E_tot", "Energy"]
VALID_STACKS = ("a-stack", "b-stack", "ch")

# ----- utils -----

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _filter_done(df: pd.DataFrame, include_incomplete: bool) -> pd.DataFrame:
    # status 列が無い or 未完了も含めたい → そのまま返す
    if include_incomplete or ("status" not in df.columns):
        return df
    mask = df["status"].astype(str).str.lower().eq("done")
    n_total = len(df)
    n_done  = int(mask.sum())
    n_bad   = n_total - n_done
    if n_bad > 0:
        counts = (df.loc[~mask, "status"]
                    .astype(str).str.lower().value_counts())
        # 概要
        print(f"[WARN] {n_bad}/{n_total} rows are not Done: " +
                ", ".join(f"{k}:{v}" for k, v in counts.items()))
        
    return df[mask]

def _estimate_step(values: np.ndarray, default: float = 0.1) -> float:
    u = np.unique(np.round(values.astype(float), 6))
    if len(u) < 2:
        return default
    diffs = np.diff(u)
    pos = diffs[diffs > 1e-8]
    return float(np.round(pos.min(), 6)) if len(pos) else default

def _reflect_alpha(x: float, a_min: float, a_max: float) -> float:
    """端点で鏡映（-5→+5, 95→85）。1ステップ分の外側のみを想定（±da）。"""
    if x < a_min:
        return 2 * a_min - x
    if x > a_max:
        return 2 * a_max - x
    return x

def _neighbors_az_reflect(a: float, z: float, da: float, dz: float,
                          a_min: float = 0.0, a_max: float = 90.0) -> List[Tuple[float, float]]:
    """
    α内部(5..85)では通常の8近傍。
    α端点(=a_min or a_max)では α を鏡映して近傍を作る。重複は除去 → 5点になる。
    """
    # ベースの8近傍オフセット
    offs = [( da,0),(-da,0),(0,dz),(0,-dz),(da,dz),(da,-dz),(-da,dz),(-da,-dz)]
    cand = []
    for dx, dz_ in offs:
        a_nb = _reflect_alpha(a + dx, a_min, a_max)
        z_nb = z + dz_
        cand.append((round(a_nb, 6), round(z_nb, 6)))

    # 端点以外は8点のまま（重複しない想定）
    if (a_min < a < a_max):
        return cand

    # 端点では鏡映により (±da, ·) が同一点に潰れるので重複を除去
    seen = set()
    uniq = []
    for p in cand:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq  # 期待通り 5 点になる

def _is_local_min_2d_reflect(a: float, z: float, e: float, grid: dict,
                             a_min: float, a_max: float) -> bool:
    """反射境界の近傍で局所最小判定。存在しない(z)はスキップ。"""
    da, dz = grid["_da"], grid["_dz"]
    nbs = _neighbors_az_reflect(a, z, da, dz, a_min=a_min, a_max=a_max)
    vals = [grid.get(p) for p in nbs]
    vals = [v for v in vals if v is not None]
    if not vals:  # 比較対象がなければ最小にしない
        return False
    return (all(e <= v for v in vals)) and any(e < v for v in vals)

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
                        top_k: Optional[int],
                        alpha_min: float = 0.0,
                        alpha_max: float = 90.0) -> pd.DataFrame:
    if "structure_type" not in df.columns:
        raise ValueError("必要列 'structure_type' が見つからない。a-stack/b-stack/ch 判定に必須。")
    if stack not in VALID_STACKS:
        raise ValueError(f"未知の stack: {stack}. 有効: {VALID_STACKS}")

    energy_col = _pick_energy_col(df)

    # --- ch: 局所最小抽出はしない。フィルタして整形のみ ---
    if stack == "ch":
        gg = df[df["structure_type"].astype(str) == "ch"].copy()
        if gg.empty:
            return df.iloc[0:0].copy()  # or raise
        # “そのまま抜き出す”解釈：行は削らない。出力整形だけ。
        out = gg.sort_values([energy_col, "z", "alpha"], ascending=[True, True, True])\
                .reset_index(drop=True)
        if top_k:
            out = out.sort_values(energy_col).head(top_k).reset_index(drop=True)
        return out

    # --- a-stack / b-stack: 反射境界で局所最小抽出 ---
    gg = df[df["structure_type"].astype(str) == stack].copy()
    if gg.empty:
        return df.iloc[0:0].copy()

    gg["alpha_r"] = gg["alpha"].round(round_alpha)
    gg["z_r"]     = gg["z"].round(round_z)
    gg = gg.sort_values(energy_col).drop_duplicates(subset=["alpha_r","z_r"], keep="first")

    da = _estimate_step(gg["alpha_r"].to_numpy(), default=5.0)
    dz = _estimate_step(gg["z_r"].to_numpy(),     default=0.1)

    grid = {(float(a), float(z)): float(e)
            for a, z, e in gg[["alpha_r","z_r", energy_col]].itertuples(index=False, name=None)}
    grid["_da"], grid["_dz"] = da, dz

    mins = []
    for r in gg.itertuples(index=False):
        a, z, e = float(r.alpha_r), float(r.z_r), float(getattr(r, energy_col))
        if _is_local_min_2d_reflect(a, z, e, grid, a_min=alpha_min, a_max=alpha_max):
            mins.append(gg.loc[(gg["alpha_r"] == a) & (gg["z_r"] == z)].iloc[0])

    if not mins:
        return gg.iloc[0:0].copy()

    out = pd.DataFrame(mins).drop(columns=["alpha_r","z_r"], errors="ignore")
    out = out.sort_values([energy_col, "z", "alpha"]).reset_index(drop=True)

    if top_k:
        out = out.sort_values(energy_col).head(top_k).reset_index(drop=True)

    return out

def select_minima_az_by_stack(
    dft_csv: str,
    out_path: str,
    round_alpha: int = 3,
    round_z: int = 3,
    top_k: int = 0,
    include_incomplete: bool = False,
    stacks: Tuple[str, ...] = VALID_STACKS,
    alpha_min: float = 0.0,
    alpha_max: float = 90.0,
) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(dft_csv)

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
            top_k=(top_k or None),
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )
        out_df = picked[out_cols].reset_index(drop=True)
        results[st] = out_df

        out_file = base.parent / f"{base.name}.{st.replace('-','_')}.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_file, index=False)
        print(f"[select_minima_az] {st}: wrote {out_file} (n={len(out_df)}) "
              f"(energy={energy_col}, round=({round_alpha},{round_z}), "
              f"alpha_range=[{alpha_min},{alpha_max}])")

    return results

# ----- CLI -----

def main():
    ap = argparse.ArgumentParser(description="(alpha, z) 2D の局所最小抽出（a-stack / b-stack 別, α端点は反射境界）")
    ap.add_argument("--dft-csv", required=True)
    ap.add_argument("--out", required=True,
                    help="出力ファイルのベース名。*.csv なら <name>.a_stack.csv / <name>.b_stack.csv を生成")
    ap.add_argument("--round-alpha", type=int, default=3, help="alpha の丸め桁")
    ap.add_argument("--round-z",     type=int, default=3, help="z の丸め桁")
    ap.add_argument("--top-k", type=int, default=0,
                    help="全体の上限件数（0で無制限）")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="status!=Done も含める")
    ap.add_argument("--stacks", nargs="+", default=list(VALID_STACKS),
                    help="対象の structure_type。デフォルト: a-stack b-stack ch")
    ap.add_argument("--alpha-min", type=float, default=0.0)
    ap.add_argument("--alpha-max", type=float, default=90.0)
    args = ap.parse_args()

    select_minima_az_by_stack(
        dft_csv=args.dft_csv,
        out_path=args.out,
        round_alpha=args.round_alpha,
        round_z=args.round_z,
        top_k=args.top_k,
        include_incomplete=args.include_incomplete,
        stacks=tuple(args.stacks),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )

if __name__ == "__main__":
    main()
