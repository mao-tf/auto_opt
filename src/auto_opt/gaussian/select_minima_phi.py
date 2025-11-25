#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dft_results.csv から (phi, z) 2D の局所最小を抽出

- structure_type ∈ {a-stack, b-stack, ch} 別に処理
- (phi, z) 平面上で E が8近傍より低い点を抽出
- alpha のときのような反射境界は考慮せず、データが存在するグリッド上での局所最小を探す
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
    if include_incomplete or ("status" not in df.columns):
        return df
    mask = df["status"].astype(str).str.lower().eq("done")
    return df[mask]

def _estimate_step(values: np.ndarray, default: float = 0.1) -> float:
    # ユニークな値をソートして最小の差分をステップ幅とみなす
    u = np.unique(np.round(values.astype(float), 6))
    if len(u) < 2:
        return default
    diffs = np.diff(u)
    pos = diffs[diffs > 1e-8]
    return float(np.round(pos.min(), 6)) if len(pos) else default

def _neighbors_phiz(phi: float, z: float, dphi: float, dz: float) -> List[Tuple[float, float]]:
    """
    (phi, z) の8近傍を返す。
    """
    offs = [( dphi,0),(-dphi,0),(0,dz),(0,-dz),
            ( dphi,dz),( dphi,-dz),(-dphi,dz),(-dphi,-dz)]
    cand = []
    for dx, dy in offs:
        p_nb = float(round(phi + dx, 6))
        z_nb = float(round(z + dy, 6))
        cand.append((p_nb, z_nb))
    return cand

def _is_local_min_2d(phi: float, z: float, e: float, grid: dict, dphi: float, dz: float) -> bool:
    """
    指定されたグリッド上に値が存在する近傍と比較して最小なら True
    """
    nbs = _neighbors_phiz(phi, z, dphi, dz)
    vals = [grid.get(p) for p in nbs]
    # None（グリッド外や欠損）は無視し、存在する値のみと比較
    vals = [v for v in vals if v is not None]
    
    # 比較対象が全くない場合（孤立点）は最小とはみなさない（あるいはみなすか？ここはFalseとする）
    if not vals:
        return False
    
    # 近傍のすべて以下 かつ 少なくとも1つより小さい（同値が連続する平坦部対策）
    # 厳密に最小なら < でもよいが、数値計算の誤差や平坦な底を考慮して <= を許容しつつ
    # 全てが等しい場合は排除するロジックにしておく
    return (all(e <= v for v in vals)) and any(e < v for v in vals)

def _pick_energy_col(df: pd.DataFrame) -> str:
    for c in ENERGY_COLS_CAND:
        if c in df.columns:
            return c
    raise ValueError(f"Energy column not found. Tried: {ENERGY_COLS_CAND}")

# ----- core -----

def _phiz_local_for_stack(df: pd.DataFrame,
                          stack: str,
                          round_phi: int,
                          round_z: int,
                          top_k: Optional[int]) -> pd.DataFrame:
    
    if "structure_type" not in df.columns:
        # 無ければ警告しつつ全データ対象にする等の処理も考えられるが、要件的に必須
        raise ValueError("必要列 'structure_type' が見つからない。")
    
    # stack フィルタ
    gg = df[df["structure_type"].astype(str) == stack].copy()
    if gg.empty:
        return df.iloc[0:0].copy()

    energy_col = _pick_energy_col(df)

    # phi, z を丸めて重複排除（最もEが低いものを残す）
    gg["phi_r"] = gg["phi"].round(round_phi)
    gg["z_r"]   = gg["z"].round(round_z)
    gg = gg.sort_values(energy_col).drop_duplicates(subset=["phi_r","z_r"], keep="first")

    # ステップ幅の推定
    dphi = _estimate_step(gg["phi_r"].to_numpy(), default=5.0)
    dz   = _estimate_step(gg["z_r"].to_numpy(),   default=0.1)

    # グリッド作成
    grid = {(float(p), float(z)): float(e)
            for p, z, e in gg[["phi_r","z_r", energy_col]].itertuples(index=False, name=None)}

    mins = []
    for r in gg.itertuples(index=False):
        p, z, e = float(r.phi_r), float(r.z_r), float(getattr(r, energy_col))
        if _is_local_min_2d(p, z, e, grid, dphi, dz):
            # 元のデータフレームから該当行を取得
            row = gg.loc[(gg["phi_r"] == p) & (gg["z_r"] == z)].iloc[0]
            mins.append(row)

    if not mins:
        return gg.iloc[0:0].copy()

    out = pd.DataFrame(mins).drop(columns=["phi_r","z_r"], errors="ignore")
    out = out.sort_values([energy_col, "z", "phi"]).reset_index(drop=True)

    if top_k:
        out = out.sort_values(energy_col).head(top_k).reset_index(drop=True)

    return out

def select_minima_phiz_by_stack(
    dft_csv: str,
    out_path: str,
    round_phi: int = 3,
    round_z: int = 3,
    top_k: int = 0,
    include_incomplete: bool = False,
    stacks: Tuple[str, ...] = VALID_STACKS,
) -> Dict[str, pd.DataFrame]:
    
    df = pd.read_csv(dft_csv)

    # 型整形 & フィルタ
    # phi, z が必須
    df = _coerce_numeric(df, ["phi","z"] + ENERGY_COLS_CAND)
    df = _filter_done(df, include_incomplete)
    df = df.dropna(subset=["phi","z"]).copy()

    energy_col = _pick_energy_col(df)

    # 出力列の選定 (alphaもあれば残す)
    base_cols   = [c for c in ["structure_type","phi","z","alpha"] if c in df.columns]
    energy_cols = [c for c in ENERGY_COLS_CAND if c in df.columns]
    extras      = [c for c in ["status","log","inp","a","b"] if c in df.columns]
    out_cols    = base_cols + energy_cols + extras
    # 重複除去して列順を整える
    out_cols = list(dict.fromkeys(out_cols))

    outp = Path(out_path)
    base_name = outp.with_suffix("") if outp.suffix.lower() == ".csv" else outp

    results: Dict[str, pd.DataFrame] = {}
    for st in stacks:
        picked = _phiz_local_for_stack(
            df=df.copy(),
            stack=st,
            round_phi=round_phi,
            round_z=round_z,
            top_k=(top_k or None),
        )
        out_df = picked[out_cols].reset_index(drop=True) if not picked.empty else pd.DataFrame(columns=out_cols)
        results[st] = out_df

        out_file = base_name.parent / f"{base_name.name}.{st.replace('-','_')}.csv"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_file, index=False)
        print(f"[select_minima] {st}: wrote {out_file} (n={len(out_df)}) "
              f"(energy={energy_col})")

    return results

# ----- CLI -----

def main():
    ap = argparse.ArgumentParser(description="(phi, z) 2D の局所最小抽出（structure_type別）")
    ap.add_argument("--dft-csv", required=True)
    ap.add_argument("--out", required=True,
                    help="出力ファイルのベース名。*.csv なら <name>.a_stack.csv 等を生成")
    ap.add_argument("--round-phi", type=int, default=3)
    ap.add_argument("--round-z",   type=int, default=3)
    ap.add_argument("--top-k", type=int, default=0, help="各スタックの上限件数（0で無制限）")
    ap.add_argument("--include-incomplete", action="store_true")
    ap.add_argument("--stacks", nargs="+", default=list(VALID_STACKS),
                    help="対象の structure_type (a-stack, b-stack, ch)")
    args = ap.parse_args()

    select_minima_phiz_by_stack(
        dft_csv=args.dft_csv,
        out_path=args.out,
        round_phi=args.round_phi,
        round_z=args.round_z,
        top_k=args.top_k,
        include_incomplete=args.include_incomplete,
        stacks=tuple(args.stacks),
    )

if __name__ == "__main__":
    main()