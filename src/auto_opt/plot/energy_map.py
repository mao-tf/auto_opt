#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
energy_map.py  ―  filtered_step1.csv のエネルギーを2D ヒートマップで表示

任意の2パラメータを軸にとり、残りは --fix で固定する。
glide: パラメータは alpha, phi, z  （軸2つ + fix 1つ）
screw: パラメータは alpha, beta, phi, z（軸2つ + fix 2つ）

Usage:
  python -m auto_opt.plot.energy_map \\
      --csv  runs/DNTT/filtered_step1.csv \\
      --x    phi  --y z \\
      --fix  alpha=65 \\
      --out  energy_phi_z.png
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def _nearest_val(series: pd.Series, target: float) -> float:
    """ユニーク値の中から target に最も近い値を返す。"""
    vals = series.dropna().unique()
    return float(vals[np.argmin(np.abs(vals - target))])


def plot_energy_map(
    csv_path: str,
    x_param: str,
    y_param: str,
    fixed_params: dict[str, float],
    out_path: str | None = None,
) -> None:
    df = pd.read_csv(csv_path)

    if 'E' not in df.columns:
        raise ValueError(f"'E' 列が {csv_path} に見つかりません。")

    for p in (x_param, y_param):
        if p not in df.columns:
            raise ValueError(f"列 '{p}' が見つかりません。利用可能列: {list(df.columns)}")

    # 固定パラメータでフィルタ（最近傍値に丸めてから一致判定）
    for col, target in fixed_params.items():
        if col not in df.columns:
            raise ValueError(f"固定パラメータ '{col}' が見つかりません。")
        snapped = _nearest_val(df[col], target)
        if abs(snapped - target) > 0.1:
            print(f"[warn] {col}={target} が見つからず、最近傍 {snapped} を使用します。")
        df = df[np.isclose(df[col], snapped, atol=1e-5)]

    if df.empty:
        print(f"[error] 指定条件に該当するデータがありません: {fixed_params}")
        return

    # 2D ピボット（重複がある場合は最小値を採用）
    pivot = df.pivot_table(index=y_param, columns=x_param, values='E', aggfunc='min')
    X = pivot.columns.values.astype(float)
    Y = pivot.index.values.astype(float)
    Z = pivot.values  # shape: (len_y, len_x)

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(7, 5))

    vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = 'RdBu_r'
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = 'viridis_r'

    Xg, Yg = np.meshgrid(X, Y)
    mesh = ax.pcolormesh(Xg, Yg, Z, cmap=cmap, norm=norm, shading='auto')

    # 等高線（10本）
    finite = Z[np.isfinite(Z)]
    if len(finite) >= 4:
        levels = np.linspace(vmin, vmax, 10)
        ax.contour(Xg, Yg, Z, levels=levels, colors='k', linewidths=0.5, alpha=0.35)

    # 最小値に星印
    min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax.scatter(
        X[min_idx[1]], Y[min_idx[0]],
        marker='*', s=250, color='gold', edgecolors='k', linewidths=0.6,
        zorder=5, label=f'min E = {Z[min_idx]:.2f} kcal/mol',
    )

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('E (kcal/mol)')
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)

    fix_str = ', '.join(f'{k}={v}' for k, v in fixed_params.items())
    title = f'E ({y_param} vs {x_param})'
    if fix_str:
        title += f'\n[{fix_str}]'
    ax.set_title(title)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"[plot] 保存 -> {out_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="filtered_step1.csv からエネルギー2D ヒートマップを表示"
    )
    ap.add_argument('--csv', required=True, help='filtered_step1.csv のパス')
    ap.add_argument('--x',   required=True, dest='x_param',
                    help='x軸パラメータ名 (例: phi)')
    ap.add_argument('--y',   required=True, dest='y_param',
                    help='y軸パラメータ名 (例: z)')
    ap.add_argument('--fix', nargs='*', default=[], metavar='PARAM=VALUE',
                    help='固定するパラメータ (例: --fix alpha=65 beta=0)')
    ap.add_argument('--out', default=None,
                    help='出力ファイル名 (.png/.pdf など)。省略時は画面表示。')
    args = ap.parse_args()

    fixed: dict[str, float] = {}
    for item in (args.fix or []):
        k, v = item.split('=', 1)
        fixed[k.strip()] = float(v.strip())

    plot_energy_map(args.csv, args.x_param, args.y_param, fixed, args.out)


if __name__ == '__main__':
    main()
