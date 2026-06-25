#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py  ―  auto_opt 可視化 UI (Streamlit)

使い方:
  streamlit run src/auto_opt/app.py

ローカル Mac で filtered_step1.csv を読み込み、
2D エネルギーマップと9分子クラスターの3D表示を行う。
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import py3Dmol

from auto_opt.plot.make_cluster_xyz import make_cluster_xyz

_MONOMER_DIR = str(Path(__file__).resolve().parents[2] / "data" / "monomer")

st.set_page_config(page_title="auto_opt viewer", layout="wide")
st.title("auto_opt — エネルギーマップ & 構造表示")

# ──────────────────────────────────────────────
#  サイドバー: データ読み込み・設定
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("設定")
    uploaded = st.file_uploader("filtered_step1.csv", type="csv")
    monomer_name = st.text_input("モノマー名", value="BTBT")
    monomer_dir  = st.text_input("モノマーデータディレクトリ", value=_MONOMER_DIR)

if uploaded is None:
    st.info("サイドバーから filtered_step1.csv を読み込んでください。")
    st.stop()

df = pd.read_csv(uploaded)
is_screw = 'beta' in df.columns
symmetry = 'screw' if is_screw else 'glide'

# 軸候補: screw は beta も含む
axis_candidates = ['alpha', 'phi', 'z'] + (['beta'] if is_screw else [])
axis_candidates = [c for c in axis_candidates if c in df.columns]

# ──────────────────────────────────────────────
#  サイドバー: 軸選択
# ──────────────────────────────────────────────

with st.sidebar:
    st.subheader("ヒートマップ軸")
    x_col = st.selectbox("X 軸", axis_candidates, index=0)
    y_options = [c for c in axis_candidates if c != x_col]
    y_col = st.selectbox("Y 軸", y_options, index=0)

    fix_cols = [c for c in axis_candidates if c not in (x_col, y_col)]
    fix_vals: dict[str, float] = {}
    if fix_cols:
        st.subheader("固定パラメータ")
        for col in fix_cols:
            unique = sorted(df[col].dropna().unique())
            fix_vals[col] = st.select_slider(col, options=unique, value=unique[0])

# ──────────────────────────────────────────────
#  ヒートマップ
# ──────────────────────────────────────────────

df_fixed = df.copy()
for col, val in fix_vals.items():
    df_fixed = df_fixed[np.isclose(df_fixed[col], val, atol=1e-5)]

if df_fixed.empty:
    st.warning("固定パラメータに合うデータがありません。")
    st.stop()

pivot = df_fixed.pivot_table(values='E', index=y_col, columns=x_col, aggfunc='min')

col_map, col_3d = st.columns([1, 1])

with col_map:
    st.subheader(f"エネルギーマップ ({x_col} vs {y_col})")
    fig = px.imshow(
        pivot,
        color_continuous_scale='RdBu_r',
        labels={'color': 'E (kcal/mol)'},
        aspect='auto',
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
#  構造表示: パラメータ選択
# ──────────────────────────────────────────────

st.subheader("構造表示パラメータ")
sel_cols = st.columns(len(axis_candidates))
sel_vals: dict[str, float] = {}
for i, col in enumerate(axis_candidates):
    unique = sorted(df[col].dropna().unique())
    default = fix_vals.get(col, unique[0])
    default_idx = min(range(len(unique)), key=lambda j: abs(unique[j] - default))
    sel_vals[col] = sel_cols[i].selectbox(col, unique, index=default_idx, key=f"sel_{col}")

# 選択条件で行を絞り込み
mask = pd.Series([True] * len(df), index=df.index)
for col, val in sel_vals.items():
    mask &= np.isclose(df[col], val, atol=1e-5)
rows = df[mask]

with col_3d:
    st.subheader("9分子クラスター 3D 表示")
    if rows.empty:
        st.warning("選択条件に合う行が見つかりません。")
    else:
        row = rows.loc[rows['E'].idxmin()] if len(rows) > 1 else rows.iloc[0]
        st.caption(
            f"E = {row['E']:.3f} kcal/mol | "
            f"a={row['a']:.2f}  b={row['b']:.2f}  z={row['z']:.2f}"
        )

        try:
            xyz_str = make_cluster_xyz(row, monomer_name, symmetry, monomer_dir)
        except Exception as e:
            st.error(f"XYZ 生成エラー: {e}")
            st.stop()

        # py3Dmol で表示
        view = py3Dmol.view(width=500, height=400)
        view.addModel(xyz_str, 'xyz')
        view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.3}})
        view.zoomTo()
        st.components.v1.html(view._make_html(), height=420)

        # XYZ ダウンロード
        st.download_button(
            label="XYZ ダウンロード",
            data=xyz_str,
            file_name=f"{monomer_name}_cluster.xyz",
            mime="text/plain",
        )
