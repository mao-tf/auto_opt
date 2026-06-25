#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py  ―  auto_opt 可視化 UI (Streamlit)

使い方:
  streamlit run src/auto_opt/app.py

ローカル Mac で filtered_step1.csv を読み込み、
2D エネルギーマップと9分子クラスターの3D表示を行う。
ヒートマップの点をクリックすると右側の3D表示が更新される。
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    mol_style = st.selectbox("表示スタイル", ["Capped sticks", "Space fill"])

if uploaded is None:
    st.info("サイドバーから filtered_step1.csv を読み込んでください。")
    st.stop()

df = pd.read_csv(uploaded)
is_screw = 'beta' in df.columns
symmetry = 'screw' if is_screw else 'glide'

axis_candidates = ['alpha', 'phi', 'z'] + (['beta'] if is_screw else [])
axis_candidates = [c for c in axis_candidates if c in df.columns]

# ──────────────────────────────────────────────
#  サイドバー: 軸選択・固定パラメータ
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
            if len(unique) == 1:
                fix_vals[col] = unique[0]
                st.text(f"{col} = {unique[0]}")
            else:
                fix_vals[col] = st.select_slider(col, options=unique, value=unique[0])

# ──────────────────────────────────────────────
#  session_state: クリックで選んだ点を保持
# ──────────────────────────────────────────────

if 'sel' not in st.session_state:
    st.session_state.sel = {}

# 軸や固定パラメータが変わったらクリック選択をリセット
_state_key = (x_col, y_col, tuple(sorted(fix_vals.items())))
if st.session_state.get('_prev_key') != _state_key:
    st.session_state.sel = {}
    st.session_state['_prev_key'] = _state_key

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
    st.caption("点をクリックすると右側の3D構造が更新されます")

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        fig = px.scatter(
            df_fixed, x=x_col, y=y_col, color='E',
            color_continuous_scale='RdBu_r',
            labels={'color': 'E (kcal/mol)'},
            hover_data=['E', 'a', 'b', 'z'],
        )
    else:
        fig = px.imshow(
            pivot,
            color_continuous_scale='RdBu_r',
            labels={'color': 'E (kcal/mol)'},
            aspect='auto',
        )

    # 現在選択中の点をマーカーで重ねて表示
    sel_x = st.session_state.sel.get(x_col)
    sel_y = st.session_state.sel.get(y_col)
    if sel_x is not None and sel_y is not None:
        fig.add_trace(go.Scatter(
            x=[sel_x], y=[sel_y],
            mode='markers',
            marker=dict(symbol='circle-open', size=18, color='white', line=dict(width=3)),
            showlegend=False, hoverinfo='skip',
        ))

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

    event = st.plotly_chart(
        fig, use_container_width=True,
        on_select='rerun', key='heatmap_chart',
    )

    # クリックされた点を session_state に保存
    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        clicked_x = pt.get('x')
        clicked_y = pt.get('y')
        if clicked_x is not None:
            st.session_state.sel[x_col] = float(clicked_x)
        if clicked_y is not None:
            st.session_state.sel[y_col] = float(clicked_y)
        st.rerun()

# ──────────────────────────────────────────────
#  構造表示: クリック値 or ドロップダウンで選択
# ──────────────────────────────────────────────

st.subheader("構造表示パラメータ")
st.caption("ヒートマップをクリックするか、下のドロップダウンで選択してください")

sel_cols_ui = st.columns(len(axis_candidates))
sel_vals: dict[str, float] = {}
for i, col in enumerate(axis_candidates):
    unique = sorted(df[col].dropna().unique())
    # クリック値 → 固定パラメータ → 先頭 の優先順位でデフォルト
    default = st.session_state.sel.get(col, fix_vals.get(col, unique[0]))
    # 最近傍の値に丸める（浮動小数点のずれ対策）
    default_idx = min(range(len(unique)), key=lambda j: abs(unique[j] - default))
    sel_vals[col] = sel_cols_ui[i].selectbox(col, unique, index=default_idx, key=f"sel_{col}")

# 選択条件で絞り込み
mask = pd.Series([True] * len(df), index=df.index)
for col, val in sel_vals.items():
    mask &= np.isclose(df[col], val, atol=1e-5)
rows = df[mask]

if 'structure_type' in rows.columns and rows['structure_type'].nunique() > 1:
    types = sorted(rows['structure_type'].dropna().unique())
    sel_type = st.selectbox("structure_type", types)
    rows = rows[rows['structure_type'] == sel_type]

# ──────────────────────────────────────────────
#  3D 表示 & ダウンロード
# ──────────────────────────────────────────────

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

        view = py3Dmol.view(width=500, height=400)
        view.addModel(xyz_str, 'xyz')
        if mol_style == "Space fill":
            view.setStyle({'sphere': {'scale': 1.0}})
        else:
            view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'radius': 0.3}})
        view.setProjection('orthographic')
        view.zoomTo()
        st.components.v1.html(view._make_html(), height=420)

        st.download_button(
            label="XYZ ダウンロード",
            data=xyz_str,
            file_name=f"{monomer_name}_cluster.xyz",
            mime="text/plain",
        )
