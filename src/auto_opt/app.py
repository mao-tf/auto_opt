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

    st.subheader("スタッキング")
    scan_axis = st.selectbox("スキャン軸", axis_candidates, key="scan_axis_sel")
    stacking_uploaded = st.file_uploader(
        "stacking_results.csv (任意)", type="csv", key="stacking_csv"
    )

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
if 'stacking_list' not in st.session_state:
    st.session_state.stacking_list = []
if 'heatmap_selection' not in st.session_state:
    st.session_state.heatmap_selection = []   # 複数選択点 [{x_col: v, y_col: v}, ...]

# 軸や固定パラメータが変わったらクリック選択をリセット
_state_key = (x_col, y_col, tuple(sorted(fix_vals.items())))
if st.session_state.get('_prev_key') != _state_key:
    st.session_state.sel = {}
    st.session_state.heatmap_selection = []
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
    n_sel = len(st.session_state.heatmap_selection)
    st.subheader(f"エネルギーマップ ({x_col} vs {y_col})")
    st.caption(
        "クリック: 3D 表示を更新 | "
        "ツールバーの□でドラッグ選択 → 複数点を一括追加 | "
        f"現在 **{n_sel}** 点選択中"
    )

    # ベース: ヒートマップ or 空の図
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
        fig = px.imshow(
            pivot,
            color_continuous_scale='RdBu_r',
            labels={'color': 'E (kcal/mol)'},
            aspect='auto',
        )
    else:
        fig = go.Figure()

    # クリック可能な散布図マーカーをデータ点に重ねる
    # （heatmap セル単体はクリックイベントを取りにくいため）
    hover_cols = [c for c in ['E', 'a', 'b', 'z'] if c in df_fixed.columns]
    hover_text = df_fixed.apply(
        lambda r: '<br>'.join(f"{c}={r[c]:.3f}" for c in hover_cols), axis=1
    )
    fig.add_trace(go.Scatter(
        x=df_fixed[x_col], y=df_fixed[y_col],
        mode='markers',
        marker=dict(size=14, color='rgba(0,0,0,0)',
                    line=dict(width=1, color='rgba(100,100,100,0.4)')),
        text=hover_text,
        hovertemplate=f"{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>%{{text}}<extra></extra>",
        showlegend=False,
    ))

    # 単一クリック: 白丸でハイライト
    sel_x = st.session_state.sel.get(x_col)
    sel_y = st.session_state.sel.get(y_col)
    if sel_x is not None and sel_y is not None:
        fig.add_trace(go.Scatter(
            x=[sel_x], y=[sel_y],
            mode='markers',
            marker=dict(symbol='circle-open', size=20, color='white',
                        line=dict(width=3, color='white')),
            showlegend=False, hoverinfo='skip',
        ))

    # 複数選択: 黄色丸でハイライト
    if st.session_state.heatmap_selection:
        fig.add_trace(go.Scatter(
            x=[p[x_col] for p in st.session_state.heatmap_selection],
            y=[p[y_col] for p in st.session_state.heatmap_selection],
            mode='markers',
            marker=dict(symbol='circle-open', size=18, color='yellow',
                        line=dict(width=2, color='yellow')),
            showlegend=False, hoverinfo='skip',
        ))

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        clickmode='event+select',
    )

    event = st.plotly_chart(
        fig, use_container_width=True,
        on_select='rerun', key='heatmap_chart',
    )

    # 選択イベントを処理
    if event and event.selection and event.selection.points:
        pts = event.selection.points
        if len(pts) == 1:
            # 単一クリック: 3D ビュー更新 + 選択リストを1点に
            pt = pts[0]
            if pt.get('x') is not None:
                st.session_state.sel[x_col] = float(pt['x'])
            if pt.get('y') is not None:
                st.session_state.sel[y_col] = float(pt['y'])
            st.session_state.heatmap_selection = [
                {x_col: float(pt['x']), y_col: float(pt['y'])}
            ]
        else:
            # 複数選択（ドラッグ選択）: 選択リストを更新、3D ビューは変えない
            st.session_state.heatmap_selection = [
                {x_col: float(p['x']), y_col: float(p['y'])}
                for p in pts if p.get('x') is not None and p.get('y') is not None
            ]
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

# ──────────────────────────────────────────────
#  スタッキング候補リスト
# ──────────────────────────────────────────────

st.divider()
st.subheader("スタッキング候補リスト")
st.caption(f"スキャン軸: **{scan_axis}**")

n_sel = len(st.session_state.heatmap_selection)
c_add1, c_add_multi, c_clear = st.columns([1, 1, 1])

with c_add1:
    if st.button("現在の構造を追加 (1点)"):
        if not rows.empty:
            row_data = (
                rows.loc[rows['E'].idxmin()] if len(rows) > 1 else rows.iloc[0]
            ).to_dict()
            scan_val = row_data.get(scan_axis)
            existing = [r.get(scan_axis) for r in st.session_state.stacking_list]
            if scan_val not in existing:
                st.session_state.stacking_list.append(row_data)
                st.rerun()
            else:
                st.warning(f"{scan_axis}={scan_val} は既にリストにあります")

with c_add_multi:
    btn_label = f"選択中の {n_sel} 点を追加" if n_sel > 0 else "選択中の点を追加 (0点)"
    if st.button(btn_label, disabled=(n_sel == 0)):
        added = 0
        existing = [r.get(scan_axis) for r in st.session_state.stacking_list]
        for sel_pt in st.session_state.heatmap_selection:
            px_val = sel_pt.get(x_col)
            py_val = sel_pt.get(y_col)
            if px_val is None or py_val is None:
                continue
            mask = (
                np.isclose(df_fixed[x_col], px_val, atol=1e-5) &
                np.isclose(df_fixed[y_col], py_val, atol=1e-5)
            )
            matches = df_fixed[mask]
            if matches.empty:
                continue
            row_data = (
                matches.loc[matches['E'].idxmin()] if len(matches) > 1 else matches.iloc[0]
            ).to_dict()
            scan_val = row_data.get(scan_axis)
            if scan_val not in existing:
                st.session_state.stacking_list.append(row_data)
                existing.append(scan_val)
                added += 1
        if added > 0:
            st.session_state.heatmap_selection = []
            st.rerun()
        else:
            st.warning("追加できる新しい点がありませんでした")

with c_clear:
    if st.button("リストをクリア"):
        st.session_state.stacking_list = []
        st.session_state.heatmap_selection = []
        st.rerun()

if st.session_state.stacking_list:
    df_candidates = (
        pd.DataFrame(st.session_state.stacking_list)
        .sort_values(scan_axis)
        .reset_index(drop=True)
    )
    st.dataframe(df_candidates, use_container_width=True)
    st.download_button(
        label="スタッキング候補 CSV ダウンロード",
        data=df_candidates.to_csv(index=False),
        file_name="stacking_candidates.csv",
        mime="text/csv",
    )

# ──────────────────────────────────────────────
#  スタッキングエネルギープロット
# ──────────────────────────────────────────────

if stacking_uploaded is not None:
    st.divider()
    st.subheader(f"スタッキングエネルギープロット ({scan_axis} vs E)")
    df_stack_res = pd.read_csv(stacking_uploaded)

    x_col_stack = scan_axis if scan_axis in df_stack_res.columns else df_stack_res.columns[0]
    fig_stack = go.Figure()
    for col, color, name in [
        ('E_layer', 'royalblue', '層内 E_layer'),
        ('E_stack', 'tomato',    '層間 E_stack'),
        ('E_total', 'seagreen',  '合計 E_total'),
    ]:
        if col in df_stack_res.columns:
            fig_stack.add_trace(go.Scatter(
                x=df_stack_res[x_col_stack], y=df_stack_res[col],
                mode='lines+markers', name=name,
                line=dict(color=color),
            ))
    fig_stack.update_layout(
        xaxis_title=x_col_stack,
        yaxis_title='E (kcal/mol)',
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_stack, use_container_width=True)
    st.download_button(
        label="スタッキング結果 CSV ダウンロード",
        data=df_stack_res.to_csv(index=False),
        file_name="stacking_results.csv",
        mime="text/csv",
    )
