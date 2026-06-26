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
#  session_state
# ──────────────────────────────────────────────

if 'sel' not in st.session_state:
    st.session_state.sel = {}
if 'stacking_list' not in st.session_state:
    st.session_state.stacking_list = []
if 'editor_key' not in st.session_state:
    st.session_state.editor_key = 0  # data_editor リセット用

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

    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
        fig = px.imshow(
            pivot,
            color_continuous_scale='RdBu_r',
            labels={'color': 'E (kcal/mol)'},
            aspect='auto',
        )
    else:
        fig = go.Figure()

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

    # 現在選択中の点をハイライト
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

    # stacking_list に登録済みの点をオレンジ丸でハイライト
    if st.session_state.stacking_list:
        df_stk = pd.DataFrame(st.session_state.stacking_list)
        if x_col in df_stk.columns and y_col in df_stk.columns:
            fig.add_trace(go.Scatter(
                x=df_stk[x_col], y=df_stk[y_col],
                mode='markers',
                marker=dict(symbol='circle-open', size=16, color='orange',
                            line=dict(width=2, color='orange')),
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

    if event and event.selection and event.selection.points:
        pt = event.selection.points[0]
        if pt.get('x') is not None:
            st.session_state.sel[x_col] = float(pt['x'])
        if pt.get('y') is not None:
            st.session_state.sel[y_col] = float(pt['y'])
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
    default = st.session_state.sel.get(col, fix_vals.get(col, unique[0]))
    default_idx = min(range(len(unique)), key=lambda j: abs(unique[j] - default))
    sel_vals[col] = sel_cols_ui[i].selectbox(col, unique, index=default_idx, key=f"sel_{col}")

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
#  スタッキング候補: テーブルでチェック選択
# ──────────────────────────────────────────────

st.divider()
st.subheader("スタッキング候補リスト")
st.caption(f"スキャン軸: **{scan_axis}** | 下のテーブルで行をチェックして追加してください")

# 表示列: 軸パラメータ + E + 格子定数
show_cols = [c for c in axis_candidates + ['E', 'a', 'b'] if c in df_fixed.columns]
df_table = (
    df_fixed[show_cols]
    .sort_values('E')
    .drop_duplicates(subset=[x_col, y_col])
    .sort_values([x_col, y_col])
    .reset_index(drop=True)
)
df_table.insert(0, '追加', False)

edited = st.data_editor(
    df_table,
    column_config={'追加': st.column_config.CheckboxColumn('追加', default=False)},
    disabled=[c for c in df_table.columns if c != '追加'],
    use_container_width=True,
    hide_index=True,
    key=f'stacking_editor_{st.session_state.editor_key}',
)

checked = edited[edited['追加']]
c_add, c_clear = st.columns([1, 1])

with c_add:
    btn_label = f"チェックした {len(checked)} 点を追加" if len(checked) > 0 else "追加 (0点チェック中)"
    if st.button(btn_label, disabled=(len(checked) == 0)):
        existing = {r.get(scan_axis) for r in st.session_state.stacking_list}
        added = 0
        for _, trow in checked.iterrows():
            # df_fixed から完全な行データを取得
            mask_t = pd.Series([True] * len(df_fixed), index=df_fixed.index)
            for c in show_cols:
                if c in df_fixed.columns and c in trow.index:
                    mask_t &= np.isclose(df_fixed[c], trow[c], atol=1e-5)
            matches = df_fixed[mask_t]
            if matches.empty:
                continue
            row_data = (
                matches.loc[matches['E'].idxmin()] if len(matches) > 1 else matches.iloc[0]
            ).to_dict()
            scan_val = row_data.get(scan_axis)
            if scan_val not in existing:
                st.session_state.stacking_list.append(row_data)
                existing.add(scan_val)
                added += 1
        if added > 0:
            st.session_state.editor_key += 1  # チェックボックスをリセット
            st.rerun()
        else:
            st.warning("追加できる新しい点がありませんでした（重複または一致なし）")

with c_clear:
    if st.button("リストをクリア"):
        st.session_state.stacking_list = []
        st.session_state.editor_key += 1
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
