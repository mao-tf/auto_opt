#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py  ―  auto_opt 可視化 UI (Streamlit)

使い方:
  streamlit run src/auto_opt/app.py
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
_SEC_PER_CALC = 0.02   # sander 1計算あたりの実測値 (秒)
_N_DIMERS = {'glide': 3, 'screw': 4}

st.set_page_config(page_title="auto_opt viewer", layout="wide")
st.title("auto_opt — 可視化 UI")

# ─── サイドバー: 共通設定 ───────────────────────────────────
with st.sidebar:
    st.header("共通設定")
    monomer_name = st.text_input("モノマー名", value="BTBT")
    monomer_dir  = st.text_input("モノマーデータディレクトリ", value=_MONOMER_DIR)
    mol_style    = st.selectbox("表示スタイル", ["Capped sticks", "Space fill"])

# ─── ヘルパー: 3D 表示 ─────────────────────────────────────
def _render_3d(row: pd.Series, sym: str, *, key_suffix: str = "") -> None:
    """9分子クラスター 3D 表示とダウンロードボタンを描画する。"""
    try:
        xyz_str = make_cluster_xyz(row, monomer_name, sym, monomer_dir)
    except Exception as e:
        st.error(f"XYZ 生成エラー: {e}")
        return
    view = py3Dmol.view(width=500, height=400)
    view.addModel(xyz_str, "xyz")
    if mol_style == "Space fill":
        view.setStyle({"sphere": {"scale": 1.0}})
    else:
        view.setStyle({"stick": {"radius": 0.15}, "sphere": {"radius": 0.3}})
    view.setProjection("orthographic")
    view.zoomTo()
    st.components.v1.html(view._make_html(), height=420)
    st.download_button(
        label="XYZ ダウンロード",
        data=xyz_str,
        file_name=f"{monomer_name}_cluster.xyz",
        mime="text/plain",
        key=f"dl_xyz_{key_suffix}",
    )


# ─── ヘルパー: ヒートマップ描画 ────────────────────────────
def _render_heatmap(
    df_all: pd.DataFrame,
    df_pts: pd.DataFrame,
    x_col: str,
    y_col: str,
    val_col: str,
    val_label: str,
    *,
    current_pt: dict | None = None,
    sel_pts: list[dict] | None = None,
    confirmed_pts: list[dict] | None = None,
    chart_key: str,
) -> dict | None:
    """ヒートマップを描画してクリックイベントを返す。"""
    pivot = df_all.pivot_table(
        values=val_col, index=y_col, columns=x_col, aggfunc="min"
    )
    if pivot.shape[0] >= 2 and pivot.shape[1] >= 2:
        fig = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            labels={"color": val_label},
            aspect="auto",
        )
    else:
        fig = go.Figure()

    hover_cols = [c for c in [val_col, "a", "b", "z"] if c in df_pts.columns]
    hover_text = df_pts.apply(
        lambda r: "<br>".join(f"{c}={r[c]:.3f}" for c in hover_cols), axis=1
    )
    fig.add_trace(go.Scatter(
        x=df_pts[x_col], y=df_pts[y_col],
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)",
                    line=dict(width=1, color="rgba(100,100,100,0.4)")),
        text=hover_text,
        hovertemplate=f"{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>%{{text}}<extra></extra>",
        showlegend=False,
    ))

    # 現在の 3D 表示点（白丸）
    if current_pt:
        fig.add_trace(go.Scatter(
            x=[current_pt[x_col]], y=[current_pt[y_col]],
            mode="markers",
            marker=dict(symbol="circle-open", size=20, color="white",
                        line=dict(width=3, color="white")),
            showlegend=False, hoverinfo="skip",
        ))

    # 選択中（黄色）
    if sel_pts:
        fig.add_trace(go.Scatter(
            x=[p[x_col] for p in sel_pts],
            y=[p[y_col] for p in sel_pts],
            mode="markers",
            marker=dict(symbol="circle-open", size=20, color="yellow",
                        line=dict(width=3, color="yellow")),
            showlegend=False, hoverinfo="skip",
        ))

    # 確定済み（オレンジ）
    if confirmed_pts:
        df_c = pd.DataFrame(confirmed_pts)
        if x_col in df_c.columns and y_col in df_c.columns:
            fig.add_trace(go.Scatter(
                x=df_c[x_col], y=df_c[y_col],
                mode="markers",
                marker=dict(symbol="circle-open", size=16, color="orange",
                            line=dict(width=2, color="orange")),
                showlegend=False, hoverinfo="skip",
            ))

    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), clickmode="event+select")
    event = st.plotly_chart(
        fig, use_container_width=True, on_select="rerun", key=chart_key
    )
    return event


# ══════════════════════════════════════════════════════════
#  タブ定義
# ══════════════════════════════════════════════════════════
tab_vdw, tab_layer, tab_stack = st.tabs(["VdW スキャン", "層内最適化", "スタッキング結果"])


# ══════════════════════════════════════════════════════════
#  Tab 1: VdW スキャン
# ══════════════════════════════════════════════════════════
with tab_vdw:
    vdw_uploaded = st.file_uploader(
        "step1_init_params.csv (VdW スキャン出力)", type="csv", key="vdw_csv"
    )

    if vdw_uploaded is None:
        st.info("step1_init_params.csv をアップロードしてください。")
    else:
        vdw_df = pd.read_csv(vdw_uploaded)
        vdw_sym = "screw" if "beta" in vdw_df.columns else "glide"

        # a*b 列を追加
        if "a" in vdw_df.columns and "b" in vdw_df.columns:
            vdw_df["a*b"] = vdw_df["a"] * vdw_df["b"]

        vdw_axes = ["alpha", "phi", "z"] + (["beta"] if vdw_sym == "screw" else [])
        vdw_axes = [c for c in vdw_axes if c in vdw_df.columns]

        # ─── 軸選択 ─────────────────────────────────────
        c1, c2 = st.columns(2)
        vdw_x = c1.selectbox("X 軸", vdw_axes, index=0, key="vdw_x")
        vdw_y_opts = [c for c in vdw_axes if c != vdw_x]
        vdw_y = c2.selectbox("Y 軸", vdw_y_opts, index=0, key="vdw_y")

        fix_axes_vdw = [c for c in vdw_axes if c not in (vdw_x, vdw_y)]
        fix_vals_vdw: dict[str, float] = {}
        if fix_axes_vdw:
            fix_cols_ui = st.columns(len(fix_axes_vdw))
            for i, col in enumerate(fix_axes_vdw):
                uniq = sorted(vdw_df[col].dropna().unique())
                if len(uniq) == 1:
                    fix_vals_vdw[col] = uniq[0]
                    fix_cols_ui[i].text(f"{col} = {uniq[0]}")
                else:
                    fix_vals_vdw[col] = fix_cols_ui[i].select_slider(
                        col, options=uniq, value=uniq[0], key=f"vdw_fix_{col}"
                    )

        # structure_type フィルタ
        if "structure_type" in vdw_df.columns:
            all_types = sorted(vdw_df["structure_type"].dropna().unique())
            sel_types = st.multiselect(
                "structure_type フィルタ", all_types, default=all_types, key="vdw_types"
            )
            vdw_df = vdw_df[vdw_df["structure_type"].isin(sel_types)]

        # 固定パラメータでフィルタ
        vdw_fixed = vdw_df.copy()
        for col, val in fix_vals_vdw.items():
            vdw_fixed = vdw_fixed[np.isclose(vdw_fixed[col], val, atol=1e-5)]

        if vdw_fixed.empty:
            st.warning("固定パラメータに合うデータがありません。")
        else:
            val_col_vdw = "a*b" if "a*b" in vdw_fixed.columns else vdw_axes[0]

            # session_state
            if "vdw_sel" not in st.session_state:
                st.session_state.vdw_sel = {}
            _vdw_key = (vdw_x, vdw_y, tuple(sorted(fix_vals_vdw.items())))
            if st.session_state.get("_vdw_prev_key") != _vdw_key:
                st.session_state.vdw_sel = {}
                st.session_state["_vdw_prev_key"] = _vdw_key

            col_map_v, col_3d_v = st.columns([1, 1])

            with col_map_v:
                st.subheader(f"a×b マップ ({vdw_x} vs {vdw_y})")
                st.caption("クリックで右側の 3D 構造を更新")

                event_v = _render_heatmap(
                    vdw_fixed, vdw_fixed,
                    vdw_x, vdw_y, val_col_vdw, "a×b (Å²)",
                    sel_pts=None,
                    chart_key="vdw_heatmap",
                )

                if event_v and event_v.selection and event_v.selection.points:
                    pt = event_v.selection.points[0]
                    if pt.get("x") is not None:
                        st.session_state.vdw_sel[vdw_x] = float(pt["x"])
                    if pt.get("y") is not None:
                        st.session_state.vdw_sel[vdw_y] = float(pt["y"])
                    st.rerun()

            with col_3d_v:
                st.subheader("9分子クラスター 3D 表示")
                sel_cols_v = st.columns(len(vdw_axes))
                sel_vals_v: dict[str, float] = {}
                for i, col in enumerate(vdw_axes):
                    uniq = sorted(vdw_df[col].dropna().unique())
                    default = st.session_state.vdw_sel.get(
                        col, fix_vals_vdw.get(col, uniq[0])
                    )
                    didx = min(range(len(uniq)), key=lambda j: abs(uniq[j] - default))
                    sel_vals_v[col] = sel_cols_v[i].selectbox(
                        col, uniq, index=didx, key=f"vdw_sel_{col}"
                    )

                mask_v = pd.Series([True] * len(vdw_df), index=vdw_df.index)
                for col, val in sel_vals_v.items():
                    mask_v &= np.isclose(vdw_df[col], val, atol=1e-5)
                rows_v = vdw_df[mask_v]

                if "structure_type" in rows_v.columns and rows_v["structure_type"].nunique() > 1:
                    sel_type_v = st.selectbox(
                        "structure_type", sorted(rows_v["structure_type"].unique()),
                        key="vdw_struct_type"
                    )
                    rows_v = rows_v[rows_v["structure_type"] == sel_type_v]

                if rows_v.empty:
                    st.warning("選択条件に合う行が見つかりません。")
                else:
                    row_v = rows_v.iloc[0]
                    info = f"a={row_v['a']:.2f}  b={row_v['b']:.2f}"
                    if "a*b" in row_v.index:
                        info += f"  a×b={row_v['a*b']:.2f} Å²"
                    st.caption(info)
                    _render_3d(row_v, vdw_sym, key_suffix="vdw")

            # ─── Amber 最適化コマンド生成 ─────────────────
            st.divider()
            st.subheader("Amber 最適化コマンド生成")
            st.caption(
                "ここで設定した内容をもとに、スパコンで実行するコマンドを生成します。"
            )

            hpc_dir = st.text_input(
                "HPC 上の実行ディレクトリ",
                placeholder="/home/user/runs/BTBT_glide",
                key="vdw_hpc_dir",
            )
            n_nodes = st.number_input(
                "並列ノード数", min_value=1, max_value=200, value=6, key="vdw_n_nodes"
            )

            n_pts   = len(vdw_fixed)
            n_dim   = _N_DIMERS[vdw_sym]
            wall_s  = n_pts * n_dim * _SEC_PER_CALC / n_nodes
            wall_str = f"{wall_s:.0f} 秒" if wall_s < 60 else f"{wall_s / 60:.1f} 分"

            st.info(
                f"計算点数: **{n_pts}** 点 ｜ "
                f"ダイマー数/点: **{n_dim}** ｜ "
                f"予想計算時間: **{wall_str}**（{n_nodes} ノード並列）"
            )

            job_mod = "auto_opt.amber.job_phi" if vdw_sym == "glide" else "auto_opt.amber.job_screw_phi"
            hpc_dir_str = hpc_dir.strip() or "<HPC_DIR>"
            cmd_lines = [
                f"python -m {job_mod} \\",
                f"    --auto-dir {hpc_dir_str} \\",
                f"    --monomer-name {monomer_name}",
            ]
            st.code("\n".join(cmd_lines), language="bash")
            st.caption(
                "※ step1_init_params.csv は VdW スキャン後に HPC の `--auto-dir` に置いてください。"
            )

            if "a*b" in vdw_fixed.columns:
                dl_cols = list(dict.fromkeys(
                    c for c in vdw_axes + ["a", "b", "a*b", "structure_type"]
                    if c in vdw_fixed.columns
                ))
                st.download_button(
                    label="現在の絞り込み結果を step1_init_params.csv としてダウンロード",
                    data=vdw_fixed[dl_cols].to_csv(index=False),
                    file_name="step1_init_params.csv",
                    mime="text/csv",
                    key="dl_vdw_filtered",
                )


# ══════════════════════════════════════════════════════════
#  Tab 2: 層内最適化
# ══════════════════════════════════════════════════════════
with tab_layer:
    layer_uploaded = st.file_uploader(
        "filtered_step1.csv (Amber 最適化結果)", type="csv", key="layer_csv"
    )
    stacking_uploaded = st.file_uploader(
        "stacking_results.csv (スタッキング結果・任意)", type="csv", key="stacking_csv"
    )

    if layer_uploaded is None:
        st.info("filtered_step1.csv をアップロードしてください。")
    else:
        df = pd.read_csv(layer_uploaded)
        sym = "screw" if "beta" in df.columns else "glide"
        axis_candidates = ["alpha", "phi", "z"] + (["beta"] if sym == "screw" else [])
        axis_candidates = [c for c in axis_candidates if c in df.columns]

        # ─── 軸選択 ─────────────────────────────────────
        c1, c2 = st.columns(2)
        x_col = c1.selectbox("X 軸", axis_candidates, index=0, key="layer_x")
        y_opts = [c for c in axis_candidates if c != x_col]
        y_col  = c2.selectbox("Y 軸", y_opts, index=0, key="layer_y")

        fix_cols = [c for c in axis_candidates if c not in (x_col, y_col)]
        fix_vals: dict[str, float] = {}
        if fix_cols:
            fix_cols_ui = st.columns(len(fix_cols))
            for i, col in enumerate(fix_cols):
                uniq = sorted(df[col].dropna().unique())
                if len(uniq) == 1:
                    fix_vals[col] = uniq[0]
                    fix_cols_ui[i].text(f"{col} = {uniq[0]}")
                else:
                    fix_vals[col] = fix_cols_ui[i].select_slider(
                        col, options=uniq, value=uniq[0], key=f"layer_fix_{col}"
                    )

        # session_state
        for k, v in [
            ("layer_sel", {}),
            ("stacking_list", []),
            ("select_mode", False),
            ("pending_stacking", []),
        ]:
            if k not in st.session_state:
                st.session_state[k] = v

        _layer_key = (x_col, y_col, tuple(sorted(fix_vals.items())))
        if st.session_state.get("_layer_prev_key") != _layer_key:
            st.session_state.layer_sel     = {}
            st.session_state.select_mode   = False
            st.session_state.pending_stacking = []
            st.session_state["_layer_prev_key"] = _layer_key

        # データ絞り込み
        df_fixed = df.copy()
        for col, val in fix_vals.items():
            df_fixed = df_fixed[np.isclose(df_fixed[col], val, atol=1e-5)]

        if df_fixed.empty:
            st.warning("固定パラメータに合うデータがありません。")
        else:
            select_mode = st.session_state.select_mode
            n_pending   = len(st.session_state.pending_stacking)

            col_map, col_3d = st.columns([1, 1])

            with col_map:
                st.subheader(f"エネルギーマップ ({x_col} vs {y_col})")
                if select_mode:
                    st.caption(
                        f"**候補選択モード** — クリックで追加/解除 ({n_pending} 点選択中)"
                    )
                else:
                    st.caption("クリックで右側の 3D 構造を更新")

                # 現在表示点（白丸）
                current_pt = None
                if not select_mode:
                    sx = st.session_state.layer_sel.get(x_col)
                    sy = st.session_state.layer_sel.get(y_col)
                    if sx is not None and sy is not None:
                        current_pt = {x_col: sx, y_col: sy}

                event = _render_heatmap(
                    df_fixed, df_fixed,
                    x_col, y_col, "E", "E (kcal/mol)",
                    current_pt=current_pt,
                    sel_pts=(
                        st.session_state.pending_stacking if select_mode else None
                    ),
                    confirmed_pts=st.session_state.stacking_list or None,
                    chart_key="layer_heatmap",
                )

                if event and event.selection and event.selection.points:
                    pt = event.selection.points[0]
                    cx, cy = pt.get("x"), pt.get("y")
                    if cx is not None and cy is not None:
                        cx, cy = float(cx), float(cy)
                        if select_mode:
                            already = any(
                                np.isclose(p[x_col], cx, atol=1e-5)
                                and np.isclose(p[y_col], cy, atol=1e-5)
                                for p in st.session_state.pending_stacking
                            )
                            if already:
                                st.session_state.pending_stacking = [
                                    p for p in st.session_state.pending_stacking
                                    if not (
                                        np.isclose(p[x_col], cx, atol=1e-5)
                                        and np.isclose(p[y_col], cy, atol=1e-5)
                                    )
                                ]
                            else:
                                st.session_state.pending_stacking.append(
                                    {x_col: cx, y_col: cy}
                                )
                        else:
                            st.session_state.layer_sel[x_col] = cx
                            st.session_state.layer_sel[y_col] = cy
                        st.rerun()

                # モード切替ボタン
                st.markdown("---")
                if not select_mode:
                    if st.button("候補構造を選ぶ", key="btn_enter_select"):
                        st.session_state.select_mode = True
                        st.rerun()
                else:
                    c_ok, c_cancel = st.columns([1, 1])
                    with c_ok:
                        if st.button(
                            f"確定 ({n_pending} 点)",
                            key="btn_confirm_select",
                            type="primary",
                            disabled=(n_pending == 0),
                        ):
                            existing_keys = {
                                tuple(sorted(
                                    (k, round(v, 5)) for k, v in r.items()
                                    if isinstance(v, float)
                                ))
                                for r in st.session_state.stacking_list
                            }
                            for p in st.session_state.pending_stacking:
                                mask_p = pd.Series(
                                    [True] * len(df_fixed), index=df_fixed.index
                                )
                                for c, v in p.items():
                                    if c in df_fixed.columns:
                                        mask_p &= np.isclose(df_fixed[c], v, atol=1e-5)
                                matches = df_fixed[mask_p]
                                if not matches.empty:
                                    row_data = (
                                        matches.loc[matches["E"].idxmin()]
                                        if len(matches) > 1 else matches.iloc[0]
                                    ).to_dict()
                                    dk = tuple(sorted(
                                        (k, round(v, 5)) for k, v in row_data.items()
                                        if isinstance(v, float)
                                    ))
                                    if dk not in existing_keys:
                                        st.session_state.stacking_list.append(row_data)
                                        existing_keys.add(dk)
                            st.session_state.pending_stacking = []
                            st.session_state.select_mode = False
                            st.rerun()
                    with c_cancel:
                        if st.button("キャンセル", key="btn_cancel_select"):
                            st.session_state.pending_stacking = []
                            st.session_state.select_mode = False
                            st.rerun()

            # ─── 3D 表示 ─────────────────────────────────
            with col_3d:
                st.subheader("9分子クラスター 3D 表示")
                sel_cols_ui = st.columns(len(axis_candidates))
                sel_vals: dict[str, float] = {}
                for i, col in enumerate(axis_candidates):
                    uniq = sorted(df[col].dropna().unique())
                    default = st.session_state.layer_sel.get(
                        col, fix_vals.get(col, uniq[0])
                    )
                    didx = min(range(len(uniq)), key=lambda j: abs(uniq[j] - default))
                    sel_vals[col] = sel_cols_ui[i].selectbox(
                        col, uniq, index=didx, key=f"layer_sel_{col}"
                    )

                mask = pd.Series([True] * len(df), index=df.index)
                for col, val in sel_vals.items():
                    mask &= np.isclose(df[col], val, atol=1e-5)
                rows = df[mask]

                if "structure_type" in rows.columns and rows["structure_type"].nunique() > 1:
                    types = sorted(rows["structure_type"].dropna().unique())
                    sel_type = st.selectbox("structure_type", types, key="layer_struct_type")
                    rows = rows[rows["structure_type"] == sel_type]

                if rows.empty:
                    st.warning("選択条件に合う行が見つかりません。")
                else:
                    row = rows.loc[rows["E"].idxmin()] if len(rows) > 1 else rows.iloc[0]
                    st.caption(
                        f"E = {row['E']:.3f} kcal/mol | "
                        f"a={row['a']:.2f}  b={row['b']:.2f}  z={row['z']:.2f}"
                    )
                    _render_3d(row, sym, key_suffix="layer")

            # ─── スタッキング候補リスト ───────────────────
            st.divider()
            st.subheader("スタッキング候補リスト")
            scan_axis = st.selectbox(
                "自動選択の軸（各軸の最安定構造を自動追加するときに使用）",
                axis_candidates, key="scan_axis_sel",
            )

            if st.button(f"各 {scan_axis} の最安定構造を自動追加", key="btn_auto_select"):
                existing_keys = {
                    tuple(sorted(
                        (k, round(v, 5)) for k, v in r.items() if isinstance(v, float)
                    ))
                    for r in st.session_state.stacking_list
                }
                added = 0
                for _, grp in df_fixed.groupby(scan_axis):
                    best_row = grp.loc[grp["E"].idxmin()].to_dict()
                    dk = tuple(sorted(
                        (k, round(v, 5)) for k, v in best_row.items()
                        if isinstance(v, float)
                    ))
                    if dk not in existing_keys:
                        st.session_state.stacking_list.append(best_row)
                        existing_keys.add(dk)
                        added += 1
                if added:
                    st.rerun()
                else:
                    st.info("追加できる新しい点がありませんでした（すでに全て登録済み）")

            if not st.session_state.stacking_list:
                st.caption(
                    "上のボタンで自動追加、またはヒートマップ上の「候補構造を選ぶ」で手動選択してください"
                )
            else:
                df_candidates = (
                    pd.DataFrame(st.session_state.stacking_list)
                    .sort_values(scan_axis)
                    .reset_index(drop=True)
                )
                show_cols = list(dict.fromkeys(
                    c for c in [scan_axis, x_col, y_col, "E", "a", "b", "z"]
                    if c in df_candidates.columns
                ))
                st.dataframe(df_candidates[show_cols], use_container_width=True)

                c_dl, c_clear = st.columns([2, 1])
                with c_dl:
                    st.download_button(
                        label="スタッキング候補 CSV ダウンロード",
                        data=df_candidates.to_csv(index=False),
                        file_name="stacking_candidates.csv",
                        mime="text/csv",
                        key="dl_stacking_candidates",
                    )
                with c_clear:
                    if st.button("リストをクリア", key="btn_clear_stacking"):
                        st.session_state.stacking_list = []
                        st.rerun()

        # ─── スタッキングエネルギープロット ──────────────
        if stacking_uploaded is not None:
            st.divider()
            st.subheader("スタッキングエネルギープロット")
            df_sr = pd.read_csv(stacking_uploaded)
            x_col_s = scan_axis if scan_axis in df_sr.columns else df_sr.columns[0]
            fig_s = go.Figure()
            for col, color, name in [
                ("E_layer", "royalblue", "層内 E_layer"),
                ("E_stack", "tomato",    "層間 E_stack"),
                ("E_total", "seagreen",  "合計 E_total"),
            ]:
                if col in df_sr.columns:
                    fig_s.add_trace(go.Scatter(
                        x=df_sr[x_col_s], y=df_sr[col],
                        mode="lines+markers", name=name,
                        line=dict(color=color),
                    ))
            fig_s.update_layout(
                xaxis_title=x_col_s,
                yaxis_title="E (kcal/mol)",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_s, use_container_width=True)
            st.download_button(
                label="スタッキング結果 CSV ダウンロード",
                data=df_sr.to_csv(index=False),
                file_name="stacking_results.csv",
                mime="text/csv",
                key="dl_stacking_results_layer",
            )


# ══════════════════════════════════════════════════════════
#  Tab 3: スタッキング結果
# ══════════════════════════════════════════════════════════
with tab_stack:
    stack_uploaded2 = st.file_uploader(
        "stacking_results.csv", type="csv", key="stack_csv2"
    )

    if stack_uploaded2 is None:
        st.info("stacking_results.csv をアップロードしてください。")
    else:
        df_sr2 = pd.read_csv(stack_uploaded2)
        stack_axes = [c for c in ["cy", "cz", "phi", "z", "beta"] if c in df_sr2.columns]
        if not stack_axes:
            stack_axes = [df_sr2.columns[0]]
        x_col_s2 = st.selectbox("X 軸", stack_axes, key="stack_x")

        fig_s2 = go.Figure()
        for col, color, name in [
            ("E_layer", "royalblue", "層内 E_layer"),
            ("E_stack", "tomato",    "層間 E_stack"),
            ("E_total", "seagreen",  "合計 E_total"),
        ]:
            if col in df_sr2.columns:
                fig_s2.add_trace(go.Scatter(
                    x=df_sr2[x_col_s2], y=df_sr2[col],
                    mode="lines+markers", name=name,
                    line=dict(color=color),
                ))
        fig_s2.update_layout(
            xaxis_title=x_col_s2,
            yaxis_title="E (kcal/mol)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_s2, use_container_width=True)
        st.download_button(
            label="スタッキング結果 CSV ダウンロード",
            data=df_sr2.to_csv(index=False),
            file_name="stacking_results.csv",
            mime="text/csv",
            key="dl_stacking_results2",
        )
