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

from auto_opt.plot.make_cluster_xyz import make_cluster_xyz, _load_symbols, _xyz_block, _cluster_screw, _cluster_glide
from auto_opt.utils import place_monomer

_MONOMER_DIR = str(Path(__file__).resolve().parents[2] / "data" / "monomer")

# 1ダイマーあたりの実効計算時間（秒、並列実行時のノード1個あたりの実測ベース）。
# sander自体の内部計算時間は0.02秒程度だが、これは.outファイルに書かれる
# 「Total time」の値であり、実際のパイプラインの所要時間ではない。
# 本番実行(num_nodes=38〜50でtleap+sanderをOSプロセスとして都度起動)では、
# tleapの起動オーバーヘッド・プロセス生成コスト・OpenMPスレッド過多による競合が
# 支配的で、BTBTの実測平均は約2.85秒/ダイマーだった。
# OMP_NUM_THREADS=1修正（本番ノード上で実測: 2.566秒→0.666秒、約3.9倍）を
# 適用した後の推定値として、tleap等の固定オーバーヘッド(~0.28秒)を残しつつ
# sander側のみ短縮されるとして概算した値。次の本番実行で実測が取れ次第、
# この値は再校正すること。
_SEC_PER_CALC = 1.0
_N_DIMERS = {'glide': 3, 'screw': 4}

st.set_page_config(page_title="auto_opt viewer", layout="wide")
st.title("auto_opt — 可視化 UI")

# ペンディング中のモノマー名をウィジェット描画前にセット（st.rerun()後の2周目で反映）
if "_pending_monomer_name" in st.session_state:
    st.session_state["monomer_name"] = st.session_state.pop("_pending_monomer_name")

# ─── サイドバー: 共通設定 ───────────────────────────────────
with st.sidebar:
    st.header("共通設定")
    monomer_name = st.text_input(
        "モノマー名",
        value=st.session_state.get("monomer_name", "BTBT"),
        key="monomer_name",
    )
    mol_style = st.selectbox("表示スタイル", ["Capped sticks", "Space fill"])

# local_work_dir はセットアップタブで定義し、他タブで参照できるよう先に初期化
local_work_dir: str = st.session_state.get("local_work_dir", "")

# モノマーデータディレクトリ: local_work_dir/data/monomer を優先、なければパッケージ付属
monomer_dir = (
    str(Path(local_work_dir) / "data" / "monomer")
    if local_work_dir.strip()
    else _MONOMER_DIR
)


def _save_file(content: str, path: Path) -> None:
    """ファイルを書き込み、成功/失敗を st.toast で通知する。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        st.toast(f"保存しました: {path}", icon="✅")
    except Exception as e:
        st.toast(f"保存失敗: {e}", icon="❌")

# ─── ヘルパー: 3D 表示 ─────────────────────────────────────
def _xyz_filename(row: pd.Series, sym: str) -> str:
    """ダウンロード用 XYZ ファイル名をパラメータから生成する。"""
    parts = [monomer_name, sym]
    for col, fmt in [("alpha", ".1f"), ("beta", ".1f"), ("phi", ".1f"), ("z", ".2f")]:
        if col in row.index:
            val = float(row[col])
            parts.append(f"{col}{val:{fmt}}")
    return "_".join(parts) + ".xyz"


def make_stacking_xyz(row: pd.Series, monomer_name: str, sym: str,
                      monomer_dir: str) -> str:
    """層クラスター(9分子) + スタッキング分子(2分子) の XYZ を返す。"""
    alpha1 = float(row.get('alpha1', row.get('alpha', 0.0)))
    alpha2 = float(row.get('alpha2', alpha1))
    beta   = float(row.get('beta',  0.0))
    phi    = float(row.get('phi',   0.0))
    z      = float(row.get('z',     0.0))
    a      = float(row.get('a',     0.0))
    bt1    = float(row.get('bt1',   0.0))
    bt2    = float(row.get('bt2',   0.0))
    b      = bt1 + bt2 if bt1 + bt2 > 0 else float(row.get('b', 0.0))
    cx     = float(row.get('cx',    0.0))
    cy     = float(row.get('cy',    0.0))
    cz     = float(row.get('cz',    0.0))

    syms = _load_symbols(monomer_name, monomer_dir)

    # 層クラスター（alpha1 で生成）
    if sym == 'screw':
        layer_syms, layer_xyz = _cluster_screw(
            monomer_name, a, b, bt1, bt2, z, alpha1, beta, phi, monomer_dir
        )
    else:
        layer_syms, layer_xyz = _cluster_glide(
            monomer_name, a, b, z, alpha1, phi, monomer_dir
        )

    # スタッキング分子（alpha2 で生成・make_io_stacking と同じ配置）
    stack_syms:  list[str]        = []
    stack_parts: list[np.ndarray] = []

    if sym == 'screw':
        # c: (cx, cy, cz, +alpha2),  c_: (a/2+cx, bt1+cy, cz+z, -alpha2)
        for tx, ty, tz, al in [
            (cx,          cy,       cz,     alpha2),
            (a/2 + cx,    bt1 + cy, cz + z, -alpha2),
        ]:
            arr = place_monomer(monomer_name, tx, ty, tz, phi, al, beta,
                                monomer_dir=monomer_dir)
            stack_syms.extend(syms)
            stack_parts.append(arr[:, :3])
    else:
        # c: (cx, cy, cz, +alpha2),  c_: (cx, cy, cz, -alpha2)
        for tx, ty, tz, al in [
            (cx, cy, cz,  alpha2),
            (cx, cy, cz, -alpha2),
        ]:
            arr = place_monomer(monomer_name, tx, ty, tz, phi, al,
                                monomer_dir=monomer_dir)
            stack_syms.extend(syms)
            stack_parts.append(arr[:, :3])

    stack_xyz = np.vstack(stack_parts)

    all_syms  = layer_syms + stack_syms
    all_xyz   = np.vstack([layer_xyz, stack_xyz])
    n_atoms   = len(all_syms)
    comment   = (f"{monomer_name} {sym} stacking | "
                 f"alpha1={alpha1:.1f} alpha2={alpha2:.1f} beta={beta:.1f} phi={phi:.1f} z={z:.3f} "
                 f"cx={cx:.2f} cy={cy:.2f} cz={cz:.2f}")
    return f"{n_atoms}\n{comment}\n" + _xyz_block(all_syms, all_xyz)


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
        file_name=_xyz_filename(row, sym),
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

    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        clickmode="event+select",
    )
    event = st.plotly_chart(
        fig, use_container_width=True,
        on_select="rerun", key=chart_key,
    )
    return event


# ══════════════════════════════════════════════════════════
#  タブ定義
# ══════════════════════════════════════════════════════════
tab_setup, tab_vdw, tab_layer, tab_stack, tab_param = st.tabs(
    ["セットアップ", "VdW スキャン", "層内最適化", "スタッキング結果", "変数説明"]
)


# ══════════════════════════════════════════════════════════
#  Tab 0: セットアップ
# ══════════════════════════════════════════════════════════
with tab_setup:
    # ─── ローカル作業ディレクトリ ────────────────────────
    st.subheader("ローカル作業ディレクトリ")
    st.caption(
        "この計算ランのローカル保存先を設定します。"
        " run_config.yaml の保存・CSV の自動読み込み・SCP コマンドの生成に使われます。"
    )
    local_work_dir = st.text_input(
        "ローカル作業ディレクトリ（絶対パス）",
        placeholder=str(Path.home() / "Working" / "auto_opt" / "runs" / "run_name"),
        key="local_work_dir",
    )
    st.divider()

    # ─── Step 1: モノマー ────────────────────────────────
    st.subheader("Step 1: モノマー xyz をドロップ")
    st.caption(
        "Gaussian で最適化する前の粗い構造ファイル（.xyz）をドロップしてください。"
        " ファイル名からモノマー名を自動検出します。"
    )
    uploaded_xyz = st.file_uploader(
        "xyz ファイルをドロップ",
        type=["xyz"],
        key="setup_xyz_upload",
        label_visibility="collapsed",
    )

    _lwd_mono = Path(local_work_dir) / "data" / "monomer" if local_work_dir.strip() else None

    if uploaded_xyz is not None:
        _stem = Path(uploaded_xyz.name).stem
        import re as _re
        _auto_name = _re.sub(r'[_-](raw|pre|input|pre_opt)$', '', _stem)

        _name_col, _mode_col = st.columns([2, 3])
        _name_mode = _mode_col.radio(
            "モノマー名",
            [f"自動（{_auto_name}）", "手動で指定"],
            horizontal=True,
            key="setup_name_mode",
            label_visibility="collapsed",
        )
        if "手動" in _name_mode:
            _resolved_name = _name_col.text_input(
                "モノマー名", value=_auto_name, key="setup_monomer_name_manual"
            )
        else:
            _resolved_name = _auto_name
            _name_col.markdown(f"**モノマー名:** `{_auto_name}`")

        if _lwd_mono:
            _save_xyz_path = _lwd_mono / f"{_resolved_name}_raw.xyz"
            if st.button(f"xyz をローカルに保存  →  {_save_xyz_path}", key="save_raw_xyz"):
                _save_file(uploaded_xyz.getvalue().decode("utf-8"), _save_xyz_path)
                st.session_state["_pending_monomer_name"] = _resolved_name
                st.rerun()
        else:
            st.caption("💡 ローカル作業ディレクトリを設定すると保存ボタンが現れます")

        _local_xyz_out = str(_lwd_mono / f"{_resolved_name}_raw.xyz") if _lwd_mono else f"<local>/{_resolved_name}_raw.xyz"
    else:
        _resolved_name = monomer_name
        _local_xyz_out = f"<local>/{monomer_name}_raw.xyz"

    c1, c2, c3 = st.columns(3)
    setup_sym    = c1.selectbox("対称性", ["glide", "screw"], key="setup_sym")
    setup_charge = c2.number_input("電荷", value=0, step=1, key="setup_charge")
    setup_mult   = c3.number_input("多重度", value=1, min_value=1, step=1, key="setup_mult")

    # ─── Step 2: HPC 設定 ────────────────────────────────
    st.subheader("Step 2: HPC 設定")
    st.caption(
        "スパコン上での作業ディレクトリを決めてください。"
        " このディレクトリ以下に全ての計算ファイルが置かれます。"
    )
    c4, c5 = st.columns(2)
    hpc_host    = c4.text_input("HPC ホスト", placeholder="user@cmdell81", key="setup_hpc_host")
    hpc_workdir = c5.text_input(
        "HPC 作業ディレクトリ（絶対パス）",
        placeholder=f"/home/user/projects/{monomer_name}",
        key="setup_hpc_workdir",
    )

    # ─── Step 3: Amber Tools パス ────────────────────────
    st.subheader("Step 3: Amber Tools 設定（スパコン上）")
    amber_bin = st.text_input(
        "Amber bin ディレクトリ", value="~/anaconda3/envs/amber/bin", key="setup_amber_bin"
    )

    c6, c7 = st.columns(2)
    with c6:
        st.caption("キュー 1")
        q1_name  = st.text_input("名前", value="gr1.q",  key="setup_q1_name")
        q1_nproc = st.number_input("nproc", value=40,    key="setup_q1_nproc")
        q1_pe    = st.text_input("pe",  value="OpenMP",  key="setup_q1_pe")
    with c7:
        st.caption("キュー 2")
        q2_name  = st.text_input("名前", value="gr2.q",  key="setup_q2_name")
        q2_nproc = st.number_input("nproc", value=52,    key="setup_q2_nproc")
        q2_pe    = st.text_input("pe",  value="OpenMP",  key="setup_q2_pe")

    ca, cb, cc = st.columns(3)
    max_jobs      = ca.number_input("max_concurrent_jobs", value=6,  min_value=1, key="setup_max_jobs")
    poll_interval = cb.number_input("poll_interval (秒)",  value=30, min_value=5, key="setup_poll")
    nproc_reserve = cc.number_input("nproc_reserve",       value=2,  min_value=0, key="setup_nproc_res")

    # ─── Step 4: 粗VdWスキャンパラメータ ────────────────
    st.subheader("Step 4: 粗VdWスキャンパラメータ")

    _DEFAULT_VDW_SETUP = {
        "glide": {
            "alpha": {"min": 0.0,   "max": 90.0, "step": 10.0},
            "phi":   {"min": -10.0, "max": 10.0, "step": 4.0},
            "z":     {"min": -2.0,  "max": 2.0,  "step": 0.5},
        },
        "screw": {
            "alpha": {"min": 0.0,   "max": 90.0, "step": 10.0},
            "phi":   {"min": -10.0, "max": 10.0, "step": 4.0},
            "z":     {"min": -2.0,  "max": 2.0,  "step": 0.5},
            "beta":  {"min": -20.0, "max": 20.0, "step": 5.0},
        },
    }
    _defs = _DEFAULT_VDW_SETUP[setup_sym]
    _axes_setup = ["alpha", "phi", "z"] + (["beta"] if setup_sym == "screw" else [])

    setup_param_cfg: dict[str, dict] = {}
    for _axis in _axes_setup:
        _d = _defs[_axis]
        with st.expander(_axis, expanded=True):
            _c1, _c2, _c3 = st.columns(3)
            _pmin  = _c1.number_input("min",  value=_d["min"],  key=f"setup_{_axis}_min")
            _pmax  = _c2.number_input("max",  value=_d["max"],  key=f"setup_{_axis}_max")
            _pstep = _c3.number_input("step", value=_d["step"], min_value=0.001,
                                      key=f"setup_{_axis}_step")
        setup_param_cfg[_axis] = {"min": _pmin, "max": _pmax, "step": _pstep}

    if setup_sym == "glide":
        setup_vdw_select = st.multiselect(
            "vdw_select",
            ["all", "a-stack", "b-stack"],
            default=["a-stack", "b-stack"],
            key="setup_vdw_select",
        )
        if not setup_vdw_select:
            setup_vdw_select = ["all"]
    else:
        setup_vdw_select = ["all"]

    setup_n_nodes = st.number_input(
        "並列ノード数", min_value=1, max_value=200, value=6, key="setup_n_nodes"
    )

    # ─── 派生値 ──────────────────────────────────────────
    _hpc_host_out    = hpc_host.strip()    or "user@hpc"
    _hpc_workdir_out = hpc_workdir.strip() or f"/path/to/{monomer_name}"
    _auto_dir_out    = f"{_hpc_workdir_out}/runs/{monomer_name}_{setup_sym}"
    _mono_xyz_hpc    = f"{_hpc_workdir_out}/data/monomer/{monomer_name}_raw.xyz"

    # ─── 生成物 ──────────────────────────────────────────
    st.divider()
    st.subheader("生成物")

    # HPC ディレクトリ構成
    with st.expander("HPC 上のディレクトリ構成", expanded=True):
        st.code(
            f"{_hpc_workdir_out}/\n"
            f"├── run_config.yaml                    ← scp で転送\n"
            f"├── data/\n"
            f"│   ├── monomer/\n"
            f"│   │   ├── {monomer_name}_raw.xyz     ← scp で転送（ファイル名は変換される）\n"
            f"│   │   ├── {monomer_name}.xyz          ← Gaussian opt 後に自動生成\n"
            f"│   │   ├── {monomer_name}.csv          ← 自動生成\n"
            f"│   │   └── {monomer_name}.mol2         ← 自動生成\n"
            f"│   └── amber_ref/\n"
            f"│       └── {monomer_name}_gaff2.out   ← 自動生成\n"
            f"└── runs/{monomer_name}_{setup_sym}/   ← VdW・Amber 計算結果\n",
            language="text",
        )

    # ~/.auto_opt.yaml テンプレート
    _amber_bin = amber_bin.rstrip("/")
    _auto_opt_yaml = (
        f"scheduler: sge\n"
        f"queues:\n"
        f"  - name: {q1_name}\n"
        f"    nproc: {int(q1_nproc)}\n"
        f"    pe: {q1_pe}\n"
        f"  - name: {q2_name}\n"
        f"    nproc: {int(q2_nproc)}\n"
        f"    pe: {q2_pe}\n"
        f"max_concurrent_jobs: {int(max_jobs)}\n"
        f"poll_interval: {int(poll_interval)}\n"
        f"nproc_reserve: {int(nproc_reserve)}\n\n"
        f"amber_tools:\n"
        f"  antechamber: {_amber_bin}/antechamber\n"
        f"  parmchk2:    {_amber_bin}/parmchk2\n"
        f"  tleap:       {_amber_bin}/tleap\n"
        f"  sander:      {_amber_bin}/sander\n"
    )
    with st.expander("~/.auto_opt.yaml", expanded=True):
        st.code(_auto_opt_yaml, language="yaml")
        if st.button("~/.auto_opt.yaml に保存", key="save_auto_opt_yaml"):
            _save_file(_auto_opt_yaml, Path.home() / ".auto_opt.yaml")

    # run_config.yaml
    _params_yaml = ""
    for _axis in ["z", "alpha", "phi"] + (["beta"] if setup_sym == "screw" else []):
        _cfg = setup_param_cfg[_axis]
        _params_yaml += (
            f"  {_axis}:\n"
            f"    min: {_cfg['min']}\n"
            f"    max: {_cfg['max']}\n"
            f"    step: {_cfg['step']}\n\n"
        )
    if len(setup_vdw_select) == 1:
        _vdw_select_yaml = setup_vdw_select[0]
    else:
        _vdw_select_yaml = "\n" + "".join(f"  - {v}\n" for v in setup_vdw_select)

    _run_config_setup = (
        f"monomer: {monomer_name}\n"
        f"symmetry: {setup_sym}\n"
        f"auto_dir: {_auto_dir_out}\n"
        f"data_dir: {_hpc_workdir_out}/data\n"
        f"monomer_xyz: {_mono_xyz_hpc}\n\n"
        f"charge: {int(setup_charge)}\n"
        f"mult: {int(setup_mult)}\n\n"
        f"parameters:\n{_params_yaml}\n"
        f"vdw_select: {_vdw_select_yaml}\n"
        f"amber:\n"
        f"  num_nodes: {int(setup_n_nodes)}\n"
    )
    with st.expander("run_config.yaml", expanded=True):
        st.code(_run_config_setup, language="yaml")
        _lwd_setup = Path(local_work_dir) if local_work_dir.strip() else None
        if _lwd_setup:
            if st.button("run_config.yaml をローカルに保存", key="save_run_config_setup"):
                _save_file(_run_config_setup, _lwd_setup / "run_config.yaml")
        else:
            st.caption("💡 サイドバーでローカル作業ディレクトリを設定すると直接保存できます")

    # コマンド
    st.subheader("コマンド")
    _lwd_str = local_work_dir.strip() or "<ローカル作業ディレクトリ>"
    _run_config_local = f"{_lwd_str}/run_config.yaml"
    _lwd_mono_str = f"{_lwd_str}/data/monomer"

    st.caption("0. HPC 側ディレクトリを作成（初回のみ）")
    st.code(
        f"ssh {_hpc_host_out} "
        f"'mkdir -p {_hpc_workdir_out}/data/monomer {_auto_dir_out}'",
        language="bash",
    )
    st.caption("1. ローカル → HPC へ転送")
    st.code(
        f"# モノマー xyz を転送\n"
        f"scp {_local_xyz_out} \\\n"
        f"    {_hpc_host_out}:{_mono_xyz_hpc}\n\n"
        f"# 設定ファイルを転送\n"
        f"scp {_run_config_local} {_hpc_host_out}:{_auto_dir_out}/\n\n"
        f"# 環境設定（初回のみ）\n"
        f"scp ~/.auto_opt.yaml {_hpc_host_out}:~/.auto_opt.yaml",
        language="bash",
    )
    st.caption("2. HPC 上で実行（モノマー最適化 → VdW スウィープ）")
    st.code(
        f"# auto_opt は pip インストール済みなのでどこからでも実行可\n"
        f"python -m auto_opt.run --config {_auto_dir_out}/run_config.yaml"
        f" --start-from monomer --stop-after vdw",
        language="bash",
    )
    st.caption(
        "2b. （任意）Tab1 で E 値付きのプレビューを見たい場合のみ実行 "
        "— Amber 最適化では使わないので省略可"
    )
    st.code(
        f"python -m auto_opt.amber.job_eval_grid --auto-dir {_auto_dir_out}"
        f" --monomer-name {monomer_name} --symmetry {setup_sym}",
        language="bash",
    )
    st.caption("3. HPC → ローカル へ結果を取得（最適化後モノマーファイル）")
    st.code(
        f"mkdir -p {_lwd_mono_str}\n"
        f"scp {_hpc_host_out}:{_hpc_workdir_out}/data/monomer/{monomer_name}.{{xyz,mol2,csv}} \\\n"
        f"    {_lwd_mono_str}/",
        language="bash",
    )


# ══════════════════════════════════════════════════════════
#  Tab 1: VdW スキャン
# ══════════════════════════════════════════════════════════
with tab_vdw:
    _vdw_local_path = (
        Path(local_work_dir) / "step1_init_params.csv"
        if local_work_dir.strip() else None
    )
    _vdw_local_exists = _vdw_local_path is not None and _vdw_local_path.exists()

    with st.expander("HPC から結果を取得（step1_init_params.csv）"):
        _lwd_vdw = local_work_dir.strip() or "<ローカル作業ディレクトリ>"
        st.code(
            f"scp {_hpc_host_out}:{_auto_dir_out}/step1_init_params.csv \\\n"
            f"    {_lwd_vdw}/",
            language="bash",
        )

    if _vdw_local_exists:
        st.caption(f"📂 {_vdw_local_path} から自動読み込み")
    vdw_uploaded = st.file_uploader(
        "step1_init_params.csv (VdW スキャン出力) — ローカルパスが設定されていれば自動読み込み",
        type="csv", key="vdw_csv"
    )

    vdw_source = vdw_uploaded or (_vdw_local_path if _vdw_local_exists else None)
    if vdw_source is None:
        st.info("step1_init_params.csv をアップロードするか、サイドバーでローカル作業ディレクトリを設定してください。")
    else:
        vdw_df = pd.read_csv(vdw_source)
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
            if len(all_types) > 1:
                sel_type = st.radio(
                    "structure_type", all_types, horizontal=True, key="vdw_types"
                )
                vdw_df = vdw_df[vdw_df["structure_type"] == sel_type]

        # 固定パラメータでフィルタ
        vdw_fixed = vdw_df.copy()
        for col, val in fix_vals_vdw.items():
            vdw_fixed = vdw_fixed[np.isclose(vdw_fixed[col], val, atol=1e-5)]

        if vdw_fixed.empty:
            st.warning("固定パラメータに合うデータがありません。")
        else:
            if "E" in vdw_fixed.columns and vdw_fixed["E"].notna().any():
                val_col_vdw  = "E"
                val_label_vdw = "E (kcal/mol)"
            elif "a*b" in vdw_fixed.columns:
                val_col_vdw  = "a*b"
                val_label_vdw = "a×b (Å²)"
            else:
                val_col_vdw  = vdw_axes[0]
                val_label_vdw = vdw_axes[0]

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

                vdw_current_pt = None
                sx_v = st.session_state.vdw_sel.get(vdw_x)
                sy_v = st.session_state.vdw_sel.get(vdw_y)
                if sx_v is not None and sy_v is not None:
                    vdw_current_pt = {vdw_x: sx_v, vdw_y: sy_v}

                event_v = _render_heatmap(
                    vdw_fixed, vdw_fixed,
                    vdw_x, vdw_y, val_col_vdw, val_label_vdw,
                    current_pt=vdw_current_pt,
                    chart_key="vdw_heatmap",
                )

                if event_v and event_v.selection and event_v.selection.points:
                    pt = event_v.selection.points[0]
                    if pt.get("x") is not None:
                        st.session_state.vdw_sel[vdw_x] = float(pt["x"])
                        # selectbox はキーに紐づく値を index より優先するため、
                        # 新しい選択を反映させるにはウィジェット自身のキーも消す必要がある
                        st.session_state.pop(f"vdw_sel_{vdw_x}", None)
                    if pt.get("y") is not None:
                        st.session_state.vdw_sel[vdw_y] = float(pt["y"])
                        st.session_state.pop(f"vdw_sel_{vdw_y}", None)
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

            # ─── 精細スキャン設定 → run_config.yaml 生成 ────
            st.divider()
            st.subheader("精細スキャン設定")
            st.caption(
                "スキャンする変数の範囲と刻みを設定して run_config.yaml を生成します。"
                " スパコンで `python -m auto_opt.run --config run_config.yaml` を実行すると"
                " VdW → Amber → 結果収集 まで一気に処理されます。"
            )

            # 基本設定
            _hpc_workdir_v = st.session_state.get("setup_hpc_workdir", "").strip()
            _auto_dir_default = (
                f"{_hpc_workdir_v}/runs/{monomer_name}_{vdw_sym}"
                if _hpc_workdir_v else f"runs/{monomer_name}_{vdw_sym}"
            )
            ca, cb = st.columns(2)
            auto_dir_str = ca.text_input(
                "HPC 上の実行ディレクトリ (auto_dir、絶対パス推奨)",
                placeholder=_auto_dir_default,
                key="vdw_auto_dir",
            )
            n_nodes = cb.number_input(
                "並列ノード数", min_value=1, max_value=200, value=6, key="vdw_n_nodes"
            )

            # 各軸のデフォルト値を現在の VdW データから取得
            def _axis_range(col):
                if col not in vdw_df.columns:
                    return None, None
                return float(vdw_df[col].min()), float(vdw_df[col].max())

            # スキャンパラメータ入力
            _DEFAULT_STEP = {"alpha": 10.0, "phi": 4.0, "z": 0.5, "beta": 5.0}
            param_cfg: dict[str, dict] = {}

            axes_to_show = ["alpha", "phi", "z"] + (["beta"] if vdw_sym == "screw" else [])
            for axis in axes_to_show:
                lo, hi = _axis_range(axis)
                default_lo = lo if lo is not None else 0.0
                default_hi = hi if hi is not None else 90.0
                default_step = _DEFAULT_STEP.get(axis, 5.0)

                with st.expander(f"{axis}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    p_min  = c1.number_input("min",  value=default_lo, key=f"p_{axis}_min")
                    p_max  = c2.number_input("max",  value=default_hi, key=f"p_{axis}_max")
                    p_step = c3.number_input("step", value=default_step, min_value=0.001,
                                             key=f"p_{axis}_step")
                param_cfg[axis] = {"min": p_min, "max": p_max, "step": p_step}

            # glide のみ: vdw_select
            if vdw_sym == "glide":
                vdw_select = st.multiselect(
                    "抽出する構造タイプ (vdw_select)",
                    ["all", "a-stack", "b-stack"],
                    default=["a-stack", "b-stack"],
                    key="p_vdw_select",
                )
                if not vdw_select:
                    st.warning("少なくとも1つ選択してください。")
                    vdw_select = ["all"]
            else:
                vdw_select = ["all"]

            # 計算点数・予想時間
            n_vdw_pts = 1
            for axis, cfg in param_cfg.items():
                if isinstance(cfg, dict):
                    n_vdw_pts *= max(1, round((cfg["max"] - cfg["min"]) / cfg["step"]) + 1)
            n_dim  = _N_DIMERS[vdw_sym]
            # 各VdW初期点で9点探索 + 平均3回×4点の更新 = 21点、各点でn_dimダイマー計算
            _N_AMBER_PER_INIT = 9 + 3 * 4
            wall_s = n_vdw_pts * _N_AMBER_PER_INIT * n_dim * _SEC_PER_CALC / max(n_nodes, 1)
            wall_str = f"{wall_s:.0f} 秒" if wall_s < 60 else f"{wall_s / 60:.1f} 分"
            st.info(
                f"VdW 初期点数: **{n_vdw_pts}** 点 ｜ "
                f"Amber 予想時間: **{wall_str}**"
                f"（初期9点 + 平均5回×4点更新 = {_N_AMBER_PER_INIT}点/初期点 × {n_dim}ダイマー、{n_nodes}ノード並列）"
            )

            # run_config.yaml 生成
            auto_dir_out = auto_dir_str.strip() or _auto_dir_default
            data_dir_out = f"{_hpc_workdir_v}/data" if _hpc_workdir_v else None
            params_yaml = ""
            for axis in ["z", "alpha", "phi"] + (["beta"] if vdw_sym == "screw" else []):
                cfg = param_cfg[axis]
                params_yaml += (
                    f"  {axis}:\n"
                    f"    min: {cfg['min']}\n"
                    f"    max: {cfg['max']}\n"
                    f"    step: {cfg['step']}\n\n"
                )
            if len(vdw_select) == 1:
                vdw_select_yaml = vdw_select[0]
            else:
                vdw_select_yaml = "\n" + "".join(f"  - {v}\n" for v in vdw_select)

            run_config_yaml = (
                f"monomer: {monomer_name}\n"
                f"symmetry: {vdw_sym}\n"
                f"auto_dir: {auto_dir_out}\n"
                + (f"data_dir: {data_dir_out}\n" if data_dir_out else "")
                + f"\nparameters:\n{params_yaml}\n"
                f"vdw_select: {vdw_select_yaml}\n"
                f"amber:\n"
                f"  num_nodes: {int(n_nodes)}\n"
            )
            if not data_dir_out:
                st.caption(
                    "⚠️ data_dir が設定されていません（Tab 0 で HPC 作業ディレクトリを"
                    "入力すると自動で入ります）。無いとパッケージ同梱の data/ が使われ、"
                    "auto_opt_dir 規約に従いません。"
                )

            st.code(run_config_yaml, language="yaml")
            _lwd_vdw = Path(local_work_dir) if local_work_dir.strip() else None
            if _lwd_vdw:
                if st.button("run_config.yaml をローカルに保存", key="save_run_config_vdw"):
                    _save_file(run_config_yaml, _lwd_vdw / "run_config.yaml")
                _hpc_host_v = st.session_state.get("setup_hpc_host", "").strip() or "<user@hpc>"
                st.code(
                    f"scp {_lwd_vdw}/run_config.yaml {_hpc_host_v}:{auto_dir_out}/\n"
                    f"ssh {_hpc_host_v} "
                    f"'nohup python -m auto_opt.run --config {auto_dir_out}/run_config.yaml "
                    f"> {auto_dir_out}/run.log 2>&1 &'",
                    language="bash",
                )
            else:
                st.caption("💡 サイドバーでローカル作業ディレクトリを設定するとSCPコマンドが表示されます")


# ══════════════════════════════════════════════════════════
#  Tab 2: 層内最適化
# ══════════════════════════════════════════════════════════
with tab_layer:
    _layer_local_path = (
        Path(local_work_dir) / "filtered_step1.csv"
        if local_work_dir.strip() else None
    )
    _layer_local_exists = _layer_local_path is not None and _layer_local_path.exists()

    with st.expander("HPC から結果を取得（filtered_step1.csv）"):
        _lwd_layer = local_work_dir.strip() or "<ローカル作業ディレクトリ>"
        st.code(
            f"scp {_hpc_host_out}:{_auto_dir_out}/filtered_step1.csv \\\n"
            f"    {_lwd_layer}/",
            language="bash",
        )

    if _layer_local_exists:
        st.caption(f"📂 {_layer_local_path} から自動読み込み")
    layer_uploaded = st.file_uploader(
        "filtered_step1.csv (Amber 最適化結果) — ローカルパスが設定されていれば自動読み込み",
        type="csv", key="layer_csv"
    )
    layer_source = layer_uploaded or (_layer_local_path if _layer_local_exists else None)
    if layer_source is None:
        st.info("filtered_step1.csv をアップロードするか、サイドバーでローカル作業ディレクトリを設定してください。")
    else:
        df = pd.read_csv(layer_source)
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

        # structure_type フィルタ（a-stack/b-stack等が混在するマップの誤読を防ぐ）
        if "structure_type" in df.columns:
            all_types = sorted(df["structure_type"].dropna().unique())
            if len(all_types) > 1:
                sel_layer_type = st.radio(
                    "structure_type", all_types, horizontal=True, key="layer_types"
                )
                df = df[df["structure_type"] == sel_layer_type]

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
                            # selectbox はキーに紐づく値を index より優先するため、
                            # 新しい選択を反映させるにはウィジェット自身のキーも消す必要がある
                            st.session_state.pop(f"layer_sel_{x_col}", None)
                            st.session_state.pop(f"layer_sel_{y_col}", None)
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
        stack_axes = [c for c in ["z", "phi", "beta", "cy", "cz"] if c in df_sr2.columns]
        if not stack_axes:
            stack_axes = [df_sr2.columns[0]]
        x_col_s2 = st.selectbox("X 軸", stack_axes, key="stack_x")

        # X軸の全値に対して他の全変数でmin → 各X値で最安定の1点を取る
        e_cols_avail = [c for c in ["E_layer", "E_stack", "E_total", "E_int"] if c in df_sr2.columns]
        sort_col = next((c for c in ["E_total", "E_int"] if c in df_sr2.columns), e_cols_avail[0] if e_cols_avail else None)

        if sort_col:
            df_plot_base = (
                df_sr2.groupby(x_col_s2, as_index=False)
                .apply(lambda g: g.loc[g[sort_col].idxmin()])
                .reset_index(drop=True)
            )
        else:
            df_plot_base = df_sr2.groupby(x_col_s2, as_index=False).first()

        # ─── グラフ ───────────────────────────────────────────
        fig_s2 = go.Figure()
        for col, color, label in [
            ("E_layer", "royalblue", "層内 E_layer"),
            ("E_total", "seagreen",  "合計 E_total"),
        ]:
            if col not in df_plot_base.columns:
                continue
            y = df_plot_base[col]
            fig_s2.add_trace(go.Scatter(
                x=df_plot_base[x_col_s2], y=y - y.iloc[0],
                mode="lines+markers", name=label, line=dict(color=color),
            ))

        # E_inter = 2 * E_stack（上下の層間を両方カウント）
        e_inter_col = "E_stack" if "E_stack" in df_plot_base.columns else "E_int" if "E_int" in df_plot_base.columns else None
        if e_inter_col:
            y = df_plot_base[e_inter_col] * 2
            fig_s2.add_trace(go.Scatter(
                x=df_plot_base[x_col_s2], y=y - y.iloc[0],
                mode="lines+markers", name="層間 E_inter (×2)", line=dict(color="tomato"),
            ))

        fig_s2.update_layout(
            xaxis_title=x_col_s2,
            yaxis_title="ΔE (kcal/mol, 相対)",
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_s2, use_container_width=True)

        # ─── パラメータテーブル ───────────────────────────────
        st.subheader("各点のパラメータ")
        show_cols = [x_col_s2] + [
            c for c in ["alpha1", "alpha2", "beta", "phi", "z", "a", "bt1", "bt2", "b",
                        "cx", "cy", "cz", "E_layer", "E_stack", "E_total"]
            if c in df_plot_base.columns and c != x_col_s2
        ]
        st.dataframe(
            df_plot_base[show_cols].reset_index(drop=True),
            use_container_width=True,
        )

        # ─── 3D 表示（行クリックで選択）──────────────────────
        st.subheader("スタッキング 3D 表示")
        best_col_s = next((c for c in ["E_total", "E_stack", "E_int"] if c in df_plot_base.columns), None)
        if best_col_s:
            x_vals   = sorted(df_plot_base[x_col_s2].unique())
            sel_x    = st.select_slider(x_col_s2, options=x_vals,
                                        value=df_plot_base.loc[df_plot_base[best_col_s].idxmin(), x_col_s2],
                                        key="stack_3d_slider")
            sel_row  = df_plot_base[np.isclose(df_plot_base[x_col_s2], sel_x, atol=1e-5)].iloc[0]
            sym_stk  = "screw" if "beta" in df_sr2.columns else "glide"
            try:
                xyz_stk = make_stacking_xyz(sel_row, monomer_name, sym_stk, monomer_dir)
                view_stk = py3Dmol.view(width=600, height=400)
                view_stk.addModel(xyz_stk, "xyz")
                if mol_style == "Space fill":
                    view_stk.setStyle({}, {"sphere": {"scale": 1.0}})
                else:
                    view_stk.setStyle({}, {"stick": {"radius": 0.15}, "sphere": {"radius": 0.3}})
                view_stk.setProjection("orthographic")
                view_stk.zoomTo()
                st.components.v1.html(view_stk._make_html(), height=420)
                e_info = "  ".join(
                    f"{c}={sel_row[c]:.3f}" for c in ["E_layer", "E_stack", "E_total"]
                    if c in sel_row
                )
                st.caption(e_info)
            except Exception as e:
                st.error(f"3D 生成エラー: {e}")

        st.download_button(
            label="スタッキングサマリー CSV ダウンロード",
            data=df_plot_base[show_cols].reset_index(drop=True).to_csv(index=False),
            file_name=f"{monomer_name}_stacking_summary.csv",
            mime="text/csv",
            key="dl_stacking_results2",
        )

# ══════════════════════════════════════════════════════════
#  Tab 4: 変数説明
# ══════════════════════════════════════════════════════════
with tab_param:
    st.markdown("""
各変数が分子の配置にどう影響するかを確認できます。
サイドバーで **モノマー名** と **対称性** を設定してから使用してください。
""")

    # 対称性選択（サイドバーにない場合のフォールバック）
    param_sym = st.radio("対称性", ["glide", "screw"], horizontal=True, key="param_sym")

    # ─── 変数の説明テキスト ────────────────────────────────
    VAR_DESC = {
        "alpha": ("alpha (°)", "分子の z 軸まわりの回転角。結晶面内での分子の向き（ヘリングボーン角）を決める。"),
        "phi":   ("phi (°)",   "分子の長軸まわりの傾き角。分子が前後にどれだけ傾いているかを表す。"),
        "z":     ("z (Å)",     "b 軸方向の隣分子との高さオフセット。glide 対称では隣分子が 2z だけずれる。"),
        "beta":  ("beta (°)",  "screw 対称のみ。x 軸まわりの追加傾き。screw 軸による非対称なねじれを表す。"),
    }

    # ─── ① インタラクティブスライダー ──────────────────────
    st.subheader("① インタラクティブ確認")
    st.caption("スライダーを動かすと右の 3D ビューがリアルタイムで更新されます。")

    col_sliders, col_3d_param = st.columns([1, 1])

    with col_sliders:
        p_alpha = st.slider("alpha (°)",  0.0,  90.0, 25.0, 5.0,  key="p_alpha")
        st.caption(VAR_DESC["alpha"][1])
        p_phi   = st.slider("phi (°)",  -15.0,  15.0,  0.0, 1.0,  key="p_phi")
        st.caption(VAR_DESC["phi"][1])
        p_z     = st.slider("z (Å)",    -2.0,   2.0,   0.0, 0.1,  key="p_z")
        st.caption(VAR_DESC["z"][1])
        if param_sym == "screw":
            p_beta = st.slider("beta (°)", -20.0, 20.0,  0.0, 1.0, key="p_beta")
            st.caption(VAR_DESC["beta"][1])
        else:
            p_beta = 0.0

        st.markdown("---")
        st.caption("a / b は VdW 接触距離（固定値）")
        p_a = st.number_input("a (Å)", value=6.0, step=0.1, key="p_a")
        p_b = st.number_input("b (Å)", value=8.0, step=0.1, key="p_b")

    with col_3d_param:
        _row_param = pd.Series({
            "alpha": p_alpha, "phi": p_phi, "z": p_z,
            "a": p_a, "b": p_b,
            "bt1": p_b / 2, "bt2": p_b / 2,
            **({"beta": p_beta} if param_sym == "screw" else {}),
        })
        try:
            _xyz_param = make_cluster_xyz(_row_param, monomer_name, param_sym, monomer_dir)
            _view_param = py3Dmol.view(width=460, height=400)
            _view_param.addModel(_xyz_param, "xyz")
            if mol_style == "Space fill":
                _view_param.setStyle({"sphere": {"scale": 1.0}})
            else:
                _view_param.setStyle({"stick": {"radius": 0.15}, "sphere": {"radius": 0.3}})
            _view_param.setProjection("orthographic")
            _view_param.zoomTo()
            st.components.v1.html(_view_param._make_html(), height=420)
        except Exception as e:
            st.error(f"3D 表示エラー: {e}")

    # ─── ② アニメーション（各変数の変化を自動再生） ─────────
    st.subheader("② 各変数のアニメーション")
    st.caption("1変数だけを動かしたときの構造変化を自動再生します。")

    def _parse_xyz(xyz_str: str):
        """XYZ 文字列から (syms, coords) を返す。"""
        lines = xyz_str.rstrip('\n').split('\n')
        n = int(lines[0])
        syms, coords = [], []
        for line in lines[2:2 + n]:
            parts = line.split()
            syms.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return lines[0], lines[1], syms, np.array(coords)

    def _shift_xyz(n_line, comment, syms, coords, offset) -> str:
        """座標を offset だけ平行移動した XYZ 文字列を返す。"""
        shifted = coords - offset
        body = '\n'.join(
            f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}"
            for s, c in zip(syms, shifted)
        )
        return f"{n_line}\n{comment}\n{body}\n"

    def _make_anim(var: str, values, base: dict, sym: str,
                   width: int = 340, height: int = 300) -> None:
        """py3Dmol アニメーションを Streamlit に埋め込む。
        最初のフレームの重心を全フレームに共通オフセットとして適用することで
        フレーム間で分子が中心に固定されたまま変化を表示する。
        """
        if mol_style == "Space fill":
            _style = {"sphere": {"scale": 1.0}}
        else:
            _style = {"stick": {"radius": 0.15}, "sphere": {"radius": 0.25}}

        raw_frames = []
        row_tmp = pd.Series(base)
        for val in values:
            row_tmp = row_tmp.copy()
            row_tmp[var] = val
            try:
                raw_frames.append(make_cluster_xyz(row_tmp, monomer_name, sym, monomer_dir))
            except Exception:
                pass

        if not raw_frames:
            st.warning("フレームを生成できませんでした。モノマー名とディレクトリを確認してください。")
            return

        # 全フレームの重心を平均してオフセットとして使用
        # （対称スイープで各フレームのズレが打ち消し合い中心が安定する）
        parsed = [_parse_xyz(f) for f in raw_frames]
        offset = np.mean([coords.mean(axis=0) for _, _, _, coords in parsed], axis=0)

        frames = []
        for (n_line, comment, syms, coords) in parsed:
            frames.append(_shift_xyz(n_line, comment, syms, coords, offset))

        combined_xyz = "".join(frames)
        view = py3Dmol.view(width=width, height=height)
        view.addModelsAsFrames(combined_xyz, "xyz")
        view.animate({"loop": "forward", "interval": 150, "reps": 0})
        view.setStyle({}, _style)
        view.setProjection("orthographic")
        view.zoomTo()
        st.components.v1.html(view._make_html(), height=height + 20)

    _base = {
        "alpha": 25.0, "phi": 0.0, "z": 0.0,
        "a": p_a, "b": p_b,
        "bt1": p_b / 2, "bt2": p_b / 2,
        "beta": 0.0,
    }

    _anim_vars = [
        ("alpha", np.linspace(0,   90,  19)),
        ("phi",   np.linspace(-15, 15,  19)),
        ("z",     np.linspace(-1.5, 1.5, 19)),
    ]
    if param_sym == "screw":
        _anim_vars.append(("beta", np.linspace(-20, 20, 19)))

    _n_panels  = len(_anim_vars)
    _panel_w   = 240 if _n_panels >= 4 else 300
    _panel_h   = 240 if _n_panels >= 4 else 270

    _anim_cols = st.columns(_n_panels)
    for col_ui, (var, vals) in zip(_anim_cols, _anim_vars):
        with col_ui:
            label, desc = VAR_DESC[var]
            st.markdown(f"**{label}**")
            st.caption(desc)
            _make_anim(var, vals, _base, param_sym,
                       width=_panel_w, height=_panel_h)
