#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_integrated.py

機能統合版:
1. phi (分子面内回転) に対応
2. 差分実行管理 (dft_status: NotYet/InProgress/Done)
3. 構造自動分類 (a-stack, b-stack, ch)
4. フィルタリング (target-z, E-threshold)

Usage:
  # 抽出のみ (step1.csv -> filtered_step1.csv)
  python -m auto_opt.gaussian.pipeline_v3 --auto-dir runs/ANT_test --monomer ANT --extract-only

  # 投入のみ (filtered_step1.csv の NotYet を投入)
  python -m auto_opt.gaussian.pipeline_v3 --auto-dir runs/ANT_test --monomer ANT --submit-only

  # 閾値を設けて投入
  python -m auto_opt.gaussian.pipeline_v3 --auto-dir runs/ANT_test --monomer ANT --submit-only --E-threshold -15.0
"""

from __future__ import annotations
import os, argparse, subprocess, time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from auto_opt.utils import Rod, R2atom

# =========================================================
#                   設定定数
# =========================================================

# デフォルトのモノマーディレクトリ
MONOMER_DIR = os.path.expanduser("~/Working/auto_opt/data/monomer")

# Gaussian 実行設定
MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}
MAX_PARALLEL = {1: 3, 2: 3}  # 各 machine の並列スロット

# 近傍判定の刻み (step1.csv の刻みに合わせる)
STEP_A = 0.1
STEP_B = 0.1

# =========================================================
#                   幾何生成ユーティリティ (phi対応)
# =========================================================

def get_monomer_xyzR(monomer_name: str, Ta: float, Tb: float, Tc: float, A2: float, A3: float) -> np.ndarray:
    """
    A2: phi (x軸回転相当だが、座標系定義による。ここでは面内回転として扱う実装に依存)
    A3: alpha (z軸回転)
    """
    path = os.path.join(MONOMER_DIR, f"{monomer_name}.csv")
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"monomer CSV not found: {path}")
    df_mono = pd.read_csv(path)
    atoms_array_xyzR = df_mono[['X','Y','Z','R']].to_numpy(dtype=float)

    ex = np.array([1.,0.,0.])
    ez = np.array([0.,0.,1.])
    xyz = atoms_array_xyzR[:, :3]
    
    # 座標変換: 定義は論文/実装に合わせる
    # phi回転 (Rod(-ex, A2) は pipeline_phi.py の実装を踏襲)
    if abs(A2) > 1e-5:
        xyz = xyz @ Rod(-ex, A2).T
    
    # alpha回転
    xyz = xyz @ Rod(ez, A3).T
    
    # 平行移動
    xyz = xyz + np.array([Ta, Tb, Tc])
    R = atoms_array_xyzR[:, 3].reshape((-1,1))
    return np.concatenate([xyz, R], axis=1)

def get_xyzR_lines(xyzR_array: np.ndarray, file_description: str, machine_type: int) -> List[str]:
    mp_num = MACHINE_SPEC[machine_type]["nproc"]
    header = [
        f'%mem=15GB\n',
        f'%nproc={mp_num}\n',
        '#P B3LYP/6-311G** EmpiricalDispersion=GD3 Counterpoise=2\n',
        '\n',
        file_description + '\n',
        '\n',
        '0 1 0 1 0 1\n',
    ]
    lines = list(header)

    n_atom_each = len(xyzR_array) // 2
    if n_atom_each * 2 != len(xyzR_array):
        raise ValueError("xyzR_array の長さが 2 の倍数ではない（ダイマー前提）")

    # 前半 Fragment=1, 後半 Fragment=2
    for i, (x, y, z, Rv) in enumerate(xyzR_array):
        frag = 1 if i < n_atom_each else 2
        atom = R2atom(Rv)
        lines.append(f'{atom}(Fragment={frag}) {float(x):.6f} {float(y):.6f} {float(z):.6f}\n')

    lines.append('\n')
    return lines

def get_file_name_from_dict(monomer_name: str, params_dict: Dict[str, float]) -> str:
    """パラメータ辞書から一意なファイル名を生成"""
    parts = [monomer_name]
    # キーの順序を固定
    for key in ("alpha", "phi", "a", "b", "z"):
        if key not in params_dict: continue
        val = params_dict[key]
        if key in ("alpha", "phi"):
            # 角度は整数丸めが見やすいが、精度が必要なら変更
            val = int(round(val))
        elif key in ("a", "b", "z"):
            val = round(float(val), 1)
        parts.append(f"{key}={val}")
    return "_".join(parts) + ".inp"

def build_dimers(monomer_name: str, alpha: float, phi: float, a: float, b: float, z: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    3ダイマーを生成:
      a1: (0,0,0) - (a, 0, 0)
      b1: (0,0,0) - (0, b, 2z)
      t1: (0,0,0) - (a/2, b/2, z) (Herringbone)
    """
    A2 = phi
    A3 = alpha
    # pipeline_phi.py のロジックを踏襲: t1ペアのみ -alpha (-A3) にする
    mon0   = get_monomer_xyzR(monomer_name, 0,   0,   0, A2,   A3)
    mon_a1 = get_monomer_xyzR(monomer_name, a,   0,   0, A2,   A3)
    mon_b1 = get_monomer_xyzR(monomer_name, 0,   b, 2*z, A2,   A3)
    mon_t1 = get_monomer_xyzR(monomer_name, a/2, b/2, z, A2,  -A3)
    mon_t2 = get_monomer_xyzR(monomer_name, -a/2, b/2, z, A2,  -A3)

    dimer_a1 = np.concatenate([mon0, mon_a1], axis=0)
    dimer_b1 = np.concatenate([mon0, mon_b1], axis=0)
    dimer_t1 = np.concatenate([mon0, mon_t1], axis=0)
    dimer_t2 = np.concatenate([mon0, mon_t2], axis=0)
    return dimer_a1, dimer_b1, dimer_t1, dimer_t2

def make_gjf_xyz(auto_dir: str, monomer_name: str, params_dict: Dict[str,float], machine_type: int) -> str:
    """Link1 で結合した gjf を作成"""
    alpha = float(params_dict.get('alpha', 0.0))
    phi   = float(params_dict.get('phi', 0.0))
    a     = float(params_dict.get('a', 0.0))
    b     = float(params_dict.get('b', 0.0))
    z     = float(params_dict.get('z', 0.0))

    d_a1, d_b1, d_t1, d_t2 = build_dimers(monomer_name, alpha, phi, a, b, z)

    desc = f'{monomer_name}_alpha={alpha}_phi={phi}_a={a}_b={b}_z={z}'
    sec1 = get_xyzR_lines(d_a1, desc + " [a1]", machine_type)
    sec2 = get_xyzR_lines(d_b1, desc + " [b1]", machine_type)
    sec3 = get_xyzR_lines(d_t1, desc + " [t1]", machine_type)
    sec4 = get_xyzR_lines(d_t2, desc + " [t2]", machine_type)

    gjf_lines = ['$ RunGauss\n'] + sec1 + ['--Link1--\n'] + sec2 + ['--Link1--\n'] + sec3 + ['--Link1--\n'] + sec4 + ['\n']

    out_dir = Path(auto_dir) / 'gaussian'
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = get_file_name_from_dict(monomer_name, params_dict)
    gjf_path = out_dir / file_name
    gjf_path.write_text("".join(gjf_lines), encoding="utf-8")
    return file_name

def get_one_exe(file_name: str, machine_type: int) -> List[str]:
    spec = MACHINE_SPEC[machine_type]
    file_basename = os.path.splitext(file_name)[0]
    return [
        '#!/bin/sh\n',
        '#$ -S /bin/sh\n',
        '#$ -cwd\n',
        '#$ -V\n',
        f'#$ -q {spec["queue"]}\n',
        f'#$ -pe OpenMP {spec["nproc"]}\n',
        f'#$ -N {file_basename}\n',
        '\n',
        'hostname\n',
        'export g16root=/home/g03\n',
        'source $g16root/g16/bsd/g16.profile\n',
        '\n',
        'export GAUSS_SCRDIR=/scr/$JOB_ID\n',
        'mkdir -p /scr/$JOB_ID\n',
        '\n',
        f'g16 < {file_basename}.inp > {file_basename}.log\n',
        '\n',
        'rm -rf /scr/$JOB_ID\n',
        '\n',
    ]

def exec_gjf(auto_dir: str, monomer_name: str, params_dict: Dict[str,float], machine_type: int,
             isTest: bool=True) -> str:
    inp_dir = Path(auto_dir) / 'gaussian'
    inp_dir.mkdir(parents=True, exist_ok=True)

    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict, machine_type)
    cc_list = get_one_exe(file_name, machine_type)

    sh_filename = Path(file_name).with_suffix('.r1').name
    sh_path = inp_dir / sh_filename
    sh_path.write_text("".join(cc_list), encoding="utf-8")

    if not isTest:
        subprocess.run(['qsub', sh_path.name], check=False, cwd=str(inp_dir))

    log_file_name = Path(file_name).with_suffix('.log').name
    return log_file_name

# =========================================================
#                   抽出（局所最小） & 分類
# =========================================================

def _neighbors(a: float, b: float) -> List[Tuple[float,float]]:
    da, db = STEP_A, STEP_B
    offs = [( da,0),(-da,0),(0,db),(0,-db),(da,db),(da,-db),(-da,db),(-da,-db)]
    return [(round(a+x,3), round(b+y,3)) for x,y in offs]

def _is_local_min(a: float, b: float, e: float, grid: Dict[Tuple[float,float], float]) -> bool:
    nb = [grid.get((round(ax,3), round(bx,3))) for (ax,bx) in _neighbors(a,b)]
    nb = [v for v in nb if v is not None]
    if not nb: return False
    return (all(e <= v for v in nb)) and any(e < v for v in nb)

def _read_step1_auto(step1_csv: str, auto_dir: str | Path | None) -> pd.DataFrame:
    p = Path(step1_csv)
    if p.is_file(): return pd.read_csv(p)
    if auto_dir is None: raise FileNotFoundError(f"step1.csv not found: {p}")
    root = Path(auto_dir)
    dfs = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir(): continue
        p2 = sub / "step1.csv"
        if p2.is_file(): dfs.append(pd.read_csv(p2))
    if not dfs: raise FileNotFoundError(f"step1.csv not found under {root}")
    return pd.concat(dfs, ignore_index=True)

def _classify_structure_by_a_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    (alpha, phi, z) ごとに格子定数 a の大小関係に基づいて structure_type を分類する
    
    ルール:
    1. 同じ (alpha, phi, z) を持つグループごとに処理
    2. その中で a の値をソートし、近い値は同じ構造とみなしてクラスタリング
    3. クラスタの個数と順位に応じてラベリング
       - 3つ: 小(a-stack) < 中(ch) < 大(b-stack)
       - 2つ: 小(a-stack) < 大(b-stack)
       - 1つ: a-stack
    """
    df = df.copy()
    if 'structure_type' not in df.columns:
        df['structure_type'] = 'unknown'

    # 列がない場合のガード
    if 'phi' not in df.columns: df['phi'] = 0.0
    if 'z' not in df.columns: return df # zがない場合は処理不能

    CLUSTER_THR = 0.5  # aの差がこれ以内なら同じ構造とみなす

    # 修正: alpha, phi に加えて 'z' もグルーピングキーに追加
    for (alpha, phi, z), group in df.groupby(['alpha', 'phi', 'z']):
        
        # 1. aの値でクラスタリング (昇順ソート)
        a_vals = sorted(group['a'].unique())
        clusters = []

        if not a_vals:
            continue
            
        # 近接するaをグループ化 (例: 5.9 と 6.0 は同じ構造とみなす)
        curr = [a_vals[0]]
        for v in a_vals[1:]:
            if v - curr[-1] > CLUSTER_THR:
                clusters.append(curr)
                curr = [v]
            else:
                curr.append(v)
        clusters.append(curr)
        
        n = len(clusters)
        
        # 2. クラスタ順位(aの小ささ)に基づくラベル決定
        labels = {} # cluster_index -> label
        if n >= 3:
            # 3つ以上ある場合: 最小=a-stack, 最大=b-stack, その他=ch
            labels[0] = 'a-stack'
            labels[n-1] = 'b-stack'
            for i in range(1, n-1):
                labels[i] = 'ch'
        elif n == 2:
            # 2つの場合: 小さい方=a-stack, 大きい方=b-stack
            labels[0] = 'a-stack'
            labels[1] = 'b-stack'
        elif n == 1:
            # 1つの場合: 基本的に最も密なパッキング(a-stack)とみなす
            labels[0] = 'a-stack'

        # 3. マッピング辞書の作成 (a_value -> label)
        val_to_label = {}
        for i, clust in enumerate(clusters): 
            lab = labels.get(i, 'unknown') 
            for v in clust:
                val_to_label[v] = lab
        
        # 4. データフレームへの適用
        for idx, val in group['a'].items():
            # その行の a値 に対応するラベルを適用
            df.at[idx, 'structure_type'] = val_to_label.get(val, 'unknown')

    return df

def extract_from_step1(step1_csv: str, out_csv: str, auto_dir: str | None = None) -> pd.DataFrame:
    """
    step1.csv から局所最小を抽出し、既存データとマージした後に分類して保存
    """
    df = _read_step1_auto(step1_csv, auto_dir)

    # 必要な列の確認
    need_base = {'alpha','a','b','z','E','status'}
    missing = [c for c in need_base if c not in df.columns]
    if missing: raise ValueError(f"step1.csv 欠落列: {missing}")

    # phi対応
    if 'phi' not in df.columns:
        df['phi'] = 0.0
    
    # Done 行のみ対象
    df = df[df['status'].astype(str).str.lower() == 'done'].copy()
    if df.empty:
        print("[extract] Done行が見つかりません。スキップします。")
        return pd.DataFrame()

    energy_cols = [c for c in ['E','E1','E2','E3'] if c in df.columns]
    for c in ['a','b','z','alpha','phi'] + energy_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['a','b','z','E'])

    # (alpha, phi, a, b, z) 重複排除: 最もエネルギーが低いものを残す
    df = df.sort_values('E').drop_duplicates(subset=['alpha','phi','a','b','z'], keep='first')

    # 局所最小の抽出 (a, b 空間)
    rows = []
    # alpha, phi, z を固定して、a,b 平面での最小を探す
    for (alpha, phi, z), g in df.groupby(['alpha','phi','z']):
        grid = {(round(r.a,3), round(r.b,3)): float(r.E) for r in g.itertuples(index=False)}
        for r in g.itertuples(index=False):
            a, b, e = float(r.a), float(r.b), float(r.E)
            if _is_local_min(a, b, e, grid):
                rec = {
                    'alpha': float(alpha),
                    'phi': float(phi),
                    'a': round(a,1),
                    'b': round(b,1),
                    'z': float(z),
                }
                for c in energy_cols:
                    rec[c] = float(getattr(r, c))
                rows.append(rec)

    if not rows:
        print("[extract] 局所最小点が見つかりませんでした。")
        return pd.DataFrame()

    df_extracted = pd.DataFrame(rows)

    # --- 既存ファイルとのマージ & Status管理 ---
    out_path = Path(out_csv)
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        if 'phi' not in existing_df.columns:
            existing_df['phi'] = 0.0
        
        # dft_status, log, inp などの管理情報は既存から引き継ぐ
        cols_to_keep = ['alpha','phi','a','b','z']
        info_cols = ['dft_status','log','inp','machine']
        info_cols = [c for c in info_cols if c in existing_df.columns]
        
        # 左外部結合: 今回の抽出結果をマスターとする
        merged_df = pd.merge(df_extracted, existing_df[cols_to_keep + info_cols], 
                             on=['alpha','phi','a','b','z'], how='left')
        
        if 'dft_status' not in merged_df.columns:
            merged_df['dft_status'] = float('nan')

        merged_df['dft_status'] = merged_df['dft_status'].fillna('NotYet')
        print(f"[extract] Extracted {len(df_extracted)} points. (Existing info merged)")
    else:
        merged_df = df_extracted
        merged_df['dft_status'] = 'NotYet'
        print(f"[extract] Created new dataset with {len(merged_df)} records")

    # --- Structure Type の再分類 ---
    merged_df = _classify_structure_by_a_rank(merged_df)

    # 列の整理
    out_cols = ['alpha','phi','a','b','z','structure_type', 'dft_status'] + energy_cols
    extra_cols = [c for c in merged_df.columns if c not in out_cols]
    merged_df = merged_df[out_cols + extra_cols]
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_path, index=False)
    print(f"[extract] updated -> {out_csv}")
    return merged_df

# =========================================================
#                   投入（qsub）
# =========================================================

def submit_from_candidates(auto_dir: str, monomer: str, cand_csv: str,
                           submit: bool=True, throttle: bool=True, 
                           target_z_list: Optional[List[float]]=None,
                           E_threshold: float=None) -> pd.DataFrame:
    """
    filtered_step1.csv を読み、条件に合う NotYet のものだけ投入。
    """
    df = pd.read_csv(cand_csv)
    
    # 互換性補完
    if 'phi' not in df.columns: df['phi'] = 0.0
    if 'dft_status' not in df.columns:
        print("[submit] dft_status列がありません。すべて NotYet とみなします。")
        df['dft_status'] = 'NotYet'

    # フィルタ条件作成
    cond = (df['dft_status'] == 'NotYet')
    
    # target_z フィルタ
    if target_z_list is not None:
        is_target = df['z'].apply(lambda z_val: any(abs(z_val - t) < 1e-5 for t in target_z_list))
        cond = cond & is_target
    
    # E_threshold フィルタ
    if E_threshold is not None:
        is_target_E = (df['E'] <= E_threshold)
        cond = cond & is_target_E

    target_indices = df[cond].index
    if len(target_indices) == 0:
        print("[submit] 新規投入対象 (NotYet) はありません。")
        return df

    print(f"[submit] {len(target_indices)} 件のジョブを処理します...")
    
    gdir = Path(auto_dir) / "gaussian"
    gdir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in target_indices:
        r = df.loc[idx]
        alpha = float(r.alpha)
        phi   = float(r.phi)
        a     = float(r.a)
        b     = float(r.b)
        z     = float(r.z)
        
        # 簡易ロードバランシング
        machine = 1 if (count % 2 == 0) else 2
        
        if submit and throttle:
            while True:
                try:
                    out = subprocess.check_output(["qstat", "-u", os.environ.get("USER","")], text=True)
                    running = sum(MACHINE_SPEC[machine]["queue"] in line for line in out.splitlines())
                    if running < MAX_PARALLEL[machine]:
                        break
                    time.sleep(2.0)
                except Exception:
                    break 

        params = {"alpha":alpha, "phi":phi, "a":a, "b":b, "z":z}
        log = exec_gjf(auto_dir, monomer, params,
                       machine_type=machine, isTest=not submit)
        
        df.at[idx, 'dft_status'] = 'InProgress' if submit else 'Written'
        df.at[idx, 'log'] = str(Path("gaussian") / log)
        df.at[idx, 'machine'] = machine
        count += 1

    df.to_csv(cand_csv, index=False)
    print(f"[submit] {count} 件投入し、CSVを更新しました -> {cand_csv}")
    return df

# =========================================================
#                       CLI
# =========================================================

def main():
    global MONOMER_DIR

    ap = argparse.ArgumentParser(description="統合版パイプライン: 局所最小抽出 & 差分DFT投入 (phi対応)")
    ap.add_argument("--auto-dir", required=True, help="作業ディレクトリ")
    ap.add_argument("--monomer", required=True, help="例: PFA")
    ap.add_argument("--step1-csv", default=None)
    ap.add_argument("--out-csv",   default=None, help="既定: <auto-dir>/filtered_step1.csv")
    ap.add_argument("--monomer-dir", default=MONOMER_DIR)
    
    # モード指定
    ap.add_argument("--extract-only", action="store_true", help="抽出・マージのみ (投入しない)")
    ap.add_argument("--submit-only",  action="store_true", help="NotYetの投入のみ (抽出しない)")
    
    # 実行オプション
    ap.add_argument("--no-throttle",  action="store_true", help="qstat待機を無効化")
    ap.add_argument("--target-z", type=float, nargs='+', default=None, help="特定のzのみ実行 (例: 0.0 1.0)")
    ap.add_argument("--E-threshold", type=float, default=None, help="Eの閾値以下の候補のみ投入")
    
    args = ap.parse_args()

    MONOMER_DIR = os.path.expanduser(args.monomer_dir)
    auto_dir = args.auto_dir
    step1_csv = args.step1_csv or str(Path(auto_dir) / "step1.csv")
    out_csv   = args.out_csv   or str(Path(auto_dir) / "filtered_step1.csv")

    if args.extract_only and args.submit_only:
        raise SystemExit("Error: extract-only と submit-only は同時指定できません")

    # 1. 抽出フェーズ
    if not args.submit_only:
        extract_from_step1(step1_csv, out_csv, auto_dir=auto_dir)

    # 2. 投入フェーズ
    if not args.extract_only:
        submit_from_candidates(
            auto_dir, args.monomer, out_csv, 
            submit=True, 
            throttle=not args.no_throttle, 
            target_z_list=args.target_z, 
            E_threshold=args.E_threshold  
        )

if __name__ == "__main__":
    main()