#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_integrated.py

機能統合版:
1. beta (分子面内回転) に対応
2. 差分実行管理 (dft_status: NotYet/InProgress/Done)
3. 構造自動分類 (a-stack, b-stack, ch)
4. フィルタリング (target-z, E-threshold)

Usage:
  # 抽出のみ (step1.csv -> filtered_step1.csv)
  python -m auto_opt.gaussian.pipeline_screw --auto-dir runs/ANT_test --monomer ANT --extract-only

  # 投入のみ (filtered_step1.csv の NotYet を投入)
  python -m auto_opt.gaussian.pipeline_screw --auto-dir runs/ANT_test --monomer ANT --submit-only

  # 閾値を設けて投入
  python -m auto_opt.gaussian.pipeline_screw --auto-dir runs/ANT_test --monomer ANT --submit-only --E-threshold -15.0
"""

from __future__ import annotations
import os, argparse, subprocess, time, itertools
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
STEP_BT1 = 0.1
STEP_BT2 = 0.1

# =========================================================
#                   幾何生成ユーティリティ (beta対応)
# =========================================================

def get_monomer_xyzR(monomer_name: str, Ta: float, Tb: float, Tc: float, A2: float, A3: float) -> np.ndarray:
    """
    A2: beta (x軸回転相当だが、座標系定義による。ここでは面内回転として扱う実装に依存)
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

    for i, (x, y, z, Rv) in enumerate(xyzR_array):
        frag = 1 if i < n_atom_each else 2
        atom = R2atom(Rv)
        lines.append(f'{atom}(Fragment={frag}) {float(x):.6f} {float(y):.6f} {float(z):.6f}\n')

    lines.append('\n')
    return lines

def get_file_name_from_dict(monomer_name: str, params_dict: Dict[str, float]) -> str:
    """パラメータ辞書から一意なファイル名を生成"""
    parts = [monomer_name]
    for key in ("alpha", "beta", "a", "b", "bt1", "bt2", "z"):
        if key not in params_dict: continue
        val = params_dict[key]
        if key in ("alpha", "beta"):
            val = int(round(val))
        elif key in ("a", "b", "bt1", "bt2", "z"):
            val = round(float(val), 1)
        parts.append(f"{key}={val}")
    return "_".join(parts) + ".inp"

def build_dimers(monomer_name: str, alpha: float, beta: float, a: float, b: float, bt1: float, bt2: float, z: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    4ダイマーを生成:
      a1: (0,0,0) - (a, 0, 0)
      b1: (0,0,0) - (0, b, 0)
      t1: (0,0,0) - (a/2, bt1, z)
      t3: (0,0,0) - (a/2, -bt2, z)
    """
    A2 = beta
    A3 = alpha
    mon0   = get_monomer_xyzR(monomer_name, 0,   0,    0, A2,   A3)
    mon_a1 = get_monomer_xyzR(monomer_name, a,   0,    0, A2,   A3)
    mon_b1 = get_monomer_xyzR(monomer_name, 0,   b,    0, A2,   A3)
    mon_t1 = mon0.copy(); mon_t1[:, 0] = -mon_t1[:, 0] + a/2; mon_t1[:, 1] += bt1; mon_t1[:, 2] += z
    mon_t3 = mon0.copy(); mon_t3[:, 0] = -mon_t3[:, 0] + a/2; mon_t3[:, 1] -= bt2; mon_t3[:, 2] += z

    dimer_a1 = np.concatenate([mon0, mon_a1], axis=0)
    dimer_b1 = np.concatenate([mon0, mon_b1], axis=0)
    dimer_t1 = np.concatenate([mon0, mon_t1], axis=0)
    dimer_t3 = np.concatenate([mon0, mon_t3], axis=0)
    return dimer_a1, dimer_b1, dimer_t1, dimer_t3

def make_gjf_xyz(auto_dir: str, monomer_name: str, params_dict: Dict[str,float], machine_type: int) -> str:
    alpha = float(params_dict.get('alpha', 0.0))
    beta   = float(params_dict.get('beta', 0.0))
    a     = float(params_dict.get('a', 0.0))
    b     = float(params_dict.get('b', 0.0))
    bt1     = float(params_dict.get('bt1', 0.0))
    bt2     = float(params_dict.get('bt2', 0.0))
    z     = float(params_dict.get('z', 0.0))

    d_a1, d_b1, d_t1, d_t3 = build_dimers(monomer_name, alpha, beta, a, b, bt1, bt2, z)

    desc = f'{monomer_name}_alpha={alpha}_beta={beta}_a={a}_b={b}_bt1={bt1}_bt2={bt2}_z={z}'
    sec1 = get_xyzR_lines(d_a1, desc + " [a1]", machine_type)
    sec2 = get_xyzR_lines(d_b1, desc + " [b1]", machine_type)
    sec3 = get_xyzR_lines(d_t1, desc + " [t1]", machine_type)
    sec4 = get_xyzR_lines(d_t3, desc + " [t3]", machine_type)

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

def _neighbors(a: float, bt1: float, bt2: float) -> List[Tuple[float,float,float]]:
    """
    itertools.product を使って (a, bt1, bt2) の3D空間における
    周囲26点（自身を除く27点）のオフセットを自動生成します。
    """
    steps_a = [-STEP_A, 0.0, STEP_A]
    steps_bt1 = [-STEP_BT1, 0.0, STEP_BT1]
    steps_bt2 = [-STEP_BT2, 0.0, STEP_BT2]
    
    neighbors = []
    # 3変数の直積 (3 x 3 x 3 = 27通り)
    for da, dbt1, dbt2 in itertools.product(steps_a, steps_bt1, steps_bt2):
        if da == 0.0 and dbt1 == 0.0 and dbt2 == 0.0:
            continue  # 自身(0, 0, 0) は除外
        
        # 浮動小数点の誤差を防ぐために round(..., 1) をかける
        neighbors.append((round(a + da, 1), round(bt1 + dbt1, 1), round(bt2 + dbt2, 1)))
        
    return neighbors

def _is_local_min(a: float, bt1: float, bt2: float, e: float, grid: Dict[Tuple[float,float,float], float]) -> bool:
    """3D空間の近傍点と比較して極小値かどうかを判定"""
    nb_coords = _neighbors(a, bt1, bt2)
    # gridに存在する近傍点のエネルギーを取得
    nb_energies = [grid[coords] for coords in nb_coords if coords in grid]
    
    if not nb_energies: 
        return False
    
    # すべての近傍点以下であり、かつ少なくとも1つの近傍点より真に小さい
    return (all(e <= v for v in nb_energies)) and any(e < v for v in nb_energies)

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
    (alpha, beta, z) ごとに格子定数 a の大小関係に基づいて structure_type を分類する
    """
    df = df.copy()
    if 'structure_type' not in df.columns:
        df['structure_type'] = 'unknown'

    if 'beta' not in df.columns: df['beta'] = 0.0
    if 'z' not in df.columns: return df

    CLUSTER_THR = 0.5

    for (alpha, beta, z), group in df.groupby(['alpha', 'beta', 'z']):
        a_vals = sorted(group['a'].unique())
        clusters = []

        if not a_vals:
            continue
            
        curr = [a_vals[0]]
        for v in a_vals[1:]:
            if v - curr[-1] > CLUSTER_THR:
                clusters.append(curr)
                curr = [v]
            else:
                curr.append(v)
        clusters.append(curr)
        
        n = len(clusters)
        
        labels = {}
        if n >= 3:
            labels[0] = 'a-stack'
            labels[n-1] = 'b-stack'
            for i in range(1, n-1):
                labels[i] = 'ch'
        elif n == 2:
            labels[0] = 'a-stack'
            labels[1] = 'b-stack'
        elif n == 1:
            labels[0] = 'a-stack'

        val_to_label = {}
        for i, clust in enumerate(clusters): 
            lab = labels.get(i, 'unknown') 
            for v in clust:
                val_to_label[v] = lab
        
        for idx, val in group['a'].items():
            df.at[idx, 'structure_type'] = val_to_label.get(val, 'unknown')

    return df

def extract_from_step1(step1_csv: str, out_csv: str, auto_dir: str | None = None) -> pd.DataFrame:
    """
    step1.csv から局所最小を抽出し、既存データとマージした後に分類して保存
    """
    df = _read_step1_auto(step1_csv, auto_dir)

    # 必要な列の確認 (bt1, bt2 を追加)
    need_base = {'alpha','a','b','bt1','bt2','z','E','status'}
    missing = [c for c in need_base if c not in df.columns]
    if missing: raise ValueError(f"step1.csv 欠落列: {missing}")

    if 'beta' not in df.columns:
        df['beta'] = 0.0
    
    df = df[df['status'].astype(str).str.lower() == 'done'].copy()
    if df.empty:
        print("[extract] Done行が見つかりません。スキップします。")
        return pd.DataFrame()

    energy_cols = [c for c in ['E','E1','E2','E3','E4'] if c in df.columns]
    for c in ['a','b','bt1','bt2','z','alpha','beta'] + energy_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['a','b','bt1','bt2','z','E'])

    # (alpha, beta, a, bt1, bt2, z) 重複排除
    df = df.sort_values('E').drop_duplicates(subset=['alpha','beta','a','bt1','bt2','z'], keep='first')

    rows = []
    # alpha, beta, z を固定して、(a, bt1, bt2) 3D空間での最小を探す
    for (alpha, beta, z), g in df.groupby(['alpha','beta','z']):
        # gridのキーを3次元に変更
        grid = {(round(r.a,1), round(r.bt1,1), round(r.bt2,1)): float(r.E) for r in g.itertuples(index=False)}
        
        for r in g.itertuples(index=False):
            a, bt1, bt2, b, e = float(r.a), float(r.bt1), float(r.bt2), float(r.b), float(r.E)
            
            # 3D空間での極小判定
            if _is_local_min(a, bt1, bt2, e, grid):
                rec = {
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'a': round(a,1),
                    'b': round(b,1),
                    'bt1': round(bt1,1),
                    'bt2': round(bt2,1),
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
        if 'beta' not in existing_df.columns:
            existing_df['beta'] = 0.0
        
        cols_to_keep = ['alpha','beta','a','b','bt1','bt2','z']
        info_cols = ['dft_status','log','inp','machine']
        info_cols = [c for c in info_cols if c in existing_df.columns]
        
        merged_df = pd.merge(df_extracted, existing_df[cols_to_keep + info_cols], 
                             on=cols_to_keep, how='left')
        
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
    out_cols = ['alpha','beta','a','b','bt1','bt2','z','structure_type', 'dft_status'] + energy_cols
    extra_cols = [c for c in merged_df.columns if c not in out_cols]
    merged_df = merged_df[out_cols + extra_cols]
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_path, index=False)
    print(f"[extract] updated -> {out_csv}")
    return merged_df

# =========================================================
#                   投入（qsub）
# =========================================================
# ※今回は実行しないため省略せずそのまま残しています。

def submit_from_candidates(auto_dir: str, monomer: str, cand_csv: str,
                           submit: bool=True, throttle: bool=True, 
                           target_z_list: Optional[List[float]]=None,
                           E_threshold: float=None) -> pd.DataFrame:
    """
    filtered_step1.csv を読み、条件に合う NotYet のものだけ投入。
    """
    df = pd.read_csv(cand_csv)
    
    if 'beta' not in df.columns: df['beta'] = 0.0
    if 'dft_status' not in df.columns:
        print("[submit] dft_status列がありません。すべて NotYet とみなします。")
        df['dft_status'] = 'NotYet'

    cond = (df['dft_status'] == 'NotYet')
    
    if target_z_list is not None:
        is_target = df['z'].apply(lambda z_val: any(abs(z_val - t) < 1e-5 for t in target_z_list))
        cond = cond & is_target
    
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
        beta   = float(r.beta)
        a     = float(r.a)
        b     = float(r.b)
        bt1   = float(r.bt1)
        bt2   = float(r.bt2)
        z     = float(r.z)
        
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

        params = {"alpha":alpha, "beta":beta, "a":a, "b":b, "bt1":bt1, "bt2":bt2, "z":z}
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

    ap = argparse.ArgumentParser(description="統合版パイプライン: 局所最小抽出 & 差分DFT投入 (beta対応)")
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