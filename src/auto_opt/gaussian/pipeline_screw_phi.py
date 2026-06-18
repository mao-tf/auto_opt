#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_screw_phi.py

pipeline_screw.py の phi 対応版。
回転順: phi(-x軸) → alpha(z軸) → beta(-x軸)

Usage:
  # 抽出のみ (step1.csv -> filtered_step1.csv)
  python -m auto_opt.gaussian.pipeline_screw_phi --auto-dir runs/DNTT_phi --monomer DNTT --extract-only

  # 投入のみ (filtered_step1.csv の NotYet を投入)
  python -m auto_opt.gaussian.pipeline_screw_phi --auto-dir runs/DNTT_phi --monomer DNTT --submit-only

  # 閾値を設けて投入
  python -m auto_opt.gaussian.pipeline_screw_phi --auto-dir runs/DNTT_phi --monomer DNTT --submit-only --E-threshold -15.0
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

MONOMER_DIR = os.path.expanduser("~/Working/auto_opt/data/monomer")

MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}
MAX_PARALLEL = {1: 3, 2: 3}

STEP_A   = 0.1
STEP_BT1 = 0.1
STEP_BT2 = 0.1

# =========================================================
#                   幾何生成ユーティリティ
# =========================================================

def get_monomer_xyzR(monomer_name: str, Ta: float, Tb: float, Tc: float,
                     A2: float, A3: float, phi: float = 0.0) -> np.ndarray:
    """
    回転順: phi(-x軸) → alpha(z軸) → beta(-x軸) → 平行移動
    make_io_gene_screw_phi.py の get_monomer_xyzR と同じ定義。
    """
    path = os.path.expanduser(os.path.join(MONOMER_DIR, f"{monomer_name}.csv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"monomer CSV not found: {path}")
    df_mono = pd.read_csv(path)
    atoms_array_xyzR = df_mono[['X', 'Y', 'Z', 'R']].to_numpy(dtype=float)

    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])
    xyz = atoms_array_xyzR[:, :3]
    xyz = xyz @ Rod(-ex, phi).T
    xyz = xyz @ Rod(ez,  A3 ).T
    xyz = xyz @ Rod(-ex, A2 ).T
    xyz = xyz + np.array([Ta, Tb, Tc])
    R = atoms_array_xyzR[:, 3].reshape((-1, 1))
    return np.concatenate([xyz, R], axis=1)


def get_xyzR_lines(xyzR_array: np.ndarray, file_description: str, machine_type: int) -> List[str]:
    mp_num = MACHINE_SPEC[machine_type]["nproc"]
    header = [
        f'%mem=15GB\n',
        f'%nproc={mp_num}\n',
        '#P PBEPBE/6-311G** EmpiricalDispersion=GD3BJ Counterpoise=2\n',
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
    parts = [monomer_name]
    for key in ("alpha", "beta", "phi", "a", "b", "bt1", "bt2", "z"):
        if key not in params_dict:
            continue
        val = params_dict[key]
        if key in ("alpha", "beta", "phi"):
            val = int(round(val))
        else:
            val = round(float(val), 1)
        parts.append(f"{key}={val}")
    return "_".join(parts) + ".inp"


def build_dimers(monomer_name: str, alpha: float, beta: float, phi: float,
                 a: float, b: float, bt1: float, bt2: float, z: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    4ダイマーを生成。t型モノマーは -alpha で make_io_gene_screw_phi.py と整合。
      a-dimer : (0,0,0) - (a,0,0)
      b-dimer : (0,0,0) - (0,b,0)
      t1-dimer: (0,0,0) - (a/2, bt1, z)  [alpha → -alpha]
      t3-dimer: (0,0,0) - (a/2,-bt2, z)  [alpha → -alpha]
    """
    mon0   = get_monomer_xyzR(monomer_name,   0,    0,   0, beta,  alpha, phi)
    mon_a1 = get_monomer_xyzR(monomer_name,   a,    0,   0, beta,  alpha, phi)
    mon_b1 = get_monomer_xyzR(monomer_name,   0,    b,   0, beta,  alpha, phi)
    mon_t1 = get_monomer_xyzR(monomer_name, a/2,  bt1,   z, beta, -alpha, phi)
    mon_t3 = get_monomer_xyzR(monomer_name, a/2, -bt2,   z, beta, -alpha, phi)

    return (
        np.concatenate([mon0, mon_a1], axis=0),
        np.concatenate([mon0, mon_b1], axis=0),
        np.concatenate([mon0, mon_t1], axis=0),
        np.concatenate([mon0, mon_t3], axis=0),
    )


def make_gjf_xyz(auto_dir: str, monomer_name: str, params_dict: Dict[str, float], machine_type: int) -> str:
    alpha = float(params_dict.get('alpha', 0.0))
    beta  = float(params_dict.get('beta',  0.0))
    phi   = float(params_dict.get('phi',   0.0))
    a     = float(params_dict.get('a',     0.0))
    b     = float(params_dict.get('b',     0.0))
    bt1   = float(params_dict.get('bt1',   0.0))
    bt2   = float(params_dict.get('bt2',   0.0))
    z     = float(params_dict.get('z',     0.0))

    d_a1, d_b1, d_t1, d_t3 = build_dimers(monomer_name, alpha, beta, phi, a, b, bt1, bt2, z)

    desc = f'{monomer_name}_alpha={alpha}_beta={beta}_phi={phi}_a={a}_b={b}_bt1={bt1}_bt2={bt2}_z={z}'
    sec1 = get_xyzR_lines(d_a1, desc + " [a1]", machine_type)
    sec2 = get_xyzR_lines(d_b1, desc + " [b1]", machine_type)
    sec3 = get_xyzR_lines(d_t1, desc + " [t1]", machine_type)
    sec4 = get_xyzR_lines(d_t3, desc + " [t3]", machine_type)

    gjf_lines = ['$ RunGauss\n'] + sec1 + ['--Link1--\n'] + sec2 + ['--Link1--\n'] + sec3 + ['--Link1--\n'] + sec4 + ['\n']

    out_dir = Path(auto_dir) / 'gaussian'
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = get_file_name_from_dict(monomer_name, params_dict)
    (out_dir / file_name).write_text("".join(gjf_lines), encoding="utf-8")
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


def exec_gjf(auto_dir: str, monomer_name: str, params_dict: Dict[str, float], machine_type: int,
             isTest: bool = True) -> str:
    inp_dir = Path(auto_dir) / 'gaussian'
    inp_dir.mkdir(parents=True, exist_ok=True)

    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict, machine_type)
    sh_path = inp_dir / Path(file_name).with_suffix('.r1').name
    sh_path.write_text("".join(get_one_exe(file_name, machine_type)), encoding="utf-8")

    if not isTest:
        subprocess.run(['qsub', sh_path.name], check=False, cwd=str(inp_dir))

    return Path(file_name).with_suffix('.log').name


# =========================================================
#                   抽出（局所最小）
# =========================================================

def _neighbors(a: float, bt1: float, bt2: float) -> List[Tuple[float, float, float]]:
    steps = [-STEP_A, 0.0, STEP_A]
    neighbors = []
    for da, dbt1, dbt2 in itertools.product(steps, steps, steps):
        if da == 0.0 and dbt1 == 0.0 and dbt2 == 0.0:
            continue
        neighbors.append((round(a + da, 1), round(bt1 + dbt1, 1), round(bt2 + dbt2, 1)))
    return neighbors


def _is_local_min(a: float, bt1: float, bt2: float, e: float,
                  grid: Dict[Tuple[float, float, float], float]) -> bool:
    nb_energies = [grid[nb] for nb in _neighbors(a, bt1, bt2) if nb in grid]
    if not nb_energies:
        return False
    return all(e <= v for v in nb_energies) and any(e < v for v in nb_energies)


def _read_step1_auto(step1_csv: str, auto_dir: str | Path | None) -> pd.DataFrame:
    p = Path(step1_csv)
    if p.is_file():
        return pd.read_csv(p)
    if auto_dir is None:
        raise FileNotFoundError(f"step1.csv not found: {p}")
    dfs = []
    for sub in sorted(Path(auto_dir).iterdir()):
        if not sub.is_dir():
            continue
        p2 = sub / "step1.csv"
        if p2.is_file():
            dfs.append(pd.read_csv(p2))
    if not dfs:
        raise FileNotFoundError(f"step1.csv が見つからない: {p} も {auto_dir}/*/step1.csv も無い")
    return pd.concat(dfs, ignore_index=True)


def _classify_structure_by_a_rank(df: pd.DataFrame) -> pd.DataFrame:
    """(alpha, beta, z, phi) ごとに a の大小で structure_type を分類"""
    df = df.copy()
    if 'structure_type' not in df.columns:
        df['structure_type'] = 'unknown'
    for col in ('beta', 'phi'):
        if col not in df.columns:
            df[col] = 0.0

    CLUSTER_THR = 0.5
    for (alpha, beta, z, phi), group in df.groupby(['alpha', 'beta', 'z', 'phi']):
        a_vals = sorted(group['a'].unique())
        if not a_vals:
            continue

        clusters, curr = [], [a_vals[0]]
        for v in a_vals[1:]:
            if v - curr[-1] > CLUSTER_THR:
                clusters.append(curr); curr = [v]
            else:
                curr.append(v)
        clusters.append(curr)

        n = len(clusters)
        labels = {}
        if n >= 3:
            labels[0] = 'a-stack'; labels[n - 1] = 'b-stack'
            for i in range(1, n - 1): labels[i] = 'ch'
        elif n == 2:
            labels[0] = 'a-stack'; labels[1] = 'b-stack'
        elif n == 1:
            labels[0] = 'a-stack'

        val_to_label = {v: labels.get(i, 'unknown') for i, clust in enumerate(clusters) for v in clust}
        for idx, val in group['a'].items():
            df.at[idx, 'structure_type'] = val_to_label.get(val, 'unknown')

    return df


def extract_from_step1(step1_csv: str, out_csv: str, auto_dir: str | None = None) -> pd.DataFrame:
    """
    step1.csv から (alpha, beta, z, phi) ごとに (a, bt1, bt2) の局所最小を抽出し、
    filtered_step1.csv として保存。
    """
    df = _read_step1_auto(step1_csv, auto_dir)

    need_base = {'alpha', 'a', 'b', 'bt1', 'bt2', 'z', 'E', 'status'}
    missing = [c for c in need_base if c not in df.columns]
    if missing:
        raise ValueError(f"step1.csv 欠落列: {missing}")

    for col in ('beta', 'phi'):
        if col not in df.columns:
            df[col] = 0.0

    df = df[df['status'].astype(str).str.lower() == 'done'].copy()
    if df.empty:
        print("[extract] Done行が見つかりません。スキップします。")
        return pd.DataFrame()

    energy_cols = [c for c in ['E', 'E1', 'E2', 'E3', 'E4'] if c in df.columns]
    for c in ['a', 'b', 'bt1', 'bt2', 'z', 'alpha', 'beta', 'phi'] + energy_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['a', 'b', 'bt1', 'bt2', 'z', 'E'])

    df = df.sort_values('E').drop_duplicates(
        subset=['alpha', 'beta', 'phi', 'a', 'bt1', 'bt2', 'z'], keep='first'
    )

    rows = []
    for (alpha, beta, z, phi), g in df.groupby(['alpha', 'beta', 'z', 'phi']):
        grid = {
            (round(r.a, 1), round(r.bt1, 1), round(r.bt2, 1)): float(r.E)
            for r in g.itertuples(index=False)
        }
        for r in g.itertuples(index=False):
            a, bt1, bt2, b, e = float(r.a), float(r.bt1), float(r.bt2), float(r.b), float(r.E)
            if _is_local_min(a, bt1, bt2, e, grid):
                rec = {
                    'alpha': float(alpha),
                    'beta':  float(beta),
                    'phi':   float(phi),
                    'a':   round(a,   1),
                    'b':   round(b,   1),
                    'bt1': round(bt1, 1),
                    'bt2': round(bt2, 1),
                    'z':   float(z),
                }
                for c in energy_cols:
                    rec[c] = float(getattr(r, c))
                rows.append(rec)

    if not rows:
        print("[extract] 局所最小点が見つかりませんでした。")
        return pd.DataFrame()

    df_extracted = pd.DataFrame(rows)

    out_path = Path(out_csv)
    if out_path.exists():
        existing_df = pd.read_csv(out_path)
        for col in ('beta', 'phi'):
            if col not in existing_df.columns:
                existing_df[col] = 0.0

        cols_key  = ['alpha', 'beta', 'phi', 'a', 'b', 'bt1', 'bt2', 'z']
        info_cols = [c for c in ['dft_status', 'log', 'inp', 'machine'] if c in existing_df.columns]
        merged_df = pd.merge(df_extracted, existing_df[cols_key + info_cols], on=cols_key, how='left')
        if 'dft_status' not in merged_df.columns:
            merged_df['dft_status'] = float('nan')
        merged_df['dft_status'] = merged_df['dft_status'].fillna('NotYet')
        print(f"[extract] Extracted {len(df_extracted)} points. (Existing info merged)")
    else:
        merged_df = df_extracted
        merged_df['dft_status'] = 'NotYet'
        print(f"[extract] Created new dataset with {len(merged_df)} records")

    merged_df = _classify_structure_by_a_rank(merged_df)

    out_cols  = ['alpha', 'beta', 'phi', 'a', 'b', 'bt1', 'bt2', 'z', 'structure_type', 'dft_status']
    out_cols += energy_cols
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
                           submit: bool = True, throttle: bool = True,
                           target_z_list: Optional[List[float]] = None,
                           E_threshold: float = None) -> pd.DataFrame:
    """filtered_step1.csv を読み、NotYet の行だけ Gaussian ジョブを投入"""
    df = pd.read_csv(cand_csv)
    for col in ('beta', 'phi'):
        if col not in df.columns:
            df[col] = 0.0
    if 'dft_status' not in df.columns:
        df['dft_status'] = 'NotYet'

    cond = (df['dft_status'] == 'NotYet')
    if target_z_list is not None:
        cond &= df['z'].apply(lambda z_val: any(abs(z_val - t) < 1e-5 for t in target_z_list))
    if E_threshold is not None:
        cond &= (df['E'] <= E_threshold)

    target_indices = df[cond].index
    if len(target_indices) == 0:
        print("[submit] 新規投入対象 (NotYet) はありません。")
        return df

    print(f"[submit] {len(target_indices)} 件のジョブを処理します...")
    (Path(auto_dir) / "gaussian").mkdir(parents=True, exist_ok=True)

    for count, idx in enumerate(target_indices):
        r = df.loc[idx]
        params = {
            'alpha': float(r.alpha),
            'beta':  float(r.beta),
            'phi':   float(r.phi),
            'a':     float(r.a),
            'b':     float(r.b),
            'bt1':   float(r.bt1),
            'bt2':   float(r.bt2),
            'z':     float(r.z),
        }
        machine = 1 if count % 2 == 0 else 2

        if submit and throttle:
            while True:
                try:
                    out = subprocess.check_output(["qstat", "-u", os.environ.get("USER", "")], text=True)
                    running = sum(MACHINE_SPEC[machine]["queue"] in line for line in out.splitlines())
                    if running < MAX_PARALLEL[machine]:
                        break
                    time.sleep(2.0)
                except Exception:
                    break

        log = exec_gjf(auto_dir, monomer, params, machine_type=machine, isTest=not submit)
        df.at[idx, 'dft_status'] = 'InProgress' if submit else 'Written'
        df.at[idx, 'log']     = str(Path("gaussian") / log)
        df.at[idx, 'machine'] = machine

    df.to_csv(cand_csv, index=False)
    print(f"[submit] {len(target_indices)} 件投入 -> {cand_csv}")
    return df


# =========================================================
#                       CLI
# =========================================================

def main():
    global MONOMER_DIR

    ap = argparse.ArgumentParser(description="pipeline_screw_phi: 局所最小抽出 & Gaussian 投入 (phi対応)")
    ap.add_argument("--auto-dir",     required=True)
    ap.add_argument("--monomer",      required=True)
    ap.add_argument("--step1-csv",    default=None)
    ap.add_argument("--out-csv",      default=None, help="既定: <auto-dir>/filtered_step1.csv")
    ap.add_argument("--monomer-dir",  default=MONOMER_DIR)
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--submit-only",  action="store_true")
    ap.add_argument("--no-throttle",  action="store_true")
    ap.add_argument("--target-z",     type=float, nargs='+', default=None)
    ap.add_argument("--E-threshold",  type=float, default=None)
    args = ap.parse_args()

    if args.extract_only and args.submit_only:
        raise SystemExit("Error: --extract-only と --submit-only は同時指定できません")

    MONOMER_DIR = os.path.expanduser(args.monomer_dir)
    auto_dir  = args.auto_dir
    step1_csv = args.step1_csv or str(Path(auto_dir) / "step1.csv")
    out_csv   = args.out_csv   or str(Path(auto_dir) / "filtered_step1.csv")

    if not args.submit_only:
        extract_from_step1(step1_csv, out_csv, auto_dir=auto_dir)

    if not args.extract_only:
        submit_from_candidates(
            auto_dir, args.monomer, out_csv,
            submit=True,
            throttle=not args.no_throttle,
            target_z_list=args.target_z,
            E_threshold=args.E_threshold,
        )


if __name__ == "__main__":
    main()
