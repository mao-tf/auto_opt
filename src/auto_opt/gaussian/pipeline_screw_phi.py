#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_screw_phi.py  ―  螺旋軸対称 (screw) Gaussian DFT 投入スクリプト

filtered_step1.csv（extract_minima.py で生成）を読み込み、
各行 (alpha, beta, phi, a, b, bt1, bt2, z) について a/b/t1/t3 の 4 ダイマーを
1 つの .inp にまとめ SGE に qsub する。

回転順: phi(-x軸) → alpha(z軸) → beta(-x軸)

Usage:
  python -m auto_opt.gaussian.pipeline_screw_phi \
      --auto-dir runs/DNTT \
      --monomer  DNTT

  # エネルギー閾値を設けて投入
  python -m auto_opt.gaussian.pipeline_screw_phi \
      --auto-dir runs/DNTT --monomer DNTT --E-threshold -15.0

局所最小の抽出は gaussian/extract_minima.py --symmetry screw を使うこと。
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

MONOMER_DIR = str(Path(__file__).resolve().parents[3] / "data" / "monomer")

MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}
MAX_PARALLEL = {1: 3, 2: 3}

# =========================================================
#                   幾何生成ユーティリティ
# =========================================================

def get_monomer_xyzR(monomer_name: str, Ta: float, Tb: float, Tc: float,
                     beta: float, alpha: float, phi: float = 0.0) -> np.ndarray:
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
    xyz = xyz @ Rod(ez,  alpha).T
    xyz = xyz @ Rod(-ex, beta).T
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

    ap = argparse.ArgumentParser(
        description="filtered_step1.csv から螺旋軸 (screw) 4ダイマー Gaussian ジョブを投入する"
    )
    ap.add_argument("--auto-dir",    required=True, help="作業ディレクトリ（gaussian/ がここに作られる）")
    ap.add_argument("--monomer",     required=True, help="モノマー名（例: DNTT）")
    ap.add_argument("--cand-csv",    default=None,  help="既定: <auto-dir>/filtered_step1.csv")
    ap.add_argument("--monomer-dir", default=MONOMER_DIR)
    ap.add_argument("--no-throttle", action="store_true", help="qstat を見ての待機をしない（即投げ）")
    ap.add_argument("--dry-run",     action="store_true", help=".inp/.r1 を生成するが qsub はしない")
    ap.add_argument("--target-z",    type=float, nargs='+', default=None,
                    help="投入対象を絞る z 値リスト（省略時: 全行）")
    ap.add_argument("--E-threshold", type=float, default=None,
                    help="この値以下の E の行だけ投入（省略時: 制限なし）")
    args = ap.parse_args()

    MONOMER_DIR = os.path.expanduser(args.monomer_dir)
    auto_dir  = args.auto_dir
    cand_csv  = args.cand_csv or str(Path(auto_dir) / "filtered_step1.csv")

    submit_from_candidates(
        auto_dir, args.monomer, cand_csv,
        submit=not args.dry_run,
        throttle=not args.no_throttle,
        target_z_list=args.target_z,
        E_threshold=args.E_threshold,
    )


if __name__ == "__main__":
    main()
