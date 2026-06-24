#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_phi.py  ―  映進対称 (glide) Gaussian DFT 投入スクリプト

filtered_step1.csv（extract_minima.py で生成）を読み込み、
各行 (alpha, phi, a, b, z) について a/b/t の 3 ダイマーを 1 つの .inp にまとめ
SGE に qsub する。

Usage:
  python -m auto_opt.gaussian.pipeline_phi \
      --auto-dir runs/DNTT \
      --monomer  DNTT

局所最小の抽出は gaussian/extract_minima.py --symmetry glide を使うこと。
"""

from __future__ import annotations
import os, argparse, subprocess, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from auto_opt.utils import Rod, R2atom

# デフォルトはパッケージ基準の data/monomer/ (CLI で --monomer-dir 上書き可)
MONOMER_DIR = str(Path(__file__).resolve().parents[3] / "data" / "monomer")

# Gaussian 実行設定
MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}
MAX_PARALLEL = {1: 3, 2: 3}  # 各 machine の並列スロット

# =========================================================
#                   幾何生成ユーティリティ
# =========================================================

def get_monomer_xyzR(monomer_name: str, Ta: float, Tb: float, Tc: float, A2: float, A3: float) -> np.ndarray:
    path = os.path.join(MONOMER_DIR, f"{monomer_name}.csv")
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"monomer CSV not found: {path}")
    df_mono = pd.read_csv(path)
    atoms_array_xyzR = df_mono[['X','Y','Z','R']].to_numpy(dtype=float)

    ex = np.array([1.,0.,0.])
    ez = np.array([0.,0.,1.])
    xyz = atoms_array_xyzR[:, :3]
    # x軸回転
    xyz = xyz @ Rod(-ex, A2).T
    # z軸回転
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

    # 前半 Fragment=1, 後半 Fragment=2
    for i, (x, y, z, Rv) in enumerate(xyzR_array):
        frag = 1 if i < n_atom_each else 2
        atom = R2atom(Rv)
        lines.append(f'{atom}(Fragment={frag}) {float(x):.6f} {float(y):.6f} {float(z):.6f}\n')

    lines.append('\n')
    return lines

def get_file_name_from_dict(monomer_name: str, params_dict: Dict[str, float]) -> str:
    """alpha は int、a/b/z は 0.1 丸めでファイル名に埋め込む"""
    parts = [monomer_name]
    for key in ("alpha","phi","a","b","z"):
        if key not in params_dict: continue
        val = params_dict[key]
        if key in ("alpha","phi"):
            val = int(round(val))
        elif key in ("a","b","z"):
            val = round(float(val), 1)
        parts.append(f"{key}={val}")
    return "_".join(parts) + ".inp"  ## リストの中身を'_'で連結する

def build_dimers(monomer_name: str, alpha: float, phi: float, a: float, b: float, z: float
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Amber側の幾何規約に合わせて 3ダイマーを作る:
      a1: (0,0,0, +alpha) と (a, 0, 0, +alpha)
      b1: (0,0,0, +alpha) と (0, b, 2z, +alpha)
      t1: (0,0,0, +alpha) と (a/2, b/2, z, -alpha)
    返り値: 各ダイマーの (x,y,z,R) を縦に 2N 並べた array
    """
    A2 = phi
    A3 = alpha
    mon0   = get_monomer_xyzR(monomer_name, 0,   0,   0, A2,   A3)
    mon_a1 = get_monomer_xyzR(monomer_name, a,   0,   0, A2,   A3)
    mon_b1 = get_monomer_xyzR(monomer_name, 0,   b, 2*z, A2,   A3)
    mon_t1 = get_monomer_xyzR(monomer_name, a/2, b/2, z, A2,  -A3)

    dimer_a1 = np.concatenate([mon0, mon_a1], axis=0)
    dimer_b1 = np.concatenate([mon0, mon_b1], axis=0)
    dimer_t1 = np.concatenate([mon0, mon_t1], axis=0)
    return dimer_a1, dimer_b1, dimer_t1

def make_gjf_xyz(auto_dir: str, monomer_name: str, params_dict: Dict[str,float], machine_type: int) -> str:
    """
    3ダイマー(a1,b1,t1)を1つの .inp に Link1 で連結して書き出す。
    戻り値: ファイル名（ベース名）
    """
    alpha = float(params_dict.get('alpha', 0.0))
    phi   = float(params_dict.get('phi', 0.0))
    a     = float(params_dict.get('a', 0.0))
    b     = float(params_dict.get('b', 0.0))
    z     = float(params_dict.get('z', 0.0))

    d_a1, d_b1, d_t1 = build_dimers(monomer_name, alpha, phi, a, b, z)

    desc = f'{monomer_name}_alpha={alpha}_phi={phi}_a={a}_b={b}_z={z}'
    sec1 = get_xyzR_lines(d_a1, desc + " [a1]", machine_type)
    sec2 = get_xyzR_lines(d_b1, desc + " [b1]", machine_type)
    sec3 = get_xyzR_lines(d_t1, desc + " [t1]", machine_type)

    gjf_lines = ['$ RunGauss\n'] + sec1 + ['--Link1--\n'] + sec2 + ['--Link1--\n'] + sec3 + ['\n']

    out_dir = Path(auto_dir) / 'gaussian'
    out_dir.mkdir(parents=True, exist_ok=True)
    file_name = get_file_name_from_dict(monomer_name, {"alpha":alpha,"phi":phi,"a":a,"b":b,"z":z})
    gjf_path = out_dir / file_name
    gjf_path.write_text("".join(gjf_lines), encoding="utf-8")
    return file_name  # 例: PFA_alpha=0_a=6.3_b=9.0_z=0.0.inp

def get_one_exe(file_name: str, machine_type: int) -> List[str]:
    """
    SGE 投入スクリプトを生成。queue名と nproc は MACHINE_SPEC 参照。
    """
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
    """
    gaussian/*.inp と *.r1 を作り、isTest=False なら qsub。
    戻り値: .log のファイル名
    """
    inp_dir = Path(auto_dir) / 'gaussian'
    inp_dir.mkdir(parents=True, exist_ok=True)

    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict, machine_type)
    cc_list = get_one_exe(file_name, machine_type)

    sh_filename = Path(file_name).with_suffix('.r1').name
    sh_path = inp_dir / sh_filename
    sh_path.write_text("".join(cc_list), encoding="utf-8")  ## Path.write_text()は文字列をPathに書き込む関数. encoding="utf-8"にすることで日本語コメントや全角スペースなども文字化けしない

    if not isTest:
        subprocess.run(['qsub', sh_path.name], check=False, cwd=str(inp_dir))  ## ターミナル上でで[]内のコマンドを実行する

    log_file_name = Path(file_name).with_suffix('.log').name
    return log_file_name

# =========================================================
#                   投入（qsub）
# =========================================================

def submit_from_candidates(auto_dir: str, monomer: str, cand_csv: str,
                           submit: bool=True, throttle: bool=True) -> pd.DataFrame:
    """
    filtered_step1.csv（列: alpha,a,b,z）を読み、
    マシン 1/2 に交互に割り当て。MAX_PARALLEL に達したときは qstat で待機（簡易）。
    """
    df = pd.read_csv(cand_csv)
    need = {'alpha','phi','a','b','z'}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{cand_csv} 欠落列: {miss}")

    jobs = []
    gdir = Path(auto_dir) / "gaussian"
    gdir.mkdir(parents=True, exist_ok=True)

    for i, r in df.iterrows():
        alpha, phi, a, b, z = float(r.alpha), float(r.phi), float(r.a), float(r.b), float(r.z)
        machine = 1 if (i % 2 == 0) else 2

        # スロット制御（簡易）：同queueの自分のジョブ数ベース
        if submit and throttle:
            while True:
                try:
                    out = subprocess.check_output(["qstat", "-u", os.environ.get("USER","")], text=True)
                    running = sum(MACHINE_SPEC[machine]["queue"] in line for line in out.splitlines())
                    if running < MAX_PARALLEL[machine]:
                        break
                    time.sleep(0.5)
                except Exception:
                    break  # qstat が無ければスキップ

        log = exec_gjf(auto_dir, monomer, {"alpha":alpha,"phi":phi,"a":a,"b":b,"z":z},
                       machine_type=machine, isTest=not submit)
        jobs.append({
            "alpha": alpha,"phi": phi, "a": a, "b": b, "z": z,
            "machine": machine,
            "inp": str(gdir / get_file_name_from_dict(monomer, {"alpha":alpha,"phi":phi,"a":a,"b":b,"z":z})),
            "log": str(gdir / log),
            "status": "submitted" if submit else "written",
        })

    out_df = pd.DataFrame(jobs)
    out_df.to_csv(Path(auto_dir)/"dft_jobs.csv", index=False)
    print(f"[submit] jobs -> {Path(auto_dir)/'dft_jobs.csv'} (n={len(out_df)})")
    return out_df

# =========================================================
#                       CLI
# =========================================================

def main():
    global MONOMER_DIR

    ap = argparse.ArgumentParser(
        description="filtered_step1.csv から映進 (glide) 3ダイマー Gaussian ジョブを投入する"
    )
    ap.add_argument("--auto-dir",    required=True, help="作業ディレクトリ（gaussian/ がここに作られる）")
    ap.add_argument("--monomer",     required=True, help="モノマー名（例: DNTT）")
    ap.add_argument("--cand-csv",    default=None,  help="既定: <auto-dir>/filtered_step1.csv")
    ap.add_argument("--monomer-dir", default=MONOMER_DIR, help="モノマーCSVディレクトリ")
    ap.add_argument("--no-throttle", action="store_true",  help="qstat を見ての待機をしない（即投げ）")
    ap.add_argument("--dry-run",     action="store_true",  help=".inp/.r1 を生成するが qsub はしない")
    args = ap.parse_args()

    MONOMER_DIR = os.path.expanduser(args.monomer_dir)
    auto_dir  = args.auto_dir
    cand_csv  = args.cand_csv or str(Path(auto_dir) / "filtered_step1.csv")

    submit_from_candidates(
        auto_dir, args.monomer, cand_csv,
        submit=not args.dry_run,
        throttle=not args.no_throttle,
    )

if __name__ == "__main__":
    main()
