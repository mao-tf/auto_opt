#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dft_pipeline.py

- 抽出: step1.csv から (alpha,z) ごとに (a,b) の局所最小を抽出 → filtered_step1.csv
python -m auto_opt.gaussian.pipeline_phi --auto-dir runs/ANT_test --monomer ANT --extract-only
- 投入: filtered_step1.csv を読み、各行(alpha,a,b,z)について a1/b1/t1 の 3ダイマーを 1つの .inp にし qsub
python -m auto_opt.gaussian.pipeline_phi --auto-dir runs/ANT_test --monomer ANT --submit-only
- モード: 抽出のみ / 投入のみ / 両方

filtered_step1.csvを作るところでstatus(Done,InProgress,NotYet)行を作って
driver.csvを作っている最中からpipelineを回してfiltered_step1.csvを作ってNotYetのものだけを流せるようにしたい。
"""

from __future__ import annotations
import os, argparse, subprocess, time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from auto_opt.utils import Rod, R2atom

# 変更: デフォルトのモノマーディレクトリ
MONOMER_DIR = os.path.expanduser("~/Working/auto_opt/data/monomer")

# Gaussian 実行設定
MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}
MAX_PARALLEL = {1: 3, 2: 3}  # 各 machine の並列スロット

# 近傍判定の刻み（step1.csv の a,b が 0.1 刻み前提ならココも 0.1）
STEP_A = 0.1
STEP_B = 0.1

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
#                   抽出（局所最小）
# =========================================================

def _neighbors(a: float, b: float) -> List[Tuple[float,float]]:
    da, db = STEP_A, STEP_B
    offs = [( da,0),(-da,0),(0,db),(0,-db),(da,db),(da,-db),(-da,db),(-da,-db)]
    return [(round(a+x,3), round(b+y,3)) for x,y in offs]

def _is_local_min(a: float, b: float, e: float, grid: Dict[Tuple[float,float], float]) -> bool:
    nb = [grid.get((round(ax,3), round(bx,3))) for (ax,bx) in _neighbors(a,b)]
    nb = [v for v in nb if v is not None]
    if not nb:  # 端点は除外
        return False
    return (all(e <= v for v in nb)) and any(e < v for v in nb)

def _read_step1_auto(step1_csv: str, auto_dir: str | Path | None) -> pd.DataFrame:
    """
    - step1_csv が実在すればそれだけ読む（旧仕様）。
    - 無ければ auto_dir 以下のサブディレクトリから step1.csv を集めて結合。
    """
    p = Path(step1_csv)
    if p.is_file():
        # 旧仕様そのまま
        return pd.read_csv(p)

    if auto_dir is None:
        raise FileNotFoundError(f"step1.csv not found: {p}")

    root = Path(auto_dir)
    dfs = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        p2 = sub / "step1.csv"
        if p2.is_file():  ## サブディレクトリの中にstep1.csvが存在したら
            dfs.append(pd.read_csv(p2))

    if not dfs:
        raise FileNotFoundError(
            f"step1.csv が見つからない: {p} も無いし {root}/*/step1.csv も無い"
        )

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all

def extract_from_step1(step1_csv: str, out_csv: str, auto_dir: str | None = None) -> pd.DataFrame:
    """
    step1.csv (列: alpha/またはtheta, a, b, z, E[, E1, E2, E3], status) から
    (alpha,z) ごとに (a,b) の局所最小を抽出し、E, E1, E2, E3 も含めて out_csv へ。
    """
    df = _read_step1_auto(step1_csv, auto_dir)

    need_base = {'alpha','phi','a','b','z','E','status'}
    missing = [c for c in need_base if c not in df.columns]
    if missing:
        raise ValueError(f"step1.csv 欠落列: {missing}")

    energy_cols = [c for c in ['E','E1','E2','E3'] if c in df.columns]

    df = df.copy()
    df = df[df['status'].astype(str).str.lower() == 'done']
    if df.empty:
        raise ValueError("Done 行がない（まず Amber を完了させる）")

    # 型整形
    for c in ['a','b','z'] + energy_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['a','b','z','E'])

    # 同一点が複数ある場合は E 最小を採用
    df = df.sort_values('E').drop_duplicates(subset=['alpha','phi','a','b','z'], keep='first')

    rows = []
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
        raise ValueError("局所最小が見つからない（グリッド刻みや Done 行を確認）")

    out_cols = ['alpha','phi','a','b','z'] + energy_cols
    out = pd.DataFrame(rows)[out_cols].drop_duplicates().sort_values(['z','alpha','phi','a','b']).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[extract] filtered minima -> {out_csv} (n={len(out)})")
    return out

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

    ap = argparse.ArgumentParser(description="step1.csv の局所最小抽出 & 3ダイマーDFT投入（Gaussian）")
    ap.add_argument("--auto-dir", required=True, help="作業ディレクトリ（step1.csv / gaussian/ がここ）")
    ap.add_argument("--monomer", required=True, help="例: PFA")
    ap.add_argument("--step1-csv", default=None, help="既定: <auto-dir>/step1.csv")
    ap.add_argument("--out-csv",   default=None, help="既定: <auto-dir>/filtered_step1.csv")
    ap.add_argument("--monomer-dir", default=MONOMER_DIR, help="モノマーCSVディレクトリ（既定: 環境既定パス）")
    ap.add_argument("--extract-only", action="store_true", help="抽出のみ行い、投入はしない")
    ap.add_argument("--submit-only",  action="store_true", help="既存 filtered_step1.csv から投入のみ行う（抽出はしない）")
    ap.add_argument("--no-throttle",  action="store_true", help="qstat を見ての待機をしない（即投げ）")
    args = ap.parse_args()

    # パスの解決
    MONOMER_DIR = os.path.expanduser(args.monomer_dir)
    auto_dir = args.auto_dir
    step1_csv = args.step1_csv or str(Path(auto_dir) / "step1.csv")
    out_csv   = args.out_csv   or str(Path(auto_dir) / "filtered_step1.csv")

    if args.extract_only and args.submit_only:
        raise SystemExit("extract-only と submit-only は同時指定できない")

    if not args.submit_only:
        extract_from_step1(step1_csv, out_csv, auto_dir=auto_dir)

    if not args.extract_only:
        submit_from_candidates(auto_dir, args.monomer, out_csv, submit=True, throttle=not args.no_throttle)

if __name__ == "__main__":
    main()
