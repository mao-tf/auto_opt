#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path

# 現在の環境に合わせて適宜インポート
from auto_opt.utils import Rod, R2atom
# utilsにget_xyzR_linesなどがあればインポートします（環境に合わせて調整してください）
from auto_opt.amber.make_io_gene_phi_asym_anti import get_xyzR_lines

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
MONO = DATA / "monomer"

# ==========================================
# 1. モノマー座標の取得（旧 make_step3.py から移植・パス修正）
# ==========================================
def get_monomer_xyzR(monomer_name, Ta, Tb, Tc, A3):
    T_vec = np.array([Ta, Tb, Tc])
    # MONOディレクトリからCSVを読み込む
    csv_path = MONO / f'{monomer_name}.csv'
    df_mono = pd.read_csv(csv_path)
    atoms_array_xyzR = df_mono[['X','Y','Z','R']].values
    
    ez = np.array([0., 0., 1.])

    xyz_array = atoms_array_xyzR[:, :3]
    # Z軸周りにA3（角度）だけ回転
    xyz_array = np.matmul(xyz_array, Rod(ez, A3).T)
    # 並進ベクトルを足す
    xyz_array = xyz_array + T_vec
    R_array = atoms_array_xyzR[:, 3].reshape((-1, 1))
    
    return np.concatenate([xyz_array, R_array], axis=1)


# ==========================================
# 2. 積層構造の組み立て
# ==========================================
def make_stacking_xyz(monomer_name, params_dict):
    a = params_dict.get('a', 0.0); b = params_dict.get('b', 0.0); z = params_dict.get('z', 0.0)
    cx = params_dict.get('cx', 0.0); cy = params_dict.get('cy', 0.0); cz = params_dict.get('cz', 0.0)
    theta1 = params_dict.get('theta1', 0.0); theta2 = params_dict.get('theta2', 0.0)

    # 上層分子 (1分子)
    monomer_array_i = get_monomer_xyzR(monomer_name, cx, cy, cz, theta2)
    
    # 下層分子 (9分子) ※もしb方向(b1,b2)が不要ならコメントアウトしてください
    monomer_array_0 = get_monomer_xyzR(monomer_name, 0, 0, 0, theta1)
    monomer_array_a1 = get_monomer_xyzR(monomer_name, a, 0, 0, theta1)
    monomer_array_a2 = get_monomer_xyzR(monomer_name, -a, 0, 0, theta1)
    monomer_array_b1 = get_monomer_xyzR(monomer_name, 0, b, 2*z, theta1)
    monomer_array_b2 = get_monomer_xyzR(monomer_name, 0, -b, -2*z, theta1)
    monomer_array_t1 = get_monomer_xyzR(monomer_name, a/2, b/2, z, -theta1)
    monomer_array_t2 = get_monomer_xyzR(monomer_name, a/2, -b/2, -z, -theta1)
    monomer_array_t3 = get_monomer_xyzR(monomer_name, -a/2, -b/2, -z, -theta1)
    monomer_array_t4 = get_monomer_xyzR(monomer_name, -a/2, b/2, z, -theta1)
    
    # 全部で10分子を結合 (N_atoms, 4) のNumpy配列を作成
    monomers_array_all = np.concatenate([
        monomer_array_i, monomer_array_0, monomer_array_a1, monomer_array_a2, 
        monomer_array_b1, monomer_array_b2, monomer_array_t1, monomer_array_t2, 
        monomer_array_t3, monomer_array_t4
    ], axis=0)
    
    return monomers_array_all


# ==========================================
# 3. 実行ラッパー (AMBER)
# ==========================================

def exec_gjf_stacking(auto_dir: str, monomer_name: str, params_dict: dict, in_file: str = "FF_calc.in", isTest: bool = True) -> str:
    """
    10分子の構造を作成し、AMBER (sander) で計算を実行する
    """
    out_dir = os.path.join(auto_dir, 'amber')
    os.makedirs(out_dir, exist_ok=True)
    
    cx_str = f"{params_dict['cx']:.2f}".replace('.', 'p').replace('-', 'm')
    cy_str = f"{params_dict['cy']:.2f}".replace('.', 'p').replace('-', 'm')
    
    # 連続実行でファイル名が被らないようにczもファイル名に含める
    cz_str = f"{params_dict['cz']:.2f}".replace('.', 'p')
    file_name = f"{monomer_name}_stack_cx{cx_str}_cy{cy_str}_cz{cz_str}"
    
    mol2_path = os.path.join(out_dir, file_name + '.mol2')
    
    # --- 中略 (座標生成とtleapの部分はそのまま) ---
    stacking_coords = make_stacking_xyz(monomer_name, params_dict)
    mol2_lines = get_xyzR_lines(stacking_coords, monomer_name)
    with open(mol2_path, 'w') as f:
        f.writelines(mol2_lines)

    tleap_in_path = os.path.join(out_dir, file_name + '_tleap.in')
    mono_gaff_mol2 = MONO / f"{monomer_name}.mol2" 
    mono_frcmod = MONO / f"{monomer_name}.frcmod"
    tleap_lines = [
        "source leaprc.gaff2\n",
        f"loadamberparams {mono_frcmod}\n",
        f"MOL = loadmol2 {mono_gaff_mol2}\n",
        f"SYS = loadmol2 {mol2_path}\n",
        f"saveamberparm SYS {file_name}.prmtop {file_name}.inpcrd\n",
        "quit\n"
    ]
    with open(tleap_in_path, 'w') as f:
        f.writelines(tleap_lines)
        
    tleap_cmd = f"tleap -f {file_name}_tleap.in > {file_name}_tleap.out"

    # ★ 変更点：in_file を変数化して、1点計算用の設定ファイルを使えるようにする
    sander_cmd = f"sander -O -i {in_file} -o {file_name}.out -p {file_name}.prmtop -c {file_name}.inpcrd -r {file_name}.rst7"
    
    full_cmd = f"cd {out_dir} && {tleap_cmd} && {sander_cmd}"
    if not isTest:
        subprocess.run(full_cmd, shell=True)
    
    return file_name + '.out'