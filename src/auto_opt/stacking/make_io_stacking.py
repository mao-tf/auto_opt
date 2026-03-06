#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path
from typing import List

# 現在の環境に合わせて適宜インポート
from auto_opt.utils import Rod, R2atom
# utilsにget_xyzR_linesなどがあればインポートします（環境に合わせて調整してください）
from auto_opt.amber.make_io_gene_phi_asym_anti import _load_mol2_params, _guess_mol2_path

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
    cz_str = f"{params_dict['cz']:.2f}".replace('.', 'p')
    file_name = f"{monomer_name}_stack_cx{cx_str}_cy{cy_str}_cz{cz_str}"
    
    mol2_path = os.path.join(out_dir, file_name + '.mol2')
    
    # --- 10分子座標の作成 ---
    stacking_coords = make_stacking_xyz(monomer_name, params_dict)
    mol2_lines = get_xyzR_lines(stacking_coords, monomer_name)
    with open(mol2_path, 'w') as f:
        f.writelines(mol2_lines)

    # ★修正：モノマーmol2のパスを自動推論し、frcmodの名前を定義
    monomer_mol2 = str(_guess_mol2_path(monomer_name))
    frcmod_name = f"{monomer_name}_gaff2.frcmod"
    
    tleap_in_path = os.path.join(out_dir, file_name + '_tleap.in')
    
    tleap_lines = [
        "source leaprc.gaff2\n",
        f"loadamberparams {frcmod_name}\n", 
        f"MOL = loadmol2 {monomer_mol2}\n",
        f"SYS = loadmol2 {file_name}.mol2\n",
        f"saveamberparm SYS {file_name}.prmtop {file_name}.inpcrd\n",
        "quit\n"
    ]
    with open(tleap_in_path, 'w') as f:
        f.writelines(tleap_lines)
        
    tleap_cmd = f"tleap -f {file_name}_tleap.in > {file_name}_tleap.out"
    sander_cmd = f"sander -O -i {in_file} -o {file_name}.out -p {file_name}.prmtop -c {file_name}.inpcrd -r {file_name}.rst7"
   
    parmchk_cmd = (
        f'if [ ! -f "{frcmod_name}" ]; then '
        f'parmchk2 -s gaff2 -i "{monomer_mol2}" -f mol2 -o "{frcmod_name}"; '
        f'fi'
    )

    full_cmd = (
        f"cd {out_dir} && "
        "source ~/anaconda3/etc/profile.d/conda.sh && "
        "conda activate amber && "
        f"{parmchk_cmd} && "  # ← ここで frcmod を自動生成！
        f"{tleap_cmd} && {sander_cmd}"
    )
    
    if not isTest:
        subprocess.run(full_cmd, shell=True, executable='/bin/bash')
    
    return file_name + '.out'

def get_xyzR_lines(xyzr_array: np.ndarray, monomer_name: str) -> List[str]:
    """
    N分子クラスター用 .mol2 全文の行を返す（MOLECULE/ATOM/BOND/SUBSTRUCTURE）。
    ATOM の atom_type/charge は monomer mol2 から抽出し、N分子分に複製。
    BOND は monomer の BOND を分子数(N)倍に複製。
    """
    # 既存のヘルパー関数（_load_mol2_params, R2atom）はそのまま使います
    types_charges, bonds = _load_mol2_params(monomer_name)
    n_mono = len(types_charges)
    n_total = xyzr_array.shape[0]
    
    # 割り切れるか（原子数に欠損がないか）をチェックし、分子数を自動算出
    assert n_total % n_mono == 0, f"全体の原子数({n_total})がモノマー原子数({n_mono})で割り切れません"
    num_molecules = n_total // n_mono

    lines: List[str] = []

    # 1. MOLECULE ヘッダ (2分子固定の _format_molecule_header を使わず、動的に生成)
    natoms = n_total
    nbonds = num_molecules * len(bonds)
    lines.append("@<TRIPOS>MOLECULE\n")
    lines.append(f"{monomer_name}_cluster\n")
    # 書式: num_atoms num_bonds num_subst 0 0
    lines.append(f"{natoms:5d} {nbonds:5d} {num_molecules:5d} 0 0\n")
    lines.append("SMALL\n")
    lines.append("GASTEIGER\n\n")

    # 2. ATOM セクション
    lines.append("@<TRIPOS>ATOM\n")
    for i in range(natoms):
        x, y, z, r = xyzr_array[i]
        atype, charge = types_charges[i % n_mono]
        
        # 何番目の分子(フラグメント)かを計算 (1始まり)
        frag = (i // n_mono) + 1
        atom_name = R2atom(r)
        
        # id name x y z type resid resname charge
        lines.append(
            f"{i+1:6d} {atom_name:<2s} {x: .6f} {y: .6f} {z: .6f} "
            f"{atype} {frag:3d} RES{frag:<3d} {charge: .6f}\n"
        )

    # 3. BOND セクション
    lines.append("@<TRIPOS>BOND\n")
    bid = 1
    for mol_idx in range(num_molecules):
        off = mol_idx * n_mono  # 2つ目の分子なら n_mono、3つ目なら 2*n_mono のオフセット
        for a, b, btype in bonds:
            lines.append(f"{bid:6d}{a+off:6d}{b+off:6d} {btype}\n")
            bid += 1

    # 4. SUBSTRUCTURE セクション (分子数分だけ自動生成)
    lines.append("@<TRIPOS>SUBSTRUCTURE\n")
    for mol_idx in range(num_molecules):
        frag = mol_idx + 1
        root_atom = mol_idx * n_mono + 1  # 各フラグメントの最初の原子をルートに指定
        # subst_id subst_name root_atom subst_type dict_type chain comment
        lines.append(f"{frag:3d} RES{frag:<3d} {root_atom:5d} GROUP 0 **** ****\n")

    return lines