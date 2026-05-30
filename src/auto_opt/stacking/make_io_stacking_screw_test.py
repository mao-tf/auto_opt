#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from auto_opt.utils import Rod, R2atom
from auto_opt.amber.make_io_gene_phi_asym_anti import _load_mol2_params, _guess_mol2_path
from auto_opt.utils import amber_get_E

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
MONO = DATA / "monomer"

def get_monomer_xyzR(monomer_name, Ta, Tb, Tc, beta, alpha):
    T_vec = np.array([Ta, Tb, Tc])
    csv_path = MONO / f'{monomer_name}.csv'
    df_mono = pd.read_csv(csv_path)
    atoms_array_xyzR = df_mono[['X','Y','Z','R']].values
    
    xyz_array = atoms_array_xyzR[:, :3]
    R_array = atoms_array_xyzR[:, 3].reshape((-1, 1))

    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])

    # 層内と整合性を取るため -ex 軸周りの回転
    xyz_array = np.matmul(xyz_array, Rod(ez, alpha).T)
    xyz_array = np.matmul(xyz_array, Rod(-ex, beta).T)
    xyz_array = xyz_array + T_vec
    
    return np.concatenate([xyz_array, R_array], axis=1)

def get_14_pairs_xyzR(monomer_name, params_dict):
    """
    大野さんのコードを基にした 14ペア (7ペア + 3ペア) のダイマー座標生成
    """
    a = float(params_dict.get('a', 0.0))
    bt1 = float(params_dict.get('bt1', 0.0))
    bt2 = float(params_dict.get('bt2', 0.0))
    b = bt1 + bt2
    z = float(params_dict.get('z', 0.0))
    beta = float(params_dict.get('beta', 0.0))
    alpha1 = float(params_dict.get('alpha1', 0.0))
    alpha2 = float(params_dict.get('alpha2', 0.0))
    cx = float(params_dict.get('cx', 0.0))
    cy = float(params_dict.get('cy', 0.0))
    cz = float(params_dict.get('cz', 0.0))

    # α の分子 (monomer_array_c) に対する 9ペア
    mon_c  = get_monomer_xyzR(monomer_name, cx, cy, cz, beta, alpha2)
    mon_i  = get_monomer_xyzR(monomer_name, 0, 0, 0, beta, alpha1)
    mon_p1 = get_monomer_xyzR(monomer_name, 0, b, 0, beta, alpha1)
    mon_p2 = get_monomer_xyzR(monomer_name, 0, -b, 0, beta, alpha1)
    mon_p3 = get_monomer_xyzR(monomer_name, a, 0, 0, beta, alpha1)
    mon_p4 = get_monomer_xyzR(monomer_name, -a, 0, 0, beta, alpha1)
    mon_t1 = get_monomer_xyzR(monomer_name, a/2, bt1, z, beta, -alpha1)
    mon_t2 = get_monomer_xyzR(monomer_name, -a/2, bt1, z, beta, -alpha1)
    mon_t3 = get_monomer_xyzR(monomer_name, a/2, -bt2, z, beta, -alpha1)
    mon_t4 = get_monomer_xyzR(monomer_name, -a/2, -bt2, z, beta, -alpha1)

    # -α の分子 (monomer_array_c_) に対する 9ペア
    mon_c_  = get_monomer_xyzR(monomer_name, a/2+cx, bt1+cy, cz+z, beta, -alpha2)
    mon_i_  = get_monomer_xyzR(monomer_name, a/2, bt1, z, beta, -alpha1)
    mon_p1_ = get_monomer_xyzR(monomer_name, a/2, b+bt1, z, beta, -alpha1)
    mon_p2_ = get_monomer_xyzR(monomer_name, a/2, -bt2, z, beta, -alpha1)
    mon_p3_ = get_monomer_xyzR(monomer_name, a/2+a, bt1, z, beta, -alpha1)
    mon_p4_ = get_monomer_xyzR(monomer_name, -a/2, bt1, z, beta, -alpha1)
    mon_t1_ = get_monomer_xyzR(monomer_name, a, b, 0, beta, alpha1)
    mon_t2_ = get_monomer_xyzR(monomer_name, a, 0, 0, beta, alpha1)
    mon_t3_ = get_monomer_xyzR(monomer_name, 0, b, 0, beta, alpha1)
    mon_t4_ = get_monomer_xyzR(monomer_name, 0, 0, 0, beta, alpha1)
    
    pairs = [
        np.concatenate([mon_c, mon_i], axis=0),
        np.concatenate([mon_c, mon_p1], axis=0),
        np.concatenate([mon_c, mon_p2], axis=0),
        np.concatenate([mon_c, mon_p3], axis=0),
        np.concatenate([mon_c, mon_p4], axis=0),
        np.concatenate([mon_c, mon_t1], axis=0),
        np.concatenate([mon_c, mon_t2], axis=0),
        np.concatenate([mon_c, mon_t3], axis=0),
        np.concatenate([mon_c, mon_t4], axis=0),
        np.concatenate([mon_c_, mon_i_], axis=0),
        np.concatenate([mon_c_, mon_p1_], axis=0),
        np.concatenate([mon_c_, mon_p2_], axis=0),
        np.concatenate([mon_c_, mon_p3_], axis=0),
        np.concatenate([mon_c_, mon_p4_], axis=0),
        np.concatenate([mon_c_, mon_t1_], axis=0),
        np.concatenate([mon_c_, mon_t2_], axis=0),
        np.concatenate([mon_c_, mon_t3_], axis=0),
        np.concatenate([mon_c_, mon_t4_], axis=0),
    ]
    return pairs

def _format_molecule_header(title: str, natoms: int, nbonds: int) -> List[str]:
    return [
        "@<TRIPOS>MOLECULE\n",
        f"{title}\n",
        f"{natoms:6d}{nbonds:7d}{2:6d}{0:6d}{0:6d}\n",
        "SMALL\n",
        "bcc\n",
        "\n",
        "@<TRIPOS>ATOM\n",
    ]

def _format_substructure(n_mono_atoms: int) -> List[str]:
    start2 = n_mono_atoms + 1
    return [
        "@<TRIPOS>SUBSTRUCTURE\n",
        f"{1:6d} RES1{1:10d} GROUP             0 **** **** 0\n",
        f"{2:6d} RES2{start2:10d} GROUP             0 **** **** 0\n",
        "\n",
    ]

def get_xyzR_lines(xyzr_array: np.ndarray, monomer_name: str) -> List[str]:
    # ダイマー専用の mol2 書き出し関数
    types_charges, bonds = _load_mol2_params(monomer_name)
    n_mono = len(types_charges)
    n_total = xyzr_array.shape[0]

    natoms = n_total
    nbonds = 2 * len(bonds)
    lines: List[str] = []
    lines += _format_molecule_header(f"{monomer_name}_dimer", natoms, nbonds)

    for i in range(natoms):
        x, y, z, r = xyzr_array[i]
        atype, charge = types_charges[i % n_mono]
        frag = 1 if i < n_mono else 2
        atom_name = R2atom(r)
        lines.append(
            f"{i+1:6d} {atom_name:<2s} {x: .6f} {y: .6f} {z: .6f} "
            f"{atype} {frag:3d} {'RES1' if frag==1 else 'RES2':<4s} {charge: .6f}\n"
        )

    lines.append("@<TRIPOS>BOND\n")
    bid = 1
    for a, b, btype in bonds:
        lines.append(f"{bid:6d}{a:6d}{b:6d} {btype}\n")
        bid += 1
    off = n_mono
    for a, b, btype in bonds:
        lines.append(f"{bid:6d}{a+off:6d}{b+off:6d} {btype}\n")
        bid += 1

    lines += _format_substructure(n_mono)
    return lines


def exec_14pairs_energy(auto_dir: str, monomer_name: str, params_dict: dict, in_file: str = "FF_calc.in", isTest: bool = False) -> float:
    pairs = get_14_pairs_xyzR(monomer_name, params_dict)
    
    out_dir = os.path.join(auto_dir, 'amber')
    os.makedirs(out_dir, exist_ok=True)
    
    cx_str = f"{params_dict['cx']:.2f}".replace('.', 'p').replace('-', 'm')
    cy_str = f"{params_dict['cy']:.2f}".replace('.', 'p').replace('-', 'm')
    cz_str = f"{params_dict['cz']:.2f}".replace('.', 'p').replace('-', 'm')
    base_file_name = f"{monomer_name}_cx{cx_str}_cy{cy_str}_cz{cz_str}"
    
    monomer_mol2 = str(_guess_mol2_path(monomer_name))
    frcmod_name = f"{monomer_name}_gaff2.frcmod"
    
    parmchk_cmd = (
        f'if [ ! -f "{frcmod_name}" ]; then '
        f'parmchk2 -s gaff2 -i "{monomer_mol2}" -f mol2 -o "{frcmod_name}"; '
        f'fi'
    )

    # 14ジョブを並べて一つの bash スクリプトとして高速一括実行
    cmds = [parmchk_cmd]
    out_files = []
    
    for i, dimer in enumerate(pairs):
        file_name = f"{base_file_name}_p{i}"
        mol2_path = os.path.join(out_dir, file_name + '.mol2')
        
        mol2_lines = get_xyzR_lines(dimer, monomer_name)
        with open(mol2_path, 'w') as f:
            f.writelines(mol2_lines)
            
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
            
        cmds.append(f"tleap -f {file_name}_tleap.in > {file_name}_tleap.out")
        cmds.append(f"sander -O -i {in_file} -o {file_name}.out -p {file_name}.prmtop -c {file_name}.inpcrd -r {file_name}.rst7")
        out_files.append(os.path.join(out_dir, f"{file_name}.out"))
        
    full_cmd = f"cd {out_dir} && source ~/anaconda3/etc/profile.d/conda.sh && conda activate amber && " + " && ".join(cmds)
    
    if not isTest:
        subprocess.run(full_cmd, shell=True, executable='/bin/bash')
        
    total_E = 0.0
    for out_file in out_files:
        try:
            E_list = amber_get_E(out_file)
            total_E += float(E_list[0])
        except:
            total_E += 9999.0
            
    return total_E