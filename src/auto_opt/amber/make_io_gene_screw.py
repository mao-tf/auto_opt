#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Optional

from auto_opt.utils import Rod, R2atom

# プロジェクトルート推定（src/auto_opt/.../このファイル という前提）
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
MONO = DATA / "monomer"

# ==========================
#      mol2 解析ユーティリティ
# ==========================

def _find_section(lines: List[str], tag: str) -> int:
    tag = tag.strip().upper()
    for i, ln in enumerate(lines):
        if ln.strip().upper() == tag:
            return i
    return -1

def _next_section_idx(lines: List[str], start: int) -> int:
    j = start + 1
    while j < len(lines) and not lines[j].startswith("@<TRIPOS>"):
        j += 1
    return j

# 先頭付近の import/定数の下あたりに追加
import shutil
import subprocess

def _guess_mol2_path(monomer_name: str) -> Path:
    cand1 = MONO / f"{monomer_name}.mol2"
    if cand1.exists(): return cand1
    cand2 = MONO / f"{monomer_name}_HF_esp.mol2"
    if cand2.exists(): return cand2
    gl = sorted(MONO.glob(f"{monomer_name}*.mol2"))
    if gl: return gl[0]
    raise FileNotFoundError(f"mol2 not found for monomer '{monomer_name}' under {MONO}")

@lru_cache(maxsize=16)
def _load_mol2_params(monomer_name: str) -> Tuple[List[Tuple[str,float]], List[Tuple[int,int,str]]]:
    """
    戻り値:
      - types_charges: [(atom_type, charge)] （atom_id昇順, 1-indexベース）
      - bonds: [(a_idx, b_idx, bond_type)]  （mol2の index 参照, 1-index）
    """
    p = _guess_mol2_path(monomer_name)
    txt = p.read_text(encoding="utf-8", errors="ignore")
    lines = txt.splitlines()

    i_atom = _find_section(lines, "@<TRIPOS>ATOM")
    if i_atom < 0:
        raise ValueError(f"AT0M section not found in {p.name}")
    j_atom = _next_section_idx(lines, i_atom)

    # ATOM 行のパース: id name x y z type resid resname charge
    types_charges: List[Tuple[str, float]] = []
    for ln in lines[i_atom+1:j_atom]:
        parts = ln.split()
        if len(parts) < 9:
            continue
        # parts[0]=id, parts[5]=type, parts[8]=charge
        try:
            _ = int(parts[0])
            atype = parts[5]
            charge = float(parts[8])
        except Exception:
            continue
        types_charges.append((atype, charge))

    if not types_charges:
        raise ValueError(f"No ATOM records parsed in {p.name}")

    i_bond = _find_section(lines, "@<TRIPOS>BOND")
    bonds: List[Tuple[int,int,str]] = []
    if i_bond >= 0:
        j_bond = _next_section_idx(lines, i_bond)
        for ln in lines[i_bond+1:j_bond]:
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                # parts: id a b type
                a = int(parts[1]); b = int(parts[2]); btype = parts[3]
                bonds.append((a, b, btype))
            except Exception:
                pass  # スキップ

    return types_charges, bonds

# ==========================
#   幾何: CSV -> (x,y,z,R)
# ==========================

def get_monomer_xyzR(monomer_name: str, Ta: float, Tb: float, Tc: float, A2: float, A3: float):
    """
    data/monomer/<monomer>.csv を読み、x回転(A2), z回転(A3) → 平行移動(Ta,Tb,Tc)。
    返り値: ndarray (N,4) = (x,y,z,R)
    """
    T_vec = np.array([Ta, Tb, Tc], float)
    df_mono = pd.read_csv(MONO / f"{monomer_name}.csv")
    atoms_array_xyzR = df_mono[['X', 'Y', 'Z', 'R']].to_numpy(dtype=float)

    ex = np.array([1., 0., 0.]); ez = np.array([0., 0., 1.])
    xyz = atoms_array_xyzR[:, :3]
    # x軸に -A2 回転 → z軸 A3 回転（あなたの元コード準拠）
    xyz = xyz @ Rod(-ex, A2).T
    xyz = xyz @ Rod(ez, A3).T
    xyz = xyz + T_vec

    R = atoms_array_xyzR[:, 3].reshape((-1, 1))
    return np.concatenate([xyz, R], axis=1)

# ==========================
#   mol2 行の生成（ダイマー）
# ==========================

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
    # RES1 の開始は 1、RES2 の開始は n_mono_atoms + 1
    start2 = n_mono_atoms + 1
    return [
        "@<TRIPOS>SUBSTRUCTURE\n",
        f"{1:6d} RES1{1:10d} GROUP             0 ****  ****    0\n",
        f"{2:6d} RES2{start2:10d} GROUP             0 ****  ****    0\n",
        "\n",
    ]

def get_xyzR_lines(xyzr_array: np.ndarray, monomer_name: str) -> List[str]:
    """
    ダイマー用 .mol2 全文の行を返す（MOLECULE/ATOM/BOND/SUBSTRUCTURE）。
    ATOM の atom_type/charge は monomer mol2 から抽出し、2残基分に複製。
    BOND は monomer の BOND を2倍に複製（片方ずつ）。
    """
    types_charges, bonds = _load_mol2_params(monomer_name)
    n_mono = len(types_charges)
    n_total = xyzr_array.shape[0]
    assert n_total % 2 == 0, "xyzr_array は 2N 行（ダイマー）である必要があります"
    assert n_total // 2 == n_mono, f"座標({n_total//2}原子)と mol2({n_mono}原子) の原子数が一致しません"

    # ヘッダ
    natoms = n_total
    nbonds = 2 * len(bonds)
    lines: List[str] = []
    lines += _format_molecule_header(f"{monomer_name}_dimer", natoms, nbonds)

    # ATOM
    for i in range(natoms):
        x, y, z, r = xyzr_array[i]
        atype, charge = types_charges[i % n_mono]
        frag = 1 if i < n_mono else 2
        atom_name = R2atom(r)  # 表示名は元素（あなたの元仕様）
        # id name x y z type resid resname charge
        lines.append(
            f"{i+1:6d} {atom_name:<2s} {x: .6f} {y: .6f} {z: .6f} "
            f"{atype} {frag:3d} {'RES1' if frag==1 else 'RES2':<4s} {charge: .6f}\n"
        )

    # BOND
    lines.append("@<TRIPOS>BOND\n")
    bid = 1
    # fragment 1
    for a, b, btype in bonds:
        lines.append(f"{bid:6d}{a:6d}{b:6d} {btype}\n")
        bid += 1
    # fragment 2 (index offset)
    off = n_mono
    for a, b, btype in bonds:
        lines.append(f"{bid:6d}{a+off:6d}{b+off:6d} {btype}\n")
        bid += 1

    # SUBSTRUCTURE
    lines += _format_substructure(n_mono)
    return lines

# ==========================
#     Amber 実行スクリプト
# ==========================

def get_one_exe(auto_dir: str, file_name: str, monomer_name: str) -> Tuple[str, str]:
    """
    amber/ に job_*.sh と *_tleap.in を書き出す。
    - parmchk2 で {file_basename}.frcmod を生成
    - tleap は loadamberparams {file_basename}.frcmod を使用（←汎用化）
    戻り値: (job_sh_path, out_log_name)
    """
    file_basename = os.path.splitext(file_name)[0]
    amber_dir = os.path.join(auto_dir, 'amber')
    os.makedirs(amber_dir, exist_ok=True)

    # モノマー mol2 の絶対パスを特定
    monomer_mol2 = str(_guess_mol2_path(monomer_name))

    # tleap 入力（汎用化）
    lines_tleap = [
        "source leaprc.gaff2\n",
        f"MOL = loadmol2 {file_basename}.mol2\n",
        f"loadamberparams {monomer_name}_gaff2.frcmod\n",
        f"saveamberparm MOL {file_basename}.prmtop {file_basename}.inpcrd\n",
        "quit\n",
    ]

    # 実行ジョブ
    lines_job = [
        "#!/bin/bash\n",
        "\n",
        f"cd {amber_dir}\n",
        "source ~/anaconda3/etc/profile.d/conda.sh\n",
        "conda activate amber\n",
        "\n",
        # amber 環境の中で、一度だけモノマー frcmod を生成
        f'if [ ! -f "{monomer_name}_gaff2.frcmod" ]; then\n',
        f'  echo "[once] parmchk2 for {monomer_name}";\n',
        f'  parmchk2 -s gaff2 -i "{monomer_mol2}" -f mol2 -o "{monomer_name}_gaff2.frcmod";\n',
        f'fi\n',
        f"tleap -f {file_basename}_tleap.in\n",
        f"sander -O -i FF_calc.in -o {file_basename}.out "
        f"-p {file_basename}.prmtop -c {file_basename}.inpcrd "
        f"-r min.rst -ref {file_basename}.inpcrd\n",
    ]

    file_job = os.path.join(amber_dir, f'job_{file_basename}.sh')
    file_tleap = os.path.join(amber_dir, f'{file_basename}_tleap.in')

    with open(file_job, 'w') as f:
        f.writelines(lines_job)
    with open(file_tleap, 'w') as f:
        f.writelines(lines_tleap)

    return file_job, f'{file_basename}.out'

# ==========================
#     GaussView 用 XYZ
# ==========================

def make_xyzfile(monomer_name: str, params_dict: dict, structure_type: int) -> List[str]:
    """
    GaussView用 XYZ （コメント行含む）
    structure_type:
      1 → a-ダイマー, 2 → b-ダイマー, 3 → t 3配置（i+p1+p2+t1+t2）
    """
    a = params_dict.get('a', 0.0)
    b = params_dict.get('b', 0.0)
    z = params_dict.get('z', 0.0)
    A2 = params_dict.get('phi', 0.0)
    A3 = params_dict.get('alpha', 0.0)

    mon_i  = get_monomer_xyzR(monomer_name, 0,    0,    0,   A2,  A3)
    mon_p1 = get_monomer_xyzR(monomer_name, a,    0,    0,   A2,  A3)
    mon_p2 = get_monomer_xyzR(monomer_name, 0,    b,    0,   A2,  A3)
    mon_t1 = get_monomer_xyzR(monomer_name, a/2,  b/2,  z,   A2, -A3)
    mon_t3 = get_monomer_xyzR(monomer_name, a/2,  -b/2, z,   A2, -A3)

    if structure_type == 1:
        arr = np.concatenate([mon_i, mon_p1], axis=0)
    elif structure_type == 2:
        arr = np.concatenate([mon_i, mon_p2], axis=0)
    elif structure_type == 3:
        arr = np.concatenate([mon_i, mon_p1, mon_p2, mon_t1, mon_t3], axis=0)
    elif structure_type == 4:
        arr = np.concatenate([mon_i, mon_p1, mon_p2, mon_t1, mon_t3], axis=0)
    
    else:
        raise ValueError("structure_type must be 1, 2, 3 or 4")

    lines = [f"{len(arr)}\n", f"{monomer_name} structure_type={structure_type}\n"]
    for x, y, zc, R in arr:
        lines.append(f"{R2atom(R)} {x:.6f} {y:.6f} {zc:.6f}\n")
    return lines

def make_xyz(monomer_name: str, params_dict: dict, structure_type: int) -> str:
    name = monomer_name
    for key, val in params_dict.items():
        if key in ['a', 'b', 'z']:
            val = np.round(val, 2)
        elif key in ['phi', 'alpha']:
            val = int(val)
        name += f"_{key}_{val}"
    return name + f'_{structure_type}.xyz'

# ==========================
#     ダイマー .mol2 の生成
# ==========================

def get_file_name_from_dict(monomer_name: str, params_dict: dict, structure_type: int) -> str:
    name = monomer_name
    for key, val in params_dict.items():
        if key in ['phi', 'alpha']:
            val = int(val)
        name += f"_{key}_{val}"
    return name + f'_{structure_type}.mol2'

def make_gjf_xyz(auto_dir: str, monomer_name: str, params_dict: dict, structure_type: int) -> str:
    """
    実際には Gaussian ではなく、Amber用の **ダイマー mol2** を作る。
    structure_type=1(a),2(b),3(t) のいずれかで1ダイマーのみを書き出す。
    """
    a = params_dict.get('a', 0.0)
    b = params_dict.get('b', 0.0)
    z = params_dict.get('z', 0.0)
    A2 = params_dict.get('phi', 0.0)
    A3 = params_dict.get('alpha', 0.0)

    mon_i  = get_monomer_xyzR(monomer_name, 0,   0,    0,   A2,  A3)
    mon_p1 = get_monomer_xyzR(monomer_name, a,   0,    0,   A2,  A3)
    mon_p2 = get_monomer_xyzR(monomer_name, 0,   b,    0,   A2,  A3)
    mon_t1 = get_monomer_xyzR(monomer_name, a/2, b/2,  z,   A2, -A3)
    mon_t3 = get_monomer_xyzR(monomer_name, a/2, -b/2, z,   A2, -A3)

    if structure_type == 1:
        dimer = np.concatenate([mon_i, mon_p1], axis=0)
    elif structure_type == 2:
        dimer = np.concatenate([mon_i, mon_p2], axis=0)
    elif structure_type == 3:
        dimer = np.concatenate([mon_i, mon_t1], axis=0)
    elif structure_type == 4:
        dimer = np.concatenate([mon_i, mon_t3], axis=0)
    else:
        raise ValueError("structure_type must be 1, 2, 3 or 4")

    mol2_lines = get_xyzR_lines(dimer, monomer_name)

    out_dir = os.path.join(auto_dir, 'amber')
    os.makedirs(out_dir, exist_ok=True)
    file_name = get_file_name_from_dict(monomer_name, params_dict, structure_type)
    path = os.path.join(out_dir, file_name)
    with open(path, 'w') as f:
        f.writelines(mol2_lines)
    return file_name

# ==========================
#     実行ラッパ
# ==========================

def exec_gjf(auto_dir: str, monomer_name: str, params_dict: dict, structure_type: int, isTest: bool = True) -> str:
    """
    - gaussview 用 .xyz を作成
    - amber 用 ダイマー .mol2 を作成
    - job スクリプトと tleap 入力を作成
    - isTest=False なら実行
    戻り値: .out（sander出力）ファイル名
    """
    # gaussview
    gv_dir = os.path.join(auto_dir, 'gaussview')
    os.makedirs(gv_dir, exist_ok=True)
    xyzfile_name = make_xyz(monomer_name, params_dict, structure_type)
    with open(os.path.join(gv_dir, xyzfile_name), 'w') as f:
        f.writelines(make_xyzfile(monomer_name, params_dict, structure_type))

    # mol2 + job
    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict, structure_type)
    file_job, out_name = get_one_exe(auto_dir, file_name, monomer_name)

    if not isTest:
        subprocess.run(['chmod', '+x', file_job], check=False)
        subprocess.run([file_job], check=False)

    return out_name
