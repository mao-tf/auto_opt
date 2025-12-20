import os
import numpy as np
import pandas as pd
import subprocess
from auto_opt.utils import Rod, R2atom
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]    # auto_opt/ の 1つ上が src/、2つ上でプロジェクトルート
DATA = ROOT / "data"
MONO = DATA / "monomer"

def get_monomer_xyzR(monomer_name,Ta,Tb,Tc,A2,A3):
    T_vec = np.array([Ta,Tb,Tc])
    df_mono = pd.read_csv(MONO / f"{monomer_name}.csv")
    atoms_array_xyzR=df_mono[['X','Y','Z','R']].values
    ex = np.array([1.,0.,0.]); ey = np.array([0.,1.,0.]); ez = np.array([0.,0.,1.])
    xyz_array = atoms_array_xyzR[:,:3]
    xyz_array = np.matmul(xyz_array,Rod(-ex,A2).T)
    xyz_array = np.matmul(xyz_array,Rod(ez,A3).T)
    xyz_array = xyz_array + T_vec
    R_array = atoms_array_xyzR[:,3].reshape((-1,1))
    return np.concatenate([xyz_array,R_array],axis=1)
        
line1='@<TRIPOS>MOLECULE\nPFA\n   48    52     2     0     0\nSMALL\nbcc\n\n\n@<TRIPOS>ATOM\n'
line2='@<TRIPOS>BOND\n'
bond_lines=[ '     1     1     2 ar  \n',
            '     2     1     3 ar  \n',
            '     3     1     7 ar  \n',
            '     4     2     4 ar  \n',
            '     5     2     9 ar  \n',
            '     6     3     5 ar  \n',
            '     7     3    11 1   \n',
            '     8     4     6 ar  \n',
            '     9     4    14 1   \n',
            '    10     5     6 ar  \n',
            '    11     5    16 1   \n',
            '    12     6    15 1   \n',
            '    13     7     8 ar  \n',
            '    14     7    12 1   \n',
            '    15     8    10 ar  \n',
            '    16     8    17 ar  \n',
            '    17     9    10 ar  \n',
            '    18     9    13 1   \n',
            '    19    10    19 ar  \n',
            '    20    17    18 ar  \n',
            '    21    17    21 1   \n',
            '    22    18    20 ar  \n',
            '    23    18    23 1   \n',
            '    24    19    20 ar  \n',
            '    25    19    22 1   \n',
            '    26    20    24 1   \n',
            '    27    25    26 ar \n',
            '    28    25    27 ar \n',
            '    29    25    31 ar \n',
            '    30    26    28 ar \n',
            '    31    26    33 ar \n',
            '    32    27    29 ar \n',
            '    33    27    35 1 \n',
            '    34    28    30 ar \n',
            '    35    28    38 1 \n',
            '    36    29    30 ar \n',
            '    37    29    40 1 \n',
            '    38    30    39 1 \n',
            '    39    31    32 ar \n',
            '    40    31    36 1 \n',
            '    41    32    34 ar \n',
            '    42    32    41 ar \n',
            '    43    33    34 ar \n',
            '    44    33    37 1 \n',
            '    45    34    43 ar \n',
            '    46    41    42 ar \n',
            '    47    41    45 1 \n',
            '    48    42    44 ar \n',
            '    49    42    47 1 \n',
            '    50    43    44 ar \n',
            '    51    43    46 1 \n',
            '    52    44    48 1 \n',
]
line3='@<TRIPOS>SUBSTRUCTURE\n     1 RES1        1 GROUP             0 ****  ****    0  \n     2 RES2       25 GROUP             0 ****  ****    0 \n\n'

para_list = []
with open(MONO / 'pfa_HF_esp.mol2') as f:
    for line in f:
        s = line.split()
        if len(s)==9:
            para_list.append([s[5], float(s[8])])
        if 'BOND' in line:
            break

def get_xyzR_lines(xyzr_array):
    lines=[]
    lines.append(line1)
    mol=int(len(xyzr_array)/2)
    for i in range(mol):
        x,y,z,r=xyzr_array[i]
        atom_type,charge=para_list[i]
        lines.append(f'  {i+1} {R2atom(r)} {x} {y} {z} {atom_type} 1 RES1 {charge}\n')
    for i in range(mol):
        x,y,z,r=xyzr_array[i+mol]
        atom_type,charge=para_list[i]
        lines.append(f'  {i+1+mol} {R2atom(r)} {x} {y} {z} {atom_type} 2 RES2 {charge}\n')   
    lines.append(line2)
    for line in bond_lines:
        lines.append(line)
    lines.append(line3)
    return lines

# 実行ファイル作成
def get_one_exe(auto_dir,file_name,structure_type):
    file_basename = os.path.splitext(file_name)[0]
    if structure_type == 3:
        n='2';fc_type='t_3'
    else:
        n='2';fc_type='p'

    lines_job = [
        "#!/bin/bash\n",
        "\n",
        "cd {0}/amber\n".format(auto_dir),   # ← これが肝。以降はこのフォルダを基準に実行される
        "source ~/anaconda3/etc/profile.d/conda.sh\n",
        "conda activate amber\n",
        "\n",
        f"parmchk2 -a Y -s gaff2 -i {file_basename}.mol2 -f mol2 -o {file_basename}.frcmod\n",
        f"tleap -f {file_basename}_tleap.in\n",
        f"sander -O -i FF_calc.in -o {file_basename}.out "
        f"-p {file_basename}.prmtop -c {file_basename}.inpcrd "
        f"-r min.rst -ref {file_basename}.inpcrd\n",
    ]

    # tleap 用入力
    lines_tleap = [
        "source leaprc.gaff2\n",
        f"MOL = loadmol2 {file_basename}.mol2\n",
        f'loadamberparams epsilon_{fc_type}.frcmod\n',#{file_basename}.frcmod\n',
        f"saveamberparm MOL {file_basename}.prmtop {file_basename}.inpcrd\n",
        "quit\n",
    ]

    file_job = os.path.join(auto_dir,f'amber/job_{file_basename}.sh')
    file_tleap = os.path.join(auto_dir,f'amber/{file_basename}_tleap.in')
    
    with open(file_job,'w')as f:
        f.writelines(lines_job)
    with open(file_tleap,'w')as f:
        f.writelines(lines_tleap)

    return file_job,f'{file_basename}.out'

######################################## 特化関数 ########################################

##################gaussview##################
def make_xyzfile(monomer_name,params_dict,structure_type):
    a = params_dict.get('a',0.0)
    b = params_dict.get('b',0.0); z = params_dict.get('z',0.0)
    A2 = params_dict.get('A2',0.0); A3 = params_dict.get('alpha',0.0)

    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A2,A3)
    
    monomer_array_p1 = get_monomer_xyzR(monomer_name,a,0,0,A2,A3)##1,2がb方向
    monomer_array_p2 = get_monomer_xyzR(monomer_name,0,b,2*z,A2,A3)##1,2がb方向
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a/2,b/2,z,A2,-A3)##1,2がb方向
    monomer_array_t2 = get_monomer_xyzR(monomer_name,-a/2,b/2,z,A2,-A3)##1,2がb方向
    
    xyz_list=['400 \n','polyacene9 \n']##4分子のxyzファイルを作成
    
    if structure_type == 1:##隣接8分子について対称性より3分子でエネルギー計算
        monomers_array_4 = np.concatenate([monomer_array_i,monomer_array_p1],axis=0)
    elif structure_type == 2:##隣接8分子について対称性より3分子でエネルギー計算
        monomers_array_4 = np.concatenate([monomer_array_i,monomer_array_p2],axis=0)
    elif structure_type == 3:##隣接8分子について対称性より3分子でエネルギー計算
        monomers_array_4 = np.concatenate([monomer_array_i,monomer_array_p1,monomer_array_p2,monomer_array_t1,monomer_array_t2],axis=0)
    
    for x,y,z,R in monomers_array_4:
        atom = R2atom(R)
        line = '{} {} {} {}\n'.format(atom,x,y,z)     
        xyz_list.append(line)
    
    return xyz_list

def make_xyz(monomer_name,params_dict,structure_type):
    xyzfile_name = ''
    xyzfile_name += monomer_name
    for key,val in params_dict.items():
        if key in ['a','b','z']:
            val = np.round(val,2)
        elif key in ['A1','A2','alpha']:
            val = int(val)
        xyzfile_name += '_{}_{}'.format(key,val)
    return xyzfile_name + f'_{structure_type}.xyz'

def make_gjf_xyz(auto_dir,monomer_name,params_dict,structure_type):
    a = params_dict.get('a',0.0); b = params_dict.get('b',0.0); z = params_dict.get('z',0.0)
    A2 = params_dict.get('A2',0.0); A3 = params_dict.get('alpha',0.0)

    monomer_array_i = get_monomer_xyzR(monomer_name,0,0,0,A2,A3)
    monomer_array_p1 = get_monomer_xyzR(monomer_name,a,0,0,A2,A3)##1,2がb方向
    monomer_array_p2 = get_monomer_xyzR(monomer_name,0,b,2*z,A2,A3)##1,2がb方向
    monomer_array_t1 = get_monomer_xyzR(monomer_name,a/2,b/2,z,A2,-A3)##1,2がb方向
    
    dimer_array_p1 = np.concatenate([monomer_array_i,monomer_array_p1]);dimer_array_p2 = np.concatenate([monomer_array_i,monomer_array_p2])
    dimer_array_t1 = np.concatenate([monomer_array_i,monomer_array_t1])
    
    line_list_dimer_p1 = get_xyzR_lines(dimer_array_p1);line_list_dimer_p2 = get_xyzR_lines(dimer_array_p2)
    line_list_dimer_t1 = get_xyzR_lines(dimer_array_t1)
    
    if structure_type == 1:##隣接8分子について対称性より3分子でエネルギー計算
        gij_xyz_lines = line_list_dimer_p1 + ['\n\n\n']
    elif structure_type == 2:##隣接8分子について対称性より3分子でエネルギー計算
        gij_xyz_lines = line_list_dimer_p2 + ['\n\n\n']
    elif structure_type == 3:##隣接8分子について対称性より3分子でエネルギー計算
        gij_xyz_lines = line_list_dimer_t1 + ['\n\n\n']
    
    file_name = get_file_name_from_dict(monomer_name,params_dict,structure_type)
    os.makedirs(os.path.join(auto_dir,'amber'),exist_ok=True)
    gij_xyz_path = os.path.join(auto_dir,'amber',file_name)
    with open(gij_xyz_path,'w') as f:
        f.writelines(gij_xyz_lines)
    
    return file_name

def get_file_name_from_dict(monomer_name,params_dict,structure_type):
    file_name = ''
    file_name += monomer_name
    for key,val in params_dict.items():
        if key in ['a','b','z']:
            val = val
        elif key in ['A2','alpha']:
            val = int(val)
        file_name += '_{}_{}'.format(key,val)
    return file_name + f'_{structure_type}.mol2'
    
def exec_gjf(auto_dir, monomer_name, params_dict,structure_type,isTest=True):
    xyz_dir = os.path.join(auto_dir,'gaussview')
    xyzfile_name = make_xyz(monomer_name, params_dict,structure_type)
    xyz_path = os.path.join(xyz_dir,xyzfile_name)
    xyz_list = make_xyzfile(monomer_name,params_dict,structure_type)
    with open(xyz_path,'w') as f:
        f.writelines(xyz_list)
    
    file_name = make_gjf_xyz(auto_dir, monomer_name, params_dict,structure_type)
    file_job,log_file_name = get_one_exe(auto_dir,file_name,structure_type)
    if not(isTest):
        subprocess.run(['chmod','+x',file_job])
        subprocess.run([file_job])
    return log_file_name
    
############################################################################################
