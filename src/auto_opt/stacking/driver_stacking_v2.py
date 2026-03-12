#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python auto_opt.stacking.driver_stacking_v2.py --auto-dir runs/ANT_stakcing_test6 --monomer-name ANT --num-nodes 10 --max-2 3
"""

import os
os.environ['HOME'] ='/home/miyoshi'
import pandas as pd
import time
import argparse
import numpy as np
import subprocess
from pathlib import Path

from auto_opt.stacking.make_io_stacking import get_14_pairs_xyzR, get_xyzR_lines
from auto_opt.amber.make_io_gene_phi_asym_anti import _guess_mol2_path
from auto_opt.utils import amber_get_E

def exec_amber_job(auto_dir, monomer_name, params_dict, machine_type, isTest=False):
    """
    14ペア分のmol2とtleap入力を作り、1つのジョブスクリプトとしてバックグラウンド実行する
    """
    out_dir = os.path.join(auto_dir, 'amber')
    os.makedirs(out_dir, exist_ok=True)
    
    cx_str = f"{params_dict['cx']:.2f}".replace('.', 'p').replace('-', 'm')
    cy_str = f"{params_dict['cy']:.2f}".replace('.', 'p').replace('-', 'm')
    cz_str = f"{params_dict['cz']:.2f}".replace('.', 'p').replace('-', 'm')
    base_file_name = f"{monomer_name}_cx{cx_str}_cy{cy_str}_cz{cz_str}" #[my_memo] ファイル名に.(ドット)や-(ハイフン)が入らないように変形
    
    pairs = get_14_pairs_xyzR(monomer_name, params_dict)
    monomer_mol2 = str(_guess_mol2_path(monomer_name))
    frcmod_name = f"{monomer_name}_gaff2.frcmod"
    
    cmds = [f'if [ ! -f "{frcmod_name}" ]; then parmchk2 -s gaff2 -i "{monomer_mol2}" -f mol2 -o "{frcmod_name}"; fi']
    
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
        cmds.append(f"sander -O -i FF_calc.in -o {file_name}.out -p {file_name}.prmtop -c {file_name}.inpcrd -r {file_name}.rst7")
        
    job_file = os.path.join(out_dir, f"job_{base_file_name}.sh")
    with open(job_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {out_dir}\n")
        f.write("source ~/anaconda3/etc/profile.d/conda.sh\n")
        f.write("conda activate amber\n")
        for cmd in cmds:
            f.write(cmd + "\n")
            
    os.chmod(job_file, 0o755)
    
    if not isTest:
        # バックグラウンドでローカル実行 (計算ノード内で並列を回すため)
        subprocess.Popen([job_file], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return base_file_name

def read_14pairs_amber(auto_dir, base_file_name):
    """
    14個のAMBER計算がすべて完了しているか確認し、エネルギーのリストを返す
    完了していなければ空リストを返す
    """
    E_list = []
    for i in range(14):
        out_file = os.path.join(auto_dir, 'amber', f"{base_file_name}_p{i}.out")
        if not os.path.exists(out_file):
            return []
        try:
            e = amber_get_E(out_file)[0]
            E_list.append(float(e))
        except:
            return [] # 計算中 または エラー
    return E_list


def main_process(args):
    args.auto_dir = os.path.abspath(args.auto_dir)
    auto_dir = args.auto_dir
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'amber'), exist_ok=True)
    
    # FF_calc.in を amber ディレクトリにコピー
    ff_src = Path(__file__).resolve().parent / "resources" / "FF_calc.in"
    if ff_src.exists():
        import shutil
        shutil.copy2(ff_src, os.path.join(auto_dir, 'amber', 'FF_calc.in'))
    
    auto_csv_path = os.path.join(auto_dir,'step1.csv')
    if not os.path.exists(auto_csv_path):
        cols = ['cx','cy','cz','alpha1','alpha2','a','b','z','phi','E',
                'E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14',
                'status','machine_type','file_name']
        df_E = pd.DataFrame(columns=cols)
        df_E.to_csv(auto_csv_path, index=False)
        print("made step1.csv")
        
    os.chdir(os.path.join(args.auto_dir,'amber'))
    isOver = False
    while True:
        if isOver:
            break
        isOver = listen(args.auto_dir, args.monomer_name, args.num_nodes, args.max_2, args.isTest)
        time.sleep(1)
    

def listen(auto_dir, monomer_name, num_nodes, max_2, isTest):
    maxnum_machine2 = max_2
    fixed_param_keys = ['alpha1','alpha2','a','b','z','phi']
    opt_param_keys = ['cx','cy','cz']
    
    auto_csv = os.path.join(auto_dir,'step1.csv')
    df_E_1 = pd.read_csv(auto_csv)
    
    df_prg_1 = df_E_1.loc[df_E_1['status']=='InProgress', fixed_param_keys+opt_param_keys+['machine_type','file_name']]
    len_prg_1 = len(df_prg_1)
    
    for idx, row in df_prg_1.iterrows():
        params_dict1_ = row[fixed_param_keys + opt_param_keys + ['file_name']].to_dict()
        file_name1 = params_dict1_['file_name']
        
        # AMBERの14ファイルの終了確認
        E_list1 = read_14pairs_amber(auto_dir, file_name1)
        
        if len(E_list1) != 14:
            continue
        else:
            len_prg_1 -= 1
            E_total = sum(E_list1)
            
            update_cols = ['E', 'E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12','E13','E14','status']
            update_vals = [E_total] + E_list1 + ['Done']
            
            df_E_1.loc[idx, update_cols] = update_vals
            df_E_1.to_csv(auto_csv, index=False)
            break
    
    df_qw_1 = df_E_1[df_E_1['status'] == 'qw']
    len_queue = len_prg_1
    len_qw_1 = len(df_qw_1)
    margin = num_nodes - len_queue

    df_inpr_1 = df_E_1.loc[df_E_1['status']=='InProgress']
    machine_counts_1 = df_inpr_1['machine_type'].value_counts().to_dict()
    machine_counts_1.setdefault(1, 0)
    machine_counts_1.setdefault(2, 0)
    num_machine2 = machine_counts_1.get(2, 0)

    if len_qw_1 > 0 and margin > 0:
        for index, row in df_qw_1.iterrows():
            if margin == 0:
                break
            params_dict = row[fixed_param_keys + opt_param_keys].to_dict()
            if num_machine2 >= maxnum_machine2:
                machine_type = 1             
            else:
                machine_type = 2
                num_machine2 += 1
            file_name = exec_amber_job(auto_dir, monomer_name, {**params_dict}, machine_type, isTest=isTest)
            len_queue += 1
            margin -= 1
            df_E_1.at[index, 'machine_type'] = machine_type
            df_E_1.at[index, 'status'] = 'InProgress'
            df_E_1.at[index, 'file_name'] = file_name
        df_E_1.to_csv(auto_csv, index=False)
    
    dict_matrix = get_params_dict(auto_dir, num_nodes)
    if len(dict_matrix) != 0:
        for i in range(len(dict_matrix)):
            params_dict = dict_matrix[i]
            params_dict1 = {k: v for k, v in params_dict.items() if (k in fixed_param_keys) or (k in opt_param_keys)}
            alreadyCalculated = check_calc_status(auto_dir, params_dict)
            
            if not(alreadyCalculated):
                auto_csv = os.path.join(auto_dir,'step1.csv')
                df_E_1 = pd.read_csv(auto_csv)
                df_sub_1 = filter_df(df_E_1, params_dict1)
                
                if len(df_sub_1) == 0:
                    isAvailable = len_queue < num_nodes
                    if isAvailable:
                        if num_machine2 >= maxnum_machine2:
                            machine_type = 1   
                        else:
                            machine_type = 2
                            num_machine2 += 1
                        file_name = exec_amber_job(auto_dir, monomer_name, {**params_dict1}, machine_type, isTest=isTest)
                        len_queue += 1
                        df_newline_1 = pd.Series({**params_dict1,'E':0.,'machine_type':machine_type,'status':'InProgress','file_name':file_name})
                        df_E_new_1 = pd.concat([df_E_1, df_newline_1.to_frame().T], axis=0, ignore_index=True)
                        df_E_new_1.to_csv(auto_csv, index=False)
                        time.sleep(1)
                    else:
                        file_name = exec_amber_job(auto_dir, monomer_name, {**params_dict1}, machine_type, isTest=True)
                        df_newline_1 = pd.Series({**params_dict1,'E':0.,'machine_type':1,'status':'qw','file_name':file_name})
                        df_E_new_1 = pd.concat([df_E_1, df_newline_1.to_frame().T], axis=0, ignore_index=True)
                        df_E_new_1.to_csv(auto_csv, index=False)
    
    init_params_csv = os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params, {'status':'Done'})
    isOver = True if len(df_init_params_done) == len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir, params_dict):
    df_E = pd.read_csv(os.path.join(auto_dir,'step1.csv'))
    if len(df_E) == 0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered, 0, 'status')
        return status == 'Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes):
    init_params_csv = os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step1.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    fixed_param_keys = ['alpha1','alpha2','a','b','z','phi']
    opt_param_keys = ['cx','cy','cz']
    
    if len(df_init_params_inprogress) < num_nodes:
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params, index, 'status', 'InProgress')
            df_init_params.to_csv(init_params_csv, index=False)
            params_dict = df_init_params.loc[index, fixed_param_keys+opt_param_keys].to_dict()
            return [params_dict]
            
    dict_matrix = []
    for index in df_init_params_inprogress.index:
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index, fixed_param_keys+opt_param_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index, fixed_param_keys].to_dict()
        
        isDone, opt_params_matrix = get_opt_params_dict(df_cur, init_params_dict, fixed_params_dict)
        
        if isDone:
            opt_params_dict = {
                'cx': np.round(opt_params_matrix[0][0], 1),
                'cy': np.round(opt_params_matrix[0][1], 1),
                'cz': np.round(opt_params_matrix[0][2], 1)
            }
            df_init_params = update_value_in_df(df_init_params, index, 'status', 'Done')
            
            if np.max(df_init_params.index) < index+1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params, index+1, 'status')
            df_init_params.to_csv(init_params_csv, index=False)
            
            if status == 'NotYet':
                opt_params_dict = get_values_from_df(df_init_params, index+1, fixed_param_keys+opt_param_keys).to_dict()
                df_init_params = update_value_in_df(df_init_params, index+1, 'status', 'InProgress')
                df_init_params.to_csv(init_params_csv, index=False)
                dict_matrix.append({**fixed_params_dict, **opt_params_dict})
            else:
                continue
        else:
            for i in range(len(opt_params_matrix)):
                opt_params_dict = {
                    'cx': np.round(opt_params_matrix[i][0], 1),
                    'cy': np.round(opt_params_matrix[i][1], 1),
                    'cz': np.round(opt_params_matrix[i][2], 1)
                }
                d = {**fixed_params_dict, **opt_params_dict}
                dict_matrix.append(d)
                
    return dict_matrix

def get_opt_params_dict(df_cur, init_params_dict, fixed_params_dict):
    """
    ★ cx, cy を固定し、cz のみを上下 0.1 Å 探索して谷底を見つける1次元最適化
    """
    df_val = filter_df(df_cur, fixed_params_dict)
    
    cx_init_prev = init_params_dict['cx']
    cy_init_prev = init_params_dict['cy']
    cz_init_prev = init_params_dict['cz']
    
    df_val = df_val[(df_val['cx'] == cx_init_prev) & (df_val['cy'] == cy_init_prev)]
    
    while True:
        E_list = []
        xyz_list = []
        para_list = []
        
        # cz のみを -0.1, 0, +0.1 動かす
        for cz in [cz_init_prev - 0.1, cz_init_prev, cz_init_prev + 0.1]:
            cz = np.round(cz, 1)
            df_val_xyz = df_val[(df_val['cz'] == cz) & (df_val['status'] == 'Done')]
            
            if len(df_val_xyz) == 0:
                para_list.append([cx_init_prev, cy_init_prev, cz])
                continue
                
            xyz_list.append([cx_init_prev, cy_init_prev, cz])
            E_list.append(df_val_xyz['E'].values[0])
            
        if len(para_list) != 0:
            return False, para_list
            
        # 3点の中で最もエネルギーが低い位置を特定
        best_idx = np.argmin(np.array(E_list))
        cx_init, cy_init, cz_init = xyz_list[best_idx]
        
        # 中心の cz が一番低ければ谷底到達
        if cz_init == cz_init_prev:
            return True, [[cx_init, cy_init, cz_init]]
        else:
            cz_init_prev = cz_init

def get_values_from_df(df, index, key):
    return df.loc[index, key]

def update_value_in_df(df, index, key, value):
    df.loc[index, key] = value
    return df

def filter_df(df, dict_filter):
    for k, v in dict_filter.items():
        df = df[df[k] == v]
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, help='path to dir which includes amber and csv')
    parser.add_argument('--monomer-name', type=str, help='monomer name')
    parser.add_argument('--num-nodes', type=int, help='num nodes')
    parser.add_argument('--max-2', type=int, help='max nodes')
    args = parser.parse_args()

    print("----main process----")
    main_process(args)
    print("----finish process----")