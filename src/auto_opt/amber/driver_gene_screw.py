#!/usr/bin/env python3
##tetracene層内計算
"""
python -m auto_opt.amber.driver_gene --auto-dir runs/PFA_test --monomer-name PFA --num-nodes 2 --isTest
"""
import pandas as pd
import time
from auto_opt.amber.make_io_gene_screw import exec_gjf
from auto_opt.utils import amber_get_E
import argparse
import numpy as np
import shutil
import subprocess
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
AMBER_REF = DATA / "amber_ref"
RES = Path(__file__).resolve().parent / "resources"

# z を fixed_param_keys に移動し、最適化の対象から外す
fixed_param_keys = ['alpha', 'beta', 'z']
opt_param_keys_1 = ['a']
opt_param_keys_2 = ['b']
opt_param_keys_3 = ['a', 'bt1']
opt_param_keys_4 = ['a', 'bt2']

def _prepare_amber_resources(auto_dir: str):
    amber_dir = Path(auto_dir) / "amber"
    amber_dir.mkdir(parents=True, exist_ok=True)
    src = RES / "FF_calc.in"
    if src.exists():
        shutil.copy2(src, amber_dir / "FF_calc.in")
    
def main_process(args):
    auto_dir = str(Path(args.auto_dir).resolve()) 
    _prepare_amber_resources(auto_dir)
    os.makedirs(os.path.join(auto_dir,'amber'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)
    
    auto_csv_path = os.path.join(auto_dir,'step1.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns = ['alpha','beta','z','a','b','bt1','bt2','E','E1','E2','E3','E4','status'])
        df_E.to_csv(auto_csv_path,index=False)

    auto_csv_path1 = os.path.join(auto_dir,'step1_1.csv')
    if not os.path.exists(auto_csv_path1):        
        df_E_1 = pd.DataFrame(columns = fixed_param_keys + opt_param_keys_1 + ['E1','status','file_name'])
        df_E_1.to_csv(auto_csv_path1,index=False)

    auto_csv_path2 = os.path.join(auto_dir,'step1_2.csv')
    if not os.path.exists(auto_csv_path2):        
        df_E_2 = pd.DataFrame(columns = fixed_param_keys + opt_param_keys_2 + ['E2','status','file_name'])
        df_E_2.to_csv(auto_csv_path2,index=False)

    auto_csv_path3 = os.path.join(auto_dir,'step1_3.csv')
    if not os.path.exists(auto_csv_path3):        
        df_E_3 = pd.DataFrame(columns = fixed_param_keys + opt_param_keys_3 + ['E3','status','file_name'])
        df_E_3.to_csv(auto_csv_path3,index=False)
    
    auto_csv_path4 = os.path.join(auto_dir,'step1_4.csv')
    if not os.path.exists(auto_csv_path4):        
        df_E_4 = pd.DataFrame(columns = fixed_param_keys + opt_param_keys_4 + ['E4','status','file_name'])
        df_E_4.to_csv(auto_csv_path4,index=False)

    os.chdir(os.path.join(args.auto_dir,'amber'))
    isOver = False
    while not(isOver):
        isOver = listen(auto_dir,args.monomer_name,args.num_nodes,args.isTest)
        time.sleep(0.1)

def listen(auto_dir,monomer_name,num_nodes,isTest):
    mono_file = str(AMBER_REF / f'{monomer_name}_HF_esp_gaff2.out')
    E_mono=amber_get_E(mono_file)[0]
    
    # ---------------- 1のチェック ----------------
    auto_csv_1 = os.path.join(auto_dir,'step1_1.csv');df_E_1 = pd.read_csv(auto_csv_1)
    df_prg_1 = df_E_1.loc[df_E_1['status']=='InProgress',fixed_param_keys+opt_param_keys_1+['file_name']]
    for idx, row in df_prg_1.iterrows():
        file_name1=row['file_name']
        log_filepath1 = os.path.join(*[auto_dir,'amber',file_name1])
        if not(os.path.exists(log_filepath1)): continue
        E_list1=amber_get_E(log_filepath1)
        if len(E_list1)==1 :
            E1 = np.round(float(E_list1[0])-2*E_mono, 4)
            df_E_1.loc[idx, ['E1','status']] = [E1,'Done']
            df_E_1.to_csv(auto_csv_1,index=False)
    
    # ---------------- 2のチェック ----------------
    auto_csv_2 = os.path.join(auto_dir,'step1_2.csv'); df_E_2 = pd.read_csv(auto_csv_2)
    df_prg_2 = df_E_2.loc[df_E_2['status']=='InProgress', fixed_param_keys+opt_param_keys_2+['file_name']]
    for idx, row in df_prg_2.iterrows():
        file_name2=row['file_name']
        log_filepath2 = os.path.join(*[auto_dir, 'amber', file_name2])
        if not(os.path.exists(log_filepath2)): continue
        E_list2 = amber_get_E(log_filepath2)
        if len(E_list2) == 1:
            E2 = np.round(float(E_list2[0]) -2*E_mono, 4)
            df_E_2.loc[idx, ['E2', 'status']] = [E2, 'Done']
            df_E_2.to_csv(auto_csv_2, index=False)
    
    # ---------------- 3のチェック ----------------
    auto_csv_3 = os.path.join(auto_dir, 'step1_3.csv'); df_E_3 = pd.read_csv(auto_csv_3)
    df_prg_3 = df_E_3.loc[df_E_3['status'] == 'InProgress', fixed_param_keys+opt_param_keys_3+['file_name']]
    for idx, row in df_prg_3.iterrows():
        file_name3=row['file_name']
        log_filepath3 = os.path.join(*[auto_dir, 'amber', file_name3])
        if not (os.path.exists(log_filepath3)): continue
        E_list3 = amber_get_E(log_filepath3)
        if len(E_list3) == 1:
            E3 = np.round(float(E_list3[0]) -2*E_mono, 4)
            df_E_3.loc[idx, ['E3', 'status']] = [E3, 'Done']
            df_E_3.to_csv(auto_csv_3, index=False)
    
    # ---------------- 4のチェック ----------------
    auto_csv_4 = os.path.join(auto_dir, 'step1_4.csv'); df_E_4 = pd.read_csv(auto_csv_4)
    df_prg_4 = df_E_4.loc[df_E_4['status'] == 'InProgress', fixed_param_keys+opt_param_keys_4+['file_name']]
    for idx, row in df_prg_4.iterrows():
        file_name4=row['file_name']
        log_filepath4 = os.path.join(*[auto_dir, 'amber', file_name4])
        if not (os.path.exists(log_filepath4)): continue
        E_list4 = amber_get_E(log_filepath4)
        if len(E_list4) == 1:
            E4 = np.round(float(E_list4[0]) -2*E_mono, 4)
            df_E_4.loc[idx, ['E4', 'status']] = [E4, 'Done']
            df_E_4.to_csv(auto_csv_4, index=False)

    # ---------------- Eの合算 ----------------
    auto_csv = os.path.join(auto_dir,'step1.csv'); df_E = pd.read_csv(auto_csv)
    df_prg = df_E.loc[df_E['status']=='InProgress']
    
    for idx,row in df_prg.iterrows():
        p1 = row[fixed_param_keys + opt_param_keys_1].to_dict()
        p2 = row[fixed_param_keys + opt_param_keys_2].to_dict()
        p3 = row[fixed_param_keys + opt_param_keys_3].to_dict()
        p4 = row[fixed_param_keys + opt_param_keys_4].to_dict()
        
        s1=filter_df(df_E_1, p1); s2=filter_df(df_E_2, p2)
        s3=filter_df(df_E_3, p3); s4=filter_df(df_E_4, p4)
        s1=s1[s1['status']=='Done']; s2=s2[s2['status']=='Done']
        s3=s3[s3['status']=='Done']; s4=s4[s4['status']=='Done']
    
        if (len(s1)==0) or (len(s2)==0) or (len(s3)==0) or (len(s4)==0):
            continue
        else:
            E1, E2, E3, E4 = s1['E1'].values[0], s2['E2'].values[0], s3['E3'].values[0], s4['E4'].values[0]
            E = 2*E1 + 2*E2 + 2*E3 + 2*E4
            df_E.loc[idx, ['E','E1','E2','E3','E4','status']] = [round(E,4),round(E1,4),round(E2,4),round(E3,4),round(E4,4),'Done']
            df_E.to_csv(auto_csv,index=False)
    
    # ---------------- 次の計算の投入 ----------------
    dict_matrix = get_params_dict(auto_dir,num_nodes)
    if len(dict_matrix)!=0:
        for params_dict in dict_matrix:
            params_dict1 = {k: v for k, v in params_dict.items() if k in fixed_param_keys + opt_param_keys_1}
            params_dict2 = {k: v for k, v in params_dict.items() if k in fixed_param_keys + opt_param_keys_2}
            params_dict3 = {k: v for k, v in params_dict.items() if k in fixed_param_keys + opt_param_keys_3}
            params_dict4 = {k: v for k, v in params_dict.items() if k in fixed_param_keys + opt_param_keys_4}
            
            alreadyCalculated = check_calc_status(auto_dir, params_dict)
            if not alreadyCalculated:
                df_E= pd.read_csv(os.path.join(auto_dir,'step1.csv'))
                df_E_filtered = filter_df(df_E, params_dict)
                if len(df_E_filtered) == 0:
                    df_newline = pd.Series({**params_dict,'E':0.,'E1':0.,'E2':0.,'E3':0.,'E4':0.,'status':'InProgress'})
                    df_E_new=pd.concat([df_E,df_newline.to_frame().T],axis=0,ignore_index=True)
                    df_E_new.to_csv(auto_csv,index=False)
                
                # 1
                auto_csv_1 = os.path.join(auto_dir,'step1_1.csv'); df_E_1 = pd.read_csv(auto_csv_1)
                if len(filter_df(df_E_1, params_dict1)) == 0:
                    file_name = exec_gjf(auto_dir, monomer_name, params_dict1, structure_type=1, isTest=isTest)
                    df_E_new_1 = pd.concat([df_E_1, pd.Series({**params_dict1,'E1':0.,'status':'InProgress','file_name':file_name}).to_frame().T], ignore_index=True)
                    df_E_new_1.to_csv(auto_csv_1,index=False)

                # 2
                auto_csv_2 = os.path.join(auto_dir,'step1_2.csv'); df_E_2 = pd.read_csv(auto_csv_2)
                if len(filter_df(df_E_2, params_dict2)) == 0:
                    file_name = exec_gjf(auto_dir, monomer_name, params_dict2, structure_type=2, isTest=isTest)
                    df_E_new_2 = pd.concat([df_E_2, pd.Series({**params_dict2,'E2':0.,'status':'InProgress','file_name':file_name}).to_frame().T], ignore_index=True)
                    df_E_new_2.to_csv(auto_csv_2,index=False)
                    
                # 3
                auto_csv_3 = os.path.join(auto_dir,'step1_3.csv'); df_E_3 = pd.read_csv(auto_csv_3)
                if len(filter_df(df_E_3, params_dict3)) == 0:
                    file_name = exec_gjf(auto_dir, monomer_name, params_dict3, structure_type=3, isTest=isTest)
                    df_E_new_3 = pd.concat([df_E_3, pd.Series({**params_dict3,'E3':0.,'status':'InProgress','file_name':file_name}).to_frame().T], ignore_index=True)
                    df_E_new_3.to_csv(auto_csv_3,index=False)
                
                # 4
                auto_csv_4 = os.path.join(auto_dir,'step1_4.csv'); df_E_4 = pd.read_csv(auto_csv_4)
                if len(filter_df(df_E_4, params_dict4)) == 0:
                    file_name = exec_gjf(auto_dir, monomer_name, params_dict4, structure_type=4, isTest=isTest)
                    df_E_new_4 = pd.concat([df_E_4, pd.Series({**params_dict4,'E4':0.,'status':'InProgress','file_name':file_name}).to_frame().T], ignore_index=True)
                    df_E_new_4.to_csv(auto_csv_4,index=False)

    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    isOver = len(filter_df(df_init_params,{'status':'Done'})) == len(df_init_params)
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step1.csv'))
    if len(df_E)==0: return False
    df_E_filtered = filter_df(df_E, params_dict).reset_index(drop=True)
    if len(df_E_filtered) > 0:
        return df_E_filtered.loc[0, 'status'] == 'Done'
    return False

def get_params_dict(auto_dir, num_nodes):
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step1.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    
    # Initパラメータ全てを使う (ここではzも含まれる)
    all_keys = ['alpha','beta','z','a','b','bt1','bt2']
    
    if len(df_init_params_inprogress) < num_nodes:
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            return [df_init_params.loc[index, all_keys].to_dict()]
            
    dict_matrix=[]
    for index in df_init_params_inprogress.index:
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index, all_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index, fixed_param_keys].to_dict()
        
        isDone, opt_params_matrix = get_opt_params_dict(df_cur, init_params_dict, fixed_params_dict)
        if isDone:
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':
                next_dict = df_init_params.loc[index+1, all_keys].to_dict()
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                dict_matrix.append(next_dict)
        else:
            for p in opt_params_matrix:
                dict_matrix.append({**fixed_params_dict, **p})
    return dict_matrix
        
def get_opt_params_dict(df_cur, init_params_dict, fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    
    a_prev = init_params_dict['a']
    bt1_prev = init_params_dict['bt1']
    bt2_prev = init_params_dict['bt2']
    
    while True:
        E_list=[]; xyz_list=[]; para_list=[]
        
        # 3Dグリッドの探索（a, bt1, bt2）※ z は fixed_params_dict に含まれるため変更しない
        for a in [a_prev-0.1, a_prev, a_prev+0.1]:
            for bt1 in [bt1_prev-0.1, bt1_prev, bt1_prev+0.1]:
                for bt2 in [bt2_prev-0.1, bt2_prev, bt2_prev+0.1]:
                        
                    a = np.round(a, 1); bt1 = np.round(bt1, 1); bt2 = np.round(bt2, 1)
                    b = np.round(bt1 + bt2, 1)
                    
                    df_val_xyz = df_val[(df_val['a']==a) & (df_val['bt1']==bt1) & (df_val['bt2']==bt2) & (df_val['status']=='Done')]
                    
                    if len(df_val_xyz)==0:
                        para_list.append({'a':a, 'b':b, 'bt1':bt1, 'bt2':bt2})
                        continue
                    
                    xyz_list.append({'a':a, 'b':b, 'bt1':bt1, 'bt2':bt2})
                    E_list.append(df_val_xyz['E'].values[0])
                        
        if len(para_list) != 0:
            return False, para_list
            
        best_idx = np.argmin(np.array(E_list))
        best = xyz_list[best_idx]
        
        if best['a']==a_prev and best['bt1']==bt1_prev and best['bt2']==bt2_prev:
            return True, [best]
        else:
            a_prev, bt1_prev, bt2_prev = best['a'], best['bt1'], best['bt2']

def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def filter_df(df, dict_filter):
    for k, v in dict_filter.items():
        # floatの丸め誤差回避
        if isinstance(v, float):
            df = df[np.isclose(df[k], v, atol=1e-5)]
        else:
            df = df[df[k]==v]
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--auto-dir',type=str,help='path to dir which includes amber, gaussview and csv')
    parser.add_argument('--monomer-name',type=str,help='monomer name')
    parser.add_argument('--num-nodes',type=int,help='num nodes')
    args = parser.parse_args()

    print("----main process----")
    main_process(args)
    print("----finish process----")