##tetracene層内計算
"""
python -m auto_opt.amber.driver_gene --auto-dir runs/PFA_test --monomer-name PFA --num-nodes 2 --isTest
"""
import pandas as pd
import time
from auto_opt.amber.make_io_gene_phi import exec_gjf ##計算した点のxyzfileを出す
from auto_opt.utils import amber_get_E, filter_df, check_calc_status, get_values_from_df, update_value_in_df
import argparse
import numpy as np
import shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
AMBER_REF = DATA / "amber_ref"
RES = Path(__file__).resolve().parent / "resources"

fixed_param_keys = ['alpha', 'phi', 'z']   # z は VdW sweep 由来の固定値
opt_param_keys_1 = ['a']
opt_param_keys_2 = ['b']
all_keys         = ['alpha', 'phi', 'z', 'a', 'b']

def _prepare_amber_resources(auto_dir: str):
    amber_dir = Path(auto_dir) / "amber"
    amber_dir.mkdir(parents=True, exist_ok=True)
    # FF_calc.in
    src = RES / "FF_calc.in"
    if src.exists():
        shutil.copy2(src, amber_dir / "FF_calc.in")
    
def main_process(args):
    auto_dir = str(Path(args.auto_dir).resolve()) 
    _prepare_amber_resources(auto_dir)
    os.makedirs(auto_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'amber'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir,'gaussview'), exist_ok=True)
    amber_path=os.path.join(auto_dir,'amber')
    auto_csv_path = os.path.join(auto_dir,'step1.csv')
    if not os.path.exists(auto_csv_path):        
        df_E = pd.DataFrame(columns=all_keys + ['E', 'E1', 'E2', 'E3', 'status'])
        df_E.to_csv(auto_csv_path, index=False)

    for n, opt_keys, e_col in [
        (1, opt_param_keys_1, 'E1'),
        (2, opt_param_keys_2, 'E2'),
        (3, opt_param_keys_1 + opt_param_keys_2, 'E3'),
    ]:
        path = os.path.join(auto_dir, f'step1_{n}.csv')
        if not os.path.exists(path):
            df = pd.DataFrame(columns=fixed_param_keys + opt_keys + [e_col, 'status', 'file_name'])
            df.to_csv(path, index=False)

    os.chdir(os.path.join(args.auto_dir,'amber'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(auto_dir,args.monomer_name,args.num_nodes,args.isTest)##argsの中身を取る
        time.sleep(0.1)

    from auto_opt.gaussian.extract_minima import extract_minima
    extract_minima(symmetry='glide', auto_dir=auto_dir)

def listen(auto_dir, monomer_name, num_nodes, isTest):
    mono_file = str(AMBER_REF / f'{monomer_name}_HF_esp_gaff2.out')
    E_mono=amber_get_E(mono_file)[0]
    auto_csv_1 = os.path.join(auto_dir,'step1_1.csv');df_E_1 = pd.read_csv(auto_csv_1)
    df_prg_1 = df_E_1.loc[df_E_1['status']=='InProgress',fixed_param_keys+opt_param_keys_1+['file_name']]
    len_prg_1=len(df_prg_1)
    for idx, row in df_prg_1.iterrows():
        params_dict1_ = row[fixed_param_keys + opt_param_keys_1 + ['file_name']].to_dict()
        file_name1=params_dict1_['file_name']##辞書をつくってそこにopt_1とopt_2でファイル名作成
        log_filepath1 = os.path.join(*[auto_dir,'amber',file_name1])
        if not(os.path.exists(log_filepath1)):#logファイルが生成される直前だとまずいので
            continue
        E_list1=amber_get_E(log_filepath1)
        if len(E_list1)!=1 :##get Eの長さは計算した分子の数
            continue
        else:
            len_prg_1-=1
            E1=float(E_list1[0])-2*E_mono##8分子に向けてep1,ep2作成　ep1:b ep2:a
            E1=np.round(E1,4)
            df_E_1.loc[idx, ['E1','status']] = [round(E1,4),'Done']
            df_E_1.to_csv(auto_csv_1,index=False)
            #time.sleep(1)
            #2つ同時に計算終わったりしたらまずいので一個で切る
    
    auto_csv_2 = os.path.join(auto_dir,'step1_2.csv')
    df_E_2 = pd.read_csv(auto_csv_2)
    df_prg_2 = df_E_2.loc[df_E_2['status']=='InProgress', fixed_param_keys+opt_param_keys_2+['file_name']]
    len_prg_2 = len(df_prg_2)

    for idx, row in df_prg_2.iterrows():
        params_dict2_ = row[fixed_param_keys + opt_param_keys_2 + ['file_name']].to_dict()
        file_name2=params_dict2_['file_name']##辞書をつくってそこにopt_1とopt_2でファイル名作成
        log_filepath2 = os.path.join(*[auto_dir, 'amber', file_name2])
        if not(os.path.exists(log_filepath2)):
            continue
        E_list2 = amber_get_E(log_filepath2)
        if len(E_list2) != 1:
            continue
        else:
            len_prg_2 -= 1
            E2 = float(E_list2[0]) -2*E_mono  # Updated to E2
            E2=np.round(E2,4)
            df_E_2.loc[idx, ['E2', 'status']] = [round(E2,4), 'Done']
            df_E_2.to_csv(auto_csv_2, index=False)  # Updated to auto_csv_2
            #time.sleep(1)
              #  after one iteration
    
    auto_csv_3 = os.path.join(auto_dir, 'step1_3.csv')
    df_E_3 = pd.read_csv(auto_csv_3)
    df_prg_3 = df_E_3.loc[df_E_3['status'] == 'InProgress', fixed_param_keys + opt_param_keys_1 + opt_param_keys_2 + ['file_name']]
    len_prg_3 = len(df_prg_3)

    for idx, row in df_prg_3.iterrows():
        params_dict3_ = row[fixed_param_keys + opt_param_keys_1 + opt_param_keys_2 + ['file_name']].to_dict()
        file_name3=params_dict3_['file_name']##辞書をつくってそこにopt_1とopt_2でファイル名作成
        log_filepath3 = os.path.join(*[auto_dir, 'amber', file_name3])
        if not (os.path.exists(log_filepath3)):
            continue
        E_list3 = amber_get_E(log_filepath3)
        if len(E_list3) != 1:
            continue
        else:
            len_prg_3 -= 1
            E3 = float(E_list3[0]) -2*E_mono # Updated to E3
            E3=np.round(E3,4)
            df_E_3.loc[idx, ['E3', 'status']] = [round(E3,4), 'Done']
            df_E_3.to_csv(auto_csv_3, index=False)  # Updated to auto_csv_3
              #  after one iteration

    auto_csv = os.path.join(auto_dir,'step1.csv')
    df_E = pd.read_csv(auto_csv)
    df_prg = df_E.loc[df_E['status']=='InProgress',fixed_param_keys+opt_param_keys_1+opt_param_keys_2]
    
    for idx,row in df_prg.iterrows():
        params_dict1_ = row[fixed_param_keys + opt_param_keys_1].to_dict()
        params_dict2_ = row[fixed_param_keys + opt_param_keys_2].to_dict()
        params_dict3_ = row[fixed_param_keys + opt_param_keys_1 + opt_param_keys_2].to_dict()
        s1=filter_df(df_E_1, params_dict1_);s2=filter_df(df_E_2, params_dict2_);s3=filter_df(df_E_3, params_dict3_)#['file_name']
        s1=s1[s1['status']=='Done'];s2=s2[s2['status']=='Done'];s3=s3[s3['status']=='Done']
    
        if (len(s1) == 0) or (len(s2) == 0) or (len(s3) == 0):
            continue
        else:
            E1 = s1['E1'].values.tolist()[0]
            E2 = s2['E2'].values.tolist()[0]
            E3 = s3['E3'].values.tolist()[0]
            
            E=2*E1+2*E2+4*E3
            df_E.loc[idx, ['E','E1','E2','E3','status']] = [round(E,4),round(E1,4),round(E2,4),round(E3,4),'Done']
            df_E.to_csv(auto_csv,index=False)
            #2つ同時に計算終わったりしたらまずいので一個で切る
    
    dict_matrix = get_params_dict(auto_dir,num_nodes)##更新分を流す a1/HOME/HASEGAWALABz2まで取得
    if len(dict_matrix)!=0:#終わりがまだ見えないなら
        for i in range(len(dict_matrix)):
            params_dict=dict_matrix[i]#print(params_dict)
            params_dict1 = {k: v for k, v in params_dict.items() if (k in fixed_param_keys) or (k in opt_param_keys_1)}
            params_dict2 = {k: v for k, v in params_dict.items() if (k in fixed_param_keys) or (k in opt_param_keys_2)}
            params_dict3 = params_dict
            alreadyCalculated = check_calc_status(auto_dir,params_dict)
            if not(alreadyCalculated):
                df_E= pd.read_csv(os.path.join(auto_dir,'step1.csv'))
                df_E_filtered = filter_df(df_E, params_dict)
                if len(df_E_filtered) == 0:
                    df_newline = pd.Series({**params_dict,'E':0.,'E1':0.,'E2':0.,'E3':0.,'status':'InProgress'})
                    df_E_new=pd.concat([df_E,df_newline.to_frame().T],axis=0,ignore_index=True);df_E_new.to_csv(auto_csv,index=False)
                
                ## 1の実行　##
                auto_csv_1 = os.path.join(auto_dir,'step1_1.csv');df_E_1 = pd.read_csv(auto_csv_1)
                df_sub_1 = filter_df(df_E_1, params_dict1)
                if len(df_sub_1) == 0:
                    file_name = exec_gjf(auto_dir, monomer_name, {**params_dict1}, structure_type=1,isTest=isTest)
                    df_newline_1 = pd.Series({**params_dict1,'E1':0.,'status':'InProgress','file_name':file_name})
                    df_E_new_1=pd.concat([df_E_1,df_newline_1.to_frame().T],axis=0,ignore_index=True);df_E_new_1.to_csv(auto_csv_1,index=False)
                    #time.sleep(0.1)

                ## 2の実行　##
                auto_csv_2 = os.path.join(auto_dir,'step1_2.csv');df_E_2 = pd.read_csv(auto_csv_2)
                df_sub_2 = filter_df(df_E_2, params_dict2)
                if len(df_sub_2) == 0:
                    file_name = exec_gjf(auto_dir, monomer_name, {**params_dict2}, structure_type=2,isTest=isTest)
                    df_newline_2 = pd.Series({**params_dict2,'E2':0.,'status':'InProgress','file_name':file_name})
                    df_E_new_2=pd.concat([df_E_2,df_newline_2.to_frame().T],axis=0,ignore_index=True);df_E_new_2.to_csv(auto_csv_2,index=False)
                    
                ## 3の実行　##
                auto_csv_3 = os.path.join(auto_dir,'step1_3.csv');df_E_3 = pd.read_csv(auto_csv_3)
                df_sub_3 = filter_df(df_E_3, params_dict3)
                if len(df_sub_3) == 0:
                    file_name = exec_gjf(auto_dir, monomer_name, {**params_dict3},  structure_type=3,isTest=isTest)
                    df_newline_3 = pd.Series({**params_dict3,'E3':0.,'status':'InProgress','file_name':file_name})
                    df_E_new_3=pd.concat([df_E_3,df_newline_3.to_frame().T],axis=0,ignore_index=True);df_E_new_3.to_csv(auto_csv_3,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def get_params_dict(auto_dir, num_nodes):
    """
    前提:
        step1_init_params.csvとstep1.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step1.csv'))
    df_inprogress = df_init_params[df_init_params['status'] == 'InProgress']

    if len(df_inprogress) < num_nodes:
        df_notyet = df_init_params[df_init_params['status'] == 'NotYet']
        for index in df_notyet.index:
            df_init_params = update_value_in_df(df_init_params, index, 'status', 'InProgress')
            df_init_params.to_csv(init_params_csv, index=False)
            return [df_init_params.loc[index, all_keys].to_dict()]

    dict_matrix = []
    for index in df_inprogress.index:
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict  = df_init_params.loc[index, all_keys].to_dict()
        fixed_params_dict = df_init_params.loc[index, fixed_param_keys].to_dict()
        isDone, opt_params_matrix = get_opt_params_dict(df_cur, init_params_dict, fixed_params_dict)
        if isDone:
            df_init_params = update_value_in_df(df_init_params, index, 'status', 'Done')
            if np.max(df_init_params.index) < index + 1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params, index + 1, 'status')
            df_init_params.to_csv(init_params_csv, index=False)

            if status == 'NotYet':
                next_params = df_init_params.loc[index + 1, all_keys].to_dict()
                df_init_params = update_value_in_df(df_init_params, index + 1, 'status', 'InProgress')
                df_init_params.to_csv(init_params_csv, index=False)
                dict_matrix.append(next_params)
            else:
                continue
        else:
            for opt in opt_params_matrix:
                dict_matrix.append({**fixed_params_dict,
                                    'a': np.round(opt[0], 1),
                                    'b': np.round(opt[1], 1)})
    return dict_matrix
        
def get_opt_params_dict(df_cur, init_params_dict, fixed_params_dict):
    # fixed_params_dict に z が含まれるので filter_df で z 固定済みの df_val を得る
    df_val = filter_df(df_cur, fixed_params_dict)
    a_prev = init_params_dict['a']
    b_prev = init_params_dict['b']
    while True:
        E_list = []; ab_list = []; para_list = []
        for a in [a_prev - 0.1, a_prev, a_prev + 0.1]:
            for b in [b_prev - 0.1, b_prev, b_prev + 0.1]:
                a = np.round(a, 1); b = np.round(b, 1)
                df_v = df_val[(df_val['a'] == a) & (df_val['b'] == b) & (df_val['status'] == 'Done')]
                if len(df_v) == 0:
                    para_list.append([a, b])
                else:
                    ab_list.append([a, b]); E_list.append(df_v['E'].values[0])
        if para_list:
            return False, para_list
        a_best, b_best = ab_list[np.argmin(E_list)]
        if a_best == a_prev and b_best == b_prev:
            return True, [[a_best, b_best]]
        a_prev, b_prev = a_best, b_best

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
