##tetracene層内計算
"""
python -m auto_opt.amber.driver_gene --auto-dir runs/PFA_test --monomer-name PFA --num-nodes 2 --isTest
"""
import pandas as pd
import time
from auto_opt.amber.make_io_gene_asym import exec_gjf ##計算した点のxyzfileを出す
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
        df_E = pd.DataFrame(columns = ['alpha','a','b','z','E','E1','E2','E3','status'])##いじる
        df_E.to_csv(auto_csv_path,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    auto_csv_path1 = os.path.join(auto_dir,'step1_1.csv')
    if not os.path.exists(auto_csv_path1):        
        df_E_1 = pd.DataFrame(columns = ['alpha','a','E1','status','file_name'])##いじる
        df_E_1.to_csv(auto_csv_path1,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    auto_csv_path2 = os.path.join(auto_dir,'step1_2.csv')
    if not os.path.exists(auto_csv_path2):        
        df_E_2 = pd.DataFrame(columns = ['alpha','b','z','E2','status','file_name'])##いじる
        df_E_2.to_csv(auto_csv_path2,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    auto_csv_path3 = os.path.join(auto_dir,'step1_3.csv')
    if not os.path.exists(auto_csv_path3):        
        df_E_3 = pd.DataFrame(columns = ['alpha','a','b','z','E3','status','file_name'])##いじる
        df_E_3.to_csv(auto_csv_path3,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく
    
    auto_csv_path3 = os.path.join(auto_dir,'step1_4.csv')
    if not os.path.exists(auto_csv_path3):        
        df_E_3 = pd.DataFrame(columns = ['alpha','a','b','z','E4','status','file_name'])##いじる
        df_E_3.to_csv(auto_csv_path3,index=False)##step3を二段階でやる場合二段階目ではinitをやらないので念のためmainにも組み込んでおく

    os.chdir(os.path.join(args.auto_dir,'amber'))
    isOver = False
    while not(isOver):
        #check
        isOver = listen(auto_dir,args.monomer_name,args.num_nodes,args.isTest)##argsの中身を取る
        time.sleep(0.1)

def listen(auto_dir,monomer_name,num_nodes,isTest):##args自体を引数に取るか中身をばらして取るかの違い
    fixed_param_keys = ['alpha'];opt_param_keys_1 = ['a'];opt_param_keys_2 = ['b','z']
    
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
            E1=np.round(E1,6)
            df_E_1.loc[idx, ['E1','status']] = [round(E1,6),'Done']
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
            E2=np.round(E2,6)
            df_E_2.loc[idx, ['E2', 'status']] = [round(E2,6), 'Done']
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
            E3=np.round(E3,6)
            df_E_3.loc[idx, ['E3', 'status']] = [round(E3,6), 'Done']
            df_E_3.to_csv(auto_csv_3, index=False)  # Updated to auto_csv_3
              #  after one iteration

    auto_csv = os.path.join(auto_dir,'step1.csv')
    df_E = pd.read_csv(auto_csv)
    df_prg = df_E.loc[df_E['status']=='InProgress',fixed_param_keys+opt_param_keys_1+opt_param_keys_2]

    auto_csv_4 = os.path.join(auto_dir, 'step1_4.csv')
    df_E_4 = pd.read_csv(auto_csv_4)
    df_prg_4 = df_E_4.loc[df_E_4['status'] == 'InProgress', fixed_param_keys + opt_param_keys_1 + opt_param_keys_2 + ['file_name']]
    len_prg_4 = len(df_prg_4)

    for idx, row in df_prg_4.iterrows():
        params_dict4_ = row[fixed_param_keys + opt_param_keys_1 + opt_param_keys_2 + ['file_name']].to_dict()
        file_name4=params_dict4_['file_name']##辞書をつくってそこにopt_1とopt_2でファイル名作成
        log_filepath4 = os.path.join(*[auto_dir, 'amber', file_name4])
        if not (os.path.exists(log_filepath4)):
            continue
        E_list3 = amber_get_E(log_filepath4)
        if len(E_list4) != 1:
            continue
        else:
            len_prg_4 -= 1
            E4 = float(E_list4[0]) -2*E_mono # Updated to E3
            E4=np.round(E4,6)
            df_E_4.loc[idx, ['E4', 'status']] = [round(E4,6), 'Done']
            df_E_4.to_csv(auto_csv_4, index=False)  # Updated to auto_csv_3
              #  after one iteration

    auto_csv = os.path.join(auto_dir,'step1.csv')
    df_E = pd.read_csv(auto_csv)
    df_prg = df_E.loc[df_E['status']=='InProgress',fixed_param_keys+opt_param_keys_1+opt_param_keys_2]
    
    for idx,row in df_prg.iterrows():
        params_dict1_ = row[fixed_param_keys + opt_param_keys_1].to_dict()
        params_dict2_ = row[fixed_param_keys + opt_param_keys_2].to_dict()
        params_dict3_ = row[fixed_param_keys + opt_param_keys_1 + opt_param_keys_2].to_dict()
        params_dict4_ = row[fixed_param_keys + opt_param_keys_1 + opt_param_keys_2].to_dict()
        s1=filter_df(df_E_1, params_dict1_);s2=filter_df(df_E_2, params_dict2_);s3=filter_df(df_E_3, params_dict3_);s4=filter_df(df_E_4, params_dict4_)#['file_name']
        s1=s1[s1['status']=='Done'];s2=s2[s2['status']=='Done'];s3=s3[s3['status']=='Done'];s4=s4[s4['status']=='Done']
    
        if (len(s1) == 0) or (len(s2) == 0) or (len(s3) == 0) or (len(s4) == 0):
            continue
        else:
            E1 = s1['E1'].values.tolist()[0]
            E2 = s2['E2'].values.tolist()[0]
            E3 = s3['E3'].values.tolist()[0]
            E4 = s4['E3'].values.tolist()[0]
            
            E=2*(E1+E2+E3+E4)
            df_E.loc[idx, ['E','E1','E2','E3','E4','status']] = [round(E,6),round(E1,6),round(E2,6),round(E3,6),round(E4,6),'Done']
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
                
                ## 4の実行　##
                auto_csv_4 = os.path.join(auto_dir,'step1_4.csv');df_E_4 = pd.read_csv(auto_csv_4)
                df_sub_4 = filter_df(df_E_4, params_dict3)
                if len(df_sub_4) == 0:
                    file_name = exec_gjf(auto_dir, monomer_name, {**params_dict4},  structure_type=4,isTest=isTest)
                    df_newline_4 = pd.Series({**params_dict3,'E4':0.,'status':'InProgress','file_name':file_name})
                    df_E_new_4=pd.concat([df_E_4,df_newline_4.to_frame().T],axis=0,ignore_index=True);df_E_new_4.to_csv(auto_csv_4,index=False)
    
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_init_params_done = filter_df(df_init_params,{'status':'Done'})
    isOver = True if len(df_init_params_done)==len(df_init_params) else False
    return isOver

def check_calc_status(auto_dir,params_dict):
    df_E= pd.read_csv(os.path.join(auto_dir,'step1.csv'))
    if len(df_E)==0:
        return False
    df_E_filtered = filter_df(df_E, params_dict)
    df_E_filtered = df_E_filtered.reset_index(drop=True)
    try:
        status = get_values_from_df(df_E_filtered,0,'status')
        return status=='Done'
    except KeyError:
        return False

def get_params_dict(auto_dir, num_nodes):
    """
    前提:
        step1_init_params.csvとstep1.csvがauto_dirの下にある
    """
    init_params_csv=os.path.join(auto_dir, 'step1_init_params.csv')
    df_init_params = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step1.csv'))
    df_init_params_inprogress = df_init_params[df_init_params['status']=='InProgress']
    fixed_param_keys = ['alpha'];opt_param_keys_1 = ['a'];opt_param_keys_2 = ['b','z']
    
    #最初の立ち上がり時
    if len(df_init_params_inprogress) < num_nodes:
        #print(1)
        df_init_params_notyet = df_init_params[df_init_params['status']=='NotYet']
        for index in df_init_params_notyet.index:
            df_init_params = update_value_in_df(df_init_params,index,'status','InProgress')
            df_init_params.to_csv(init_params_csv,index=False)
            params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys_1+opt_param_keys_2].to_dict()
            return [params_dict]
    dict_matrix=[]
    for index in df_init_params_inprogress.index:##こちら側はinit_params内のある業に関する探索が終わった際の新しい行での探索を開始するもの ###ここを改良すればよさそう
        df_init_params = pd.read_csv(init_params_csv)
        init_params_dict = df_init_params.loc[index,fixed_param_keys+opt_param_keys_1+opt_param_keys_2].to_dict()
        fixed_params_dict = df_init_params.loc[index,fixed_param_keys].to_dict()
        isDone, opt_params_matrix = get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict)
        if isDone:
            opt_params_dict={'a':np.round(opt_params_matrix[0][0],1),'b':np.round(opt_params_matrix[0][1],1),'z':np.round(opt_params_matrix[0][2],1)}
            # df_init_paramsのstatusをupdate
            df_init_params = update_value_in_df(df_init_params,index,'status','Done')
            if np.max(df_init_params.index) < index+1:##もうこれ以上は新しい計算は進まない
                status = 'Done'
            else:
                status = get_values_from_df(df_init_params,index+1,'status')
            df_init_params.to_csv(init_params_csv,index=False)
            
            if status=='NotYet':##計算が始まっていないものがあったらこの時点で開始する　ここでダメでもまた直にlistenでgrt_params_dictまでいけば新しいのが始まる            
                opt_params_dict = get_values_from_df(df_init_params,index+1,fixed_param_keys+opt_param_keys_1+opt_param_keys_2)
                df_init_params = update_value_in_df(df_init_params,index+1,'status','InProgress')
                df_init_params.to_csv(init_params_csv,index=False)
                dict_matrix.append({**fixed_params_dict,**opt_params_dict})
            else:
                continue

        else:
            for i in range(len(opt_params_matrix)):
                opt_params_dict={'a':np.round(opt_params_matrix[i][0],1),'b':np.round(opt_params_matrix[i][1],1),'z':np.round(opt_params_matrix[i][2],1)}
                d={**fixed_params_dict,**opt_params_dict}
                dict_matrix.append(d)
                    #print(d)
    return dict_matrix
        
def get_opt_params_dict(df_cur, init_params_dict,fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    a_init_prev = init_params_dict['a'];b_init_prev = init_params_dict['b'];z_init_prev=init_params_dict['z']
    while True:
        E_list=[];xyz_list=[]
        para_list=[]
        for a in [a_init_prev-0.1,a_init_prev,a_init_prev+0.1]:
                    for b in [b_init_prev-0.1,b_init_prev,b_init_prev+0.1]:
                                for z in [z_init_prev]:   
                                    a = np.round(a,1);b = np.round(b,1);z=np.round(z,1)
                                    df_val_xyz = df_val[(df_val['a']==a)&(df_val['b']==b)&(df_val['z']==z)&(df_val['status']=='Done')]
                                    if len(df_val_xyz)==0:
                                        para_list.append([a,b,z])
                                        continue
                                    xyz_list.append([a,b,z]);E_list.append(df_val_xyz['E'].values[0])
        if len(para_list) != 0:
            return False,para_list
        a_init,b_init,z_init = xyz_list[np.argmin(np.array(E_list))]
        if a_init==a_init_prev and b_init==b_init_prev and z_init==z_init_prev:
            return True,[[a_init,b_init,z_init]]
        else:
            a_init_prev=a_init;b_init_prev=b_init;z_init_prev=z_init

def get_values_from_df(df,index,key):
    return df.loc[index,key]

def update_value_in_df(df,index,key,value):
    df.loc[index,key]=value
    return df

def filter_df(df, dict_filter):
    for k, v in dict_filter.items():
        if type(v)==str:
            df=df[df[k]==v]
        else:
            df=df[df[k]==v]
    df_filtered=df
    return df_filtered

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
