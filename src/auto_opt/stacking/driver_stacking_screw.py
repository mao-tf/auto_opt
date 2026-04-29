#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['HOME'] ='/home/miyoshi'
import pandas as pd
import time
import argparse
import numpy as np
import subprocess
from pathlib import Path

from auto_opt.stacking.make_io_stacking_screw import get_14_pairs_xyzR, get_xyzR_lines
from auto_opt.amber.make_io_gene_phi_asym_anti import _guess_mol2_path
from auto_opt.utils import amber_get_E

def exec_amber_job(auto_dir, monomer_name, params_dict, isTest=False):
    """
    14ペア分のmol2とtleap入力を作り、1つのジョブスクリプトとしてバックグラウンド実行する
    """
    out_dir = os.path.join(auto_dir, 'amber')
    os.makedirs(out_dir, exist_ok=True)
    
    beta_str = f"{params_dict['beta']:.1f}".replace('.', 'p').replace('-', 'm')
    z_str = f"{params_dict['z']:.1f}".replace('.', 'p').replace('-', 'm')
    cx_str = f"{params_dict['cx']:.2f}".replace('.', 'p').replace('-', 'm')
    cy_str = f"{params_dict['cy']:.2f}".replace('.', 'p').replace('-', 'm')
    cz_str = f"{params_dict['cz']:.2f}".replace('.', 'p').replace('-', 'm')
    base_file_name = f"{monomer_name}_beta{beta_str}_z{z_str}_cx{cx_str}_cy{cy_str}_cz{cz_str}"

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
        subprocess.Popen([job_file], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return base_file_name


def read_14pairs_amber(auto_dir, base_file_name):
    """
    14個のAMBER計算がすべて完了しているか確認し、エネルギーのリストを返す
    """
    E_list = []
    for i in range(13):
        out_file = os.path.join(auto_dir, 'amber', f"{base_file_name}_p{i}.out")
        if not os.path.exists(out_file):
            return []
        try:
            e = amber_get_E(out_file)[0]
            E_list.append(float(e))
        except:
            return [] 
    return E_list


def main_process(args):
    args.auto_dir = os.path.abspath(args.auto_dir)
    auto_dir = args.auto_dir
    os.makedirs(auto_dir, exist_ok=True)
    out_amber_dir = os.path.join(auto_dir, 'amber')
    os.makedirs(out_amber_dir, exist_ok=True)
    
    ff_src = Path(__file__).resolve().parent / "resources" / "FF_calc.in"
    if ff_src.exists():
        import shutil
        shutil.copy2(ff_src, os.path.join(out_amber_dir, 'FF_calc.in'))
    
    init_csv_path = os.path.join(auto_dir, 'step1_init_params.csv')
    auto_csv_path = os.path.join(auto_dir, 'step1.csv')
    
    df_init = pd.read_csv(init_csv_path)
    
    # --- メモリ上での状態管理 ---
    job_queue = []               
    running_jobs = {}            
    completed_czs = {}           
    queued_or_running_czs = {}   
    results_list = []            
    group_status = {}            

    fixed_keys = ['a', 'bt1', 'bt2', 'b', 'z', 'beta', 'alpha1', 'alpha2']
    
    print("🚀 初期タスクのセットアップ中...")
    for _, row in df_init.iterrows():
        beta = np.round(row['beta'], 1)
        cx = np.round(row['cx'], 1)
        cy = np.round(row['cy'], 1)
        cz_vdw = np.round(row['cz'], 1)
        group_key = (beta, cx, cy)  # ← beta をキーに追加！
        
        completed_czs[group_key] = {}
        queued_or_running_czs[group_key] = set()
        group_status[group_key] = 'Working'
        
        fixed_params = row[fixed_keys].to_dict()
        
        for d_cz in [-0.2, -0.1, 0.0, 0.1, 0.2]:
            target_cz = np.round(cz_vdw + d_cz, 1)
            task = {'cx': cx, 'cy': cy, 'cz': target_cz, **fixed_params}
            job_queue.append(task)
            queued_or_running_czs[group_key].add(target_cz)

    print(f"📦 初期キューに {len(job_queue)} 個のタスクを装填完了。非同期ループを開始します。")
    
    last_save_time = time.time()
    
    # 非同期メインループ
    while True:
        finished_files = []
        for file_name, task in running_jobs.items():
            E_list = read_14pairs_amber(auto_dir, file_name)
            if len(E_list) == 13:
                E_total = sum(E_list)
                beta = task['beta']
                cx = task['cx']
                cy = task['cy']
                cz = task['cz']
                
                # 結果をメモリに保存（キーにbetaを含む）
                completed_czs[(beta, cx, cy)][cz] = E_total
                
                result_dict = task.copy()
                result_dict['E'] = E_total
                for i in range(13):
                    result_dict[f'E{i+1}'] = E_list[i]
                result_dict['status'] = 'Done'
                result_dict['file_name'] = file_name
                results_list.append(result_dict)
                
                finished_files.append(file_name)
        
        for f in finished_files:
            del running_jobs[f]

        # 谷底の判定と追加投入
        for group_key, known_czs in completed_czs.items():
            if group_status[group_key] == 'Done' or not known_czs:
                continue
                
            best_cz = min(known_czs, key=known_czs.get)
            cz_down = np.round(best_cz - 0.1, 1)
            cz_up   = np.round(best_cz + 0.1, 1)
            
            needs_down = cz_down not in known_czs
            needs_up   = cz_up not in known_czs
            
            if not needs_down and not needs_up:
                group_status[group_key] = 'Done'
            else:
                task_base = {k: v for k, v in df_init.loc[
                    (np.round(df_init['beta'], 1) == group_key[0]) &
                    (np.round(df_init['cx'], 1) == group_key[1]) & 
                    (np.round(df_init['cy'], 1) == group_key[2])
                ].iloc[0].to_dict().items() if k in fixed_keys}
                
                if needs_down and cz_down not in queued_or_running_czs[group_key]:
                    new_task = {'cx': group_key[1], 'cy': group_key[2], 'cz': cz_down, **task_base}
                    job_queue.append(new_task)
                    queued_or_running_czs[group_key].add(cz_down)
                    
                if needs_up and cz_up not in queued_or_running_czs[group_key]:
                    new_task = {'cx': group_key[1], 'cy': group_key[2], 'cz': cz_up, **task_base}
                    job_queue.append(new_task)
                    queued_or_running_czs[group_key].add(cz_up)

        while len(running_jobs) < args.num_nodes and len(job_queue) > 0:
            task = job_queue.pop(0)
            file_name = exec_amber_job(auto_dir, args.monomer_name, task, isTest=args.isTest)
            running_jobs[file_name] = task

        current_time = time.time()
        if current_time - last_save_time > 10.0 and len(results_list) > 0:
            df_out = pd.DataFrame(results_list)
            df_out.to_csv(auto_csv_path, index=False)
            last_save_time = current_time
            done_groups = sum(1 for status in group_status.values() if status == 'Done')
            print(f"[{time.strftime('%H:%M:%S')}] 進捗: {done_groups}/{len(group_status)} 箇所で谷底確定. (実行中: {len(running_jobs)}個, キュー待機: {len(job_queue)}個)")

        if all(s == 'Done' for s in group_status.values()) and len(running_jobs) == 0 and len(job_queue) == 0:
            if len(results_list) > 0:
                df_out = pd.DataFrame(results_list)
                df_out.to_csv(auto_csv_path, index=False)
            print("🎉 すべての探索が完了しました！")
            break

        time.sleep(1) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, required=True)
    parser.add_argument('--num-nodes', type=int, required=True)
    parser.add_argument('--max-2', type=int, default=0)
    args = parser.parse_args()

    print("----main process start----")
    main_process(args)
    print("----finish process----")