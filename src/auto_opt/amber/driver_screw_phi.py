#!/usr/bin/env python3
"""
python -m auto_opt.amber.driver_screw_phi --auto-dir runs/DNTT_test --monomer-name DNTT --num-nodes 2 --isTest
"""
import pandas as pd
import time
from auto_opt.amber.make_io_gene_screw_phi import exec_gjf
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

fixed_param_keys  = ['alpha', 'beta', 'z', 'phi']
all_keys          = ['alpha', 'beta', 'z', 'phi', 'a', 'b', 'bt1', 'bt2']
# 各ダイマーの識別キー: 平行移動に z が出てこないダイマーは z を含まない
DIMER_KEYS = {
    1: ['alpha', 'beta', 'phi', 'a'],            # a-dimer: 平行移動 (a, 0, 0) → z 無依存
    2: ['alpha', 'beta', 'phi', 'b'],            # b-dimer: 平行移動 (0, b, 0) → z 無依存
    3: ['alpha', 'beta', 'z', 'phi', 'a', 'bt1'], # t1-dimer: 平行移動 (a/2, bt1, z) → z 依存
    4: ['alpha', 'beta', 'z', 'phi', 'a', 'bt2'], # t3-dimer: 平行移動 (a/2, -bt2, z) → z 依存
}


def _prepare_amber_resources(auto_dir: str):
    amber_dir = Path(auto_dir) / "amber"
    amber_dir.mkdir(parents=True, exist_ok=True)
    src = RES / "FF_calc.in"
    if src.exists():
        shutil.copy2(src, amber_dir / "FF_calc.in")


def main_process(args):
    auto_dir = str(Path(args.auto_dir).resolve())
    _prepare_amber_resources(auto_dir)
    os.makedirs(os.path.join(auto_dir, 'amber'), exist_ok=True)
    os.makedirs(os.path.join(auto_dir, 'gaussview'), exist_ok=True)

    auto_csv_path = os.path.join(auto_dir, 'step1.csv')
    if not os.path.exists(auto_csv_path):
        df_E = pd.DataFrame(columns=all_keys + ['E', 'E1', 'E2', 'E3', 'E4', 'status'])
        df_E.to_csv(auto_csv_path, index=False)

    for n, e_col in [(1, 'E1'), (2, 'E2'), (3, 'E3'), (4, 'E4')]:
        path = os.path.join(auto_dir, f'step1_{n}.csv')
        if not os.path.exists(path):
            df = pd.DataFrame(columns=DIMER_KEYS[n] + [e_col, 'status', 'file_name'])
            df.to_csv(path, index=False)

    os.chdir(os.path.join(args.auto_dir, 'amber'))
    isOver = False
    while not isOver:
        isOver = listen(auto_dir, args.monomer_name, args.num_nodes, args.isTest)
        time.sleep(0.1)

    from auto_opt.gaussian.extract_minima import extract_minima
    extract_minima(symmetry='screw', auto_dir=auto_dir)


def listen(auto_dir, monomer_name, num_nodes, isTest):
    mono_file = str(AMBER_REF / f'{monomer_name}_HF_esp_gaff2.out')
    E_mono = amber_get_E(mono_file)[0]

    # 各ダイマーの計算完了チェックと E 更新
    e_cols = {1: 'E1', 2: 'E2', 3: 'E3', 4: 'E4'}
    dfs = {}
    for n, e_col in e_cols.items():
        csv_n = os.path.join(auto_dir, f'step1_{n}.csv')
        df_n = pd.read_csv(csv_n)
        for idx, row in df_n.loc[df_n['status'] == 'InProgress', DIMER_KEYS[n] + ['file_name']].iterrows():
            log = os.path.join(auto_dir, 'amber', row['file_name'])
            if not os.path.exists(log):
                continue
            E_list = amber_get_E(log)
            if len(E_list) == 1:
                E = np.round(float(E_list[0]) - 2 * E_mono, 4)
                df_n.loc[idx, [e_col, 'status']] = [E, 'Done']
                df_n.to_csv(csv_n, index=False)
        dfs[n] = df_n

    # step1.csv の集計: 4ダイマーが全て Done になったら合計 E を記録
    auto_csv = os.path.join(auto_dir, 'step1.csv')
    df_E = pd.read_csv(auto_csv)
    for idx, row in df_E.loc[df_E['status'] == 'InProgress'].iterrows():
        sub = {n: filter_df(dfs[n], {k: row[k] for k in DIMER_KEYS[n]}) for n in [1, 2, 3, 4]}
        sub = {n: sub[n][sub[n]['status'] == 'Done'] for n in [1, 2, 3, 4]}
        if not all(len(sub[n]) > 0 for n in [1, 2, 3, 4]):
            continue
        E1, E2, E3, E4 = (sub[1]['E1'].values[0], sub[2]['E2'].values[0],
                           sub[3]['E3'].values[0], sub[4]['E4'].values[0])
        E = 2*E1 + 2*E2 + 2*E3 + 2*E4
        df_E.loc[idx, ['E', 'E1', 'E2', 'E3', 'E4', 'status']] = [
            round(E, 4), round(E1, 4), round(E2, 4), round(E3, 4), round(E4, 4), 'Done'
        ]
        df_E.to_csv(auto_csv, index=False)

    # 新規計算の投入
    dict_matrix = get_params_dict(auto_dir, num_nodes)
    for params_dict in dict_matrix:
        if check_calc_status(auto_dir, params_dict):
            continue

        df_E = pd.read_csv(auto_csv)
        if len(filter_df(df_E, params_dict)) == 0:
            new_row = pd.Series({**params_dict, 'E': 0., 'E1': 0., 'E2': 0., 'E3': 0., 'E4': 0., 'status': 'InProgress'})
            df_E = pd.concat([df_E, new_row.to_frame().T], ignore_index=True)
            df_E.to_csv(auto_csv, index=False)

        for n, e_col in e_cols.items():
            p_n = {k: v for k, v in params_dict.items() if k in DIMER_KEYS[n]}
            csv_n = os.path.join(auto_dir, f'step1_{n}.csv')
            df_n = pd.read_csv(csv_n)
            if len(filter_df(df_n, p_n)) == 0:
                file_name = exec_gjf(auto_dir, monomer_name, p_n, structure_type=n, isTest=isTest)
                new = pd.Series({**p_n, e_col: 0., 'status': 'InProgress', 'file_name': file_name})
                df_n = pd.concat([df_n, new.to_frame().T], ignore_index=True)
                df_n.to_csv(csv_n, index=False)

    init_params_csv = os.path.join(auto_dir, 'step1_init_params.csv')
    df_init = pd.read_csv(init_params_csv)
    return len(filter_df(df_init, {'status': 'Done'})) == len(df_init)


def get_params_dict(auto_dir, num_nodes):
    init_params_csv = os.path.join(auto_dir, 'step1_init_params.csv')
    df_init = pd.read_csv(init_params_csv)
    df_cur = pd.read_csv(os.path.join(auto_dir, 'step1.csv'))
    df_inprogress = df_init[df_init['status'] == 'InProgress']

    if len(df_inprogress) < num_nodes:
        df_notyet = df_init[df_init['status'] == 'NotYet']
        for index in df_notyet.index:
            df_init = update_value_in_df(df_init, index, 'status', 'InProgress')
            df_init.to_csv(init_params_csv, index=False)
            return [df_init.loc[index, all_keys].to_dict()]

    dict_matrix = []
    for index in df_inprogress.index:
        df_init = pd.read_csv(init_params_csv)
        init_params_dict  = df_init.loc[index, all_keys].to_dict()
        fixed_params_dict = df_init.loc[index, fixed_param_keys].to_dict()

        isDone, opt_params_matrix = get_opt_params_dict(df_cur, init_params_dict, fixed_params_dict)
        if isDone:
            df_init = update_value_in_df(df_init, index, 'status', 'Done')
            if np.max(df_init.index) < index + 1:
                status = 'Done'
            else:
                status = get_values_from_df(df_init, index + 1, 'status')
            df_init.to_csv(init_params_csv, index=False)

            if status == 'NotYet':
                next_dict = df_init.loc[index + 1, all_keys].to_dict()
                df_init = update_value_in_df(df_init, index + 1, 'status', 'InProgress')
                df_init.to_csv(init_params_csv, index=False)
                dict_matrix.append(next_dict)
        else:
            for p in opt_params_matrix:
                dict_matrix.append({**fixed_params_dict, **p})
    return dict_matrix


def get_opt_params_dict(df_cur, init_params_dict, fixed_params_dict):
    df_val = filter_df(df_cur, fixed_params_dict)
    a_prev   = init_params_dict['a']
    bt1_prev = init_params_dict['bt1']
    bt2_prev = init_params_dict['bt2']

    while True:
        E_list = []; xyz_list = []; para_list = []
        for a in [a_prev-0.1, a_prev, a_prev+0.1]:
            for bt1 in [bt1_prev-0.1, bt1_prev, bt1_prev+0.1]:
                for bt2 in [bt2_prev-0.1, bt2_prev, bt2_prev+0.1]:
                    a   = np.round(a,   1)
                    bt1 = np.round(bt1, 1)
                    bt2 = np.round(bt2, 1)
                    b   = np.round(bt1 + bt2, 1)
                    df_v = df_val[(df_val['a'] == a) & (df_val['bt1'] == bt1) &
                                  (df_val['bt2'] == bt2) & (df_val['status'] == 'Done')]
                    if len(df_v) == 0:
                        para_list.append({'a': a, 'b': b, 'bt1': bt1, 'bt2': bt2})
                    else:
                        xyz_list.append({'a': a, 'b': b, 'bt1': bt1, 'bt2': bt2})
                        E_list.append(df_v['E'].values[0])

        if para_list:
            return False, para_list

        best = xyz_list[np.argmin(E_list)]
        if best['a'] == a_prev and best['bt1'] == bt1_prev and best['bt2'] == bt2_prev:
            return True, [best]
        a_prev, bt1_prev, bt2_prev = best['a'], best['bt1'], best['bt2']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, required=True)
    parser.add_argument('--num-nodes', type=int, required=True)
    args = parser.parse_args()

    print("----main process----")
    main_process(args)
    print("----finish process----")
