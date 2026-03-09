#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
import os
import shutil
from pathlib import Path
from scipy.optimize import minimize_scalar

# 既存のIO関数に加えて、モノマーの座標を取得する関数もインポートします
from auto_opt.stacking.make_io_stacking import exec_gjf_stacking, get_monomer_xyzR
from auto_opt.utils import amber_get_E

RES = Path(__file__).resolve().parent / "resources"

def _prepare_amber_resources(auto_dir: str):
    amber_dir = Path(auto_dir) / "amber"
    amber_dir.mkdir(parents=True, exist_ok=True)
    src = RES / "FF_calc.in"
    if src.exists():
        shutil.copy2(src, amber_dir / "FF_calc.in")

def estimate_vdw_cz(monomer_name, params_dict):
    a = params_dict['a']; b = params_dict['b']; z = params_dict['z']
    theta1 = params_dict['theta1']; theta2 = params_dict['theta2']
    cx = params_dict['cx']; cy = params_dict['cy']
    
    top_mol = get_monomer_xyzR(monomer_name, cx, cy, 0.0, theta2)
    b0 = get_monomer_xyzR(monomer_name, 0, 0, 0, theta1)
    ba1 = get_monomer_xyzR(monomer_name, a, 0, 0, theta1)
    ba2 = get_monomer_xyzR(monomer_name, -a, 0, 0, theta1)
    bb1 = get_monomer_xyzR(monomer_name, 0, b, 2*z, theta1)
    bb2 = get_monomer_xyzR(monomer_name, 0, -b, -2*z, theta1)
    bt1 = get_monomer_xyzR(monomer_name, a/2, b/2, z, -theta1)
    bt2 = get_monomer_xyzR(monomer_name, a/2, -b/2, -z, -theta1)
    bt3 = get_monomer_xyzR(monomer_name, -a/2, -b/2, -z, -theta1)
    bt4 = get_monomer_xyzR(monomer_name, -a/2, b/2, z, -theta1)
    
    bottom_layer = np.concatenate([b0, ba1, ba2, bb1, bb2, bt1, bt2, bt3, bt4], axis=0)
    
    b_xyz = bottom_layer[:, :3]; b_r = bottom_layer[:, 3]
    t_xyz = top_mol[:, :3];      t_r = top_mol[:, 3]
    
    dx = t_xyz[:, 0].reshape(-1, 1) - b_xyz[:, 0]
    dy = t_xyz[:, 1].reshape(-1, 1) - b_xyz[:, 1]
    dxy2 = dx**2 + dy**2
    
    sr = t_r.reshape(-1, 1) + b_r
    sr2 = sr**2
    
    mask = dxy2 < sr2
    dz = b_xyz[:, 2] - t_xyz[:, 2].reshape(-1, 1)
    required_cz = dz[mask] + np.sqrt(sr2[mask] - dxy2[mask])
    
    if len(required_cz) > 0:
        return np.max(required_cz)
    else:
        return 4.0

def get_amber_energy_for_cz(cz_val, cx_val, cy_val, base_params, auto_dir, monomer_name):
    params = base_params.copy()
    params.update({'cx': cx_val, 'cy': cy_val, 'cz': float(cz_val)})
    
    out_file = exec_gjf_stacking(auto_dir, monomer_name, params, in_file="FF_calc.in", isTest=False)
    out_path = os.path.join(auto_dir, 'amber', out_file)
    try:
        energy_list = amber_get_E(out_path)
        return float(energy_list[0])
    except Exception:
        return 9999.0

def main_process(args):
    auto_dir = str(Path(args.auto_dir).resolve()) 
    _prepare_amber_resources(auto_dir)
    
    csv_path = Path(args.input_csv).resolve()
    df_input = pd.read_csv(csv_path)
    
    out_csv_path = os.path.join(auto_dir, 'cy_scan_results.csv')
    
    calculated_keys = set()
    if os.path.exists(out_csv_path) and not args.isTest:
        df_existing = pd.read_csv(out_csv_path)
        print(f"既存の計算結果 '{out_csv_path}' を読み込みました。続きから再開します。")
        for _, r in df_existing.iterrows():
            cx_val_hist = round(r['cx'], 2) if 'cx' in r else 0.0
            # ★修正: a, b, z をキーに追加！
            key = (round(r['a'], 2), round(r['b'], 2), round(r['alpha'], 2), round(r['z'], 2), round(r['phi'], 2), round(r['theta2'], 2), cx_val_hist, round(r['cy'], 2))
            calculated_keys.add(key)

    print(f"入力CSVから {len(df_input)} 件の構造を読み込みました。")
    print(f"モード: {args.mode} スキャンを開始します (cz精度: {args.cz_tol} Å)...\n")
    
    for index, row in df_input.iterrows():
        a_opt = row['a']; b_opt = row['b']; z_opt = row['z']
        alpha_opt = row['alpha']; phi_opt = row['phi']
        
        print(f"\n--- 構造 {index+1}: a={a_opt}, b={b_opt}, alpha={alpha_opt}, z={z_opt}, phi={phi_opt} ---")
        
        for theta2_val in [alpha_opt]:#, -alpha_opt]:
            base_params = {
                'a': a_opt, 'b': b_opt, 'alpha': alpha_opt, 
                'z': z_opt, 'phi': phi_opt,
                'theta1': alpha_opt, 'theta2': theta2_val 
            }
            print(f"  >> theta2 = {theta2_val} のスキャンを開始")

            step_size_x = args.step_x
            step_size_y = args.step_y

            if args.mode == '1D':
                cx_list = [0.0]
            elif args.mode == '2D':
                num_points_x = int(a_opt / step_size_x) + 1
                cx_list = np.linspace(-a_opt/2, a_opt/2, num_points_x)

            num_points_y = int(b_opt / step_size_y) + 1
            cy_list = np.linspace(-b_opt/2, b_opt/2, num_points_y)

            for cx in cx_list:
                cx_val = np.round(cx, 2)
                for cy in cy_list:
                    cy_val = np.round(cy, 2)
                    
                    # ★修正: a, b, z をキーに追加！
                    current_key = (round(a_opt, 2), round(b_opt, 2), round(alpha_opt, 2), round(z_opt, 2), round(phi_opt, 2), round(theta2_val, 2), cx_val, cy_val)
                    if current_key in calculated_keys:
                        print(f"    Skip: cx={cx_val:5.2f}, cy={cy_val:5.2f} は計算済みのためスキップします。")
                        continue
                    
                    params_for_vdw = base_params.copy()
                    params_for_vdw.update({'cx': cx_val, 'cy': cy_val, 'cz': 0.0})
                    
                    cz_vdw = estimate_vdw_cz(args.monomer_name, params_for_vdw)
                    search_bounds = (cz_vdw - 1.0, cz_vdw + 1.0)
                    
                    res = minimize_scalar(
                        get_amber_energy_for_cz, 
                        args=(cx_val, cy_val, base_params, auto_dir, args.monomer_name),
                        bounds=search_bounds,
                        method='bounded',
                        options={'xatol': args.cz_tol} 
                    )
                    
                    best_cz = res.x
                    best_E = res.fun
                    v_z = cy_val * (2.0 * z_opt / b_opt)
                        
                    d_rise = best_cz - v_z
                    d_vdw_rise = cz_vdw - v_z
                    
                    res_dict = base_params.copy()
                    res_dict.update({
                        'cx': cx_val, 'cy': cy_val, 'cz_abs': best_cz,    
                        'd_vdw': d_vdw_rise, 'd_opt': d_rise, 'E_total': best_E
                    })
                    
                    if not args.isTest:
                        df_single = pd.DataFrame([res_dict])
                        if not os.path.exists(out_csv_path):
                            df_single.to_csv(out_csv_path, index=False)
                        else:
                            df_single.to_csv(out_csv_path, mode='a', header=False, index=False)
                    
                    calculated_keys.add(current_key)
                    print(f"    Done: cx={cx_val:5.2f}, cy={cy_val:5.2f} | d={d_rise:.2f} | E={best_E:.4f}")

    print(f"\nすべての計算が完了しました！")
    
    if not args.isTest and os.path.exists(out_csv_path):
        df_res = pd.read_csv(out_csv_path)
        if 'cx' not in df_res.columns:
            df_res['cx'] = 0.0
        # ★修正: a, b, z をソートと重複削除のキーに追加！
        sort_keys = ['a', 'b', 'alpha', 'z', 'phi', 'theta2', 'cx', 'cy']
        df_res = df_res.drop_duplicates(subset=sort_keys, keep='last')
        df_res = df_res.sort_values(sort_keys)
        df_res.to_csv(out_csv_path, index=False)
        print(f"結果をソートして '{out_csv_path}' に最終保存しました。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-dir', type=str, required=True)
    parser.add_argument('--monomer-name', type=str, required=True)
    parser.add_argument('--input-csv', type=str, required=True)
    parser.add_argument('--cz-tol', type=float, default=0.1, help="SciPyでの cz 最適化の収束条件")
    parser.add_argument('--mode', type=str, choices=['1D', '2D'], default='1D', help="スキャンモード (1D or 2D)")
    parser.add_argument('--step-x', type=float, default=0.1, help="cx のスキャン間隔")
    parser.add_argument('--step-y', type=float, default=0.1, help="cy のスキャン間隔")
    parser.add_argument('--isTest', action='store_true')
    args = parser.parse_args()
    
    main_process(args)