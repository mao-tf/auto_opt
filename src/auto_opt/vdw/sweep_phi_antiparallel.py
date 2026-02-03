#!/usr/bin/env python3
"""
Usage Example:
1. Range mode (Original):
   python -m auto_opt.vdw.sweep_phi ... --alpha-min 60 --alpha-max 70 --alpha-step 5

2. List mode (New):
   python -m auto_opt.vdw.sweep_phi ... --alpha-list 60 65 66.5 70
"""
from __future__ import annotations
import math, argparse, pathlib
from typing import List, Iterable, Tuple, Optional
import numpy as np
import pandas as pd
from auto_opt.utils import Rod, vdw_radius

# --- IO ----------------------------------------------------------------------
def read_xyz(path: str) -> List[List[object]]:
    rows = []
    with open(path) as f:
        for line in f:
            s = line.split()
            if len(s) == 4:
                try:
                    x, y, z = float(s[1]), float(s[2]), float(s[3])
                except ValueError:
                    continue
                rows.append([x, y, z, s[0]])
    if not rows:
        raise ValueError(f"No XYZ rows parsed from {path}. Expect lines: 'El x y z'")
    return rows

# --- 幾何 --------------------------------------------------------------------
def t_shaped_pair(base_axyz, Rx, Rz, z):
    """
    T字配置 (t1): 隣接分子を反転（Head-to-Tail）させる
    操作: Y軸周りの180度回転 (-x, y, -z) をしてから、Rx, Rzで回転
    """
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        # --- Center (そのままで回転) ---
        # 1. 初期座標
        v0 = np.array([x, y, zz])
        # 2. 回転適用 (Rx -> Rz)
        rot1 = np.matmul(v0, Rx)
        rot1 = np.matmul(rot1, Rz)
        
        axyz_1.append([rot1[0], rot1[1], rot1[2], sym])
        
        # --- Neighbor (反転させてから回転) ---
        # 1. 初期座標をY軸180度回転 (x->-x, z->-z)
        # これでHead-to-Tailの「種」を作る
        v0_flip = np.array([x, -y, -zz]) 
        
        # 2. 同じ回転行列 (Rx -> Rz) を適用
        # これで「同じ傾き」を持ちつつ「逆向き」になる
        rot2 = np.matmul(v0_flip, Rx)
        rot2 = np.matmul(rot2, Rz)
        
        # 3. 配置 (T字の位置へシフト)
        axyz_2.append([-rot2[0], rot2[1], rot2[2] + z, sym])
        
    return axyz_1, axyz_2

def parallel_pair1(base_axyz, Rx, Rz, z):
    """
    a軸平行配置 (p1): 隣接分子を反転（Head-to-Tail）させる
    操作: X軸周りの180度回転 (x, -y, -z) をしてから、Rx, Rzで回転
    """
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        # --- Center ---
        v0 = np.array([x, y, zz])
        rot1 = np.matmul(v0, Rx)
        rot1 = np.matmul(rot1, Rz)
        
        axyz_1.append([rot1[0], rot1[1], rot1[2], sym])
        
        # --- Neighbor (反転) ---
        # 1. 初期座標をX軸180度回転 (y->-y, z->-z)
        # これが物理的に正しいp1方向の反転
        v0_flip = np.array([x, y, zz])
        
        # 2. 回転適用
        rot2 = np.matmul(v0_flip, Rx)
        rot2 = np.matmul(rot2, Rz)
        
        # 3. 配置 (シフトなし = 相対距離計算用)
        axyz_2.append([rot2[0], rot2[1], rot2[2], sym]) 
        
    return axyz_1, axyz_2

def parallel_pair2(base_axyz, Rx, Rz, z):
    """
    b軸平行配置 (p2): 反転させない（Head-to-Head）
    操作: そのまま回転してシフト
    """
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        # --- Center ---
        v0 = np.array([x, y, zz])
        rot1 = np.matmul(v0, Rx)
        rot1 = np.matmul(rot1, Rz)
        
        axyz_1.append([rot1[0], rot1[1], rot1[2], sym])
        
        # --- Neighbor (Parallel) ---
        # Centerと同じ向き (v0を使用)
        # シフト (+2z) だけ加える
        axyz_2.append([rot1[0], rot1[1], rot1[2] + 2*z, sym]) 
        
    return axyz_1, axyz_2


# --- 接触半径（剛体球・二分法） -------------------------------------------
def _axyz_to_arrays(axyz: list) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.array([[a[0], a[1], a[2]] for a in axyz], float)
    rad = np.array([vdw_radius(a[3]) for a in axyz], float)
    return xyz, rad

def vdw_R(axyz_1, axyz_2, theta_deg: float) -> float:
    R1 = np.asarray([[x, y, z] for x, y, z, _ in axyz_1], float)
    R2 = np.asarray([[x, y, z] for x, y, z, _ in axyz_2], float)
    r1 = np.asarray([vdw_radius(a[3]) for a in axyz_1], float)
    r2 = np.asarray([vdw_radius(a[3]) for a in axyz_2], float)

    ct, st = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    eR = np.array([ct, st, 0.0], float)

    D = R2[None,:,:] - R1[:,None,:]
    R12b = D @ eR
    D2   = (D*D).sum(axis=2)
    R12a2 = D2 - R12b*R12b

    rad_sum = r1[:,None] + r2[None,:]
    sq = rad_sum*rad_sum - R12a2
    sq = np.maximum(sq, 0.0)
    twoR_need = -R12b + np.sqrt(sq)
    twoR_need = np.maximum(twoR_need, 0.0)

    R = float(np.max(twoR_need))
    return R


# --- スイープ本体 ------------------------------------------------------------
def sweep(monomer_path: str, out_dir: str, z_min:float, z_max: float, z_step: float,
          alpha_config: dict, # Changed to dict to handle logic inside
          phi_min: float, phi_max: float, phi_step: float,
          theta_step: float, eps_a: float, eps_b: float) -> None:

    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    monomer_name = pathlib.Path(monomer_path).stem

    # --- Alpha List Logic ---
    if alpha_config.get('list') is not None:
        # リストが指定されていればそれを使う
        alphas = sorted(list(set(alpha_config['list']))) # 重複排除とソート
        print(f"Mode: Explicit Alpha List -> {alphas}")
    else:
        # リストがなければ範囲生成
        a_min = alpha_config.get('min', 0)
        a_max = alpha_config.get('max', 90)
        a_step = alpha_config.get('step', 5)
        alphas = [float(a) for a in np.arange(a_min, a_max + 1e-9, a_step)]
        print(f"Mode: Range Alpha Scan -> {a_min} to {a_max} step {a_step}")
    # ------------------------

    z_vals = [round(z, 1) for z in np.arange(z_min, z_max + 1e-9, z_step)]
    phis = [float(p) for p in np.arange(phi_min, phi_max + 1e-9, phi_step)]
    thetas = [float(t) for t in np.arange(0, 91, theta_step)]
    
    # Pre-calculate trig values
    cosb = {b: math.cos(math.radians(b)) for b in thetas}
    sinb = {b: math.sin(math.radians(b)) for b in thetas}

    base_axyz = read_xyz(monomer_path)
    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])

    all_rows = []
    
    # Progress check for long loops
    total_iter = len(z_vals) * len(phis) * len(alphas)
    print(f"Starting sweep... Total configurations to check: {total_iter}")

    count = 0
    for z in z_vals:
        for phi in phis:
            Rx = Rod(-ex, phi)
            for alpha in alphas:
                Rz = Rod(ez, alpha)
                
                # Calculation Logic
                axyz_c, axyz_a = parallel_pair1(base_axyz, Rx, Rz, 0)
                axyz_c, axyz_b = parallel_pair2(base_axyz, Rx, Rz, z)
                R_a = vdw_R(axyz_c, axyz_a, 0.0)
                R_b = vdw_R(axyz_c, axyz_b, 90.0)
                axyz_1, axyz_2 = t_shaped_pair(base_axyz, Rx, Rz, z)
                
                for beta in thetas:
                    R_clps = vdw_R(axyz_1, axyz_2, beta)
                    ca = R_a - 2.0 * R_clps * cosb[beta]
                    cb = R_b - 2.0 * R_clps * sinb[beta]
                    ok = (ca <= eps_a) and (cb <= eps_b)
                    all_rows.append([alpha, phi, beta, z, R_clps, ok])
                
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count}/{total_iter}...")# Print progress every 1,000 iterations to monitor status.

    df = pd.DataFrame(all_rows, columns=['alpha','phi','beta','z','R_clps','TorF'])
    df = df.sort_values(['z','alpha','phi','beta']).reset_index(drop=True)
    out_csv = out / f"vdW_r_contact_{monomer_name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (n={len(df)})")

# --- CLI ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="vdW sweep for Step 1 (Intralayer)")
    ap.add_argument('--monomer-path', required=True)
    ap.add_argument('--out-dir', required=True)
    
    # Z params
    ap.add_argument('--z-min', type=float, default=0.0)
    ap.add_argument('--z-max', type=float, required=True)
    ap.add_argument('--z-step', type=float, default=0.1)
    
    # Alpha params (Range OR List)
    ap.add_argument('--alpha-min', type=float, default=0)
    ap.add_argument('--alpha-max', type=float, default=90)
    ap.add_argument('--alpha-step', type=float, default=5)
    ap.add_argument('--alpha-list', type=float, nargs='+', help='Specific alpha values list (e.g. 60 65 70). Overrides min/max/step.')
    
    # Phi params
    ap.add_argument('--phi-min', type=float, default=0.0)
    ap.add_argument('--phi-max', type=float, default=0.0)
    ap.add_argument('--phi-step', type=float, default=1)
    
    # Step 1 specific params
    ap.add_argument('--theta-step', type=float, default=5)
    ap.add_argument('--eps-a', type=float, default=1e-3)
    ap.add_argument('--eps-b', type=float, default=1e-2)
    
    args = ap.parse_args()
    
    # Pack alpha config
    alpha_config = {
        'min': args.alpha_min,
        'max': args.alpha_max,
        'step': args.alpha_step,
        'list': args.alpha_list
    }

    sweep(args.monomer_path, args.out_dir, args.z_min, args.z_max, args.z_step,
          alpha_config, # Pass dict
          args.phi_min, args.phi_max, args.phi_step,
          args.theta_step, args.eps_a, args.eps_b)

if __name__ == '__main__':
    main()