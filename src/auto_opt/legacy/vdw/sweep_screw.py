#!/usr/bin/env python3
"""
Usage Example:
1. Range mode (Original):
   python -m auto_opt.vdw.sweep_screw --monomer-path data/monomer/.xyz --out-dir runs/ --z-min --z-max --z-step --beta-min --beta-max --beta-step --alpha-min 60 --alpha-max 70 --alpha-step 5

2. List mode (New):
   python -m auto_opt.vdw.sweep_screw ... --alpha-list 60 65 66.5 70
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

# --- 幾何 -------------------------------------------------------------------
def t_shaped_pair(base_axyz, Rx, Rz, b, z):
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        rot = np.matmul(np.array([x, y, zz]), Rz)
        rot = np.matmul(rot, Rx)
        axyz_1.append([ rot[0],      rot[1], rot[2],   sym])
        axyz_2.append([ -rot[0], rot[1]+b/2, rot[2]+z, sym]) # Note: Check if 2*z is intended for your specific model
    return axyz_1, axyz_2


def parallel_pair(base_axyz, Rx, Rz, z):
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        rot = np.matmul(np.array([x, y, zz]), Rz)
        rot = np.matmul(rot, Rx)
        axyz_1.append([ rot[0],  rot[1],  rot[2], sym])
        axyz_2.append([ rot[0],  rot[1],  rot[2], sym]) # Note: Check if 2*z is intended for your specific model
    return axyz_1, axyz_2

# --- 接触半径（剛体球・二分法） -------------------------------------------
def _axyz_to_arrays(axyz: list) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.array([[a[0], a[1], a[2]] for a in axyz], float)
    rad = np.array([vdw_radius(a[3]) for a in axyz], float)
    return xyz, rad

def vdw_R(axyz_1, axyz_2, theta_deg: float) -> float:
    # 座標と半径の抽出（高速化のため、ループ外で抽出済みの配列を渡すのが理想です）
    R1 = np.asarray([[x, y, z] for x, y, z, _ in axyz_1], float)
    R2 = np.asarray([[x, y, z] for x, y, z, _ in axyz_2], float)
    r1 = np.asarray([vdw_radius(a[3]) for a in axyz_1], float)
    r2 = np.asarray([vdw_radius(a[3]) for a in axyz_2], float)

    ct, st = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    eR = np.array([ct, st, 0.0], float)

    # 全ペアの相対ベクトル
    D = R2[None,:,:] - R1[:,None,:]
    
    # 射影成分（指定方向への距離）
    R12b = D @ eR
    
    # 垂直方向の距離の2乗（d_perp^2 = 全距離^2 - 射影^2）
    D2 = (D*D).sum(axis=2)
    R12a2 = D2 - R12b*R12b

    # 半径の和（の2乗）
    rad_sum = r1[:,None] + r2[None,:]
    rad_sum_sq = rad_sum**2
    
    # 垂直距離的に「ぶつかる可能性がある」ペアだけを抽出
    mask = R12a2 < rad_sum_sq
    
    # 誰もぶつからない場合は距離0.0を返す
    if not np.any(mask):
        return 0.0
    
    # マスクを適用して、必要なスライド距離Rを計算
    sq = rad_sum_sq[mask] - R12a2[mask]
    twoR_need = -R12b[mask] + np.sqrt(sq)
    
    # 負の値（既に離れている）は0にする
    twoR_need = np.maximum(twoR_need, 0.0)

    return float(np.max(twoR_need))


# --- スイープ本体 ------------------------------------------------------------
def sweep(monomer_path: str, out_dir: str, z_min:float, z_max: float, z_step: float,
          alpha_config: dict, # Changed to dict to handle logic inside
          beta_min: float, beta_max: float, beta_step: float) -> None:

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
    betas = [float(p) for p in np.arange(beta_min, beta_max + 1e-9, beta_step)]

    base_axyz = read_xyz(monomer_path)
    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])

    all_rows = []
    
    # Progress check for long loops
    total_iter = len(z_vals) * len(betas) * len(alphas)
    print(f"Starting sweep... Total configurations to check: {total_iter}")

    count = 0
    for z in z_vals:
        for beta in betas:
            Rx = Rod(-ex, beta)
            for alpha in alphas:
                Rz = Rod(ez, alpha)
                
                # Calculation Logic
                axyz_c, axyz_a = parallel_pair(base_axyz, Rx, Rz, 0)
                R_b = vdw_R(axyz_c, axyz_a, 90)
                axyz_c, axyz_t = t_shaped_pair(base_axyz, Rx, Rz, R_b, z)
                R_a = vdw_R(axyz_c, axyz_t, 0.0)
                R_a *= 2
                # beta を beta 列に、R_a, R_b を a, b 列に当てはめ、Amber用のステータスも追加
                all_rows.append([alpha, beta, round(R_a, 1), round(R_b, 1), round(R_b/2, 1), round(R_b/2, 1), z, "NotYet", "vdW_min"])

    # ドライバがそのまま読み込めるヘッダー名に変更
    df = pd.DataFrame(all_rows, columns=['alpha', 'beta', 'a', 'b', 'bt1', 'bt2', 'z', 'status', 'structure_type'])
    df = df.sort_values(['z','alpha','beta']).reset_index(drop=True)
    out_csv = out / f"step1_init_params.csv"
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
    
    # Beta params
    ap.add_argument('--beta-min', type=float, default=0.0)
    ap.add_argument('--beta-max', type=float, default=0.0)
    ap.add_argument('--beta-step', type=float, default=1)
    
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
          args.beta_min, args.beta_max, args.beta_step)

if __name__ == '__main__':
    main()