#!/usr/bin/env python3
"""
Usage Example:
1. Range mode (両タイプ出力):
   python -m auto_opt.vdw.sweep_screw_phi --monomer-path data/monomer/DNTT.xyz --out-dir runs/ \
       --z-min 0 --z-max 3 --z-step 0.5 \
       --beta-min 0 --beta-max 10 --beta-step 5 \
       --alpha-min 60 --alpha-max 70 --alpha-step 5 \
       --phi-min 0 --phi-max 10 --phi-step 5

2. List mode (alpha):
   python -m auto_opt.vdw.sweep_screw_phi ... --alpha-list 60 65 70

3. a-stack のみ出力:
   python -m auto_opt.vdw.sweep_screw_phi ... --select a-stack

4. b-stack のみ出力:
   python -m auto_opt.vdw.sweep_screw_phi ... --select b-stack
"""
from __future__ import annotations
import math, argparse, pathlib
from typing import List, Tuple
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
# 回転順: phi(x軸) → alpha(z軸) → beta(x軸)

def t_shaped_pair(base_axyz, Rx_phi, Rz, Rx, a, b, z):
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        rot = np.matmul(np.array([x, y, zz]), Rx_phi)
        rot = np.matmul(rot, Rz)
        rot = np.matmul(rot, Rx)
        axyz_1.append([ rot[0],       rot[1],       rot[2],   sym])
        axyz_2.append([-rot[0]+a/2,  rot[1]+b/2,  rot[2]+z,  sym])
    return axyz_1, axyz_2

def parallel_pair(base_axyz, Rx_phi, Rz, Rx):
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        rot = np.matmul(np.array([x, y, zz]), Rx_phi)
        rot = np.matmul(rot, Rz)
        rot = np.matmul(rot, Rx)
        axyz_1.append([rot[0], rot[1], rot[2], sym])
        axyz_2.append([rot[0], rot[1], rot[2], sym])
    return axyz_1, axyz_2

# --- 接触半径 ----------------------------------------------------------------
def vdw_R(axyz_1, axyz_2, theta_deg: float) -> float:
    R1 = np.asarray([[x, y, z] for x, y, z, _ in axyz_1], float)
    R2 = np.asarray([[x, y, z] for x, y, z, _ in axyz_2], float)
    r1 = np.asarray([vdw_radius(a[3]) for a in axyz_1], float)
    r2 = np.asarray([vdw_radius(a[3]) for a in axyz_2], float)

    ct, st = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    eR = np.array([ct, st, 0.0], float)

    D = R2[None, :, :] - R1[:, None, :]
    R12b = D @ eR
    D2 = (D * D).sum(axis=2)
    R12a2 = D2 - R12b * R12b
    rad_sum = r1[:, None] + r2[None, :]
    rad_sum_sq = rad_sum ** 2
    mask = R12a2 < rad_sum_sq

    if not np.any(mask):
        return 0.0

    sq = rad_sum_sq[mask] - R12a2[mask]
    twoR_need = -R12b[mask] + np.sqrt(sq)
    twoR_need = np.maximum(twoR_need, 0.0)
    return float(np.max(twoR_need))


# --- スイープ本体 ------------------------------------------------------------
def sweep(monomer_path: str, out_dir: str,
          z_min: float, z_max: float, z_step: float,
          alpha_config: dict,
          beta_min: float, beta_max: float, beta_step: float,
          phi_min: float, phi_max: float, phi_step: float,
          select_types: List[str] | None = None) -> None:

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    monomer_name = pathlib.Path(monomer_path).stem

    if alpha_config.get('list') is not None:
        alphas = sorted(set(alpha_config['list']))
        print(f"Mode: Explicit Alpha List -> {alphas}")
    else:
        a_min  = alpha_config.get('min',  0)
        a_max  = alpha_config.get('max',  90)
        a_step = alpha_config.get('step', 5)
        alphas = [float(a) for a in np.arange(a_min, a_max + 1e-9, a_step)]
        print(f"Mode: Range Alpha Scan -> {a_min} to {a_max} step {a_step}")

    z_vals = [round(z, 1) for z in np.arange(z_min, z_max + 1e-9, z_step)]
    betas  = [float(b) for b in np.arange(beta_min, beta_max + 1e-9, beta_step)]
    phis   = [float(p) for p in np.arange(phi_min,  phi_max  + 1e-9, phi_step)]

    base_axyz = read_xyz(monomer_path)
    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])

    accept_all = select_types is None or "all" in select_types
    accept_set = set(select_types) if select_types else set()

    total_iter = len(z_vals) * len(betas) * len(alphas) * len(phis)
    print(f"Starting sweep... Total configurations: {total_iter}")

    all_rows = []
    for z in z_vals:
        for beta in betas:
            Rx = Rod(-ex, beta)
            for alpha in alphas:
                Rz = Rod(ez, alpha)
                for phi in phis:
                    Rx_phi = Rod(-ex, phi)

                    # b-stack: b方向を平行接触で決め、a方向をT字接触で決める
                    if accept_all or "b-stack" in accept_set:
                        axyz_c, axyz_b = parallel_pair(base_axyz, Rx_phi, Rz, Rx)
                        R_b = vdw_R(axyz_c, axyz_b, 90)
                        axyz_c, axyz_t = t_shaped_pair(base_axyz, Rx_phi, Rz, Rx, 0, R_b, z)
                        R_a = vdw_R(axyz_c, axyz_t, 0.0) * 2
                        all_rows.append([
                            alpha, beta, phi,
                            round(R_a, 1), round(R_b, 1),
                            round(R_b/2, 1), round(R_b/2, 1),
                            z, "NotYet", "b-stack"
                        ])

                    # a-stack: a方向を平行接触で決め、b方向をT字接触で決める
                    if accept_all or "a-stack" in accept_set:
                        axyz_c, axyz_a = parallel_pair(base_axyz, Rx_phi, Rz, Rx)
                        R_a = vdw_R(axyz_c, axyz_a, 0)
                        axyz_c, axyz_t = t_shaped_pair(base_axyz, Rx_phi, Rz, Rx, R_a, 0, z)
                        R_b = vdw_R(axyz_c, axyz_t, 90) * 2
                        all_rows.append([
                            alpha, beta, phi,
                            round(R_a, 1), round(R_b, 1),
                            round(R_b/2, 1), round(R_b/2, 1),
                            z, "NotYet", "a-stack"
                        ])

    df = pd.DataFrame(all_rows, columns=[
        'alpha', 'beta', 'phi', 'a', 'b', 'bt1', 'bt2', 'z', 'status', 'structure_type'
    ])
    df = df.sort_values(['z', 'alpha', 'beta', 'phi']).reset_index(drop=True)
    out_csv = out / "step1_init_params.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (n={len(df)})")


# --- CLI ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="vdW sweep for screw+phi")
    ap.add_argument('--monomer-path', required=True)
    ap.add_argument('--out-dir', required=True)

    ap.add_argument('--z-min',   type=float, default=0.0)
    ap.add_argument('--z-max',   type=float, required=True)
    ap.add_argument('--z-step',  type=float, default=0.1)

    ap.add_argument('--alpha-min',  type=float, default=0)
    ap.add_argument('--alpha-max',  type=float, default=90)
    ap.add_argument('--alpha-step', type=float, default=5)
    ap.add_argument('--alpha-list', type=float, nargs='+',
                    help='alpha値をリスト指定 (e.g. 60 65 70). min/max/stepを上書き.')

    ap.add_argument('--beta-min',  type=float, default=0.0)
    ap.add_argument('--beta-max',  type=float, default=0.0)
    ap.add_argument('--beta-step', type=float, default=1.0)

    ap.add_argument('--phi-min',  type=float, default=0.0)
    ap.add_argument('--phi-max',  type=float, default=0.0)
    ap.add_argument('--phi-step', type=float, default=1.0)

    ap.add_argument('--select', nargs='+', default=['all'],
                    choices=['all', 'a-stack', 'b-stack'],
                    help='出力する構造タイプ (デフォルト: all = 両方出力)')

    args = ap.parse_args()

    alpha_config = {
        'min':  args.alpha_min,
        'max':  args.alpha_max,
        'step': args.alpha_step,
        'list': args.alpha_list,
    }

    sweep(args.monomer_path, args.out_dir,
          args.z_min, args.z_max, args.z_step,
          alpha_config,
          args.beta_min, args.beta_max, args.beta_step,
          args.phi_min,  args.phi_max,  args.phi_step,
          select_types=args.select)


if __name__ == '__main__':
    main()
