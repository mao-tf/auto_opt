#!/usr/bin/env python3
"""
python -m auto_opt.vdw.sweep --monomer-path data/monomer/PFA.xyz --out-dir runs/PFA_test3 --z-max 1.3 --z-step 0.1 --theta-step 1
"""
from __future__ import annotations
import math, argparse, pathlib
from typing import List, Iterable, Tuple
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
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        rot = np.matmul(np.array([x, y, zz]), Rx)
        rot = np.matmul(rot, Rz)  # row @ R ## Rzはz周りのrodrigues
        axyz_1.append([ rot[0],  rot[1],  rot[2],     sym])
        axyz_2.append([-rot[0],  rot[1],  rot[2] + z, sym])
    return axyz_1, axyz_2

def parallel_pair(base_axyz, Rx, Rz, z):
    axyz_1, axyz_2 = [], []
    for x, y, zz, sym in base_axyz:
        rot = np.matmul(np.array([x, y, zz]), Rx)
        rot = np.matmul(rot, Rz)  # row @ R ## Rzはz周りのrodrigues
        axyz_1.append([ rot[0],  rot[1],  rot[2],       sym])
        axyz_2.append([ rot[0],  rot[1],  rot[2] + 2*z, sym])
    return axyz_1, axyz_2

# --- 接触半径（剛体球・二分法） -------------------------------------------
def _axyz_to_arrays(axyz: list) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.array([[a[0], a[1], a[2]] for a in axyz], float)
    rad = np.array([vdw_radius(a[3]) for a in axyz], float)
    return xyz, rad

def vdw_R(axyz_1, axyz_2, theta_deg: float) -> float:
    """片側シフト量 R（反復なしの閉形式）。速い＆分かりやすい。"""
    R1 = np.asarray([[x, y, z] for x, y, z, _ in axyz_1], float)  # (n1,3)
    R2 = np.asarray([[x, y, z] for x, y, z, _ in axyz_2], float)  # (n2,3)
    r1 = np.asarray([vdw_radius(a[3]) for a in axyz_1], float)    # (n1,)
    r2 = np.asarray([vdw_radius(a[3]) for a in axyz_2], float)    # (n2,)

    ct, st = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    eR = np.array([ct, st, 0.0], float)

    D = R2[None,:,:] - R1[:,None,:]          # (n1,n2,3)
    R12b = D @ eR                             # (n1,n2)
    D2   = (D*D).sum(axis=2)                  # (n1,n2) = ||D||^2
    R12a2 = D2 - R12b*R12b                    # 直交成分^2

    rad_sum = r1[:,None] + r2[None,:]         # (n1,n2)
    sq = rad_sum*rad_sum - R12a2              # √の中身
    sq = np.maximum(sq, 0.0)                  # 実数域のみ（負ならこのペアは制約しない）
    twoR_need = -R12b + np.sqrt(sq)           # 各ペアの要求する合計シフト量 2R
    twoR_need = np.maximum(twoR_need, 0.0)    # 負要求は0扱い

    R = float(np.max(twoR_need))        # シフト量
    return R


# --- スイープ本体 ------------------------------------------------------------
def sweep(monomer_path: str, out_dir: str, z_min:float, z_max: float, z_step: float,
          alpha_min:float, alpha_max:float, alpha_step: float,
          phi_min: float, phi_max: float, phi_step: float,
          theta_step: float, eps_a: float, eps_b: float) -> None:

    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    monomer_name = pathlib.Path(monomer_path).stem

    z_vals = [round(z, 1) for z in np.arange(z_min, z_max + 1e-9, z_step)]
    phis = [float(p) for p in np.arange(phi_min, phi_max + 1e-9, phi_step)]
    alphas = [float(a) for a in np.arange(alpha_min, alpha_max + 1e-9, alpha_step)]
    thetas = [float(t) for t in np.arange(0, 91, theta_step)]
    cosb = {b: math.cos(math.radians(b)) for b in thetas}
    sinb = {b: math.sin(math.radians(b)) for b in thetas}

    base_axyz = read_xyz(monomer_path)
    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])

    all_rows = []
    for z in z_vals:
        for phi in phis:
            for alpha in alphas:
                Rx = Rod(-ex, phi)
                Rz = Rod(ez, alpha)
                axyz_c, axyz_a = parallel_pair(base_axyz, Rx, Rz, 0)
                axyz_c, axyz_b = parallel_pair(base_axyz, Rx, Rz, z)
                R_a = vdw_R(axyz_c, axyz_a, 0.0)
                R_b = vdw_R(axyz_c, axyz_b, 90.0)
                axyz_1, axyz_2 = t_shaped_pair(base_axyz, Rx, Rz, z)
                for beta in thetas:
                    R_clps = vdw_R(axyz_1, axyz_2, beta)
                    ca = R_a - 2.0 * R_clps * cosb[beta]
                    cb = R_b - 2.0 * R_clps * sinb[beta]
                    ok = (ca <= eps_a) and (cb <= eps_b)
                    all_rows.append([alpha, phi, beta, z, R_clps, ok])

    df = pd.DataFrame(all_rows, columns=['alpha','phi','beta','z','R_clps','TorF'])
    df = df.sort_values(['z','alpha','phi','beta']).reset_index(drop=True)
    out_csv = out / f"vdW_r_contact_{monomer_name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (n={len(df)})")

# --- CLI ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="vdW sweep without notebooks")
    ap.add_argument('--monomer-path', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--z-min', type=float, default=0.0)
    ap.add_argument('--z-max', type=float, required=True)
    ap.add_argument('--z-step', type=float, default=0.1)
    ap.add_argument('--alpha-min', type=float, default=0)
    ap.add_argument('--alpha-max', type=float, default=90)
    ap.add_argument('--alpha-step', type=float, default=5)
    ap.add_argument('--phi-min', type=float, default=0.0)
    ap.add_argument('--phi-max', type=float, default=0.0)
    ap.add_argument('--phi-step', type=float, default=1)
    ap.add_argument('--theta-step', type=float, default=5)
    ap.add_argument('--eps-a', type=float, default=1e-3)
    ap.add_argument('--eps-b', type=float, default=1e-2)
    args = ap.parse_args()
    sweep(args.monomer_path, args.out_dir, args.z_min, args.z_max, args.z_step,
          args.alpha_min, args.alpha_max,args.alpha_step,
          args.phi_min, args.phi_max, args.phi_step,
          args.theta_step, args.eps_a, args.eps_b)

if __name__ == '__main__':
    main()

