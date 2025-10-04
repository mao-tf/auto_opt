#!/usr/bin/env python3
"""
vdw_sweep.py — Run the vdW (rigid-sphere) parameter sweep **without any notebook**.

- Reads a monomer XYZ (format: `Element  x  y  z` per line; header lines are ignored)
- For a grid of (alpha, z, beta) it computes:
    * R_clps(theta=beta) between two monomers placed as in your notebook/code
    * R_a = self-contact at beta=0°, R_b = self-contact at 90°
    * contact_a = R_a - 2*R_clps*cos(beta), contact_b = R_b - 2*R_clps*sin(beta)
    * TorF = (contact_a <= eps_a) and (contact_b <= eps_b)
- Writes a **single CSV** with a `z` column to --out-dir as  `vdW_r_contact_<monomer>.csv`

Backends:
- default "auto": use your `C6_dimir_vdW.vdw_R` if available; otherwise fall back to a built-in solver
- built-in solver reproduces the rigid-sphere contact radius via bisection (uses internal vdW radii table)

CLI example:
  python vdw_sweep.py \
    --monomer-path /path/to/PFA.xyz \
    --out-dir /path/to/vdw_out \
    --z-max 3.0 --z-step 0.1 \
    --alpha-step 5 --theta-step 5 \
    --eps-a 1e-3 --eps-b 1e-2
"""
from __future__ import annotations
import os, math, argparse, pathlib
from typing import List, Tuple, Iterable, Optional
import numpy as np
import pandas as pd

# --- optional imports (your environment) ---
C6_AVAILABLE = False
try:
    from C6_dimir_vdW import vdw_R as vdw_R_c6  # your implementation
    C6_AVAILABLE = True
except Exception:
    pass

ROD_AVAILABLE = False
try:
    from utils import Rod as Rod_external  # your rotation (row-vector * R)
    ROD_AVAILABLE = True
except Exception:
    pass

# --- fallback helpers ---------------------------------------------------------

def rodrigues(axis: Iterable[float], angle_deg: float) -> np.ndarray:
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n == 0:
        return np.eye(3)
    a = a / n
    th = math.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], float)
    I = np.eye(3)
    return I + math.sin(th)*K + (1 - math.cos(th))*(K @ K)


def Rod(axis: Iterable[float], angle_deg: float) -> np.ndarray:
    """Fallback compatible with your utils.Rod: returns a 3x3 rotation matrix
    intended to be used as: row_vector(1x3) @ R (or np.matmul(row_vec, R))."""
    if ROD_AVAILABLE:
        return Rod_external(axis, angle_deg)
    # our Rodrigues returns R where col-vector usage is R @ v
    # but row-vector * R is also valid; we'll stick to your calling convention
    return rodrigues(axis, angle_deg)

_VDW = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'CL': 1.75, 'BR': 1.85, 'I': 1.98,
}

def vdw_radius(sym: str) -> float:
    s = sym.strip().upper()
    if s in _VDW: return _VDW[s]
    if s[:2] in _VDW: return _VDW[s[:2]]
    if s[:1] in _VDW: return _VDW[s[:1]]
    return _VDW['C']

# --- IO ----------------------------------------------------------------------

def read_xyz(path: str) -> List[List[object]]:
    """Read XYZ-like file with lines like: "C   0.0  1.0  2.0".
    Returns list of [x,y,z,element]. Ignores lines with len!=4."""
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

# --- geometry builders (your convention) -------------------------------------

def get_t_z(monomer_path: str, alpha: float, z: float):
    """Rotate around z by alpha; make monomer1 as-is and monomer2 mirrored in x and lifted by +z.
    Returns (axyz_1, axyz_2) where each is [[x,y,z,element], ...]."""
    axyz_1, axyz_2 = [], []
    ez = np.array([0., 0., 1.])
    Rz = Rod(ez, alpha)
    for x, y, zz, sym in read_xyz(monomer_path):
        vec = np.array([x, y, zz])
        rot = np.matmul(vec, Rz)  # row-vector * R (your code style)
        axyz_1.append([ rot[0],  rot[1],  rot[2],     sym])
        axyz_2.append([-rot[0],  rot[1],  rot[2] + z, sym])
    return axyz_1, axyz_2

# --- contact radius backends --------------------------------------------------

def _axyz_to_arrays(axyz):
    xyz = np.array([[a[0], a[1], a[2]] for a in axyz], float)
    rad = np.array([vdw_radius(a[3]) for a in axyz], float)
    return xyz, rad


def vdw_R_builtin(axyz_1, axyz_2, theta_deg: float, tol=1e-4, max_iter=80) -> float:
    """Rigid-sphere contact radius via bisection along direction theta (deg)."""
    xyz1, r1 = _axyz_to_arrays(axyz_1)
    xyz2, r2 = _axyz_to_arrays(axyz_2)
    ct, st = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    d = np.array([ct, st, 0.0])
    def clearance(R: float) -> float:
        d12 = (xyz2 + R*d) - (xyz1 - R*d)
        dist = np.linalg.norm(d12[:, None, :] - 0, axis=2)  # wrong shape; implement properly below
        # The above line isn't optimal; do proper pairwise distances:
        D = xyz2[None, :, :] - xyz1[:, None, :]
        D = D + R*(d[None, None, :] + d[None, None, :])  # shift both ways: +R*d - (-R*d) = 2R*d
        # Actually simpler: place directly as used:
        p1 = xyz1 - R*d
        p2 = xyz2 + R*d
        D = p2[None, :, :] - p1[:, None, :]
        dist = np.linalg.norm(D, axis=2)
        clr = dist - (r1[:, None] + r2[None, :])
        return float(np.min(clr))
    # if already non-overlapping at R=0
    c0 = clearance(0.0)
    if c0 >= 0:
        return 0.0
    hi = 0.5
    for _ in range(64):
        if clearance(hi) >= 0: break
        hi *= 1.7
    lo = 0.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        cm = clearance(mid)
        if abs(cm) < tol or (hi-lo) < tol:
            return float(mid)
        if cm < 0: lo = mid
        else: hi = mid
    return float(0.5*(lo+hi))


def vdw_R(axyz_1, axyz_2, theta_deg: float, backend: str) -> float:
    if backend == 'c6':
        if not C6_AVAILABLE:
            raise RuntimeError("backend 'c6' requested but C6_dimir_vdW not importable")
        return float(vdw_R_c6(axyz_1, axyz_2, theta_deg))
    if backend == 'auto':
        if C6_AVAILABLE:
            return float(vdw_R_c6(axyz_1, axyz_2, theta_deg))
        return vdw_R_builtin(axyz_1, axyz_2, theta_deg)
    return vdw_R_builtin(axyz_1, axyz_2, theta_deg)

# --- sweep -------------------------------------------------------------------

def sweep(monomer_path: str, out_dir: str, z_max: float, z_step: float,
          alpha_step: float, theta_step: float, eps_a: float, eps_b: float,
          backend: str = 'auto') -> None:
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    monomer_name = pathlib.Path(monomer_path).stem

    z_vals = [round(z, 1) for z in np.arange(0.0, z_max + 1e-9, z_step)]
    alphas = [float(a) for a in np.arange(0, 91, alpha_step)]
    thetas = [float(t) for t in np.arange(0, 91, theta_step)]

    all_rows = []
    for z in z_vals:
        for alpha in alphas:
            axyz_1, axyz_2 = get_t_z(monomer_path, alpha, z)
            # self-contact once per alpha
            R_a = vdw_R(axyz_1, axyz_1, 0.0, backend)
            R_b = vdw_R(axyz_1, axyz_1, 90.0, backend)
            for beta in thetas:
                R_clps = vdw_R(axyz_1, axyz_2, beta, backend)
                ca = R_a - 2.0 * R_clps * math.cos(math.radians(beta))
                cb = R_b - 2.0 * R_clps * math.sin(math.radians(beta))
                ok = (ca <= eps_a) and (cb <= eps_b)
                all_rows.append([alpha, beta, z, R_clps, ok])

    df = pd.DataFrame(all_rows, columns=['alpha','beta','z','R_clps','TorF'])
    df = df.sort_values(['z','alpha','beta']).reset_index(drop=True)
    out_csv = out / f"vdW_r_contact_{monomer_name}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (n={len(df)})")

# --- CLI ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="vdW sweep without notebooks")
    ap.add_argument('--monomer-path', required=True, help='path to monomer .xyz (Element x y z per line)')
    ap.add_argument('--out-dir', required=True, help='directory to write CSVs')
    ap.add_argument('--z-max', type=float, required=True)
    ap.add_argument('--z-step', type=float, default=0.1)
    ap.add_argument('--alpha-step', type=float, default=5)
    ap.add_argument('--theta-step', type=float, default=5)
    ap.add_argument('--eps-a', type=float, default=1e-3)
    ap.add_argument('--eps-b', type=float, default=1e-2)
    ap.add_argument('--backend', choices=['auto','c6','builtin'], default='auto')
    args = ap.parse_args()

    sweep(args.monomer_path, args.out_dir, args.z_max, args.z_step,
          args.alpha_step, args.theta_step, args.eps_a, args.eps_b,
          backend=args.backend)

if __name__ == '__main__':
    main()
