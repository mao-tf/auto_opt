from __future__ import annotations
import numpy as np
from typing import List

# VdW radii (Å) — Bondi values
_VDW: dict[str, float] = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
    'F': 1.47, 'P': 1.80, 'S': 1.80,
    'CL': 1.75, 'BR': 1.85, 'I': 1.98,
}

# 逆引き: 半径値 → 元素記号（R2atom 用）
_RADIUS_TO_ATOM: dict[float, str] = {v: k for k, v in _VDW.items()}


def vdw_radius(sym: str) -> float:
    """元素記号 → VdW 半径 (Å)。未知元素は C の値を返す。"""
    return _VDW.get(sym.strip().upper(), _VDW['C'])


def R2atom(R: float) -> str:
    """VdW 半径値 → 元素記号。未知の値は 'C' を返す。"""
    return _RADIUS_TO_ATOM.get(round(R, 2), 'C')


def Rod(n: np.ndarray, theta_in: float) -> np.ndarray:
    """ロドリゲス回転行列: 軸 n まわりに theta_in 度回転。"""
    nx, ny, nz = n
    c = np.cos(np.radians(theta_in))
    s = np.sin(np.radians(theta_in))
    return np.array([
        [c + nx*nx*(1-c),     nx*ny*(1-c) - nz*s,  nx*nz*(1-c) + ny*s],
        [nx*ny*(1-c) + nz*s,  c + ny*ny*(1-c),     ny*nz*(1-c) - nx*s],
        [nx*nz*(1-c) - ny*s,  ny*nz*(1-c) + nx*s,  c + nz*nz*(1-c)   ],
    ])


def amber_get_E(filepath: str) -> List[float]:
    """
    Amber sander 出力ファイルからエネルギー (kcal/mol) を読む。
    見つかれば [E] を、見つからなければ [] を返す。
    """
    found_rms = False
    with open(filepath) as f:
        for line in f:
            if found_rms:
                s = line.split()
                return [float(s[1])]
            if ' RMS ' in line:
                found_rms = True
    return []
