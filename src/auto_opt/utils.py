from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

_MONOMER_DIR = Path(__file__).resolve().parents[2] / "data" / "monomer"

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


def read_xyz(path: str) -> List[List]:
    """XYZ ファイルを読み込み [[x, y, z, symbol], ...] のリストを返す。"""
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
        raise ValueError(f"XYZ 行が読み取れません: {path}  (形式: 'El x y z')")
    return rows


def vdw_R(axyz_1: List[List], axyz_2: List[List], theta_deg: float) -> float:
    """
    剛体球モデルで分子2の接触距離を返す。
    theta_deg: eR 方向（ab面内の接触方向角, degree）
    戻り値: 分子2を eR 方向に押した時の最小接触距離 (Å)。接触不可なら 0.0。
    """
    R1 = np.asarray([[x, y, z] for x, y, z, _ in axyz_1], float)
    R2 = np.asarray([[x, y, z] for x, y, z, _ in axyz_2], float)
    r1 = np.asarray([vdw_radius(a[3]) for a in axyz_1], float)
    r2 = np.asarray([vdw_radius(a[3]) for a in axyz_2], float)

    ct = math.cos(math.radians(theta_deg))
    st = math.sin(math.radians(theta_deg))
    eR = np.array([ct, st, 0.0], float)

    D      = R2[None, :, :] - R1[:, None, :]
    R12b   = D @ eR
    R12a2  = (D * D).sum(axis=2) - R12b ** 2

    rad_sum    = r1[:, None] + r2[None, :]
    rad_sum_sq = rad_sum ** 2
    mask = R12a2 < rad_sum_sq

    if not np.any(mask):
        return 0.0

    sq        = rad_sum_sq[mask] - R12a2[mask]
    twoR_need = np.maximum(-R12b[mask] + np.sqrt(sq), 0.0)
    return float(np.max(twoR_need))


def place_monomer(
    monomer_name: str,
    tx: float, ty: float, tz: float,
    phi: float, alpha: float, beta: float = 0.0,
    monomer_dir: str | None = None,
) -> np.ndarray:
    """
    モノマー CSV を読み、回転と平行移動を適用して (N,4) 配列を返す。

    回転順: phi(-x軸) → alpha(z軸) → beta(-x軸) → 平行移動(tx,ty,tz)
    glide では beta=0.0 (デフォルト) のまま使う。
    """
    _dir = Path(monomer_dir) if monomer_dir else _MONOMER_DIR
    df_mono = pd.read_csv(_dir / f"{monomer_name}.csv")
    arr = df_mono[['X', 'Y', 'Z', 'R']].to_numpy(dtype=float)

    ex = np.array([1., 0., 0.])
    ez = np.array([0., 0., 1.])
    xyz = arr[:, :3]
    xyz = xyz @ Rod(-ex, phi).T
    xyz = xyz @ Rod(ez, alpha).T
    xyz = xyz @ Rod(-ex, beta).T
    xyz = xyz + np.array([tx, ty, tz])

    R = arr[:, 3].reshape((-1, 1))
    return np.concatenate([xyz, R], axis=1)


def filter_df(df: "pd.DataFrame", dict_filter: dict) -> "pd.DataFrame":
    """各キーの値で DataFrame をフィルタリング。float は np.isclose で比較。"""
    for k, v in dict_filter.items():
        if isinstance(v, float):
            # 空CSVを読み返したとき列がobject型になる場合があるので数値変換してから比較
            col = pd.to_numeric(df[k], errors='coerce')
            df = df[np.isclose(col, v, atol=1e-5)]
        else:
            df = df[df[k] == v]
    return df


def check_calc_status(auto_dir: str, params_dict: dict) -> bool:
    """step1.csv を読み、params_dict に対応する行が Done かどうかを返す。"""
    df_E = pd.read_csv(Path(auto_dir) / "step1.csv")
    if len(df_E) == 0:
        return False
    df_f = filter_df(df_E, params_dict).reset_index(drop=True)
    if len(df_f) > 0:
        return bool(df_f.loc[0, "status"] == "Done")
    return False


def get_values_from_df(df: "pd.DataFrame", index: int, key) -> object:
    return df.loc[index, key]


def update_value_in_df(df: "pd.DataFrame", index: int, key: str, value) -> "pd.DataFrame":
    df.loc[index, key] = value
    return df


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
