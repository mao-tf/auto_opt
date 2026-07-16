#!/usr/bin/env python3
"""
スタッキング Amber 最適化ドライバー (glide / screw 共通)

step1_init_params.csv の各 (cx, cy) について cz を±0.1 ずつ広げながら
エネルギー最小値を探す。各計算は非同期バックグラウンドで実行。

Usage:
  python -m auto_opt.stacking.driver_stacking \
      --auto-dir runs/BTBT_glide/split_0 \
      --monomer-name BTBT \
      --symmetry glide \
      --num-nodes 38
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from auto_opt.cluster import get_amber_tool
from auto_opt.utils import amber_get_E
from auto_opt.amber.make_io_gene_phi import _guess_mol2_path

_FF_CALC = Path(__file__).resolve().parent / "resources" / "FF_calc.in"

# ──────────────────────────────────────────────────────────────
#  対称性ごとの設定
# ──────────────────────────────────────────────────────────────

_FIXED_KEYS = {
    'glide': ['a', 'b', 'z', 'phi', 'alpha1', 'alpha2'],
    'screw': ['a', 'bt1', 'bt2', 'b', 'z', 'beta', 'phi', 'alpha1', 'alpha2'],
}

_GROUP_COLS = {
    'glide': ['z', 'phi', 'cx', 'cy'],
    'screw': ['beta', 'z', 'phi', 'cx', 'cy'],
}


def _group_key(params: dict, symmetry: str) -> tuple:
    return tuple(round(float(params[c]), 1) for c in _GROUP_COLS[symmetry])


def _fmt(v: float, dec: int) -> str:
    return f"{v:.{dec}f}".replace('.', 'p').replace('-', 'm')


def _base_name(monomer: str, params: dict, symmetry: str) -> str:
    parts = [monomer]
    if symmetry == 'screw':
        parts += [f"beta{_fmt(params['beta'], 1)}"]
    parts += [
        f"z{_fmt(params['z'],   1)}",
        f"phi{_fmt(params['phi'], 1)}",
        f"cx{_fmt(params['cx'],  2)}",
        f"cy{_fmt(params['cy'],  2)}",
        f"cz{_fmt(params['cz'],  2)}",
    ]
    return '_'.join(parts)


# ──────────────────────────────────────────────────────────────
#  1 タスク実行
# ──────────────────────────────────────────────────────────────

def _exec_job(out_dir: Path, monomer: str, params: dict,
              symmetry: str, is_test: bool, monomer_dir: str | None = None) -> str:
    """mol2・tleap 入力を書き出してバックグラウンドシェルジョブを起動。"""
    if symmetry == 'screw':
        from auto_opt.stacking.make_io_stacking_screw import get_pairs_xyzR, get_mol2_lines
    else:
        from auto_opt.stacking.make_io_stacking import get_pairs_xyzR, get_mol2_lines

    base         = _base_name(monomer, params, symmetry)
    pairs        = get_pairs_xyzR(monomer, params, monomer_dir=monomer_dir)
    monomer_mol2 = str(_guess_mol2_path(monomer, monomer_dir))
    frcmod       = f"{monomer}_gaff2.frcmod"

    parmchk2 = get_amber_tool('parmchk2')
    tleap    = get_amber_tool('tleap')
    sander   = get_amber_tool('sander')

    cmds: List[str] = [
        f'if [ ! -f "{frcmod}" ]; then '
        f'"{parmchk2}" -s gaff2 -i "{monomer_mol2}" -f mol2 -o "{frcmod}"; fi',
    ]

    for i, dimer in enumerate(pairs):
        fn       = f"{base}_p{i}"
        mol2_txt = "".join(get_mol2_lines(dimer, monomer, monomer_dir=monomer_dir))
        (out_dir / f"{fn}.mol2").write_text(mol2_txt)

        tleap_txt = (
            "source leaprc.gaff2\n"
            f"loadamberparams {frcmod}\n"
            f"MOL = loadmol2 {monomer_mol2}\n"
            f"SYS = loadmol2 {fn}.mol2\n"
            f"saveamberparm SYS {fn}.prmtop {fn}.inpcrd\n"
            "quit\n"
        )
        (out_dir / f"{fn}_tleap.in").write_text(tleap_txt)

        cmds += [
            f'"{tleap}" -f {fn}_tleap.in > {fn}_tleap.out',
            f'"{sander}" -O -i FF_calc.in -o {fn}.out '
            f'-p {fn}.prmtop -c {fn}.inpcrd -r {fn}.rst7',
        ]

    done_path = out_dir / f"{base}.done"
    cmds.append(f'touch "{done_path}"')

    job_path = out_dir / f"job_{base}.sh"
    job_path.write_text(
        "#!/bin/bash\n"
        # 1ダイマー(数十原子)の1点評価にOpenMPスレッド並列化の恩恵はなく、
        # num_nodes分のsanderが同時に全コアを奪い合ってスレッド過多になるのを防ぐ
        "export OMP_NUM_THREADS=1\n"
        "export MKL_NUM_THREADS=1\n"
        "export OPENBLAS_NUM_THREADS=1\n"
        f"cd {out_dir}\n"
        + "\n".join(cmds) + "\n"
    )
    job_path.chmod(0o755)

    if not is_test:
        subprocess.Popen(
            [str(job_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return base


def _read_results(out_dir: Path, base: str,
                  n_pairs: int, calc_E_total) -> List[float]:
    """done マークがあれば全ペアのエネルギーリストを返す。未完了は []。"""
    if not (out_dir / f"{base}.done").exists():
        return []
    E_list = []
    for i in range(n_pairs):
        try:
            E_list.append(float(amber_get_E(str(out_dir / f"{base}_p{i}.out"))[0]))
        except Exception:
            E_list.append(99999.0)
    return E_list


# ──────────────────────────────────────────────────────────────
#  メインプロセス
# ──────────────────────────────────────────────────────────────

def main_process(args):
    if args.symmetry == 'screw':
        from auto_opt.stacking.make_io_stacking_screw import N_PAIRS, calc_E_total
    else:
        from auto_opt.stacking.make_io_stacking import N_PAIRS, calc_E_total

    auto_dir  = Path(args.auto_dir).resolve()
    out_amber = auto_dir / 'amber'
    out_amber.mkdir(parents=True, exist_ok=True)

    if _FF_CALC.exists():
        shutil.copy2(_FF_CALC, out_amber / 'FF_calc.in')

    df_init   = pd.read_csv(auto_dir / 'step1_init_params.csv')
    step1_csv = auto_dir / 'step1.csv'
    fkeys     = _FIXED_KEYS[args.symmetry]

    # 初期状態を構築
    fixed_by_key: Dict[tuple, dict] = {}
    job_queue:    List[dict]         = []
    running:      Dict[str, dict]    = {}
    done_czs:     Dict[tuple, dict]  = {}
    queued_czs:   Dict[tuple, Set]   = {}
    group_status: Dict[tuple, str]   = {}
    results:      List[dict]         = []

    for _, row in df_init.iterrows():
        key = _group_key(row.to_dict(), args.symmetry)
        fixed_by_key[key] = {k: row[k] for k in fkeys if k in row.index}
        done_czs[key]     = {}
        queued_czs[key]   = set()
        group_status[key] = 'Working'

        cz_vdw = round(float(row['cz']), 1)
        cx     = round(float(row['cx']), 1)
        cy     = round(float(row['cy']), 1)
        for d in (-0.2, -0.1, 0.0, 0.1, 0.2):
            cz = round(cz_vdw + d, 1)
            job_queue.append({'cx': cx, 'cy': cy, 'cz': cz, **fixed_by_key[key]})
            queued_czs[key].add(cz)

    print(f"初期キュー: {len(job_queue)} タスク / {len(group_status)} グループ")
    last_save = time.time()

    while True:
        # ── 完了チェック ───────────────────────────────────────
        finished: List[str] = []
        for base, task in running.items():
            E_list = _read_results(out_amber, base, N_PAIRS, calc_E_total)
            if not E_list:
                continue
            E   = calc_E_total(E_list)
            key = _group_key(task, args.symmetry)
            done_czs[key][task['cz']] = E
            r = dict(task, E=E, status='Done', file_name=base)
            for i, e in enumerate(E_list):
                r[f'E{i+1}'] = e
            results.append(r)
            finished.append(base)
        for b in finished:
            del running[b]

        # ── 谷底判定・追加投入 ────────────────────────────────
        for key, czs in done_czs.items():
            if group_status[key] == 'Done' or not czs:
                continue
            best = min(czs, key=czs.get)
            needs = [round(best + d, 1) for d in (-0.1, 0.1) if round(best + d, 1) not in czs]
            if not needs:
                group_status[key] = 'Done'
                continue
            fixed  = fixed_by_key[key]
            cx_k   = key[_GROUP_COLS[args.symmetry].index('cx')]
            cy_k   = key[_GROUP_COLS[args.symmetry].index('cy')]
            for nb in needs:
                if nb not in queued_czs[key]:
                    job_queue.append({'cx': cx_k, 'cy': cy_k, 'cz': nb, **fixed})
                    queued_czs[key].add(nb)

        # ── ジョブ投入 ────────────────────────────────────────
        while len(running) < args.num_nodes and job_queue:
            task = job_queue.pop(0)
            base = _exec_job(out_amber, args.monomer_name, task,
                             args.symmetry, args.isTest,
                             monomer_dir=getattr(args, 'monomer_dir', None))
            running[base] = task

        # ── 定期保存 ──────────────────────────────────────────
        now = time.time()
        if now - last_save > 10.0 and results:
            pd.DataFrame(results).to_csv(step1_csv, index=False)
            last_save = now
            done_n = sum(s == 'Done' for s in group_status.values())
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"{done_n}/{len(group_status)} 完了 "
                  f"(実行中: {len(running)}, キュー: {len(job_queue)})")

        # ── 終了判定 ──────────────────────────────────────────
        if (all(s == 'Done' for s in group_status.values())
                and not running and not job_queue):
            if results:
                pd.DataFrame(results).to_csv(step1_csv, index=False)
            print("すべての探索が完了しました。")
            break

        time.sleep(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--isTest',       action='store_true')
    ap.add_argument('--auto-dir',     required=True)
    ap.add_argument('--monomer-name', required=True)
    ap.add_argument('--symmetry',     choices=['glide', 'screw'], required=True)
    ap.add_argument('--num-nodes',    type=int, required=True)
    ap.add_argument('--monomer-dir',  default=None,
                    help='モノマー CSV/mol2 の探索先（省略時はパッケージ付属の data/monomer）')
    args = ap.parse_args()

    print('---- driver_stacking start ----')
    main_process(args)
    print('---- driver_stacking finish ----')
