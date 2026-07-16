#!/usr/bin/env python3
"""
VdW グリッド全点の力場1点エネルギー計算。

step1_init_params.csv の各行についてダイマー mol2 を生成し
sander maxcyc=0 を実行する。
  glide: E = 2*E1 + 2*E2 + 4*E3
  screw: E = 2*E1 + 2*E2 + 2*E3 + 2*E4

結果は step1_init_params.csv に E カラムとして追記される。

律速改善版: 各グリッド点は座標降下法と違い完全に独立（逐次依存なし）なので、
driver_stacking.py / driver_gene_phi.py と同じ非同期 Popen 並列実行に書き換えた。
旧実装は df.iterrows() で1点ずつ subprocess.run（ブロッキング）しており、
--num-nodes を渡しても並列度ゼロだった。

Usage (単独/分割後の split dir で実行):
  python -m auto_opt.amber.eval_vdw_grid \
      --auto-dir runs/BTBT_glide/eval_split_0 \
      --monomer-name BTBT \
      --symmetry glide \
      --num-nodes 38
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd

from auto_opt.utils import amber_get_E

_RES      = Path(__file__).resolve().parent / "resources"
_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_AMBER_REF = _DATA_DIR / "amber_ref"

# structure_type → 重み（E = Σ weight * E_dimer）
_GLIDE_WEIGHTS = {1: 2, 2: 2, 3: 4}
_SCREW_WEIGHTS = {1: 2, 2: 2, 3: 2, 4: 2}


def _get_e_mono(monomer_name: str) -> float:
    cand = _AMBER_REF / f'{monomer_name}_gaff2.out'
    if not cand.exists():
        cands = sorted(_AMBER_REF.glob(f'{monomer_name}*.out'))
        if not cands:
            raise FileNotFoundError(f"amber_ref not found for {monomer_name}")
        cand = cands[0]
    return amber_get_E(str(cand))[0]


def _prepare_resources(auto_dir: Path) -> None:
    amber_dir = auto_dir / 'amber'
    amber_dir.mkdir(parents=True, exist_ok=True)
    src = _RES / 'FF_calc.in'
    if src.exists():
        shutil.copy2(src, amber_dir / 'FF_calc.in')


def _base_name(monomer_name: str, idx: int) -> str:
    return f"{monomer_name}_evalrow{idx}"


def run_eval(auto_dir: str, monomer_name: str, symmetry: str,
             monomer_dir: str | None = None, num_nodes: int = 4) -> None:
    base = Path(auto_dir).resolve()
    amber_dir = base / 'amber'
    _prepare_resources(base)

    csv_path = base / 'step1_init_params.csv'
    df = pd.read_csv(csv_path)

    e_mono = _get_e_mono(monomer_name)
    print(f"E_mono = {e_mono:.4f} kcal/mol")
    print(f"{len(df)} 点を評価します ({symmetry}, num_nodes={num_nodes})...")

    if symmetry == 'glide':
        from auto_opt.amber.make_io_gene_phi import write_dimer_inputs, ensure_frcmod
        weights = _GLIDE_WEIGHTS
    else:
        from auto_opt.amber.make_io_gene_screw_phi import write_dimer_inputs, ensure_frcmod
        weights = _SCREW_WEIGHTS

    ensure_frcmod(str(base), monomer_name, monomer_dir=monomer_dir)

    # ── ジョブキュー構築: 1行 = 1ジョブ（その行に必要な全ダイマーを束ねる） ──
    job_queue = []
    for idx, row in df.iterrows():
        params = row.to_dict()
        base_name = _base_name(monomer_name, idx)
        dimers = []
        for stype, w in weights.items():
            out_name, tleap_cmd, sander_cmd = write_dimer_inputs(
                str(base), monomer_name, params, structure_type=stype,
                monomer_dir=monomer_dir,
            )
            dimers.append((stype, w, out_name, tleap_cmd, sander_cmd))
        job_queue.append({'idx': idx, 'base': base_name, 'dimers': dimers})

    def launch(job):
        done_path = amber_dir / f"{job['base']}.done"
        cmds = []
        for _, _, _, tleap_cmd, sander_cmd in job['dimers']:
            cmds.append(tleap_cmd)
            cmds.append(sander_cmd)
        cmds.append(f'touch "{done_path}"')
        job_path = amber_dir / f"job_{job['base']}.sh"
        job_path.write_text(
            "#!/bin/bash\n"
            # 1ダイマー(数十原子)の1点評価にOpenMPスレッド並列化の恩恵はなく、
            # num_nodes分のsanderが同時に全コアを奪い合ってスレッド過多になるのを防ぐ
            "export OMP_NUM_THREADS=1\n"
            "export MKL_NUM_THREADS=1\n"
            "export OPENBLAS_NUM_THREADS=1\n"
            f"cd {amber_dir}\n" + "\n".join(cmds) + "\n"
        )
        os.chmod(job_path, 0o755)
        subprocess.Popen([str(job_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    running: dict = {}
    results: dict = {}
    total = len(job_queue)
    n_done = 0

    while job_queue or running:
        # 完了チェック
        finished = []
        for base_key, job in running.items():
            if not (amber_dir / f"{base_key}.done").exists():
                continue
            total_e = 0.0
            ok = True
            for stype, w, out_name, _, _ in job['dimers']:
                e_list = amber_get_E(str(amber_dir / out_name))
                if not e_list:
                    ok = False
                    break
                total_e += w * (e_list[0] - 2 * e_mono)
            results[job['idx']] = round(total_e, 4) if ok else None
            finished.append(base_key)
        for b in finished:
            del running[b]
            n_done += 1
            print(f"\r  [{n_done}/{total}] 完了", end='', flush=True)

        # ジョブ投入（同時実行数は num_nodes で制限）
        while len(running) < num_nodes and job_queue:
            job = job_queue.pop(0)
            launch(job)
            running[job['base']] = job

        if job_queue or running:
            time.sleep(0.1)

    print()
    df['E'] = [results.get(idx) for idx in df.index]
    df.to_csv(csv_path, index=False)
    print(f"完了: {csv_path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='VdW グリッド全点の力場1点エネルギーを計算して E カラムを追加する'
    )
    ap.add_argument('--auto-dir',     required=True)
    ap.add_argument('--monomer-name', required=True)
    ap.add_argument('--symmetry',     default='glide', choices=['glide', 'screw'])
    ap.add_argument('--monomer-dir',  default=None,
                    help='モノマー CSV/XYZ ディレクトリ（省略時はパッケージ付属の data/monomer）')
    ap.add_argument('--num-nodes',    type=int, default=4,
                    help='同時実行 Amber ジョブ数')
    args = ap.parse_args()
    run_eval(args.auto_dir, args.monomer_name, args.symmetry,
              monomer_dir=args.monomer_dir, num_nodes=args.num_nodes)
