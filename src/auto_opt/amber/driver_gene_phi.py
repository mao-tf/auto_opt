##tetracene層内計算
"""
python -m auto_opt.amber.driver_gene --auto-dir runs/PFA_test --monomer-name PFA --num-nodes 2 --isTest

律速改善版: driver_screw_phi.py と同じ設計（driver_stacking.py 踏襲）。
ファイルポーリング + subprocess.run による直列ブロッキング実行を廃止し、
メモリ上のジョブ管理 + subprocess.Popen による非同期並列実行に変更した。
--num-nodes は「同時実行 Amber ジョブ数」を意味する
（job_phi.py で SGE ノードの空きコア数から算出される値）。
"""
import pandas as pd
import time
import subprocess
from auto_opt.amber.make_io_gene_phi import write_dimer_inputs, ensure_frcmod
from auto_opt.utils import amber_get_E
import argparse
import numpy as np
import shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
AMBER_REF = DATA / "amber_ref"
RES = Path(__file__).resolve().parent / "resources"

fixed_param_keys = ['alpha', 'phi']
all_keys         = ['alpha', 'phi', 'z', 'a', 'b']
# 各ダイマーの識別キー: 平行移動に z が出てこないダイマーは z を含まない
DIMER_KEYS = {
    1: ['alpha', 'phi', 'a'],            # a-dimer: 平行移動 (a, 0, 0) → z 無依存
    2: ['alpha', 'phi', 'b', 'z'],       # b-dimer: 平行移動 (0, b, 2z) → z 依存
    3: ['alpha', 'phi', 'a', 'b', 'z'],  # t1-dimer: 平行移動 (a/2, b/2, z) → z 依存
    # t2-dimer: 平行移動 (-a/2, b/2, z)。モノマーに反転対称性が無い場合は
    # t1と別エネルギーになるため --asymmetric 指定時のみ使う
    4: ['alpha', 'phi', 'a', 'b', 'z'],
}


def _prepare_amber_resources(auto_dir: str):
    amber_dir = Path(auto_dir) / "amber"
    amber_dir.mkdir(parents=True, exist_ok=True)
    src = RES / "FF_calc.in"
    if src.exists():
        shutil.copy2(src, amber_dir / "FF_calc.in")


def _round_key(params_dict, keys):
    return tuple(np.round(float(params_dict[k]), 2) for k in keys)


def _fmt(v):
    return f"{np.round(float(v), 2)}".replace('.', 'p').replace('-', 'm')


def _base_name(monomer_name, params_dict):
    parts = [monomer_name] + [f"{k}{_fmt(params_dict[k])}" for k in all_keys]
    return '_'.join(parts)


def _opt_search_step(grid_results, fixed_params_dict, init_point):
    """
    座標降下法で次に評価すべき (a, b) を求める。z は init_point 固定のまま。
    step1.csv は (alpha, phi, z, a, b) を持つが、fixed_params_dict は (alpha, phi) のみなので
    z を明示的に固定して絞り込む。
    """
    z = init_point['z']
    a_prev, b_prev = init_point['a'], init_point['b']
    while True:
        E_list = []; ab_list = []; para_list = []
        for a in [a_prev - 0.1, a_prev, a_prev + 0.1]:
            for b in [b_prev - 0.1, b_prev, b_prev + 0.1]:
                a, b = np.round(a, 1), np.round(b, 1)
                point = {**fixed_params_dict, 'z': z, 'a': a, 'b': b}
                key = _round_key(point, all_keys)
                if key in grid_results:
                    ab_list.append(point)
                    E_list.append(grid_results[key]['E'])
                else:
                    para_list.append(point)
        if para_list:
            return para_list, None
        best = ab_list[int(np.argmin(E_list))]
        if best['a'] == a_prev and best['b'] == b_prev:
            return None, best
        a_prev, b_prev = best['a'], best['b']


def main_process(args):
    auto_dir = str(Path(args.auto_dir).resolve())
    _prepare_amber_resources(auto_dir)
    amber_dir = os.path.join(auto_dir, 'amber')
    os.makedirs(amber_dir, exist_ok=True)
    os.makedirs(os.path.join(auto_dir, 'gaussview'), exist_ok=True)
    if not args.isTest:
        ensure_frcmod(auto_dir, args.monomer_name, monomer_dir=args.monomer_dir)

    init_csv = os.path.join(auto_dir, 'step1_init_params.csv')
    df_init = pd.read_csv(init_csv)
    # num_nodes は「同時実行 Amber ジョブ数」の意味に変わったため、探索ブランチは
    # 最初から全て有効化する（実際の実行速度は job_queue の draining 側で絞られる）
    df_init['status'] = 'InProgress'
    df_init.to_csv(init_csv, index=False)

    amber_ref_dir = Path(args.amber_ref_dir) if args.amber_ref_dir else AMBER_REF
    mono_file = str(amber_ref_dir / f'{args.monomer_name}_gaff2.out')
    E_mono = amber_get_E(mono_file)[0]

    # 非対称モノマー（t1≠t2）は4ダイマー・対称モノマー（t1=t2想定）は3ダイマーで計算する
    n_dimer_types = 4 if args.asymmetric else 3

    dimer_energy  = {n: {} for n in range(1, n_dimer_types + 1)}   # key(tuple) -> E (2*E_mono 差し引き済み)
    dimer_pending = {n: {} for n in range(1, n_dimer_types + 1)}   # key(tuple) -> 計算中ジョブの base 名
    pending_points = {}   # key(tuple) -> point(dict)  … 依頼済みでまだ未解決
    grid_results   = {}   # key(tuple) -> row(dict)    … 解決済み(Done)
    job_queue = []
    running   = {}
    step1_csv = os.path.join(auto_dir, 'step1.csv')
    last_save = time.time()

    def try_resolve(key):
        point = pending_points.get(key)
        if point is None:
            return
        keys_n = {n: _round_key(point, DIMER_KEYS[n]) for n in range(1, n_dimer_types + 1)}
        if not all(keys_n[n] in dimer_energy[n] for n in range(1, n_dimer_types + 1)):
            return
        rec = {**point, 'status': 'Done'}
        if n_dimer_types == 4:
            E1, E2, E3, E4 = (dimer_energy[n][keys_n[n]] for n in range(1, 5))
            E = 2 * E1 + 2 * E2 + 2 * E3 + 2 * E4
            rec['E4'] = round(E4, 4)
        else:
            E1, E2, E3 = (dimer_energy[n][keys_n[n]] for n in range(1, 4))
            E = 2 * E1 + 2 * E2 + 4 * E3
        rec.update({'E': round(E, 4), 'E1': round(E1, 4), 'E2': round(E2, 4), 'E3': round(E3, 4)})
        grid_results[key] = rec
        del pending_points[key]

    def request_point(point):
        key = _round_key(point, all_keys)
        if key in grid_results or key in pending_points:
            return
        pending_points[key] = point

        needed = []
        for n in range(1, n_dimer_types + 1):
            key_n = _round_key(point, DIMER_KEYS[n])
            if key_n in dimer_energy[n] or key_n in dimer_pending[n]:
                continue
            needed.append((n, key_n))

        if not needed:
            try_resolve(key)
            return

        base = _base_name(args.monomer_name, point)
        for n, key_n in needed:
            dimer_pending[n][key_n] = base
        # ダイマー入力ファイル(mol2/tleap.in)の実書き出しは launch() まで遅延させる。
        # ここで書き出すと起動前に全探索点ぶんのI/Oが一気に走り、num_nodesの並列度が
        # 効き始めるまでに長い無並列区間ができてしまう（driver_stacking.py と同じ設計に統一）。
        job_queue.append({'base': base, 'point': point, 'needed': needed})

    def launch(job):
        done_path = os.path.join(amber_dir, f"{job['base']}.done")

        dimers = []
        for n, key_n in job['needed']:
            out_name, tleap_cmd, sander_cmd = write_dimer_inputs(
                auto_dir, args.monomer_name, job['point'], structure_type=n,
                monomer_dir=args.monomer_dir,
            )
            dimers.append((n, key_n, out_name, tleap_cmd, sander_cmd))
        job['dimers'] = dimers

        if args.isTest:
            # 実 Amber を呼ばずに疑似エネルギーで即完了させ、状態遷移ロジックだけを検証できるようにする
            for _, _, out_name, _, _ in job['dimers']:
                fake_E = -float(abs(hash(out_name)) % 1000) / 10.0
                with open(os.path.join(amber_dir, out_name), 'w') as f:
                    f.write(" RMS \n")
                    f.write(f"1 {fake_E} 0.0\n")
            Path(done_path).touch()
            return

        cmds = []
        for _, _, _, tleap_cmd, sander_cmd in job['dimers']:
            cmds.append(tleap_cmd)
            cmds.append(sander_cmd)
        cmds.append(f'touch "{done_path}"')
        job_path = os.path.join(amber_dir, f"job_{job['base']}.sh")
        with open(job_path, 'w') as f:
            f.write(
                "#!/bin/bash\n"
                # 1ダイマー(数十原子)の1点評価にOpenMPスレッド並列化の恩恵はなく、
                # num_nodes分のsanderが同時に全コアを奪い合ってスレッド過多になるのを防ぐ
                "export OMP_NUM_THREADS=1\n"
                "export MKL_NUM_THREADS=1\n"
                "export OPENBLAS_NUM_THREADS=1\n"
                f"cd {amber_dir}\n" + "\n".join(cmds) + "\n"
            )
        os.chmod(job_path, 0o755)
        subprocess.Popen([job_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    while True:
        # ── 完了チェック ────────────────────────────────────
        finished = []
        for base, job in running.items():
            if not os.path.exists(os.path.join(amber_dir, f'{base}.done')):
                continue
            for n, key_n, out_name, _, _ in job['dimers']:
                try:
                    E_list = amber_get_E(os.path.join(amber_dir, out_name))
                    E = round(float(E_list[0]) - 2 * E_mono, 4) if len(E_list) == 1 else 99999.0
                except Exception:
                    E = 99999.0
                dimer_energy[n][key_n] = E
                dimer_pending[n].pop(key_n, None)
            finished.append(base)
        for b in finished:
            del running[b]
        if finished:
            for key in list(pending_points.keys()):
                try_resolve(key)

        # ── 座標降下法: 各探索ブランチの次の要求点 ──────────────
        for idx, row in df_init.iterrows():
            if row['status'] == 'Done':
                continue
            fixed_params_dict = {k: row[k] for k in fixed_param_keys}
            init_point = {k: row[k] for k in all_keys}
            para_list, best = _opt_search_step(grid_results, fixed_params_dict, init_point)
            if para_list is None:
                df_init.loc[idx, 'status'] = 'Done'
                continue
            for point in para_list:
                request_point(point)

        # ── ジョブ投入（同時実行数は num_nodes で制限） ───────────
        while len(running) < args.num_nodes and job_queue:
            job = job_queue.pop(0)
            launch(job)
            running[job['base']] = job

        # ── 定期保存 ────────────────────────────────────────
        now = time.time()
        if now - last_save > 10.0 and grid_results:
            pd.DataFrame(list(grid_results.values())).to_csv(step1_csv, index=False)
            df_init.to_csv(init_csv, index=False)
            last_save = now
            n_done = int((df_init['status'] == 'Done').sum())
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"{n_done}/{len(df_init)} 探索完了 "
                  f"(実行中: {len(running)}, キュー: {len(job_queue)}, 解決済み点: {len(grid_results)})")

        # ── 終了判定 ────────────────────────────────────────
        if (df_init['status'] == 'Done').all() and not running and not job_queue:
            if grid_results:
                pd.DataFrame(list(grid_results.values())).to_csv(step1_csv, index=False)
            df_init.to_csv(init_csv, index=False)
            break

        time.sleep(0.2 if args.isTest else 0.1)

    from auto_opt.gaussian.extract_minima import extract_minima
    extract_minima(symmetry='glide', auto_dir=auto_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--isTest', action='store_true')
    parser.add_argument('--auto-dir', type=str, help='path to dir which includes amber, gaussview and csv')
    parser.add_argument('--monomer-name', type=str, help='monomer name')
    parser.add_argument('--num-nodes', type=int, help='num nodes')
    parser.add_argument('--monomer-dir', type=str, default=None,
                        help='monomer mol2 の探索先（省略時はパッケージ内 data/monomer）')
    parser.add_argument('--amber-ref-dir', type=str, default=None,
                        help='amber_ref の探索先（省略時はパッケージ内 data/amber_ref）')
    parser.add_argument('--asymmetric', action='store_true',
                        help='モノマーに反転対称性が無く t1≠t2 の場合に指定（4ダイマー計算になる）')
    args = parser.parse_args()

    print("----main process----")
    main_process(args)
    print("----finish process----")
