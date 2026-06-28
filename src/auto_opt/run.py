#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py  ―  Step 1→4 オーケストレーター

run_config.yaml を読み込み、VdW スウィープ → Amber 最適化 → 局所最小収集 を
順番に実行する。途中で止めて結果を確認したい場合は --stop-after を使う。

Usage:
  # 全ステップ実行（VdW → Amber → collect）
  python -m auto_opt.run --config runs/BTBT_glide/run_config.yaml

  # VdW スウィープだけ実行して止める
  python -m auto_opt.run --config runs/BTBT_glide/run_config.yaml --stop-after vdw

  # VdW 結果が既にある場合、Amber から再開
  python -m auto_opt.run --config runs/BTBT_glide/run_config.yaml --start-from amber

  # 全 qsub ジョブ完了後に結果を収集するだけ
  python -m auto_opt.run --config runs/BTBT_glide/run_config.yaml --start-from collect

  # コマンドを確認するだけ（実行しない）
  python -m auto_opt.run --config runs/BTBT_glide/run_config.yaml --dry-run
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import yaml

# 実行ステップの順序
STEPS = ['monomer', 'vdw', 'amber', 'collect', 'stacking', 'merge']

# デフォルト data ディレクトリ（パッケージ相対・後方互換用）
_DATA_DIR    = Path(__file__).resolve().parents[2] / "data"
_MONOMER_DIR = _DATA_DIR / "monomer"


def _resolve_data_dirs(config: dict) -> tuple[Path, Path]:
    """config から monomer_dir と amber_ref_dir を解決する。
    data_dir が指定されていればそこを使い、なければパッケージ付属の data/ にフォールバック。
    """
    if 'data_dir' in config:
        data = Path(config['data_dir'])
    else:
        data = _DATA_DIR
    return data / 'monomer', data / 'amber_ref'


def run_monomer_step(config: dict, dry_run: bool = False) -> None:
    """モノマー準備: opt → RESP → amber_ref を連続実行する。"""
    from auto_opt.monomer.prep_monomer import run_all

    mon = config['monomer']
    mono_dir, amber_ref_dir = _resolve_data_dirs(config)

    mol2 = mono_dir      / f'{mon}.mol2'
    ref  = amber_ref_dir / f'{mon}_gaff2.out'

    if mol2.exists() and ref.exists():
        print(f'[monomer] {mol2.name} と {ref.name} が既に存在します。スキップします。')
        return

    if 'monomer_xyz' not in config:
        raise SystemExit(
            f'[monomer] run_config.yaml に monomer_xyz が必要です。\n'
            f'  monomer_xyz: {mono_dir}/{mon}_raw.xyz'
        )

    xyz_path = Path(config['monomer_xyz'])
    if not xyz_path.exists():
        raise SystemExit(f'[monomer] XYZ ファイルが見つかりません: {xyz_path}')

    if dry_run:
        print(f'[dry-run] prep_monomer.run_all({mon}, {xyz_path}, mono_dir={mono_dir})')
        return

    run_all(
        monomer=mon,
        xyz_path=xyz_path,
        make_amber_ref=True,
        charge=config.get('charge', 0),
        mult=config.get('mult', 1),
        opt_level=config.get('opt_level', 'B3LYP/6-31G(d)'),
        esp_level=config.get('esp_level', 'HF/6-31G(d)'),
        mono_dir=mono_dir,
        amber_ref_dir=amber_ref_dir,
    )


def _run(args: list[str], dry_run: bool = False) -> None:
    cmd = [sys.executable, '-m'] + args
    print(f"\n[auto_opt] $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def _range_args(params: dict, key: str) -> list[str]:
    """params[key] の min/max/step を '--{key}-min ... --{key}-max ... --{key}-step ...' に変換。"""
    p = params.get(key)
    if p is None or not isinstance(p, dict):
        return []
    result = []
    for attr in ('min', 'max', 'step'):
        if attr in p:
            result += [f'--{key}-{attr}', str(p[attr])]
    return result


def collect_results(auto_dir: str, dry_run: bool = False) -> None:
    """
    各 phi/alpha サブディレクトリの filtered_step1.csv を集約して
    auto_dir/filtered_step1.csv に書き出す。
    """
    root = Path(auto_dir)
    csvs = sorted(root.glob('*/filtered_step1.csv'))

    if not csvs:
        print(f"[collect] filtered_step1.csv が見つかりません: {root}/*/")
        return

    print(f"\n[collect] {len(csvs)} 件の filtered_step1.csv を結合します")
    for p in csvs:
        print(f"  {p}")

    if dry_run:
        print(f"[dry-run] -> {root / 'filtered_step1.csv'}")
        return

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    out = root / 'filtered_step1.csv'
    df.to_csv(out, index=False)
    print(f"[collect] -> {out}  (n={len(df)})")


def run_pipeline(
    config: dict,
    stop_after: str = 'collect',
    start_from: str = 'vdw',
    dry_run: bool = False,
) -> None:
    monomer   = config['monomer']
    symmetry  = config['symmetry'].lower()
    auto_dir  = config['auto_dir']
    params    = config.get('parameters', {})

    if symmetry not in ('glide', 'screw'):
        raise ValueError(f"symmetry は 'glide' または 'screw' を指定してください。got: {symmetry}")

    mono_dir, _ = _resolve_data_dirs(config)
    monomer_path = config.get('monomer_path', str(mono_dir / f"{monomer}.xyz"))

    start_i = STEPS.index(start_from)
    stop_i  = STEPS.index(stop_after)
    steps_to_run = set(STEPS[start_i: stop_i + 1])

    Path(auto_dir).mkdir(parents=True, exist_ok=True)

    # ── Step 0: モノマー準備 ──────────────────────────────────────────
    if 'monomer' in steps_to_run:
        print("\n" + "=" * 60)
        print(f"  Step 0: モノマー準備  (opt → RESP → amber_ref)")
        print("=" * 60)
        run_monomer_step(config, dry_run=dry_run)

    # ── Step 1: VdW スウィープ ────────────────────────────────────────
    if 'vdw' in steps_to_run:
        print("\n" + "=" * 60)
        print(f"  Step 1: VdW スウィープ  [{symmetry}]")
        print("=" * 60)

        common = (
            _range_args(params, 'z') +
            _range_args(params, 'alpha') +
            _range_args(params, 'phi')
        )

        if symmetry == 'glide':
            theta_step = str(params.get('theta_c_step', 5))
            _run([
                'auto_opt.vdw.sweep_phi',
                '--monomer-path', monomer_path,
                '--out-dir', auto_dir,
                *common,
                '--theta-step', theta_step,
            ], dry_run=dry_run)

            print("\n" + "-" * 60)
            print("  Step 2: 初期点抽出  [glide のみ]")
            print("-" * 60)
            vdw_csv    = str(Path(auto_dir) / f"vdW_r_contact_{monomer}.csv")
            init_csv   = str(Path(auto_dir) / "step1_init_params.csv")
            raw_select = config.get('vdw_select', 'all')
            vdw_select = [raw_select] if isinstance(raw_select, str) else list(raw_select)
            _run([
                'auto_opt.vdw.extract_init_phi',
                '--vdw-csv', vdw_csv,
                '--out', init_csv,
                '--select', *vdw_select,
            ], dry_run=dry_run)

        else:  # screw
            beta_args  = _range_args(params, 'beta')
            raw_select = config.get('vdw_select', 'all')
            vdw_select = [raw_select] if isinstance(raw_select, str) else list(raw_select)
            _run([
                'auto_opt.vdw.sweep_screw_phi',
                '--monomer-path', monomer_path,
                '--out-dir', auto_dir,
                *common,
                *beta_args,
                '--select', *vdw_select,
            ], dry_run=dry_run)

    # ── Step 3: Amber 最適化 ─────────────────────────────────────────
    if 'amber' in steps_to_run:
        print("\n" + "=" * 60)
        print(f"  Step 3: Amber 層内最適化  [{symmetry}]")
        print("  ※ 各 phi/alpha ジョブ完了後に Step 4（局所最小抽出）が自動実行されます")
        print("=" * 60)

        if symmetry == 'glide':
            _run([
                'auto_opt.amber.job_phi',
                '--auto-dir', auto_dir,
                '--monomer-name', monomer,
            ], dry_run=dry_run)
        else:
            _run([
                'auto_opt.amber.job_screw_phi',
                '--auto-dir', auto_dir,
                '--monomer-name', monomer,
            ], dry_run=dry_run)

        # collect も対象なら全ジョブ完了を待つ
        if 'collect' in steps_to_run and not dry_run:
            from auto_opt.cluster import load_env, get_my_job_count
            poll = load_env().get('poll_interval', 60)
            print(f"\n[auto_opt] 全ジョブ投入完了。終了を待機中... ({poll}秒ごとに確認)")
            while True:
                n = get_my_job_count()
                if n == 0:
                    print("[auto_opt] 全ジョブ完了。")
                    break
                print(f"  残りジョブ数: {n}  ({poll}秒後に再確認...)")
                time.sleep(poll)
        else:
            print(f"\n[auto_opt] 全ジョブ投入完了。qsub ジョブ終了後に collect を実行してください:")
            print(f"  python -m auto_opt.run --config <config> --start-from collect")

    # ── Step 5: 結果収集 ──────────────────────────────────────────────
    if 'collect' in steps_to_run:
        print("\n" + "=" * 60)
        print("  Step 5: 結果収集")
        print("=" * 60)
        collect_results(auto_dir, dry_run=dry_run)
        out = Path(auto_dir) / 'filtered_step1.csv'
        print(f"\n[auto_opt] 完了。ローカルにダウンロードして Streamlit で確認:")
        print(f"  scp <server>:{out.resolve()} ~/Downloads/filtered_step1.csv")
        print(f"  streamlit run src/auto_opt/app.py")


    # ── Step 6: スタッキング計算 ──────────────────────────────────────
    if 'stacking' in steps_to_run:
        print("\n" + "=" * 60)
        print(f"  Step 6: スタッキング計算  [{symmetry}]")
        print("=" * 60)

        stacking_dir = config.get('stacking_dir', str(Path(auto_dir).parent / f"{Path(auto_dir).name}_stacking"))
        Path(stacking_dir).mkdir(parents=True, exist_ok=True)

        # amber/collect で生成された filtered_step1.csv をスタッキングの入力に使う
        src_csv = Path(auto_dir) / 'filtered_step1.csv'
        dst_csv = Path(stacking_dir) / 'step1_init_params.csv'
        if src_csv.exists() and not dst_csv.exists():
            shutil.copy2(src_csv, dst_csv)
            print(f"[stacking] {src_csv.name} → {stacking_dir}/step1_init_params.csv")

        _run([
            'auto_opt.stacking.job_stacking',
            '--auto-dir', stacking_dir,
            '--monomer-name', monomer,
            '--symmetry', symmetry,
        ], dry_run=dry_run)

        if 'merge' in steps_to_run and not dry_run:
            from auto_opt.cluster import load_env, get_my_job_count
            poll = load_env().get('poll_interval', 60)
            print(f"\n[auto_opt] スタッキングジョブ投入完了。終了を待機中... ({poll}秒ごとに確認)")
            while True:
                n = get_my_job_count()
                if n == 0:
                    print("[auto_opt] 全ジョブ完了。")
                    break
                print(f"  残りジョブ数: {n}  ({poll}秒後に再確認...)")
                time.sleep(poll)
        else:
            print(f"\n[auto_opt] ジョブ投入完了。完了後に merge を実行してください:")
            print(f"  python -m auto_opt.run --config <config> --start-from merge")

    # ── Step 7: スタッキング結果マージ ────────────────────────────────
    if 'merge' in steps_to_run:
        print("\n" + "=" * 60)
        print("  Step 7: スタッキング結果マージ")
        print("=" * 60)

        stacking_dir = config.get('stacking_dir', str(Path(auto_dir).parent / f"{Path(auto_dir).name}_stacking"))
        mono_dir, _ = _resolve_data_dirs(config)

        _run([
            'auto_opt.stacking.merge_results',
            '--auto-dir', stacking_dir,
            '--monomer-name', monomer,
            '--data-dir', str(mono_dir.parent),
        ], dry_run=dry_run)

        out = Path(stacking_dir) / 'stacking_results.csv'
        print(f"\n[auto_opt] 完了。ローカルにダウンロードして Streamlit で確認:")
        print(f"  scp <server>:{out.resolve()} ~/Downloads/stacking_results.csv")
        print(f"  streamlit run src/auto_opt/app.py")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="run_config.yaml に従い VdW → Amber → スタッキング を順番に実行する"
    )
    ap.add_argument('--config',      required=True,
                    help='設定ファイルのパス (run_config.yaml)')
    ap.add_argument('--stop-after',  choices=STEPS, default='collect',
                    help='このステップで止まる (デフォルト: collect)')
    ap.add_argument('--start-from',  choices=STEPS, default='vdw',
                    help='このステップから再開する (デフォルト: vdw)')
    ap.add_argument('--dry-run',     action='store_true',
                    help='コマンドを表示するだけで実行しない')
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] 設定ファイルが見つかりません: {config_path}")
        sys.exit(1)

    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.dry_run:
        print("[dry-run] コマンドのみ表示します（実行しません）")

    run_pipeline(
        config,
        stop_after=args.stop_after,
        start_from=args.start_from,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
