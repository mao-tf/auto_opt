#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py  ―  Step 1→4 オーケストレーター

run_config.yaml を読み込み、VdW スウィープ → Amber 最適化 → 局所最小抽出 を
順番に実行する。途中で止めて結果を確認したい場合は --stop-after を使う。

Usage:
  # 全ステップ実行（VdW → Amber → 局所最小）
  python -m auto_opt.run --config runs/DNTT_glide/run_config.yaml

  # VdW スウィープだけ実行して止める
  python -m auto_opt.run --config runs/DNTT_glide/run_config.yaml --stop-after vdw

  # VdW 結果が既にある場合、Amber から再開
  python -m auto_opt.run --config runs/DNTT_glide/run_config.yaml --start-from amber

  # コマンドを確認するだけ（実行しない）
  python -m auto_opt.run --config runs/DNTT_glide/run_config.yaml --dry-run
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import yaml

# 実行ステップの順序
STEPS = ['vdw', 'amber']

# デフォルト monomer ディレクトリ（パッケージ相対）
_MONOMER_DIR = Path(__file__).resolve().parents[2] / "data" / "monomer"


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


def run_pipeline(
    config: dict,
    stop_after: str = 'amber',
    start_from: str = 'vdw',
    dry_run: bool = False,
) -> None:
    monomer   = config['monomer']
    symmetry  = config['symmetry'].lower()
    auto_dir  = config['auto_dir']
    params    = config.get('parameters', {})
    amber_cfg = config.get('amber', {})

    if symmetry not in ('glide', 'screw'):
        raise ValueError(f"symmetry は 'glide' または 'screw' を指定してください。got: {symmetry}")

    # monomer_path: config で指定がなければパッケージ付属の data/monomer/ を使う
    monomer_path = config.get('monomer_path', str(_MONOMER_DIR / f"{monomer}.xyz"))

    # 実行するステップ範囲を決定
    start_i = STEPS.index(start_from)
    stop_i  = STEPS.index(stop_after)
    steps_to_run = set(STEPS[start_i: stop_i + 1])

    Path(auto_dir).mkdir(parents=True, exist_ok=True)

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

            # Step 2: 初期点抽出（glide のみ、VdW 直後に実施）
            print("\n" + "-" * 60)
            print("  Step 2: 初期点抽出  [glide のみ]")
            print("-" * 60)
            vdw_csv  = str(Path(auto_dir) / f"vdW_r_contact_{monomer}.csv")
            init_csv = str(Path(auto_dir) / "step1_init_params.csv")
            _run([
                'auto_opt.vdw.extract_init_phi',
                '--vdw-csv', vdw_csv,
                '--out', init_csv,
                '--select', 'all',
            ], dry_run=dry_run)

        else:  # screw
            beta_args  = _range_args(params, 'beta')
            vdw_select = config.get('vdw_select', 'all')
            _run([
                'auto_opt.vdw.sweep_screw_phi',
                '--monomer-path', monomer_path,
                '--out-dir', auto_dir,
                *common,
                *beta_args,
                '--select', vdw_select,
            ], dry_run=dry_run)

    # ── Step 3: Amber 最適化（driver が終わると Step 4 を自動実行） ──
    if 'amber' in steps_to_run:
        print("\n" + "=" * 60)
        print(f"  Step 3: Amber 層内最適化  [{symmetry}]")
        print("  ※ 完了後に Step 4（局所最小抽出）が自動実行されます")
        print("=" * 60)

        num_nodes = str(amber_cfg.get('num_nodes', 38))

        if symmetry == 'glide':
            _run([
                'auto_opt.amber.job_phi',
                '--auto-dir', auto_dir,
                '--monomer-name', monomer,
                '--num-nodes', num_nodes,
            ], dry_run=dry_run)
        else:
            _run([
                'auto_opt.amber.job_screw_phi',
                '--auto-dir', auto_dir,
                '--monomer-name', monomer,
                '--num-nodes', num_nodes,
            ], dry_run=dry_run)

        print(f"\n[auto_opt] 完了。結果 -> {auto_dir}/filtered_step1.csv")
        print(f"[auto_opt] エネルギーマップ確認:")
        print(f"  python -m auto_opt.plot.energy_map \\")
        print(f"      --csv {auto_dir}/filtered_step1.csv \\")
        print(f"      --x phi --y z --fix alpha=<値>")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="run_config.yaml に従い VdW → Amber → 局所最小 を順番に実行する"
    )
    ap.add_argument('--config',      required=True,
                    help='設定ファイルのパス (run_config.yaml)')
    ap.add_argument('--stop-after',  choices=STEPS, default='amber',
                    help='このステップで止まる (デフォルト: amber)')
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

    if dry_run_check := args.dry_run:
        print("[dry-run] コマンドのみ表示します（実行しません）")

    run_pipeline(
        config,
        stop_after=args.stop_after,
        start_from=args.start_from,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
