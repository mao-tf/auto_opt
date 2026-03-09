#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
層間積層（1次元スキャン）のジョブ投入スクリプト
指定したCSVを読み込み、phiごとにディレクトリを分割して計算ジョブを分散投入します。

実行例:
python -m auto_opt.stacking.job --auto-dir runs/PFA_stacking/phi_20.0 --monomer-name PFA --input-csv runs/PFA_stacking/input_for_stacking.csv --mode 2D --step-x 0.25 --step-y 0.25
"""

import os
os.environ['HOME'] = '/data/group1/z40145w'

import pandas as pd
import argparse
import subprocess
from pathlib import Path

# cmdell81 のキュー仕様
MACHINE_SPEC = {
    1: {"queue": "gr1.q", "nproc": 40},
    2: {"queue": "gr2.q", "nproc": 52},
}

def init_process(args):
    """
    args.input_csv を読み込み、phiごとにジョブを分割して投入するプロセス
    """
    # ベースディレクトリと入力CSVのパスを絶対パスで解決
    auto_dir_root = Path(args.auto_dir).resolve()
    input_csv_path = Path(args.input_csv).resolve()

    if not input_csv_path.exists():
        raise FileNotFoundError(f"入力CSVが見つかりません: {input_csv_path}")

    # Step 1 で最適化されたパラメータのCSVを読む
    df_input = pd.read_csv(input_csv_path)

    # phi ごとにサブディレクトリを作る
    phi_list = sorted(df_input['phi'].unique())

    for i, phi in enumerate(phi_list):
        # 交互にキューを割り当てる
        if i % 2 == 0: 
            machine_type = 1
        else: 
            machine_type = 2
            
        spec = MACHINE_SPEC[machine_type]
        queue = spec["queue"]
        nproc = spec["nproc"]
        
        # サブディレクトリの作成
        dir_name = f'phi_{phi}' # phi=... というディレクトリ名にしておくと後で分かりやすいです
        subdir = auto_dir_root / dir_name
        subdir.mkdir(parents=True, exist_ok=True)

        # この phi の行だけ抜き出す
        df_phi = df_input[df_input['phi'] == phi]
        if df_phi.empty:
            continue

        # サブディレクトリ専用の入力CSVを保存
        job_input_csv = subdir / 'input_for_stacking.csv'
        df_phi.to_csv(job_input_csv, index=False)

        auto_dir_for_driver = str(subdir)

        # 実際に投げるコマンドを組み立てる (driver_stacking_1D を呼び出す)
        cmd = (
            'python -m auto_opt.stacking.driver_stacking '
            f'--auto-dir {auto_dir_for_driver} '
            f'--monomer-name {args.monomer_name} '
            f'--input-csv {str(job_input_csv)} '
            f'--cz-tol {args.cz_tol}'
        )
        if args.isTest:
            cmd += ' --isTest'

        job_lines = [
            "#!/bin/sh\n",
            "#$ -S /bin/sh\n",
            "#$ -cwd\n",
            "#$ -V\n",
            f"#$ -q {queue}\n",
            f"#$ -pe OpenMP {nproc}\n",
            "\n",
            "hostname\n",
            "\n",
            cmd + "\n",
            "\n",
            "#sleep 5\n",
        ]

        job_path = subdir / 'job.sh'
        with open(job_path, 'w') as f:
            f.writelines(job_lines)

        # ジョブ投げ (cwdを指定することで、.o や .e のログファイルがサブディレクトリに保存されます)
        if not args.isTest:
            print(f"Submitting job for phi={phi} into {queue}...")
            subprocess.run(['qsub', 'job.sh'], cwd=str(subdir))
        else:
            print(f"[TEST] phi={phi} のジョブ作成完了 (投入はスキップ)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--isTest', action='store_true',
        help='テスト実行 (ジョブスクリプトを作成するだけで qsub はしない)',
    )
    parser.add_argument(
        '--auto-dir', type=str, required=True,
        help='計算を実行するベースディレクトリ (例: runs/PFA_stacking)',
    )
    parser.add_argument(
        '--monomer-name', type=str, required=True,
        help='対象の分子名 (例: PFA)',
    )
    parser.add_argument(
        '--input-csv', type=str, required=True,
        help='Step1/2で最適化された構造が入ったCSVファイルのパス',
    )
    parser.add_argument(
        '--cz-tol', type=float, default=0.1,
        help='SciPyでの cz 最適化の収束条件 (デフォルト: 0.1 Å)',
    )
    parser.add_argument(
        '--mode', type=str, choices=['1D', '2D'], default='1D', 
        help="スキャンモードの選択 (1D または 2D)"
        )
    parser.add_argument(
        '--step-x', type=float, default=0.1, 
        help="cx のスキャン間隔 (デフォルト: 0.1)"
        )
    parser.add_argument(
        '--step-y', type=float, default=0.1, 
        help="cy のスキャン間隔 (デフォルト: 0.1)"
        )
    
    args = parser.parse_args()

    print("---- main process ----")
    init_process(args)
    print("---- finish process ----")