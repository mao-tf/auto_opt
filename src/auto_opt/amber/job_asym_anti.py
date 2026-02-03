#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1 (Intralayer Search) のジョブ投入スクリプト
指定されたディレクトリ内の step1_init_params.csv を読み込み、
alpha (分子の傾き) ごとにサブディレクトリを作成して計算ジョブを分散投入します。

Usage:
    python -m auto_opt.amber.job_asym_anti --auto-dir runs/PFA_test --monomer-name PFA --num-nodes 10
"""

import os
# 必要に応じて環境変数を設定
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
    Step 1 の初期パラメータを読み込み、alpha ごとにジョブを分割して投入するプロセス。
    
    args.auto_dir :
        step1_init_params.csv が置いてあるディレクトリ
        （絶対パスでも相対パスでもOK）
    """

    # ベースディレクトリ（絶対パスにそろえる）
    auto_dir_root = Path(args.auto_dir).resolve() # [MY_MEMO] resolve() はpwdとauto_dirを繋いで絶対パスにするやつ！計算ノードが迷子にならないようにするため。
    params_csv = auto_dir_root / 'step1_init_params.csv'

    if not params_csv.exists(): # [my_memo] step1_init_params.csvがなかったらクラッシュしてメッセージを出す
        raise FileNotFoundError(f"初期パラメータファイルが見つかりません: {params_csv}")

    # まとめてある init を読む
    df_init = pd.read_csv(params_csv)

    # 【修正箇所】CSVに含まれるユニークな alpha 値を動的に取得してソートする
    # これにより、任意の角度刻みや範囲に対応可能になります
    if 'alpha' not in df_init.columns:
         raise ValueError(f"{params_csv} に 'alpha' カラムが含まれていません。")
         
    alpha_list = sorted(df_init['alpha'].unique()) #[my_memo] .uniqueは[30,30,40,40,45]を[30,40,45]にするもの

    print(f"Detected alpha values: {alpha_list}") #[my_memo] alphaのリストを出力

    for i, alpha in enumerate(alpha_list):
        # 交互にキュー（マシンタイプ）を振り分ける
        if i % 2 == 0: 
            machine_type = 1
        else: 
            machine_type = 2
            
        spec = MACHINE_SPEC[machine_type]
        queue = spec["queue"]
        nproc = spec["nproc"]

        # ディレクトリ名は alpha の値そのもの
        dir_name = f'{alpha}'
        subdir = auto_dir_root / dir_name
        subdir.mkdir(parents=True, exist_ok=True) #[my_memo] 親ディレクトリもまとめて作ってくれる！ もともと存在していてもOK! 途中でクラッシュして再計算になった時に再度作らなくていいように

        # この alpha の行だけ抜き出して保存
        df_alpha = df_init[df_init['alpha'] == alpha]
        
        if df_alpha.empty:
            continue

        # サブディレクトリ用にパラメータファイルを保存
        df_alpha.to_csv(subdir / 'step1_init_params.csv', index=False)

        # driver_gene に渡す auto-dir は「その alpha のサブディレクトリ」
        auto_dir_for_driver = str(subdir) #[mymemo] Path型をstr型にする。なぜなら外部のLinuxコマンド(qsubや driver_gene)に引数を渡す時は、Path型ではなく文字列型(str)じゃないと受け取ってくれないから。

        # 実際に投げるコマンドを組み立てる
        cmd = (
            'python -m auto_opt.amber.driver_gene_asym '
            f'--auto-dir {auto_dir_for_driver} '
            f'--monomer-name {args.monomer_name} '
            f'--num-nodes {args.num_nodes}'
        )
        if args.isTest:
            cmd += ' --isTest'

        job_lines = [
            "#!/bin/sh\n",
            "#$ -S /bin/sh\n",
            "#$ -cwd\n", #[mymemo]実行環境の指定。計算ノードの環境設定（Pythonのバージョンや必要なライブラリ）を統一し、どこで動かしても同じ結果が出るようにする。
            "#$ -V\n",
            f"#$ -q {queue}\n", #[mymemo]計算資源の確保。1000種類のリスクシナリオを、1000コア使って同時に計算させる。
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

        # ジョブ投入
        print(f"Submitting job for alpha={alpha} into {queue}...")
        subprocess.run(['qsub', str(job_path)]) #[mymemo]計算の実行。計算が集中したとき、システムが「このジョブは夜中にやろう」「このジョブは空いている別のサーバーでやろう」と自動で順番待ちや振り分けをしてくれる。


def update_value_in_df(df, index, key, value):
    """
    DataFrame内の値を更新するヘルパー関数
    (このスクリプト内では直接使用されていませんが、将来的な拡張のために維持します)
    """
    df.loc[index, key] = value
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Step 1 Job Dispatcher')
    ## argparseを使うと外から変数を変えられる

    parser.add_argument(
        '--isTest', action='store_true',
        help='driver_gene に --isTest を付けてテスト実行にする',
    )
    parser.add_argument(
        '--auto-dir', type=str, required=True,
        help='step1_init_params.csv があるディレクトリ '
             '(例: runs/PFA_test)',
    )
    parser.add_argument(
        '--monomer-name', type=str, default='pentacene',
        help='driver_gene に渡す monomer 名 (例: PFA)',
    )
    parser.add_argument(
        '--num-nodes', type=int, default=10,
        help='driver_gene の --num-nodes (1つのalphaあたりの計算ノード数/並列数)',
    )
    
    args = parser.parse_args()

    print("----main process start----")
    init_process(args)
    print("----finish process----")