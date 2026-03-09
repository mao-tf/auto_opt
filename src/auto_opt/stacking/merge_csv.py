#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
各 phi_XXX ディレクトリにある cy_scan_results.csv を一つに結合するスクリプト。

実行例:
python -m auto_opt.stacking.merge_csv --target-dir runs/PFA_stacking
"""

import os
import glob
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="CSVファイルを結合します")
    parser.add_argument('--target-dir', type=str, required=True, help="phi_XXX ディレクトリが含まれる親ディレクトリのパス")
    parser.add_argument('--out-name', type=str, default='ALL_results.csv', help="出力するファイル名")
    args = parser.parse_args()

    # 指定されたディレクトリの絶対パスを取得
    target_dir = os.path.abspath(args.target_dir)
    
    # target_dir の下にあるすべての phi_*/cy_scan_results.csv を探す
    # 例: runs/PFA_stacking/phi_*/cy_scan_results.csv
    search_pattern = os.path.join(target_dir, 'phi_*', 'cy_scan_results.csv')
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"エラー: '{search_pattern}' に一致するCSVファイルが見つかりません。")
        return

    print(f"{len(csv_files)} 個のCSVファイルを発見しました。結合を開始します...")

    # 全部読み込んで縦にくっつける
    df_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    # 綺麗に並び替える
    # 万が一、古いデータで cx 列がない場合にも対応できるように、列が存在するかチェックしてソート
    sort_cols = [col for col in ['alpha', 'phi', 'theta2', 'cx', 'cy'] if col in df_all.columns]
    df_all = df_all.sort_values(sort_cols)
    
    # 1つのファイルとして出力 (target_dir の直下に出力する)
    out_path = os.path.join(target_dir, args.out_name)
    df_all.to_csv(out_path, index=False)
    
    print(f"\n【結合完了】")
    print(f"'{out_path}' を作成しました！")
    print(f"総データ数: {len(df_all)} 行")

if __name__ == '__main__':
    main()