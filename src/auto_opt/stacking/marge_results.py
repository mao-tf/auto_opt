import pandas as pd
import argparse
from pathlib import Path

def merge_csvs(base_dir):
    base_path = Path(base_dir).resolve()
    dfs = []
    
    print(f"Looking for step1.csv in {base_path}/split_* ...")
    
    # split_0, split_1... の中にある step1.csv を順番に探して読み込む
    for split_dir in base_path.glob("split_*"):
        csv_path = split_dir / "step1.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
            print(f"Loaded {len(df)} rows from {split_dir.name}")
        else:
            print(f"Warning: {csv_path} not found.")

    if not dfs:
        print("結合するファイルが見つかりませんでした。")
        return

    # 全てのデータフレームを縦に結合 (ヘッダーは自動で1つにまとまります)
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # データを cx, cy の順番で綺麗に並べ直す（見やすさのため）
    merged_df = merged_df.sort_values(by=['cx', 'cy'])
    
    # 元のフォルダの直下に step1_merged.csv として保存
    out_path = base_path / "step1_merged.csv"
    merged_df.to_csv(out_path, index=False)
    
    print("-" * 30)
    print(f"✅ Successfully merged {len(dfs)} files!")
    print(f"Saved to: {out_path}")
    print(f"Total rows: {len(merged_df)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge scattered step1.csv files')
    parser.add_argument('--auto-dir', type=str, required=True, help='Base directory containing split_* folders')
    args = parser.parse_args()
    
    merge_csvs(args.auto_dir)