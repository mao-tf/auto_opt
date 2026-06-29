# auto_opt

有機半導体結晶の**層内・層間分子配置**を自動最適化するパイプライン。

モノマーの XYZ ファイルを入力として、ファンデルワールス接触幾何のスウィープ → Amber 力場による格子パラメータ最適化 → 局所最小の特定 → 層間最適化 を自動実行する。

## 対応する結晶対称性

| 対称性 | 説明 | ダイマー種 | エネルギー式 |
|--------|------|-----------|------------|
| **glide**（映進対称） | 鏡映＋平行移動で関連する2分子配置 | a, b, t | `E = 2E1 + 2E2 + 4E3` |
| **screw**（螺旋軸対称） | 回転＋平行移動で関連する2分子配置 | a, b, t1, t3 | `E = 2E1 + 2E2 + 2E3 + 2E4` |

## 必要なソフトウェア

| ソフトウェア | 用途 |
|-------------|------|
| Python ≥ 3.8 | 実行環境 |
| [Amber](https://ambermd.org/)（GAFF2, sander, tleap, antechamber） | 力場最適化 |
| [Gaussian 16](https://gaussian.com/) | DFT 構造最適化・ESP 電荷計算 |
| SGE ジョブスケジューラー（`qsub` / `qstat`） | スパコンへのジョブ投入 |

## インストール

```bash
git clone https://github.com/mao-tf/auto_opt.git
cd auto_opt
pip install -e .
```

## クイックスタート

`run_config.yaml` を用意して `run.py` から一括実行するのが推奨の使い方です：

```bash
python -m auto_opt.run --config /path/to/run_config.yaml
```

テンプレートは `examples/run_config.yaml` を参照してください。主要フィールド：

```yaml
monomer: DNTT
symmetry: glide
auto_dir: /home/user/work/runs/DNTT_glide

data_dir:    /home/user/work/data
monomer_xyz: /home/user/work/data/monomer/DNTT_raw.xyz

charge: 0
mult:   1

parameters:
  z:     {min: -2.0, max: 2.0, step: 0.5}
  alpha: {min: 0,    max: 90,  step: 10}
  phi:   {min: -10,  max: 10,  step: 4}

vdw_select: all   # または a-stack / b-stack

amber:
  num_nodes: 38
```

特定のステップだけ実行する場合：

```bash
# モノマー前処理 + VdW スウィープのみ
python -m auto_opt.run --config run_config.yaml --start-from monomer --stop-after vdw

# Amber 最適化のみ（VdW スウィープ済みの場合）
python -m auto_opt.run --config run_config.yaml --start-from amber --stop-after amber

# 局所最小の収集
python -m auto_opt.run --config run_config.yaml --start-from collect
```

## ワークフロー

```
monomer_raw.xyz
    │
    ▼  Step 0: モノマー前処理
    │  prep_monomer.py
    │  → {monomer}.xyz（PCA 整列済み）, {monomer}.mol2（RESP 電荷付き）, {monomer}_gaff2.out
    │
    ▼  Step 1: VdW 接触スウィープ
    │  [glide]  vdw/sweep_phi.py       → step1_init_params.csv
    │  [screw]  vdw/sweep_screw_phi.py → step1_init_params.csv
    │
    ▼  Step 2: Amber 層内最適化
    │  [glide]  amber/job_phi.py
    │  [screw]  amber/job_screw_phi.py
    │  → filtered_step1.csv
    │
    ▼  Step 3: スタッキング VdW スウィープ
    │  stacking/sweep_stacking_vdw.py
    │  → step1_init_params.csv（cx, cy, cz）
    │
    ▼  Step 4: Amber 層間最適化
    │  stacking/job_stacking.py
    │
    ▼  Step 5: 結果収集
       stacking/merge_results.py
       → stacking_results.csv
```

## 個別スクリプトの使い方

### VdW スウィープ（glide）

```bash
python -m auto_opt.vdw.sweep_phi \
    --monomer-path /path/to/DNTT.xyz \
    --out-dir runs/DNTT_glide \
    --z-min -2.0 --z-max 2.0 --z-step 0.5 \
    --alpha-min 0 --alpha-max 90 --alpha-step 10 \
    --phi-min -10 --phi-max 10 --phi-step 4 \
    --select all    # または a-stack / b-stack
```

### VdW スウィープ（screw）

```bash
python -m auto_opt.vdw.sweep_screw_phi \
    --monomer-path /path/to/DNTT.xyz \
    --out-dir runs/DNTT_screw \
    --z-min -2.0 --z-max 2.0 --z-step 0.5 \
    --alpha-min 0 --alpha-max 90 --alpha-step 10 \
    --beta-min -20 --beta-max 20 --beta-step 5 \
    --phi-min -10 --phi-max 10 --phi-step 4
```

### Amber 最適化

```bash
# glide
python -m auto_opt.amber.job_phi \
    --auto-dir runs/DNTT_glide \
    --monomer-name DNTT \
    --num-nodes 38

# screw
python -m auto_opt.amber.job_screw_phi \
    --auto-dir runs/DNTT_screw \
    --monomer-name DNTT \
    --num-nodes 38
```

### Gaussian DFT ジョブ投入

```bash
# glide
python -m auto_opt.gaussian.pipeline_phi \
    --auto-dir runs/DNTT_glide \
    --monomer  DNTT

# screw
python -m auto_opt.gaussian.pipeline_screw_phi \
    --auto-dir runs/DNTT_screw \
    --monomer  DNTT \
    --E-threshold -10.0    # 任意: E ≤ -10 kcal/mol の行だけ投入
```

## 可視化（Streamlit UI）

可視化用の依存パッケージをインストールします（ローカルで実行、HPC 上では不要）：

```bash
pip install -e ".[viz]"
```

UI を起動します：

```bash
streamlit run src/auto_opt/app.py
```

ブラウザで `http://localhost:8501` を開いてください。セットアップタブで**ローカル作業ディレクトリ**（リポジトリ外）を指定すると、HPC の計算結果をそのパス経由で読み込めます。

インタラクティブなヒートマップ・3D 分子クラスタービュー・スタッキング計算の候補選択を提供します。

## ディレクトリ構成

```
auto_opt/
├── src/auto_opt/
│   ├── run.py          # オーケストレーター（推奨エントリーポイント）
│   ├── app.py          # Streamlit 可視化 UI
│   ├── cluster.py      # SGE ジョブ管理
│   ├── utils.py        # 共通ユーティリティ
│   ├── monomer/        # Step 0: モノマー前処理
│   ├── vdw/            # Step 1: VdW 接触スウィープ
│   ├── amber/          # Step 2: Amber 力場最適化
│   ├── stacking/       # Steps 3-5: 層間最適化
│   ├── gaussian/       # DFT ジョブ投入
│   └── plot/           # 可視化ツール
├── examples/
│   ├── run_config.yaml # run_config.yaml テンプレート
│   └── auto_opt.yaml   # ~/.auto_opt.yaml テンプレート
└── docs/
    └── spec_overview.md
```

`data/`・`runs/`・`legacy/` は git 管理外です。作業データはリポジトリ外のディレクトリに置いてください。

## パラメータの意味

| パラメータ | 説明 |
|-----------|------|
| `alpha` | z 軸まわりの分子回転角（°） |
| `phi` | x 軸まわりの傾き・長軸傾斜（°） |
| `beta` | x 軸まわりの追加傾き（°）、screw のみ |
| `a`, `b` | 結晶格子定数（Å） |
| `z` | T ダイマーの積層方向オフセット（Å） |

## DFT 計算設定

- 手法: PBEPBE / 6-311G\*\* ＋ Grimme D3BJ 分散補正
- 基底関数重複誤差（BSSE）補正: Counterpoise 法
- 相互作用エネルギーの基準: 孤立モノマー（Amber GAFF2）

## ライセンス

MIT
