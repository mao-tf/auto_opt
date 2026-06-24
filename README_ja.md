# auto_opt

有機半導体結晶の**層内分子配置**を自動最適化するパイプライン。

モノマーの XYZ ファイルを入力として、ファンデルワールス接触幾何のスウィープ → Amber 力場による格子パラメータ最適化 → Gaussian DFT による局所最小の精密化（相互作用エネルギー・基底関数重複補正つき）を自動実行する。

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
| [Gaussian 16](https://gaussian.com/) | DFT 一点計算 |
| SGE ジョブスケジューラー（`qsub` / `qstat`） | スパコンへのジョブ投入 |

## インストール

```bash
git clone https://github.com/<your-org>/auto_opt.git
cd auto_opt
pip install -e .
```

## ワークフロー

```
monomer.xyz
    │
    ▼  Step 0: モノマー前処理
    │  python -m auto_opt.monomer.prep_monomer ...
    │  → monomer.csv, monomer.mol2, monomer_gaff2.frcmod
    │
    ▼  Step 1: VdW 接触スウィープ
    │  [glide]  python -m auto_opt.vdw.sweep_phi ...
    │           → vdW_r_contact_<monomer>.csv
    │  [screw]  python -m auto_opt.vdw.sweep_screw_phi ...
    │           → step1_init_params.csv（Step 3 へ直接進む）
    │
    ▼  Step 2: 初期点抽出（glide のみ）
    │  python -m auto_opt.vdw.extract_init_phi ...
    │  → step1_init_params.csv
    │
    ▼  Step 3: Amber 層内最適化
    │  [glide]  python -m auto_opt.amber.job_phi ...
    │  [screw]  python -m auto_opt.amber.job_screw_phi ...
    │  → step1.csv（完了後、Step 4 を自動実行）
    │
    ▼  Step 4: 局所最小抽出（Step 3 が自動実行）
    │  python -m auto_opt.gaussian.extract_minima --symmetry glide|screw ...
    │  → filtered_step1.csv
    │
    ▼  Step 5: Gaussian DFT 計算
       [glide]  python -m auto_opt.gaussian.pipeline_phi ...
       [screw]  python -m auto_opt.gaussian.pipeline_screw_phi ...
       → *.inp, *.log
```

## 使い方

### Step 1 — VdW スウィープ（glide）

```bash
python -m auto_opt.vdw.sweep_phi \
    --monomer-path data/monomer/DNTT.xyz \
    --out-dir runs/DNTT_glide \
    --z-min 0.0 --z-max 3.0 --z-step 0.5 \
    --alpha-min 60 --alpha-max 70 --alpha-step 5 \
    --phi-min 0 --phi-max 10 --phi-step 5 \
    --theta-step 5
```

### Step 1 — VdW スウィープ（screw）

```bash
python -m auto_opt.vdw.sweep_screw_phi \
    --monomer-path data/monomer/DNTT.xyz \
    --out-dir runs/DNTT_screw \
    --z-min 0.0 --z-max 3.0 --z-step 0.5 \
    --alpha-min 60 --alpha-max 70 --alpha-step 5 \
    --beta-min 0 --beta-max 10 --beta-step 5 \
    --phi-min 0 --phi-max 10 --phi-step 5 \
    --select all          # または --select a-stack / --select b-stack
```

### Step 2 — 初期点抽出（glide のみ）

```bash
python -m auto_opt.vdw.extract_init_phi \
    --vdw-csv runs/DNTT_glide/vdW_r_contact_DNTT.csv \
    --out     runs/DNTT_glide/step1_init_params.csv \
    --select  all         # または --select a-stack b-stack
```

### Step 3 — Amber 最適化

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

Step 3 が完了すると Step 4（`extract_minima`）が自動的に実行されます。

### Step 4 — 局所最小抽出（手動で再実行する場合）

```bash
python -m auto_opt.gaussian.extract_minima \
    --symmetry glide \          # または screw
    --auto-dir runs/DNTT_glide
# → runs/DNTT_glide/filtered_step1.csv
```

### Step 5 — Gaussian DFT ジョブ投入

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

## ディレクトリ構成

```
auto_opt/
├── data/
│   └── monomer/          # monomer.xyz, monomer.csv, monomer.mol2, *.frcmod
├── docs/
│   └── spec_overview.md  # 詳細設計書
├── runs/                 # 実行時に自動生成される出力ディレクトリ
└── src/auto_opt/
    ├── monomer/          # Step 0: モノマー前処理
    ├── vdw/              # Step 1-2: VdW スウィープ・初期点抽出
    ├── amber/            # Step 3: Amber 力場最適化
    ├── gaussian/         # Step 4-5: 局所最小抽出・DFT ジョブ投入
    ├── plot/             # 可視化ツール（エネルギーマップ・XYZ 出力）
    └── utils.py          # 共通ユーティリティ（回転行列・VdW 半径など）
```

## パラメータの意味

| パラメータ | 説明 |
|-----------|------|
| `alpha` | z 軸まわりの分子回転角（°） |
| `phi` | 分子面内の回転角（°）、どちらの対称性でも共通 |
| `beta` | x 軸まわりの分子傾き角（°）、screw のみ |
| `a`, `b` | 結晶格子定数（Å） |
| `z` | glide の場合: t ダイマーの積層方向オフセット（Å） |
| `bt1`, `bt2` | screw の場合: t1/t3 ダイマーの b 方向オフセット（Å） |

## 可視化ツール

### エネルギーマップ（Step 4 完了後）

`filtered_step1.csv` の結果を任意の2パラメータで2Dヒートマップ表示する。

```bash
# phi vs z のエネルギーマップ（alpha=65 で固定）
python -m auto_opt.plot.energy_map \
    --csv  runs/DNTT_glide/filtered_step1.csv \
    --x    phi  --y z \
    --fix  alpha=65 \
    --out  energy_phi_z.png

# screw: alpha vs phi のマップ（beta=0, z=1.5 で固定）
python -m auto_opt.plot.energy_map \
    --csv  runs/DNTT_screw/filtered_step1.csv \
    --x    phi  --y alpha \
    --fix  beta=0 z=1.5 \
    --out  energy_alpha_phi.png
```

最低エネルギー点に星印、等高線つきで出力される。`--out` を省略すると画面表示。

### 分子構造 XYZ 出力（VESTA・Molden 等で可視化）

特定パラメータの結晶構造を `.xyz` ファイルで出力する。

```bash
# glide: phi=-10 の構造を 2×2 タイルで出力
python -m auto_opt.plot.export_xyz \
    --csv     runs/DNTT_glide/filtered_step1.csv \
    --monomer DNTT \
    --alpha 65 --phi -10 --z 1.5 \
    --tiles 2 2 \
    --out  DNTT_phi-10.xyz

# screw
python -m auto_opt.plot.export_xyz \
    --csv     runs/DNTT_screw/filtered_step1.csv \
    --monomer DNTT \
    --alpha 65 --beta 0 --phi -10 --z 1.5 \
    --out  DNTT_screw_phi-10.xyz
```

`--tiles na nb` でユニットセルを a/b 方向に繰り返せる（デフォルト 2×2）。
対称性（glide/screw）は CSV の列から自動判定する。

---

## DFT 計算設定

- 手法: PBEPBE / 6-311G\*\* ＋ Grimme D3BJ 分散補正
- 基底関数重複誤差（BSSE）の補正: Counterpoise 法
- 相互作用エネルギーの基準: 孤立モノマー（Amber GAFF2 最適化済み）

## ライセンス

MIT
