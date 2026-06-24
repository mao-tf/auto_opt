# auto_opt システム仕様書

## 1. システム概要

有機半導体結晶の**層内分子配置**を自動最適化するツール。
ユーザーが用意したモノマーの XYZ ファイルを入力として、Amber 力場による格子エネルギー最適化と Gaussian DFT 計算を自動実行する。

### 対象とする結晶対称性

ユーザーは以下の2種類から対称性を選択する：

| 対称性 | 説明 | コード識別子 |
|--------|------|------------|
| 映進対称（glide） | 鏡映＋平行移動で関連する2分子配置 | `glide` |
| 螺旋軸対称（screw） | 回転＋平行移動で関連する2分子配置 | `screw` |

phi（分子面内回転角）はどちらの対称性でも共通して扱う。

---

## 2. 全体ワークフロー

```
[ユーザー入力]
  monomer.xyz  +  対称性の選択（glide / screw）
      │
      ▼
┌─────────────────────────────────┐
│  Step 0: モノマー前処理          │
│  monomer/prep_monomer.py        │
└──────────────┬──────────────────┘
               │ monomer.csv, monomer.mol2,
               │ monomer_gaff2.frcmod, amber_ref/*.out
               ▼
┌──────────────────────────────────────────────────────┐
│  Step 1: VdW スウィープ                               │
│                                                      │
│  [glide]  vdw/sweep_phi.py       → vdW_r_contact.csv│
│  [screw]  vdw/sweep_screw_phi.py → step1_init_      │
│                                    params.csv (直接) │
└──────────────┬───────────────────────────────────────┘
               │
               ▼ [glide のみ]
┌─────────────────────────────────┐
│  Step 2: 初期点抽出              │
│  vdw/extract_init_phi.py        │
│  → step1_init_params.csv        │
└──────────────┬──────────────────┘
               │ step1_init_params.csv
               ▼
┌──────────────────────────────────────────────────────┐
│  Step 3: Amber 層内最適化                             │
│                                                      │
│  [glide]  amber/job_phi.py                           │
│             └ driver_gene_phi.py                     │
│                └ make_io_gene_phi.py                 │
│                                                      │
│  [screw]  amber/job_screw_phi.py                     │
│             └ driver_screw_phi.py                    │
│                └ make_io_gene_screw_phi.py            │
│                                                      │
│           → step1.csv                                │
│           → ★完了後に自動で Step 4 を呼び出す         │
└──────────────┬───────────────────────────────────────┘
               │ step1.csv
               ▼
┌─────────────────────────────────┐
│  Step 4: 局所最小抽出 ★新設     │
│  extract_minima.py              │
│  （driver 完了後に自動実行）     │
│  → filtered_step1.csv           │
└──────────────┬──────────────────┘
               │ filtered_step1.csv
               ▼
┌──────────────────────────────────────────────────────┐
│  Step 5: Gaussian DFT 計算                           │
│                                                      │
│  [glide]  gaussian/pipeline_phi.py    ← 投入のみ     │
│  [screw]  gaussian/pipeline_screw_phi.py ← 投入のみ  │
│                                                      │
│  ジョブ管理: gaussian/driver_dft_jobs.py             │
│  → *.inp, *.log                                      │
└──────────────────────────────────────────────────────┘
```

---

## 3. ステップ別詳細

### Step 0: モノマー前処理

| 項目 | 内容 |
|------|------|
| スクリプト | `monomer/prep_monomer.py` |
| 入力 | `monomer.xyz` |
| 出力 | `monomer.csv`（原子座標＋VdW半径）, `monomer.mol2`, `monomer_gaff2.frcmod`, `amber_ref/*.out`（Amber基準エネルギー） |
| 外部依存 | Gaussian（DFT最適化・RESP電荷）, antechamber（GAFF2パラメータ）, sander（Amber E） |

---

### Step 1: VdW スウィープ

#### 映進（glide）
| 項目 | 内容 |
|------|------|
| スクリプト | `vdw/sweep_phi.py` |
| 入力 | `monomer.xyz`, パラメータ範囲（alpha, beta, phi, z の最小/最大/刻み） |
| 出力 | `vdW_r_contact_<monomer>.csv`（列: alpha, phi, beta, z, R_clps, TorF） |

#### 螺旋軸（screw）
| 項目 | 内容 |
|------|------|
| スクリプト | `vdw/sweep_screw_phi.py` |
| 入力 | `monomer.xyz`, パラメータ範囲（alpha, beta, phi, z の最小/最大/刻み） |
| 出力 | `step1_init_params.csv`（列: alpha, beta, phi, a, b, bt1, bt2, z, status, structure_type） |
| 備考 | Step 2 は不要（初期点まで一括生成） |

---

### Step 2: 初期点抽出（映進のみ）

| 項目 | 内容 |
|------|------|
| スクリプト | `vdw/extract_init_phi.py` |
| 入力 | `vdW_r_contact_<monomer>.csv` |
| 出力 | `step1_init_params.csv`（列: alpha, phi, a, b, z, status, structure_type） |
| 備考 | phi=0 固定の場合も `extract_init_phi.py` をそのまま使用（phi=0 を含むケースをカバー済み） |

---

### Step 3: Amber 層内最適化

#### 映進（glide）

| 項目 | 内容 |
|------|------|
| ジョブ投入 | `amber/job_phi.py` |
| ドライバー | `amber/driver_gene_phi.py` |
| 幾何生成 | `amber/make_io_gene_phi.py` |
| 固定パラメータ | alpha, phi |
| 最適化パラメータ | a（E1用）, b+z（E2用）|
| ダイマー種類 | 3種: a-dimer (E1), b-dimer (E2), t-dimer (E3) |
| エネルギー式 | `E = 2*E1 + 2*E2 + 4*E3` |
| 出力列 | alpha, phi, a, b, z, E, E1, E2, E3, status |

#### 螺旋軸（screw）

| 項目 | 内容 |
|------|------|
| ジョブ投入 | `amber/job_screw_phi.py` |
| ドライバー | `amber/driver_screw_phi.py` |
| 幾何生成 | `amber/make_io_gene_screw_phi.py` |
| 固定パラメータ | alpha, beta, phi |
| 最適化パラメータ | a, bt1, bt2（b = bt1+bt2） |
| ダイマー種類 | 4種: a-dimer (E1), b-dimer (E2), t1-dimer (E3), t3-dimer (E4) |
| エネルギー式 | `E = 2*E1 + 2*E2 + 2*E3 + 2*E4` |
| 出力列 | alpha, beta, phi, a, b, bt1, bt2, z, E, E1, E2, E3, E4, status |

---

### Step 4: 局所最小抽出（新設）★

driver 完了後に自動実行される独立スクリプト。
現在は `pipeline_phi.py` / `pipeline_screw_phi.py` の内部に `extract_from_step1()` として埋め込まれているものを独立させる。

| 項目 | 内容 |
|------|------|
| スクリプト | `gaussian/extract_minima.py`（新規作成） |
| 起動 | driver の最後に自動呼び出し |
| 入力 | `step1.csv` |
| 処理（glide） | (alpha, phi, z) ごとに (a, b) の2D局所最小を抽出 |
| 処理（screw） | (alpha, beta, phi, z) ごとに (a, bt1, bt2) の3D局所最小を抽出 |
| 出力 | `filtered_step1.csv`（局所最小行のみ） |

---

### Step 5: Gaussian DFT 計算

| 役割 | スクリプト |
|------|-----------|
| ジョブ作成＋投入（glide） | `gaussian/pipeline_phi.py`（投入のみに整理） |
| ジョブ作成＋投入（screw） | `gaussian/pipeline_screw_phi.py`（投入のみに整理） |
| 空きノード管理・自動投入 | `gaussian/driver_dft_jobs.py` |

---

## 4. ファイル整理方針

### 残す（正式版）

```
src/auto_opt/
├── utils.py
├── monomer/
│   └── prep_monomer.py
├── vdw/
│   ├── sweep_phi.py              ← 映進: VdWスウィープ
│   ├── sweep_screw_phi.py        ← 螺旋軸: VdWスウィープ＋初期点生成
│   └── extract_init_phi.py       ← 映進: 初期点抽出（phi=0も対応）
├── amber/
│   ├── make_io_gene_phi.py       ← 映進: ダイマー幾何
│   ├── driver_gene_phi.py        ← 映進: 最適化ドライバー
│   ├── job_phi.py                ← 映進: SGEジョブ投入
│   ├── make_io_gene_screw_phi.py ← 螺旋軸: ダイマー幾何
│   ├── driver_screw_phi.py       ← 螺旋軸: 最適化ドライバー
│   └── job_screw_phi.py          ← 螺旋軸: SGEジョブ投入
└── gaussian/
    ├── extract_minima.py         ← ★新規作成: 局所最小抽出
    ├── pipeline_phi.py           ← 映進: Gaussian投入（抽出部分を分離）
    ├── pipeline_screw_phi.py     ← 螺旋軸: Gaussian投入（抽出部分を分離）
    └── driver_dft_jobs.py        ← 空きノード管理・自動投入
```

### アーカイブ（`legacy/` フォルダへ移動）

```
vdw/
  sweep.py                   ← phi なし旧版
  extract_init.py            ← phi なし旧版（extract_init_phi.py でカバー済み）
  sweep_phi_antiparallel.py  ← 実験的

amber/
  driver_gene.py             ← phi なし旧版
  driver_gene_screw.py       ← phi なし旧版
  driver_gene_asym.py        ← 非対称（除外）
  driver_gene_phi_asym.py    ← 非対称（除外）
  driver_gene_phi_asym_anti.py ← 非対称（除外）
  driver_pfa.py              ← PFA専用（除外）
  driver_crystal_energy.py   ← 別用途
  calc_pentamer_interaction.py ← 別用途
  job.py / job_screw.py
  job_asym.py / job_phi_asym.py / job_phi_asym_anti.py
  make_io_gene.py / make_io_gene_screw.py
  make_io_gene_asym.py / make_io_gene_phi_asym.py
  make_io_gene_phi_asym_anti.py
  make_io_pfa.py

gaussian/
  pipeline.py / pipeline_v2.py / pipeline_v3.py ← 旧版
  pipeline_v3_asym.py / pipeline_screw.py        ← 旧版
  collect.py / collect_asym.py / collect_phi_asym.py / collect_phi.py
  select_minima.py / select_minima_phi.py
  make_qe_xyz.py / collect_qe_energy.py / collect_zscan_dft.py / collect_cif.py
  make_inp_from_xyz.py

flow/
  run.py                     ← 未使用
```

---

## 5. 今後の作業リスト

| 状態 | 優先度 | 作業 | 内容 |
|------|--------|------|------|
| ✅ 完了 | 高 | `extract_minima.py` 新規作成 | `pipeline_phi.py` / `pipeline_screw_phi.py` の `extract_from_step1()` を独立化 |
| ✅ 完了 | 高 | driver への自動呼び出し追加 | `driver_gene_phi.py`, `driver_screw_phi.py` の最後で `extract_minima.py` を実行 |
| ✅ 完了 | 高 | `pipeline_phi.py` の整理 | 抽出部分を除き、Gaussian 投入のみに |
| ✅ 完了 | 高 | `pipeline_screw_phi.py` の整理 | 同上 |
| ✅ 完了 | 中 | legacy フォルダ作成・移動 | 旧バージョンファイルを整理 |
| ✅ 完了 | 中 | `os.environ['HOME']` の除去 | 全8ファイルから削除。`MONOMER_DIR` を `__file__` 相対パスに変更 |
| ✅ 完了 | 中 | `sweep_phi.py` 変数名整理 | 接触方向角 `beta` → `theta_c` に改名（`extract_init_phi.py` も追従） |
| ✅ 完了 | 中 | `vdw_R` のバグ修正 | `sweep_phi.py` を mask 方式に統一（接触不可ペアの誤計上を修正） |
| ✅ 完了 | 中 | screw VdW sweep に構造分類を追加 | `sweep_screw_phi.py` に a-stack / b-stack 計算と `--select` オプションを追加 |
| 🔲 未着手 | 中 | 環境設定ファイルの導入 | `~/.auto_opt.yaml` でパスを外出し（詳細は §6 参照） |
| 🔲 未着手 | 中 | `utils.py` の整理 | 共通関数を整備 |
| 🔲 未着手 | 低 | README.md 作成 | インストール・使い方 |
| 🔲 未着手 | 低 | pyproject.toml 整備 | 依存パッケージ明記 |

---

## 6. 環境移植性の方針

他の研究機関でも使えるようにするため、環境依存箇所を段階的に取り除く。

### 依存の種類と対応方針

| 種類 | 現状の問題 | 対応方針 |
|------|-----------|---------|
| ハードコードパス | `os.environ['HOME'] = '/home/miyoshi'` など | 即削除（`Path.home()` で代替） |
| ソフトウェアパス | `source ~/anaconda3/...`, `conda activate amber`, `export g16root=/home/g03` | 設定ファイルで外出し |
| ジョブスケジューラー | SGE (`qsub`/`qstat`) 固有 | まずはSGEのみ対応、将来スケジューラー抽象化 |

### フェーズ 1（近いうち）: 設定ファイルの導入

ユーザーが一度だけ書く設定ファイル `~/.auto_opt.yaml`：

```yaml
amber:
  conda_init: ~/anaconda3/etc/profile.d/conda.sh  # conda の初期化スクリプト
  conda_env: amber                                  # conda 環境名

gaussian:
  g16root: /home/g03                               # Gaussian インストール先
  scrdir: /scr/$JOB_ID                             # 一時ファイル置き場

scheduler:
  type: sge                                         # sge / slurm / local
  queues:
    - name: gr1.q
      nproc: 40
    - name: gr2.q
      nproc: 52
  max_concurrent: 6
```

コードはこのファイルを起動時に読み込み、ジョブスクリプトの中身を動的に生成する。

### フェーズ 2（将来）: スケジューラー抽象化

```
SchedulerBase
├── SGEScheduler   (qsub / qstat)   ← 現状
├── SLURMScheduler (sbatch / squeue) ← 将来
└── LocalScheduler (subprocess並列)  ← ローカル実行用
```

設定ファイルの `scheduler.type` で切り替え。
