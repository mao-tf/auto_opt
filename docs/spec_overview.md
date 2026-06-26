# auto_opt システム仕様書

最終更新: 2026-06-26（Tab 0 追加）

---

## 1. システム概要

有機半導体結晶の**層内・層間分子配置**を自動最適化するツール。
ユーザーが用意したモノマーの XYZ ファイルを入力として、以下を自動実行する：

1. **モノマー前処理**（Gaussian DFT 最適化 → RESP 電荷 → GAFF2 パラメータ）
2. **層内最適化**（VdW スウィープ → Amber 力場最適化 → 局所最小抽出）
3. **層間最適化**（スタッキング VdW スウィープ → Amber 力場最適化）

### 対象とする結晶対称性

| 対称性 | 説明 | コード識別子 |
|--------|------|------------|
| 映進対称（glide） | 鏡映＋平行移動。b方向の隣分子は z が 2z ずれる | `glide` |
| 螺旋軸対称（screw） | 回転＋平行移動。b方向の隣分子は z オフセットなし | `screw` |

---

## 2. 全体ワークフロー

```
[ユーザー入力]
  monomer_raw.xyz  +  対称性（glide / screw）
      │
      ▼
┌──────────────────────────────────────┐
│  Step 0: モノマー前処理               │
│  monomer/prep_monomer.py --mode all  │
│  出力: data/monomer/{MON}.mol2       │
│        data/amber_ref/{MON}_gaff2.out│
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  Step 1: 層内 VdW スウィープ                          │
│                                                      │
│  [glide]  vdw/sweep_phi.py       → vdW_r_contact.csv│
│             ↓ extract_init_phi.py                    │
│  [screw]  vdw/sweep_screw_phi.py → step1_init_      │
│                                    params.csv (直接) │
└──────────────┬───────────────────────────────────────┘
               │ step1_init_params.csv
               ▼
┌──────────────────────────────────────────────────────┐
│  Step 2: 層内 Amber 最適化                            │
│                                                      │
│  [glide]  amber/job_phi.py → driver_gene_phi.py      │
│  [screw]  amber/job_screw_phi.py → driver_screw_phi  │
│                                                      │
│  → split_*/step1.csv → filtered_step1.csv            │
└──────────────┬───────────────────────────────────────┘
               │ filtered_step1.csv
               ↓ [ローカル Mac へ scp]
┌──────────────────────────────────────────────────────┐
│  可視化 (app.py / Streamlit)                          │
│  ① ヒートマップで層内安定構造を確認                   │
│  ② 「候補構造を選ぶ」モードで複数点を選択             │
│  ③ stacking_candidates.csv をダウンロード            │
└──────────────┬───────────────────────────────────────┘
               │ stacking_candidates.csv
               ↓ [スパコンへ scp]
┌──────────────────────────────────────────────────────┐
│  Step 3: スタッキング VdW スウィープ                  │
│  stacking/sweep_stacking_vdw.py                      │
│  → step1_init_params.csv（cy × cz の初期値）         │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  Step 4: スタッキング Amber 最適化                    │
│  stacking/job_stacking.py → driver_stacking.py       │
│  → split_*/step1.csv                                 │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  Step 5: 結果収集                                     │
│  stacking/merge_results.py                           │
│  → stacking_results.csv                              │
└──────────────┬───────────────────────────────────────┘
               ↓ [ローカル Mac へ scp]
┌──────────────────────────────────────────────────────┐
│  可視化 (app.py)                                      │
│  スタッキングエネルギープロット                       │
│  (E_layer / E_stack / E_total vs scan_axis)          │
└──────────────────────────────────────────────────────┘
```

---

## 3. ステップ別詳細

### Step 0: モノマー前処理

| 項目 | 内容 |
|------|------|
| スクリプト | `monomer/prep_monomer.py --mode all` |
| 入力 | `{MON}_raw.xyz`（最適化前の粗い構造可） |
| 出力 | `data/monomer/{MON}.mol2`, `data/amber_ref/{MON}_gaff2.out` |
| 処理 | Gaussian B3LYP/6-31G(d) SCF=Tight 最適化 → PCA 主軸整列 → HF/6-31G(d) ESP → RESP → GAFF2 |
| 外部依存 | Gaussian, antechamber, parmchk2, tleap, sander（パスは `~/.auto_opt.yaml` の `amber_tools` で指定） |

---

### Step 1: 層内 VdW スウィープ

#### 映進（glide）

| 項目 | 内容 |
|------|------|
| スクリプト | `vdw/sweep_phi.py` → `vdw/extract_init_phi.py` |
| パラメータ | alpha, phi, z の範囲と刻み |
| 出力 | `step1_init_params.csv` |

#### 螺旋軸（screw）

| 項目 | 内容 |
|------|------|
| スクリプト | `vdw/sweep_screw_phi.py` |
| パラメータ | alpha, beta, phi, z の範囲と刻み |
| 出力 | `step1_init_params.csv`（a-stack / b-stack 分類付き） |

---

### Step 2: 層内 Amber 最適化

| 項目 | glide | screw |
|------|-------|-------|
| ジョブ投入 | `amber/job_phi.py` | `amber/job_screw_phi.py` |
| ドライバー | `amber/driver_gene_phi.py` | `amber/driver_screw_phi.py` |
| 幾何生成 | `amber/make_io_gene_phi.py` | `amber/make_io_gene_screw_phi.py` |
| ダイマー種類 | 3種 (a/b/t-dimer) | 4種 (a/b/t1/t3-dimer) |
| エネルギー式 | `E = 2E1 + 2E2 + 4E3` | `E = 2E1 + 2E2 + 2E3 + 2E4` |
| 出力 | `filtered_step1.csv` | `filtered_step1.csv` |

ジョブ投入は `cluster.py` 経由で SGE キューを管理。キュー定義は `~/.auto_opt.yaml` で指定。

---

### 可視化（app.py / Streamlit）

```
streamlit run src/auto_opt/app.py
```

4タブ構成。

**Tab 0: セットアップ**

| 機能 | 説明 |
|------|------|
| monomer.xyz アップロード | 前処理前の粗い構造 → HPC 転送用にダウンロード |
| 対称性 / 電荷 / 多重度 | glide or screw、Gaussian 計算パラメータ |
| HPC 設定 | user@hostname、作業ディレクトリ、auto_dir |
| Amber Tools パス | スパコン上の bin ディレクトリ。キュー設定（gr1.q / gr2.q）も入力 |
| 粗VdWパラメータ | デフォルト値入り（glide: α0-90/10°, φ-10-10/4°, z-2-2/0.5; screw: +β-20-20/5°） |
| ~/.auto_opt.yaml 生成 | 入力値から生成、コードブロック表示 + ダウンロード |
| run_config.yaml 生成 | monomer_xyz パス付きで生成 + ダウンロード |
| コマンド表示 | scp 転送コマンド（ローカル実行）＋ `python -m auto_opt.run --start-from monomer --stop-after vdw`（スパコン実行） |

**Tab 1: VdW スキャン**

| 機能 | 説明 |
|------|------|
| a×b ヒートマップ | `step1_init_params.csv` から任意の2軸で格子面積マップを表示 |
| 3D 表示 | クリックした点の9分子クラスターを py3Dmol で表示（E列なしでも動作） |
| structure_type フィルタ | a-stack / b-stack / local_min を選択 |
| 精細スキャン設定 | 各変数の min/max/step を設定 → `run_config.yaml` をダウンロード |
| コマンド表示 | `python -m auto_opt.run --config run_config.yaml` を表示（VdW→Amber→collect を一括実行） |

**Tab 2: 層内最適化**

| 機能 | 説明 |
|------|------|
| E ヒートマップ | `filtered_step1.csv` から任意の2軸でエネルギーマップを表示 |
| 3D 表示 | クリックした点の9分子クラスターを py3Dmol で表示 |
| 候補構造を選ぶモード | ボタンで入る。クリックで複数点をトグル選択し、確定で追加 |
| 自動追加 | 指定軸の各値でEが最小の構造を一括追加 |
| CSV ダウンロード | `stacking_candidates.csv` として出力 |

**Tab 3: スタッキング結果**

| 機能 | 説明 |
|------|------|
| 結果プロット | `stacking_results.csv` を読み込み E_layer/E_stack/E_total を1Dプロット |

**XYZ ファイル名**: `{モノマー}_{symmetry}_alpha{val}_phi{val}_z{val}.xyz` 形式（screw は beta も含む）

---

### Step 3: スタッキング VdW スウィープ

| 項目 | 内容 |
|------|------|
| スクリプト | `stacking/sweep_stacking_vdw.py` |
| 入力 | `stacking_candidates.csv`（app.py から出力）, `data/monomer/{MON}.csv` |
| 処理 | cy を `b/2` の範囲でスキャンし、各点で VdW 接触 cz を計算 |
| 出力 | `step1_init_params.csv`（cx, cy, cz, 固定パラメータ） |
| 対称性 | `--symmetry [glide\|screw]` で切り替え |

分子の Z>0.1 (上半分) / Z<-0.1 (下半分) に分割して VdW 接触を計算。
- **glide**: 平行ペア `(0,±b,±2z)` ＋ T字ペア `(±a/2, ±b/2, ±z)`
- **screw**: 平行ペア `(0,±b,0)` ＋ T字ペア `(±a/2, ±bt1/2, ±z)`

---

### Step 4: スタッキング Amber 最適化

| 項目 | 内容 |
|------|------|
| ジョブ投入 | `stacking/job_stacking.py --symmetry [glide\|screw]` |
| ドライバー | `stacking/driver_stacking.py` |
| 幾何生成 | `stacking/make_io_stacking.py` (glide) / `stacking/make_io_stacking_screw.py` (screw) |
| ダイマー種類 | glide: 14ペア, screw: 13ペア |
| エネルギー式 | glide: `E = Σ14`, screw: `E = Σ[0:5] + Σ[5:13]/2` |
| 処理 | VdW 初期 cz を±0.1 ずつ拡張して最小エネルギー cz を探索 |
| 出力 | `split_*/step1.csv` |

Amber ツールのパスは `cluster.get_amber_tool()` で `~/.auto_opt.yaml` から解決。

---

### Step 5: 結果収集

| 項目 | 内容 |
|------|------|
| スクリプト | `stacking/merge_results.py` |
| 入力 | `split_*/step1.csv` |
| 出力 | `stacking_results.csv`（cy, cz, E 等） |

---

## 4. ファイル構成（現行）

```
src/auto_opt/
├── cluster.py                  ← SGE ジョブ管理・設定読み込み
├── run.py                      ← Step 0-3 オーケストレーター
├── app.py                      ← Streamlit 可視化 UI
├── utils.py
├── monomer/
│   └── prep_monomer.py
├── vdw/
│   ├── sweep_phi.py
│   ├── sweep_screw_phi.py
│   └── extract_init_phi.py
├── amber/
│   ├── make_io_gene_phi.py
│   ├── driver_gene_phi.py
│   ├── job_phi.py
│   ├── make_io_gene_screw_phi.py
│   ├── driver_screw_phi.py
│   └── job_screw_phi.py
├── stacking/
│   ├── sweep_stacking_vdw.py
│   ├── make_io_stacking.py       ← glide ダイマーペア生成
│   ├── make_io_stacking_screw.py ← screw ダイマーペア生成
│   ├── driver_stacking.py        ← glide/screw 統合ドライバー
│   ├── job_stacking.py           ← glide/screw 統合ジョブ投入
│   └── merge_results.py          ← 結果収集
├── plot/
│   └── make_cluster_xyz.py
└── gaussian/
    ├── pipeline_phi.py
    └── pipeline_screw_phi.py

legacy/                           ← 旧版（参照のみ）
data/
  monomer/                        ← {MON}.csv, {MON}.mol2
  amber_ref/                      ← {MON}_gaff2.out
examples/
  auto_opt_env.yaml               ← ~/.auto_opt.yaml のテンプレート
  run_config.yaml                 ← run.py 設定ファイルのテンプレート
```

---

## 5. 環境設定（~/.auto_opt.yaml）

```yaml
scheduler: sge
queues:
  - name: gr1.q
    nproc: 40
    pe: OpenMP
  - name: gr2.q
    nproc: 52
    pe: OpenMP
max_concurrent_jobs: 6
poll_interval: 30
nproc_reserve: 2

amber_tools:
  antechamber: ~/anaconda3/envs/amber/bin/antechamber
  parmchk2:    ~/anaconda3/envs/amber/bin/parmchk2
  tleap:       ~/anaconda3/envs/amber/bin/tleap
  sander:      ~/anaconda3/envs/amber/bin/sander
```

---

## 6. 今後の作業リスト

| 状態 | 優先度 | 作業 |
|------|--------|------|
| ✅ | 高 | モノマー前処理 `--mode all` 実装 |
| ✅ | 高 | `cluster.get_amber_tool()` による AmberTools パス解決 |
| ✅ | 高 | `run.py` に monomer ステップ追加 |
| ✅ | 高 | スタッキング IO・ドライバー・ジョブ投入を glide/screw 統合で書き直し |
| ✅ | 高 | `sweep_stacking_vdw.py` glide/screw の分子配置を正しく分離 |
| ✅ | 中 | `app.py` スタッキング候補選択 UI（ヒートマップ複数選択） |
| ✅ | 高 | `app.py` Tab 0 セットアップUI実装 |
| ✅ | 中 | `data_dir` を `run_config.yaml` で指定可能に変更 |
| 🔲 | 高 | **セットアップ〜モノマー前処理の動作確認** |
| 🔲 | 高 | `run.py` にスタッキングステップ追加 |
| 🔲 | 中 | スタッキング動作確認（isTest → 実計算） |
| 🔲 | 低 | Gaussian DFT ステップの整理 |

### セットアップ〜モノマー前処理 動作確認チェックリスト

Tab 0 で生成したファイルとコマンドを使って、以下を順に確認する。

**準備（ローカル）**
- [ ] Tab 0 でモノマー名・対称性・HPC設定を入力
- [ ] `run_config.yaml` をダウンロード（`data_dir` / `monomer_xyz` が正しいパスか確認）
- [ ] `auto_opt.yaml` をダウンロード（Amber パス・キュー設定が正しいか確認）

**転送（ローカルで実行）**
- [ ] `scp <local_xyz> user@hpc:<workdir>/data/monomer/<MON>_raw.xyz`
- [ ] `scp run_config.yaml user@hpc:<workdir>/`
- [ ] `scp auto_opt.yaml user@hpc:~/.auto_opt.yaml`（初回のみ）

**実行（スパコンで実行）**
- [ ] `python -m auto_opt.run --config run_config.yaml --start-from monomer --stop-after vdw`

**確認（スパコン上）**
- [ ] `<workdir>/data/monomer/<MON>.xyz` が生成されている（PCA 整列済み）
- [ ] `<workdir>/data/monomer/<MON>.csv` が生成されている
- [ ] `<workdir>/data/monomer/<MON>.mol2` が生成されている（RESP 電荷付き）
- [ ] `<workdir>/data/amber_ref/<MON>_gaff2.out` が生成されている
- [ ] VdW スウィープが走り `<workdir>/runs/<MON>_<sym>/vdW_r_contact_<MON>.csv` が生成されている
- [ ] `step1_init_params.csv` が生成されている

**確認（ローカル）**
- [ ] `step1_init_params.csv` を scp でダウンロードし Tab 1 (VdW スキャン) で読み込める
- [ ] a×b ヒートマップが表示され、3D 構造が表示できる
