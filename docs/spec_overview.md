# auto_opt システム仕様書

最終更新: 2026-06-29（glide VdWスウィープ直接出力化・UI改善・ディレクトリ構成整理）

---

## 1. システム概要

有機半導体結晶の**層内・層間分子配置**を自動最適化するツール。
ユーザーが用意したモノマーの XYZ ファイルを入力として、以下を自動実行する：

1. **モノマー前処理**（Gaussian DFT 最適化 → RESP 電荷 → GAFF2 パラメータ）
2. **層内最適化**（VdW スウィープ → Amber 力場最適化 → 局所最小抽出）
3. **層間最適化**（スタッキング VdW スウィープ → Amber 力場最適化）

### 実行環境の考え方

各ステップはユーザーの環境に応じてローカルまたは HPC で実行できる。

| ステップ | ローカル実行 | HPC 実行 | 備考 |
|---------|------------|---------|------|
| VdW スウィープ | ✅ 推奨 | ✅ 可 | 純Python・Amber不要。数分で終わる |
| Amber 層内最適化 | ✅ 可（Amberあれば） | ✅ 推奨 | 計算量が多く、HPC が現実的 |
| モノマー前処理 | ❌ Gaussian が必要 | ✅ 推奨 | Gaussian ライセンスは通常 HPC |
| スタッキング Amber | ✅ 可（Amberあれば） | ✅ 推奨 | 同上 |
| 結果収集・可視化 | ✅ 常にローカル | — | app.py はローカルで動作 |

想定する典型的な構成例：

**構成A（標準）**: VdW ローカル、Amber/Gaussian → HPC  
**構成B（フル HPC）**: 全ステップを HPC、可視化のみローカル  
**構成C（フルローカル）**: Amber と Gaussian をローカルに持つユーザー（稀）

SSH 連携機能（Section 6）では、各ステップをローカル or HPC どちらで実行するかを UI から選択できる設計を目指す。

---

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
│  [glide]  vdw/sweep_phi.py → step1_init_params.csv  │
│  [screw]  vdw/sweep_screw_phi.py → step1_init_      │
│                                    params.csv        │
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
| スクリプト | `vdw/sweep_phi.py` |
| パラメータ | alpha, phi, z の範囲と刻み、vdw_select (a-stack/b-stack/all) |
| 出力 | `step1_init_params.csv`（a-stack / b-stack 分類付き、直接出力） |

#### 螺旋軸（screw）

| 項目 | 内容 |
|------|------|
| スクリプト | `vdw/sweep_screw_phi.py` |
| パラメータ | alpha, beta, phi, z の範囲と刻み |
| 出力 | `step1_init_params.csv`（a-stack / b-stack 分類付き） |

### Step 1.5: VdW グリッド力場1点評価（オプション）

VdW スウィープは幾何学的な接触距離しか評価しないため 2D マップの精度が低い。
全グリッド点について **最適化なし（sander maxcyc=0）** の力場エネルギーを計算し、
Tab 1 の 2D マップをエネルギーで色付けすることで精度を高める。

| 項目 | 内容 |
|------|------|
| ジョブ投入 | `amber/job_eval_grid.py` |
| ドライバー | `amber/eval_vdw_grid.py`（各 eval_split_N/ で実行） |
| 使用 sander 入力 | `resources/FF_calc.in`（既存、maxcyc=0） |
| エネルギー式 | glide: `E = 2*E1 + 2*E2 + 4*E3`、screw: `E = 2*E1 + 2*E2 + 2*E3 + 2*E4` |
| 出力 | `step1_init_params.csv` に `E` カラムを追加 |

**所要時間の目安（glide 540点、6ノード）**: ~10〜15分。VdW スウィープより遅いが、Tab 1 のマップ品質が大幅に向上する。速度が問題な場合は本ステップをスキップして v0.1.0 に戻す（`git checkout v0.1.0`）。

**実行方法**:
```bash
python -m auto_opt.amber.job_eval_grid \
    --auto-dir runs/BTBT_glide \
    --monomer-name BTBT \
    --symmetry glide
```

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

**Streamlit について**:
Python だけでブラウザ UI を作れるライブラリ。HTML/CSS/JavaScript を書かずに済む。
`streamlit run src/auto_opt/app.py` で起動する。ローカル Mac 上で動かす想定。

HTML/JavaScript（React 等）の方が UI の自由度・レスポンスは高いが、開発コストが大きく
研究者が Python のみで保守できなくなる。研究ツールとして Streamlit で十分と判断。
（将来的に UI を凝りたい場合は移行を検討する。）

4タブ構成。

**Tab 0: セットアップ**

| 機能 | 説明 |
|------|------|
| ローカル作業ディレクトリ | セットアップタブ先頭で指定。yaml/xyz 保存先の基準パスになる |
| monomer.xyz ドラッグ&ドロップ | ファイル名から自動推定 or 手動でモノマー名を指定 → `{local_work_dir}/data/monomer/{name}_raw.xyz` に直接保存 |
| 対称性 / 電荷 / 多重度 | glide or screw、Gaussian 計算パラメータ |
| HPC 設定 | user@hostname、HPC 作業ディレクトリ（プロジェクトルート）を指定。`auto_dir`・`data_dir`・`monomer_xyz` を自動導出 |
| Amber Tools パス | スパコン上の bin ディレクトリ。キュー設定（gr1.q / gr2.q）も入力 |
| 粗VdWパラメータ | デフォルト値入り（glide: α0-90/10°, φ-10-10/4°, z-2-2/0.5; screw: +β-20-20/5°） |
| HPC ディレクトリ構成プレビュー | 入力値から生成されるディレクトリ構造を表示 |
| ~/.auto_opt.yaml 生成 | 入力値から生成、コードブロック表示 + `~/.auto_opt.yaml` に保存ボタン |
| run_config.yaml 生成 | `auto_dir`/`data_dir`/`monomer_xyz` パス付きで生成 → `{local_work_dir}/run_config.yaml` に直接保存 |
| コマンド表示 | scp 転送コマンド ＋ `python -m auto_opt.run --start-from monomer --stop-after vdw`（cd 不要・絶対パス指定） |
| **[計画] SSH 連携** | Paramiko を使い「ファイル転送」「コマンド実行」「ログ表示」をボタン1つで実行（→ SSH連携機能 参照） |

**Tab 1: VdW スキャン**

| 機能 | 説明 |
|------|------|
| a×b ヒートマップ | `step1_init_params.csv` から任意の2軸で格子面積マップを表示 |
| 3D 表示 | クリックした点の9分子クラスターを py3Dmol で表示（E列なしでも動作） |
| structure_type フィルタ | a-stack / b-stack をラジオボタンで選択（単一選択） |
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
| 結果プロット | `stacking_results.csv` を読み込み E_layer / E_inter(×2) / E_total を相対エネルギーで1Dプロット |
| X軸選択 | z / phi / beta / cy / cz から選択。全X値で他変数をまたいで最安定点を自動選択 |
| パラメータテーブル | 各X値点の全パラメータ（alpha, beta, phi, z, a, bt1, bt2, cx, cy, cz, E_layer, E_stack, E_total）を表示 |
| 3D表示 | スライダーでX軸値を選択 → 層クラスター(9分子)＋スタッキング分子(2分子)を py3Dmol で表示 |
| CSV ダウンロード | テーブル表示列のみのサマリーを `{MON}_stacking_summary.csv` として出力 |
| エネルギー定義 | E_total = E_layer + 2×E_stack（上下2層分をカウント） |

**Tab 4: 変数説明**（お試し機能、v0.2.0 で戻せる）

| 機能 | 説明 |
|------|------|
| インタラクティブ確認 | alpha/phi/z/beta のスライダーを動かすと 3D ビューがリアルタイム更新 |
| アニメーション | 1変数だけをスイープしたフレームを py3Dmol で自動再生（各変数ごとに独立） |
| screw 対応 | 対称性を screw にすると beta スライダー・アニメーションが追加される |
| a/b 設定 | VdW 接触距離を手動入力（scan CSV 不要で動作） |

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
| 入力 | `split_*/step1.csv`、`stacking_candidates.csv`（E_layer 取得用） |
| 出力 | `stacking_results.csv`（cy, cz, E_stack, E_layer, E_total 等） |
| オプション | `--candidates stacking_candidates.csv` で E_layer・E_total を自動計算 |
| エネルギー | `E_stack = E_amber - 18×E_mono`、`E_total = E_layer + 2×E_stack` |

---

## 4. ファイル構成（現行）

```
src/auto_opt/
├── cluster.py                  ← SGE ジョブ管理・設定読み込み
├── run.py                      ← Step 0-3 オーケストレーター
├── app.py                      ← Streamlit 可視化 UI
├── utils.py                    ← 回転行列・vdW半径・Amberパーサ等
├── monomer/
│   └── prep_monomer.py
├── vdw/
│   ├── sweep_phi.py            ← glide VdWスウィープ（step1_init_params.csv を直接出力）
│   └── sweep_screw_phi.py
├── amber/
│   ├── make_io_gene_phi.py
│   ├── driver_gene_phi.py
│   ├── job_phi.py
│   ├── make_io_gene_screw_phi.py
│   ├── driver_screw_phi.py
│   ├── job_screw_phi.py
│   ├── eval_vdw_grid.py          ← VdW グリッド全点 力場1点評価ドライバー
│   ├── job_eval_grid.py          ← 同 SGE ジョブ投入スクリプト
│   └── resources/
│       ├── FF_calc.in
│       └── epsilon_*.frcmod
├── stacking/
│   ├── sweep_stacking_vdw.py
│   ├── make_io_stacking.py       ← glide ダイマーペア生成
│   ├── make_io_stacking_screw.py ← screw ダイマーペア生成
│   ├── driver_stacking.py        ← glide/screw 統合ドライバー
│   ├── job_stacking.py           ← glide/screw 統合ジョブ投入
│   ├── merge_results.py          ← 結果収集
│   ├── merge_csv.py
│   └── resources/
│       └── FF_calc.in
├── plot/
│   ├── make_cluster_xyz.py
│   ├── energy_map.py
│   └── export_xyz.py
└── gaussian/
    ├── pipeline_phi.py
    ├── pipeline_screw_phi.py
    ├── driver_dft_jobs.py
    └── extract_minima.py

legacy/                           ← 旧版スクリプト（git 管理外）
data/                             ← モノマー・計算データ（git 管理外）
examples/
  auto_opt.yaml                   ← ~/.auto_opt.yaml のテンプレート
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

## 6. SSH 連携機能（計画）

現在の Tab 0 はコマンドをテキストで表示するだけだが、Paramiko（Python SSH ライブラリ）を使って UI 上でそのまま実行できるようにする。

### 前提条件

- ユーザーのローカルマシンに `~/.ssh/id_rsa`（または `id_ed25519`）が存在し、HPC にパスワードなしで SSH できること
- アプリはローカルで `streamlit run` することが前提（クラウドへのデプロイは対象外。秘密鍵をサーバーに置けないため）

### ステップごとの実行場所の設定

各ステップをローカル or HPC どちらで動かすかをUIから選択できるようにする。

| ステップ | デフォルト実行場所 | 変更可否 |
|---------|----------------|---------|
| VdW スウィープ | ローカル | ✅ HPC も選べる |
| Amber 層内最適化 | HPC | ✅ ローカル Amber があれば選べる |
| モノマー前処理 | HPC（Gaussian 必須） | ❌ 固定 |
| スタッキング VdW | ローカル | ✅ HPC も選べる |
| スタッキング Amber | HPC | ✅ ローカル Amber があれば選べる |

`~/.auto_opt.yaml` にて各ステップの `execution: local | hpc` を設定で切り替える方針：

```yaml
execution:
  vdw:      local   # ローカルで実行（デフォルト）
  amber:    hpc     # HPC で実行（デフォルト）
  stacking_vdw:   local
  stacking_amber: hpc
```

ローカル実行時は `subprocess` で直接呼び出し、HPC 実行時は Paramiko 経由で SSH 実行する。

### サイドバーに追加する設定項目

| 設定 | 例 |
|------|----|
| HPC ホスト | `133.11.68.31` |
| HPC ユーザー名 | `miyoshi` |
| SSH 秘密鍵パス | `~/.ssh/id_rsa` |
| HPC 作業ディレクトリ | `/home/miyoshi/Working/auto_opt` |
| **ローカル作業ディレクトリ** | `~/Working/auto_opt/runs` |

ローカル作業ディレクトリを指定することで、結果ファイルが自動的にそのディレクトリ以下に保存される。

### 各タブに追加するボタン・機能

| タブ | 追加機能 |
|------|---------|
| Tab 0 | ① `monomer_raw.xyz` を HPC へアップロード ② `run_config.yaml` / `~/.auto_opt.yaml` を HPC へ転送 ③ `python -m auto_opt.run --stop-after vdw` を（ローカルor SSH で）実行 |
| Tab 1 | ① `step1_init_params.csv` をローカル作業ディレクトリに自動ダウンロード ② `python -m auto_opt.run --start-from amber --stop-after amber` を（ローカルorSSH で）実行 |
| Tab 2 | ① `stacking_candidates.csv` を（ローカルorHPC に）転送 ② `sweep_stacking_vdw` を（ローカルorSSH で）実行 ③ `job_stacking.py` を（ローカルorSSH で）実行 → ジョブ完了までポーリングで進捗表示 ④ 完了後 `merge_results.py` を実行 ⑤ `stacking_results.csv` をローカルに自動ダウンロード |
| Tab 3 | 結果が自動的に表示される（手動アップロード不要になる） |

### 実装方針

```python
import paramiko, subprocess

def run_remote(cmd, host, user, key_path):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(host, username=user, key_filename=key_path)
    stdin, stdout, stderr = ssh.exec_command(cmd)
    for line in stdout:
        st.write(line)

def run_local(cmd):
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)
    for line in proc.stdout:
        st.write(line)

# ファイル転送（ローカル→HPC）
sftp.put(local_path, remote_path)

# ファイル取得（HPC→ローカル作業ディレクトリ）
local_save = Path(local_work_dir) / Path(remote_path).name
sftp.get(remote_path, str(local_save))
```

### 依存ライブラリ

```bash
pip install paramiko
```

---

## 7. 今後の作業リスト

| 状態 | 優先度 | 作業 |
|------|--------|------|
| ✅ | 高 | モノマー前処理 `--mode all` 実装 |
| ✅ | 高 | `cluster.get_amber_tool()` による AmberTools パス解決 |
| ✅ | 高 | `run.py` に monomer・stacking・merge ステップ追加 |
| ✅ | 高 | スタッキング IO・ドライバー・ジョブ投入を glide/screw 統合で書き直し |
| ✅ | 高 | `sweep_stacking_vdw.py` glide/screw の分子配置を正しく分離 |
| ✅ | 中 | `app.py` スタッキング候補選択 UI（ヒートマップ複数選択） |
| ✅ | 高 | `app.py` Tab 0 セットアップUI実装 |
| ✅ | 中 | `data_dir` を `run_config.yaml` で指定可能に変更 |
| ✅ | 高 | `merge_results.py` に E_layer / E_stack / E_total 計算を追加 |
| ✅ | 高 | Tab 3 スタッキング結果UI（相対エネルギープロット・パラメータテーブル・3D表示）実装 |
| ✅ | 高 | **スタッキング動作確認**（DNTT screwで実計算・結果確認済み） |
| ✅ | 高 | **セットアップ〜モノマー前処理の動作確認** |
| 🔲 | 中 | `--monomer-dir` を amber/stacking 全スクリプトに通す（`data_dir` 対応の完成） |
| ✅ | 高 | **VdW グリッド力場1点評価の動作確認**（`job_eval_grid.py` → Tab 1 マップ品質確認） |
| 🔲 | 低 | Gaussian DFT ステップの整理 |
| 🔲 | 高 | **Amber ドライバーの律速改善**（詳細は下記セクション参照） |
| 🔲 | 中 | **SSH 連携機能の実装**（ローカル作業ディレクトリ指定・ファイル自動授受・HPC コマンド実行・進捗表示） |
| 🔲 | 中 | **実行環境の柔軟化**（VdW/Amber 各ステップをローカルor HPC から選択。`~/.auto_opt.yaml` に `execution:` セクション追加） |
| 🔲 | 低 | HPC 可搬性対応（分子科学研究所 PBS/SLURM スケジューラー対応） |
| 🔲 | 低 | ドキュメント整備（計算化学を知らない材料研究者向け README） |
| 🔲 | 低 | **HPC シングルマシンモード**（SSH ポートフォワーディングで HPC 上の Streamlit を操作。ローカル/HPC の区別をなくし SCP 不要に。app.py にモード切り替えラジオを追加） |

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
- [ ] VdW スウィープが走り `<auto_dir>/step1_init_params.csv` が生成されている

**確認（ローカル）**
- [ ] `step1_init_params.csv` を scp でダウンロードし Tab 1 (VdW スキャン) で読み込める
- [ ] a×b ヒートマップが表示され、3D 構造が表示できる

---

## 8. Amber ドライバー律速問題の詳細分析

### なぜ計算時間が理論値の5倍かかるか

Amber 計算本体は約 **20ms/ジョブ**（出力ファイルの TIMING セクションより）だが、
現在のファイルベース・ポーリング方式により1ジョブあたりのトータルが約 **100ms** になっている。

| ステップ | 時間 | 備考 |
|---|---|---|
| 入力ファイル書き込み | ~5ms | |
| プロセス起動（subprocess） | ~10ms | |
| **Amber 計算本体** | **~20ms** | ← 有効な時間 |
| 出力ファイル書き込み | ~5ms | |
| ポーリング検知待ち | **~50ms** | `sleep(0.1)` で最大 100ms、平均 50ms |
| CSV read × 4本 | ~10ms | NFS 上では特に遅い |
| CSV write | ~5ms | |
| **合計** | **~105ms** | **理論値の約5倍（実測と一致）** |

### なぜ sleep を長くしても解決しないか

計算が 20ms で終わるのに `sleep(2)` にすると平均 **1000ms** 待つことになり、
逆に遅くなる。計算が速いほど poll 間隔は短くしなければならないが、
短くしても per-job のファイル I/O オーバーヘッドが消えるわけではない。
**polling 自体をやめる**のが唯一の根本解決。

### 推奨改善：multiprocessing.Pool への移行

```python
# 現在：ファイルポーリング（遅い）
while not done:
    check_csv_files()   # NFS read × 4本
    sleep(0.1)          # 平均50ms 無駄待ち

# 改善後：Pool（速い＋進捗も見える）
completed = 0
def on_done(result):
    global completed
    completed += 1
    print(f"[{completed}/{total}] 完了: {result}")  # ジョブごとにリアルタイム表示

with Pool(processes=N) as pool:
    for job in all_jobs:
        pool.apply_async(run_amber, args=(job,), callback=on_done)
    pool.close()
    pool.join()

write_csv(results)  # 最後に1回だけ書き出し
```

**改善効果：**
- ポーリング待ち（50ms）→ 0ms（プロセス終了を直接検知）
- CSV read/write の繰り返し → 最後に1回のみ
- 期待速度改善：**約3〜4倍**

**進捗確認について：**
`apply_async` のコールバックでジョブ完了ごとに即座に通知されるため、
現在より細かく・リアルタイムに進捗が確認できる。CSV の途中状態を見る必要もなくなる。

**実装コスト：** 半日〜1日程度。`driver_screw_phi.py` と `driver_gene_phi.py` の
`listen()` / メインループを書き直す必要がある。
