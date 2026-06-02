#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DFT (Gaussian) ジョブを 6ノード（gr1.q×3 + gr2.q×3）に自動分配するドライバー。

動作:
  1. 指定ディレクトリ内の .inp ファイルを収集してキューに積む
  2. qstat -f でノードの空き状況をリアルタイムに確認
  3. 空いているノードに順次ジョブを投入（最大 MAX_CONCURRENT=6 並列）
  4. 1ジョブ完了 → 空きを確認 → 次のジョブを投入、を繰り返す
  5. 全ジョブ完了で終了

使い方:
  # HPCサーバー上で実行
  python -m auto_opt.gaussian.driver_dft_jobs \\
      --inp-dirs /home/miyoshi/Working/DNTT/inp/test21 \\
                 /home/miyoshi/Working/DNTT/inp/test22

  # テストモード（ジョブスクリプト生成のみ、qsubしない）
  python -m auto_opt.gaussian.driver_dft_jobs \\
      --inp-dirs /path/to/inp --test

注意:
  実際のノード名は qstat -f の出力で確認し、
  下の QUEUE_SPEC の node_prefix を環境に合わせてください。
"""

import os
import re
import sys
import time
import subprocess
import argparse
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional, Tuple

# ─── クラスター設定（環境に合わせて変更） ────────────────────────────
USER = os.environ.get("USER", "miyoshi")

# キュー仕様: {queue_name: nproc}
QUEUE_SPEC = {
    "gr1.q": 40,
    "gr2.q": 52,
}
# 各キューの最大同時使用ノード数
MAX_PER_QUEUE = {
    "gr1.q": 3,
    "gr2.q": 3,
}
MAX_CONCURRENT = sum(MAX_PER_QUEUE.values())  # = 6

POLL_INTERVAL = 30   # ノード確認の間隔（秒）

# Gaussian 環境設定
G16_SETUP = [
    "export g16root=/home/g03\n",
    "source $g16root/g16/bsd/g16.profile\n",
]


# ─────────────────────────────────────────────────────────────────────
#  qstat ユーティリティ
# ─────────────────────────────────────────────────────────────────────

def _run_qstat_f() -> str:
    """qstat -f -u '*' の出力を返す。失敗時は空文字列。"""
    try:
        r = subprocess.run(
            ["qstat", "-f", "-u", "*"],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout
    except Exception:
        return ""


def _run_qstat_user() -> str:
    """qstat -u {USER} の出力を返す。"""
    try:
        r = subprocess.run(
            ["qstat", "-u", USER],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout
    except Exception:
        return ""


def get_free_queue_instances() -> Dict[str, List[str]]:
    """
    qstat -f を解析し、空き（used==0）のqueue instanceを返す。

    Returns:
        {"gr1.q": ["gr1.q@node01", "gr1.q@node03"],
         "gr2.q": ["gr2.q@node05"], ...}
    """
    output = _run_qstat_f()
    free: Dict[str, List[str]] = {q: [] for q in QUEUE_SPEC}

    # 例: "gr1.q@cmdell01              BIPC   0/0/40        0.000    lx-amd64"
    #       queuename@host              qtype  resv/used/tot  load     arch     [states]
    pat = re.compile(
        r'^((' + '|'.join(re.escape(q) for q in QUEUE_SPEC) + r')@\S+)'
        r'\s+\S+'           # qtype
        r'\s+\d+/(\d+)/\d+' # resv/used/tot → used = group(3)
        r'\s+[\d.]+',       # load_avg
        re.MULTILINE,
    )
    for m in pat.finditer(output):
        qi    = m.group(1)                   # "gr1.q@node01"
        qname = m.group(2)                   # "gr1.q"
        used  = int(m.group(3))
        if used == 0:
            free[qname].append(qi)

    return free


def get_my_running_job_ids() -> List[str]:
    """自分のジョブIDのリストを返す（実行中 + キュー待ち）。"""
    output = _run_qstat_user()
    ids = []
    for line in output.splitlines():
        # "123456  0.55500 jobname  user  r  MM/DD/HH:MM:SS  1"
        m = re.match(r'^\s*(\d+)', line)
        if m:
            ids.append(m.group(1))
    return ids


def count_my_jobs_in_queue(queue_name: str) -> int:
    """指定キューで自分が実行中+待機中のジョブ数を返す。"""
    output = _run_qstat_user()
    return sum(queue_name in line for line in output.splitlines())


# ─────────────────────────────────────────────────────────────────────
#  ジョブスクリプト生成
# ─────────────────────────────────────────────────────────────────────

def make_job_script(
    inp_path: Path,
    queue_name: str,
    nproc: int,
) -> Path:
    """
    .inp ファイルに対応する job.sh を生成して返す。
    queue_instance は qsub のコマンドライン引数で指定するので
    スクリプト内では queue_name のみ記述（フォールバック用）。
    """
    base = inp_path.stem
    sh_path = inp_path.with_suffix(".sh")
    lines = [
        "#!/bin/sh\n",
        "#$ -S /bin/sh\n",
        "#$ -cwd\n",
        "#$ -V\n",
        f"#$ -q {queue_name}\n",
        f"#$ -pe OpenMP {nproc}\n",
        f"#$ -N {base}\n",
        "\n",
        "hostname\n",
        "\n",
    ] + G16_SETUP + [
        "\n",
        "export GAUSS_SCRDIR=/scr/$JOB_ID\n",
        "mkdir -p /scr/$JOB_ID\n",
        "\n",
        f"g16 < {base}.inp > {base}.log\n",
        "\n",
        "rm -rf /scr/$JOB_ID\n",
        "\n",
    ]
    sh_path.write_text("".join(lines), encoding="utf-8")
    return sh_path


def submit_job(
    sh_path: Path,
    queue_instance: str,   # "gr1.q@node01"
    isTest: bool = False,
) -> Optional[str]:
    """
    qsub でジョブを投入し、ジョブIDを返す。
    queue_instance を -q 引数で指定してそのノードに固定投入する。
    """
    if isTest:
        print(f"    [TEST] qsub -q {queue_instance} {sh_path.name}")
        return "TEST_JOB_ID"

    try:
        r = subprocess.run(
            ["qsub", "-q", queue_instance, sh_path.name],
            cwd=str(sh_path.parent),
            capture_output=True, text=True, timeout=15,
        )
        # "Your job 123456 ("name") has been submitted"
        m = re.search(r"Your job (\d+)", r.stdout)
        if m:
            return m.group(1)
        print(f"  [WARNING] qsub 応答が解析できませんでした: {r.stdout.strip()}")
        return None
    except Exception as e:
        print(f"  [ERROR] qsub 失敗: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
#  メインループ
# ─────────────────────────────────────────────────────────────────────

def collect_inp_files(inp_dirs: List[str]) -> deque:
    """
    指定ディレクトリ内の .inp ファイルを収集してキューを返す。
    .log が既に存在するものはスキップ（再実行しない）。
    """
    pending = deque()
    for d in inp_dirs:
        p = Path(d)
        if not p.exists():
            print(f"  [SKIP] ディレクトリが見つかりません: {d}")
            continue
        for inp in sorted(p.glob("*.inp")):
            log = inp.with_suffix(".log")
            if log.exists():
                print(f"  [SKIP] ログ既存: {inp.name}")
                continue
            pending.append(inp)
    return pending


def _pick_queue_instance(
    free_by_queue: Dict[str, List[str]],
    used_counts: Dict[str, int],
    prefer: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """
    空きのあるキューインスタンスを1つ選んで返す。
    prefer が指定されていれば優先的にそのキューを使う。
    Returns: (queue_name, queue_instance) or None
    """
    order = list(QUEUE_SPEC.keys())
    if prefer and prefer in order:
        order = [prefer] + [q for q in order if q != prefer]

    for qname in order:
        if used_counts.get(qname, 0) >= MAX_PER_QUEUE[qname]:
            continue  # このキューは上限到達
        instances = free_by_queue.get(qname, [])
        if instances:
            return qname, instances[0]
    return None


def run_all(
    inp_dirs: List[str],
    isTest: bool = False,
    poll_interval: int = POLL_INTERVAL,
) -> None:
    """
    全 .inp ファイルを 6ノードに分散投入してすべての完了を待つ。
    """
    pending = collect_inp_files(inp_dirs)
    total = len(pending)
    if total == 0:
        print("投入対象の .inp ファイルがありません（.log が存在しないもの）。")
        return

    print(f"投入対象: {total} ファイル  /  最大同時実行: {MAX_CONCURRENT} ジョブ\n")

    # 実行中ジョブの管理: {job_id: (inp_path, queue_name)}
    running: Dict[str, Tuple[Path, str]] = {}
    # キューごとの使用ノード数
    used_counts: Dict[str, int] = {q: 0 for q in QUEUE_SPEC}

    submitted = 0
    completed = 0

    while pending or running:
        # ── 完了ジョブの確認 ──────────────────────────────────────────
        if running and not isTest:
            current_ids = set(get_my_running_job_ids())
            finished = [jid for jid in running if jid not in current_ids]
            for jid in finished:
                inp_path, qname = running.pop(jid)
                used_counts[qname] = max(0, used_counts[qname] - 1)
                completed += 1
                log = inp_path.with_suffix(".log")
                status = "✓ done" if log.exists() else "? log未生成"
                print(f"  [{completed}/{total}] 完了: {inp_path.name}  {status}")

        # ── 空きノードの確認 ─────────────────────────────────────────
        if pending:
            if isTest:
                free_by_queue = {q: [f"{q}@node{i+1:02d}" for i in range(MAX_PER_QUEUE[q])]
                                 for q in QUEUE_SPEC}
            else:
                free_by_queue = get_free_queue_instances()

            _print_node_status(free_by_queue, used_counts, len(running), len(pending))

            # 空きがある限り投入
            prefer_queue = None  # 交互に gr1/gr2 を選ぶ用
            while pending:
                pick = _pick_queue_instance(free_by_queue, used_counts, prefer=prefer_queue)
                if pick is None:
                    break  # 空きなし

                qname, qi = pick
                inp_path = pending.popleft()

                # ジョブスクリプト生成
                sh_path = make_job_script(inp_path, qname, QUEUE_SPEC[qname])

                # 投入
                print(f"  投入: {inp_path.name}  →  {qi}")
                job_id = submit_job(sh_path, qi, isTest=isTest)

                if job_id:
                    running[job_id] = (inp_path, qname)
                    used_counts[qname] += 1
                    submitted += 1
                    # 同じノードを連続で使わないよう次は別キューを優先
                    other = [q for q in QUEUE_SPEC if q != qname]
                    prefer_queue = other[0] if other else None
                    # 使ったインスタンスをfree_by_queueから除く
                    if qi in free_by_queue.get(qname, []):
                        free_by_queue[qname].remove(qi)
                else:
                    pending.appendleft(inp_path)  # 失敗 → 戻す
                    break

        if pending or (running and not isTest):
            if isTest:
                break   # テストモードはループしない
            time.sleep(poll_interval)

    print(f"\n全ジョブ完了: {completed}/{total}")


def _print_node_status(
    free_by_queue: Dict[str, List[str]],
    used_counts: Dict[str, int],
    n_running: int,
    n_pending: int,
) -> None:
    parts = []
    for qname, max_n in MAX_PER_QUEUE.items():
        n_free = len(free_by_queue.get(qname, []))
        n_used = used_counts.get(qname, 0)
        parts.append(f"{qname}: {n_used}使用/{n_free}空き/{max_n}台")
    print(f"[ノード状況] {' | '.join(parts)}  "
          f"(実行中:{n_running} / 待機:{n_pending})")


# ─────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inp-dirs", nargs="+", required=True,
        help=".inp ファイルが入ったディレクトリ（複数指定可）",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="テストモード: ジョブスクリプト生成のみ、qsub は実行しない",
    )
    parser.add_argument(
        "--poll", type=int, default=POLL_INTERVAL,
        help=f"ノード確認の間隔（秒、デフォルト: {POLL_INTERVAL}）",
    )
    args = parser.parse_args()

    run_all(inp_dirs=args.inp_dirs, isTest=args.test, poll_interval=args.poll)


if __name__ == "__main__":
    main()
