#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster.py  ―  HPC クラスター設定 & SGE ジョブキュー管理

~/.auto_opt.yaml からユーザー環境設定を読み込む。
ファイルがなければ組み込みデフォルト値（cmdell81 相当）を使う。

設定例 (~/.auto_opt.yaml):
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
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ── デフォルト設定（cmdell81 相当） ────────────────────────────────────
_DEFAULTS: dict = {
    'scheduler': 'sge',
    'queues': [
        {'name': 'gr1.q', 'nproc': 40, 'pe': 'OpenMP'},
        {'name': 'gr2.q', 'nproc': 52, 'pe': 'OpenMP'},
    ],
    'max_concurrent_jobs': 6,
    'poll_interval': 30,
    'nproc_reserve': 2,
}

_cached: dict | None = None


def load_env(config_path: str | Path | None = None) -> dict:
    """
    環境設定を返す。
    優先順位: config_path 引数 > ~/.auto_opt.yaml > デフォルト値

    2 回目以降はキャッシュを返す。
    """
    global _cached
    if _cached is not None:
        return _cached

    cfg = dict(_DEFAULTS)
    search = [Path.home() / '.auto_opt.yaml']
    if config_path:
        search = [Path(config_path)] + search

    for p in search:
        if p.exists():
            with open(p, encoding='utf-8') as f:
                user = yaml.safe_load(f) or {}
            cfg.update(user)
            break

    _cached = cfg
    return cfg


def get_amber_tool(name: str) -> str:
    """AmberTools 実行ファイルのパスを返す。
    ~/.auto_opt.yaml の amber_tools セクション → PATH の順で探す。
    """
    import shutil
    cfg = load_env()
    path = cfg.get('amber_tools', {}).get(name)
    if path:
        return os.path.expanduser(path)
    found = shutil.which(name)
    if found is None:
        raise SystemExit(
            f'{name} が見つかりません。'
            f'~/.auto_opt.yaml の amber_tools に {name} のパスを設定してください。'
        )
    return found


def _queue_spec() -> Dict[str, dict]:
    """キュー名 → spec dict のマップ。"""
    return {q['name']: q for q in load_env()['queues']}


def _queue_names() -> List[str]:
    return [q['name'] for q in load_env()['queues']]


# ── SGE ジョブ数確認 ────────────────────────────────────────────────────

def get_my_job_count() -> int:
    """自分のジョブ数（実行中 + 待機中）を返す。"""
    user = os.environ.get('USER', '')
    try:
        r = subprocess.run(['qstat', '-u', user],
                           capture_output=True, text=True, timeout=15)
        return sum(1 for ln in r.stdout.splitlines() if re.match(r'^\s*\d+', ln))
    except Exception:
        return 0


def get_free_queue_instances() -> Dict[str, List[str]]:
    """qstat -f を解析して used==0 のキューインスタンスを返す。"""
    names = _queue_names()
    free: Dict[str, List[str]] = {n: [] for n in names}
    q_pat = '|'.join(re.escape(n) for n in names)
    pat = re.compile(rf'^(({q_pat})@\S+)\s+\S+\s+\d+/(\d+)/\d+', re.MULTILINE)
    try:
        r = subprocess.run(['qstat', '-f', '-u', '*'],
                           capture_output=True, text=True, timeout=15)
        for m in pat.finditer(r.stdout):
            qi, qname, used = m.group(1), m.group(2), int(m.group(3))
            if used == 0:
                free[qname].append(qi)
    except Exception:
        pass
    return free


def pick_free_instance(
    free_by_queue: Dict[str, List[str]],
    prefer_queue: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """空きキューインスタンスを 1 つ選んで (qname, qi) を返す。"""
    names = _queue_names()
    order = names if not prefer_queue else \
            [prefer_queue] + [q for q in names if q != prefer_queue]
    for qname in order:
        if free_by_queue.get(qname):
            return qname, free_by_queue[qname][0]
    return None


def wait_for_free_node(
    prefer_queue: Optional[str] = None,
    is_test: bool = False,
    test_index: int = 0,
) -> Tuple[str, str, int]:
    """
    空きノードが見つかるまで待機し (qname, queue_instance, nproc_actual) を返す。

    nproc_actual = nproc - nproc_reserve  (ドライバに渡す --num-nodes に使う)
    is_test=True の場合は qstat を呼ばずダミー値を返す。
    """
    cfg = load_env()
    spec = _queue_spec()
    names = _queue_names()
    max_conc = cfg['max_concurrent_jobs']
    poll = cfg['poll_interval']
    reserve = cfg['nproc_reserve']

    if is_test:
        qname = names[test_index % len(names)]
        nproc = spec[qname]['nproc']
        return qname, f"{qname}@node{test_index + 1:02d}", nproc - reserve

    while True:
        n_jobs = get_my_job_count()
        if n_jobs < max_conc:
            free = get_free_queue_instances()
            pick = pick_free_instance(free, prefer_queue)
            if pick:
                qname, qi = pick
                nproc = spec[qname]['nproc']
                return qname, qi, nproc - reserve
            print(f"  空きノードなし。{poll}秒後に再確認...")
        else:
            print(f"  実行中ジョブ数={n_jobs}（上限{max_conc}）。{poll}秒後に再確認...")
        time.sleep(poll)


# ── ジョブスクリプト生成 ─────────────────────────────────────────────────

def make_job_script(
    cmd: str,
    queue: str,
    queue_instance: Optional[str] = None,
    job_name: Optional[str] = None,
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> str:
    """SGE ジョブスクリプトの文字列を生成する。"""
    spec = _queue_spec()
    q = spec[queue]
    nproc = q['nproc']
    pe = q.get('pe', 'OpenMP')
    q_directive = queue_instance if queue_instance else queue

    lines = ['#!/bin/sh\n', '#$ -S /bin/sh\n', '#$ -cwd\n', '#$ -V\n']
    if job_name:
        lines.append(f'#$ -N {job_name}\n')
    lines.append(f'#$ -q {q_directive}\n')
    lines.append(f'#$ -pe {pe} {nproc}\n')
    if stdout:
        lines.append(f'#$ -o {stdout}\n')
    if stderr:
        lines.append(f'#$ -e {stderr}\n')
    lines += ['\n', 'hostname\n', '\n', cmd + '\n', '\n']
    return ''.join(lines)


# ── 分割ジョブの入力フィンガープリント ─────────────────────────────────

def invalidate_stale_splits(
    auto_dir: Path,
    df_init: "object",
    fingerprint_name: str,
    split_glob: str,
) -> None:
    """同じ auto_dir を異なるパラメータで再実行した際に、
    古い split_*/eval_split_* ディレクトリを誤って再利用しないようにする。

    df_init の内容（status/E など処理中に変わる列を除く）からハッシュを計算し、
    前回実行時の値と比較する。異なれば（＝run_config.yaml のパラメータが変わった
    等）、対象の分割ディレクトリを削除してから新しいハッシュを保存する。
    同じであれば何もしない（＝既存の分割ジョブの再利用・resumeは維持される）。
    """
    import hashlib
    import shutil

    key_cols = [c for c in df_init.columns if c not in ('status', 'E')]
    payload = df_init[key_cols].round(6).to_csv(index=False).encode('utf-8')
    new_hash = hashlib.sha256(payload).hexdigest()

    fp_path = Path(auto_dir) / fingerprint_name
    if fp_path.exists() and fp_path.read_text().strip() == new_hash:
        return

    if fp_path.exists():
        removed = 0
        for old_dir in Path(auto_dir).glob(split_glob):
            shutil.rmtree(old_dir, ignore_errors=True)
            removed += 1
        if removed:
            print(
                f"  [警告] 入力パラメータが前回実行と異なるため、"
                f"古い分割ディレクトリを{removed}個削除しました（{split_glob}）"
            )

    fp_path.write_text(new_hash)
