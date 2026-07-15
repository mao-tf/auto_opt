#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モノマー準備パイプライン: XYZ → Amber 計算用ファイル一式

ステップ 1: ジオメトリ最適化（Gaussian opt）
  python -m auto_opt.monomer.prep_monomer --in initial.xyz --monomer BTBT --mode opt --submit
  # Gaussian ジョブ終了後
  python -m auto_opt.monomer.prep_monomer --in initial.xyz --monomer BTBT --mode opt --finalize
  # → data/monomer/BTBT.xyz (PCA 整列済み)  data/monomer/BTBT.csv

ステップ 2: ESP / RESP → mol2 → Amber 基準エネルギー
  python -m auto_opt.monomer.prep_monomer --in data/monomer/BTBT.xyz --monomer BTBT --mode resp --submit
  # Gaussian ジョブ終了後
  python -m auto_opt.monomer.prep_monomer --in data/monomer/BTBT.xyz --monomer BTBT --mode resp --finalize --amber-ref

BCC を使う場合（ステップ 2 の代替）:
  python -m auto_opt.monomer.prep_monomer --in data/monomer/BTBT.xyz --monomer BTBT --mode bcc --amber-ref
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from auto_opt.utils import vdw_radius

_CLEAN_PERL_VARS = [
    "PERL5OPT", "PERL5LIB", "PERL_LOCAL_LIB_ROOT", "PERL_MB_OPT", "PERL_MM_OPT"
]

# ── パス ─────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve()
ROOT     = HERE.parents[3]
DATA     = ROOT / "data"
MONO_DIR = DATA / "monomer"
AMBER_REF_DIR = DATA / "amber_ref"
RES      = ROOT / "src" / "auto_opt" / "amber" / "resources"

# 原子番号 → 元素記号
_Z_TO_EL = {
    1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F',
    14:'Si', 15:'P', 16:'S', 17:'Cl', 35:'Br', 53:'I',
}


# ── 環境ユーティリティ ──────────────────────────────────────────────────

def _clean_env() -> dict:
    env = os.environ.copy()
    for k in _CLEAN_PERL_VARS:
        env.pop(k, None)
    return env


def _gauss_queue_nproc() -> Tuple[str, int]:
    """~/.auto_opt.yaml からキュー名と nproc を取得。なければデフォルト値。"""
    try:
        from auto_opt.cluster import load_env
        cfg = load_env()
        q = cfg['queues'][0]
        return q['name'], q['nproc']
    except Exception:
        return 'gr1.q', 40


# ── shell helper ────────────────────────────────────────────────────────

def _run(cmd: str | List[str], cwd: Optional[Path] = None, check: bool = True) -> int:
    msg = cmd if isinstance(cmd, str) else ' '.join(map(shlex.quote, cmd))
    print(f'[cmd] {msg}')
    r = subprocess.run(
        cmd if isinstance(cmd, list) else shlex.split(cmd),
        cwd=str(cwd) if cwd else None,
        env=_clean_env(),
    )
    if check and r.returncode != 0:
        raise RuntimeError(f'command failed: {cmd}')
    return r.returncode


def _which(x: str) -> bool:
    return shutil.which(x) is not None


def _amber_tool(name: str) -> str:
    """AmberTools 実行ファイルのパスを返す（cluster.py 経由 → PATH フォールバック）。"""
    try:
        from auto_opt.cluster import get_amber_tool
        return get_amber_tool(name)
    except ImportError:
        found = shutil.which(name)
        if found is None:
            raise SystemExit(
                f'{name} が見つかりません。~/.auto_opt.yaml の amber_tools に設定してください。'
            )
        return found


# ── XYZ / CSV IO ────────────────────────────────────────────────────────

def read_xyz(xyz_path: Path) -> List[Tuple[str, float, float, float]]:
    atoms = []
    with open(xyz_path, encoding='utf-8') as f:
        for line in f:
            t = line.split()
            if len(t) == 4:
                try:
                    atoms.append((t[0], float(t[1]), float(t[2]), float(t[3])))
                except ValueError:
                    pass
    if not atoms:
        raise ValueError(f'atoms not found: {xyz_path}')
    return atoms


def write_xyz(atoms: List[Tuple[str, float, float, float]],
              out_path: Path, comment: str = '') -> None:
    lines = [f'{len(atoms)}\n', f'{comment}\n']
    for el, x, y, z in atoms:
        lines.append(f'{el:2s}  {x: .6f}  {y: .6f}  {z: .6f}\n')
    out_path.write_text(''.join(lines), encoding='utf-8')
    print(f'[xyz] wrote {out_path}  (n={len(atoms)})')


def write_csv_from_xyz(xyz_path: Path, out_csv: Path) -> None:
    atoms = read_xyz(xyz_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8') as f:
        f.write('X,Y,Z,R\n')
        for el, x, y, z in atoms:
            f.write(f'{x:.6f},{y:.6f},{z:.6f},{vdw_radius(el):.2f}\n')
    print(f'[csv] wrote {out_csv}  (n={len(atoms)})')


# ── PCA 整列 ─────────────────────────────────────────────────────────────

def pca_align(atoms: List[Tuple[str, float, float, float]]
              ) -> List[Tuple[str, float, float, float]]:
    """
    PCA で分子を標準配向に整列する。
      PC1 (最大分散 = 長軸) → z 軸
      PC2 (第二分散)        → y 軸
      PC3 (最小分散 = 法線) → x 軸
    """
    coords = np.array([[x, y, z] for _, x, y, z in atoms])
    center = coords.mean(axis=0)
    centered = coords - center

    # SVD: centered = U S Vt  →  Vt[i] が PC_i の方向
    _, _, Vt = np.linalg.svd(centered, full_matrices=True)

    # 回転行列: 新 x = PC3, 新 y = PC2, 新 z = PC1
    R = np.array([Vt[2], Vt[1], Vt[0]])

    # 右手系を保証
    if np.linalg.det(R) < 0:
        R[0] = -R[0]

    new_coords = centered @ R.T
    return [(el, float(c[0]), float(c[1]), float(c[2]))
            for (el, _, _, _), c in zip(atoms, new_coords)]


# ── Gaussian ジオメトリ最適化 ────────────────────────────────────────────

def write_gaussian_inp_for_opt(
    xyz_path: Path, monomer: str, out_inp: Path,
    level: str = 'B3LYP/6-31G(d) SCF=Tight',
    charge: int = 0, mult: int = 1,
    nproc: int = 40,
) -> Path:
    """Gaussian ジオメトリ最適化インプットを生成する。"""
    atoms = read_xyz(xyz_path)
    lines = [f'%mem=15GB\n', f'%nproc={nproc}\n']
    lines += [f'#P {level} Opt\n\n']
    lines += [f'{monomer} geometry optimization\n\n']
    lines += [f'{charge} {mult}\n']
    for el, x, y, z in atoms:
        lines.append(f'{el:2s}  {x: .6E}  {y: .6E}  {z: .6E}\n')
    lines.append('\n')
    out_inp.write_text(''.join(lines), encoding='utf-8')
    print(f'[gaussian] wrote {out_inp}')
    return out_inp


def read_opt_xyz_from_log(log_path: Path) -> List[Tuple[str, float, float, float]]:
    """
    Gaussian 最適化ログから最終 "Standard orientation:" ブロックを読む。
    """
    atoms_last: List[Tuple[str, float, float, float]] = []

    with open(log_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if 'Standard orientation:' in lines[i]:
            i += 5  # ヘッダー行（破線×2 + 列名×2）をスキップ
            cur: List[Tuple[str, float, float, float]] = []
            while i < len(lines) and '---' not in lines[i]:
                parts = lines[i].split()
                if len(parts) == 6:
                    try:
                        z_num = int(parts[1])
                        xc, yc, zc = float(parts[3]), float(parts[4]), float(parts[5])
                        el = _Z_TO_EL.get(z_num, f'Z{z_num}')
                        cur.append((el, xc, yc, zc))
                    except ValueError:
                        pass
                i += 1
            if cur:
                atoms_last = cur
        i += 1

    if not atoms_last:
        raise ValueError(f'Standard orientation が見つかりません: {log_path}')
    return atoms_last


# ── Gaussian ESP 計算 ────────────────────────────────────────────────────

def write_gaussian_inp_for_gesp(
    xyz_path: Path, monomer: str, out_inp: Path,
    level: str = 'HF/6-31G(d)',
    charge: int = 0, mult: int = 1,
    nproc: int = 40,
) -> Path:
    atoms = read_xyz(xyz_path)
    base = out_inp.with_suffix('')
    esp_dat = base.with_suffix('.dat')
    lines = [f'%mem=15GB\n', f'%nproc={nproc}\n']
    lines += [f'#P {level} SCF=Tight Pop=MK IOp(6/50=1)\n\n']
    lines += [f'{monomer}\n\n', f'{charge} {mult}\n']
    for el, x, y, z in atoms:
        lines.append(f'{el:2s}  {x: .6E}  {y: .6E}  {z: .6E}\n')
    lines += ['\n', esp_dat.name, '\n']
    out_inp.write_text(''.join(lines), encoding='utf-8')
    print(f'[gaussian] wrote {out_inp}  (ESP -> {esp_dat.name})')
    return esp_dat


# ── qsub スクリプト生成 ──────────────────────────────────────────────────

def write_gaussian_qsub(inp_path: Path, nproc: int, queue: str) -> Path:
    """Gaussian 用 qsub スクリプトを生成する。"""
    base = inp_path.with_suffix('')
    r1 = base.with_suffix('.r1')
    name = base.name
    r1.write_text('\n'.join([
        '#!/bin/sh',
        '#$ -S /bin/sh',
        '#$ -cwd',
        '#$ -V',
        f'#$ -q {queue}',
        f'#$ -pe OpenMP {nproc}',
        f'#$ -N {name}',
        'hostname',
        'export g16root=/home/g03',
        'source $g16root/g16/bsd/g16.profile',
        'export GAUSS_SCRDIR=/scr/$JOB_ID',
        'mkdir -p /scr/$JOB_ID',
        f'g16 < {name}.inp > {name}.log',
        'rm -rf /scr/$JOB_ID',
        '',
    ]), encoding='utf-8')
    r1.chmod(0o755)
    print(f'[qsub] wrote {r1}')
    return r1


# ── Gaussian ジョブ完了待ち ──────────────────────────────────────────────

def _wait_for_normal_termination(log_path: Path, poll_interval: int = 30) -> None:
    print(f'[poll] {log_path.name} の完了待ち ({poll_interval}s 間隔)...')
    while True:
        if log_path.exists():
            text = log_path.read_text(encoding='utf-8', errors='replace')
            if 'Normal termination' in text:
                print(f'[poll] 完了: {log_path.name}')
                return
            if 'Error termination' in text:
                raise RuntimeError(f'Gaussian が異常終了しました: {log_path}')
        time.sleep(poll_interval)


# ── antechamber (RESP) ───────────────────────────────────────────────────

def resp_mol2_from_gesp(gesp_dat: Path, xyz_path: Path,
                        out_mol2: Path, residue: str = 'MOL') -> None:
    antechamber = _amber_tool('antechamber')
    work = out_mol2.parent
    work.mkdir(parents=True, exist_ok=True)
    cmds = [
        f'{antechamber} -i {gesp_dat} -fi gesp -o {out_mol2} -fo mol2 -c resp -rn {residue} -s 2 -ic {xyz_path} -fc xyz',
        f'{antechamber} -i {gesp_dat} -fi gesp -o {out_mol2} -fo mol2 -c resp -rn {residue} -s 2',
    ]
    for c in cmds:
        try:
            _run(c, cwd=work)
            print(f'[mol2] wrote {out_mol2}')
            return
        except Exception:
            continue
    raise SystemExit('antechamber(RESP) failed')


# ── tleap / sander ───────────────────────────────────────────────────────

def write_tleap_and_run(mol2_path: Path, out_prefix: Path,
                        frcmods: List[Path]) -> Tuple[Path, Path]:
    tleap_in = out_prefix.with_suffix('.tleap.in')
    prmtop   = out_prefix.with_suffix('.prmtop')
    inpcrd   = out_prefix.with_suffix('.inpcrd')
    frcmod   = f'{out_prefix.name}.frcmod'

    lines = ['source leaprc.gaff2\n', f'MOL = loadmol2 {mol2_path.name}\n']
    lines += [f'loadamberparams {frcmod}\n']
    for frc in frcmods:
        if frc.exists():
            lines += [f'loadamberparams {frc.name}\n']
    lines += [f'saveamberparm MOL {prmtop.name} {inpcrd.name}\n', 'quit\n']
    tleap_in.write_text(''.join(lines), encoding='utf-8')

    _run(f'{_amber_tool("parmchk2")} -s gaff2 -i {mol2_path.name} -f mol2 -o {frcmod}',
         cwd=mol2_path.parent, check=False)
    _run(f'{_amber_tool("tleap")} -f {tleap_in.name}', cwd=mol2_path.parent)
    return prmtop, inpcrd


def _ensure_ff_calc_in(dst_dir: Path) -> Path:
    dst = dst_dir / 'FF_calc.in'
    if dst.exists():
        return dst
    src = RES / 'FF_calc.in'
    if src.exists():
        dst.write_text(src.read_text(), encoding='utf-8')
    else:
        dst.write_text(
            'Single-point\n &cntrl\n  imin=1, maxcyc=0, ntb=0, igb=0, cut=999.0, ntpr=1,\n /\n',
            encoding='utf-8',
        )
    return dst


def run_sander_energy(workdir: Path, base: str) -> Path:
    prmtop = workdir / f'{base}.prmtop'
    inpcrd = workdir / f'{base}.inpcrd'
    out    = workdir / f'{base}.out'
    ff     = _ensure_ff_calc_in(workdir)
    _run(
        f'{_amber_tool("sander")} -O -i {ff.name} -o {out.name} -p {prmtop.name} -c {inpcrd.name}'
        f' -r min.rst -ref {inpcrd.name}',
        cwd=workdir,
    )
    print(f'[amber] energy -> {out}')
    return out


# ── 公開 API ────────────────────────────────────────────────────────────

def run_all(
    monomer: str,
    xyz_path: Path,
    out_mol2: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    make_amber_ref: bool = True,
    queue: Optional[str] = None,
    nproc: Optional[int] = None,
    opt_level: str = 'B3LYP/6-31G(d)',
    esp_level: str = 'HF/6-31G(d)',
    charge: int = 0,
    mult: int = 1,
    mono_dir: Optional[Path] = None,
    amber_ref_dir: Optional[Path] = None,
) -> None:
    """opt → resp → amber_ref を連続実行する。run.py から直接呼べる。"""
    _mono_dir      = Path(mono_dir)      if mono_dir      else MONO_DIR
    _amber_ref_dir = Path(amber_ref_dir) if amber_ref_dir else AMBER_REF_DIR

    _mono_dir.mkdir(parents=True, exist_ok=True)
    _amber_ref_dir.mkdir(parents=True, exist_ok=True)

    _q, _n = _gauss_queue_nproc()
    queue    = queue    or _q
    nproc    = nproc    or _n
    out_mol2 = out_mol2 or (_mono_dir / f'{monomer}.mol2')
    out_csv  = out_csv  or (_mono_dir / f'{monomer}.csv')

    work = _mono_dir
    # --- opt ---
    inp_opt = work / f'{monomer}_opt.inp'
    log_opt = work / f'{monomer}_opt.log'
    write_gaussian_inp_for_opt(
        xyz_path, monomer, inp_opt,
        level=opt_level, charge=charge, mult=mult, nproc=nproc,
    )
    r1_opt = write_gaussian_qsub(inp_opt, nproc=nproc, queue=queue)
    _run(['qsub', r1_opt.name], cwd=work)
    print('[all] opt ジョブを投入しました。完了を待ちます...')
    _wait_for_normal_termination(log_opt)
    atoms = read_opt_xyz_from_log(log_opt)
    atoms = pca_align(atoms)
    out_xyz = work / f'{monomer}.xyz'
    write_xyz(atoms, out_xyz,
              comment=f'{monomer} optimized ({opt_level}), PCA aligned')
    write_csv_from_xyz(out_xyz, out_csv)
    # --- resp ---
    base_esp = work / f'{monomer}_HF_esp'
    inp_esp  = base_esp.with_suffix('.inp')
    log_esp  = base_esp.with_suffix('.log')
    gesp_dat = write_gaussian_inp_for_gesp(
        out_xyz, monomer, inp_esp,
        level=esp_level, charge=charge, mult=mult, nproc=nproc,
    )
    r1_esp = write_gaussian_qsub(inp_esp, nproc=nproc, queue=queue)
    _run(['qsub', r1_esp.name], cwd=work)
    print('[all] resp ジョブを投入しました。完了を待ちます...')
    _wait_for_normal_termination(log_esp)
    if not gesp_dat.exists():
        cands = list(work.glob(f'{monomer}_HF_esp*.dat'))
        if not cands:
            raise SystemExit(f'GESP .dat が見つかりません: {gesp_dat}')
        gesp_dat = cands[0]
    resp_mol2_from_gesp(gesp_dat, out_xyz, out_mol2, residue=monomer)
    # --- amber_ref ---
    if make_amber_ref:
        wd    = out_mol2.parent
        base2 = f'{monomer}_gaff2'
        write_tleap_and_run(out_mol2, wd / base2, frcmods=[])
        tmp_out    = run_sander_energy(wd, base2)
        single_out = _amber_ref_dir / f'{monomer}_gaff2.out'
        single_out.write_text(tmp_out.read_text(), encoding='utf-8')
        print(f'[amber_ref] wrote {single_out}')


# ── CLI ─────────────────────────────────────────────────────────────────

@dataclass
class Args:
    xyz: Path
    monomer: str
    mode: str          # 'opt' | 'resp' | 'bcc'
    out_mol2: Optional[Path]
    out_csv: Optional[Path]
    make_amber_ref: bool
    submit: bool
    finalize: bool
    queue: str
    nproc: int
    opt_level: str
    esp_level: str
    charge: int
    mult: int
    mono_dir: Path
    amber_ref_dir: Path


def _parse_args() -> Args:
    import argparse
    ap = argparse.ArgumentParser(
        description='XYZ → Gaussian opt → RESP/BCC mol2 → Amber パラメータ'
    )
    ap.add_argument('--in',       dest='xyz',      required=True)
    ap.add_argument('--monomer',  required=True)
    ap.add_argument('--mode',     choices=['opt', 'resp', 'bcc', 'all'], default='resp')
    ap.add_argument('--out-mol2', default=None)
    ap.add_argument('--out-csv',  default=None)
    ap.add_argument('--amber-ref', action='store_true',
                    help='Amber 単分子エネルギーも計算する')
    ap.add_argument('--submit',   action='store_true',
                    help='Gaussian ジョブを qsub して終了')
    ap.add_argument('--finalize', action='store_true',
                    help='Gaussian ログから結果を取り出す')
    ap.add_argument('--queue',    default=None,
                    help='SGE キュー名（省略時は ~/.auto_opt.yaml から取得）')
    ap.add_argument('--nproc',    type=int, default=None,
                    help='Gaussian nproc（省略時は ~/.auto_opt.yaml から取得）')
    ap.add_argument('--opt-level',  default='B3LYP/6-31G(d) SCF=Tight',
                    help='ジオメトリ最適化の計算レベル')
    ap.add_argument('--esp-level',  default='HF/6-31G(d)',
                    help='ESP 計算の計算レベル')
    ap.add_argument('--charge',   type=int, default=0)
    ap.add_argument('--mult',     type=int, default=1)
    ap.add_argument('--mono-dir', default=None,
                    help='モノマー出力先（省略時はパッケージ内 data/monomer）')
    ap.add_argument('--amber-ref-dir', default=None,
                    help='amber_ref 出力先（省略時はパッケージ内 data/amber_ref）')
    a = ap.parse_args()

    default_queue, default_nproc = _gauss_queue_nproc()
    xyz = Path(os.path.expanduser(a.xyz)).resolve()
    mon = a.monomer
    mono_dir = (Path(os.path.expanduser(a.mono_dir)).resolve() if a.mono_dir
                else MONO_DIR)
    amber_ref_dir = (Path(os.path.expanduser(a.amber_ref_dir)).resolve() if a.amber_ref_dir
                     else AMBER_REF_DIR)
    out_mol2 = (Path(os.path.expanduser(a.out_mol2)).resolve() if a.out_mol2
                else mono_dir / f'{mon}.mol2')
    out_csv  = (Path(os.path.expanduser(a.out_csv)).resolve() if a.out_csv
                else mono_dir / f'{mon}.csv')

    return Args(
        xyz=xyz, monomer=mon, mode=a.mode,
        out_mol2=out_mol2, out_csv=out_csv,
        make_amber_ref=a.amber_ref,
        submit=a.submit, finalize=a.finalize,
        queue=a.queue or default_queue,
        nproc=a.nproc or default_nproc,
        opt_level=a.opt_level,
        esp_level=a.esp_level,
        charge=a.charge, mult=a.mult,
        mono_dir=mono_dir, amber_ref_dir=amber_ref_dir,
    )


# ── main ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    args.mono_dir.mkdir(parents=True, exist_ok=True)
    args.amber_ref_dir.mkdir(parents=True, exist_ok=True)

    # ── mode: all (opt → resp → amber_ref を自動で連続実行) ──────────────
    if args.mode == 'all':
        run_all(
            monomer=args.monomer,
            xyz_path=args.xyz,
            out_mol2=args.out_mol2,
            out_csv=args.out_csv,
            make_amber_ref=args.make_amber_ref,
            queue=args.queue,
            nproc=args.nproc,
            opt_level=args.opt_level,
            esp_level=args.esp_level,
            charge=args.charge,
            mult=args.mult,
            mono_dir=args.mono_dir,
            amber_ref_dir=args.amber_ref_dir,
        )
        return

    # ── mode: opt ────────────────────────────────────────────────────────
    if args.mode == 'opt':
        work = args.mono_dir
        inp  = work / f'{args.monomer}_opt.inp'
        log  = work / f'{args.monomer}_opt.log'

        if args.submit:
            write_gaussian_inp_for_opt(
                args.xyz, args.monomer, inp,
                level=args.opt_level,
                charge=args.charge, mult=args.mult, nproc=args.nproc,
            )
            r1 = write_gaussian_qsub(inp, nproc=args.nproc, queue=args.queue)
            _run(['qsub', r1.name], cwd=work)
            print('[opt] submitted. 収束後に --finalize を実行してください。')
            return

        if args.finalize:
            if not log.exists():
                raise SystemExit(f'ログが見つかりません: {log}')
            atoms = read_opt_xyz_from_log(log)
            atoms = pca_align(atoms)
            out_xyz = args.mono_dir / f'{args.monomer}.xyz'
            write_xyz(atoms, out_xyz,
                      comment=f'{args.monomer} optimized ({args.opt_level}), PCA aligned')
            write_csv_from_xyz(out_xyz, args.out_csv)
            print(f'[opt] 完了。次のステップ:')
            print(f'  python -m auto_opt.monomer.prep_monomer '
                  f'--in {out_xyz} --monomer {args.monomer} --mode resp --submit')
            return

        print('[opt] --submit か --finalize を指定してください。')
        return

    # ── mode: bcc ────────────────────────────────────────────────────────
    if args.mode == 'bcc':
        write_csv_from_xyz(args.xyz, args.out_csv)
        if not _which('antechamber'):
            raise SystemExit('antechamber が見つかりません')
        _run(f'antechamber -i {args.xyz} -fi xyz -o {args.out_mol2} -fo mol2 '
             f'-c bcc -rn {args.monomer} -s 2')
        print(f'[mol2] wrote {args.out_mol2}')

    # ── mode: resp ───────────────────────────────────────────────────────
    else:
        write_csv_from_xyz(args.xyz, args.out_csv)
        work = args.out_mol2.parent
        work.mkdir(parents=True, exist_ok=True)
        base    = work / f'{args.monomer}_HF_esp'
        inp     = base.with_suffix('.inp')
        gesp_dat = write_gaussian_inp_for_gesp(
            args.xyz, args.monomer, inp,
            level=args.esp_level,
            charge=args.charge, mult=args.mult, nproc=args.nproc,
        )
        r1 = write_gaussian_qsub(inp, nproc=args.nproc, queue=args.queue)
        # XYZ を作業ディレクトリにコピー
        (work / f'{args.monomer}.xyz').write_text(
            Path(args.xyz).read_text(), encoding='utf-8'
        )

        if args.submit:
            _run(['qsub', r1.name], cwd=work)
            print('[resp] submitted. 収束後に --finalize を実行してください。')
            return

        if args.finalize:
            if not gesp_dat.exists():
                cands = list(work.glob(f'{args.monomer}_HF_esp*.dat'))
                if not cands:
                    raise SystemExit(f'GESP .dat が見つかりません: {gesp_dat}')
                gesp_dat = cands[0]
            resp_mol2_from_gesp(
                gesp_dat, work / f'{args.monomer}.xyz',
                args.out_mol2, residue=args.monomer,
            )
        else:
            print('[resp] --submit か --finalize を指定してください。')
            return

    # ── Amber 単分子エネルギー（resp / bcc 共通） ──────────────────────
    if args.make_amber_ref:
        wd   = args.out_mol2.parent
        base2 = f'{args.monomer}_gaff2'
        write_tleap_and_run(args.out_mol2, wd / base2, frcmods=[])
        tmp_out = run_sander_energy(wd, base2)
        single_out = args.amber_ref_dir / f'{args.monomer}_gaff2.out'
        single_out.write_text(tmp_out.read_text(), encoding='utf-8')
        print(f'[amber_ref] wrote {single_out}')


if __name__ == '__main__':
    main()
