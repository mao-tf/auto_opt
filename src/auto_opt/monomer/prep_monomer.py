# path: src/auto_opt/monomer/prep_monomer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, shlex, subprocess, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from auto_opt.utils import vdw_radius

# --- config / paths ----------------------------------------------------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                  # プロジェクトルート（~/Working/auto_opt 想定）
DATA = ROOT / "data"
MONO_DIR = DATA / "monomer"
AMBER_REF = DATA / "amber_ref"
RES = ROOT / "src" / "auto_opt" / "amber" / "resources"  # FF_calc.in, frcmod があれば使う

# --- shell helpers -----------------------------------------------------------
def run(cmd: str | List[str], cwd: Optional[Path]=None, check: bool=True) -> int:
    print(f"[cmd] {cmd if isinstance(cmd,str) else ' '.join(map(shlex.quote,cmd))}")
    r = subprocess.run(cmd if isinstance(cmd,list) else shlex.split(cmd), cwd=str(cwd) if cwd else None)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return r.returncode

def which(x: str) -> bool:
    return subprocess.call(['bash','-lc', f'type {shlex.quote(x)} >/dev/null 2>&1']) == 0

# --- io ----------------------------------------------------------------------
def read_xyz(xyz_path: Path) -> List[Tuple[str,float,float,float]]:
    rows = []
    with open(xyz_path) as f:
        for line in f:
            s = line.split()
            if len(s) == 4:
                try:
                    el, x, y, z = s[0], float(s[1]), float(s[2]), float(s[3])
                except ValueError:
                    continue
                rows.append((el, x, y, z))
    if not rows:
        raise ValueError(f"no atoms parsed from {xyz_path}")
    return rows

def write_csv_from_xyz(xyz_path: Path, out_csv: Path) -> None:
    rows = read_xyz(xyz_path)
    df = pd.DataFrame([{"X":x, "Y":y, "Z":z, "R": vdw_radius(el)} for (el,x,y,z) in rows])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[csv] wrote {out_csv} (n={len(df)})")

# --- Gaussian (RESP) ---------------------------------------------------------
GAUSS_QUEUE_DEFAULT = "gr1.q"
GAUSS_NPROC_DEFAULT = 40

def write_gaussian_inp_from_xyz(xyz_path: Path, monomer: str, out_inp: Path,
                                level: str="HF/6-31G*", charge:int=0, mult:int=1) -> None:
    atoms = read_xyz(xyz_path)
    lines = []
    lines += [f"%mem=15GB\n", f"%nproc={GAUSS_NPROC_DEFAULT}\n"]
    # 最適化＋ESP（Merz–Kollman）を一発で
    lines += ["#P " + level + " Opt Pop=MK IOp(6/33=2,6/42=6)\n\n"]
    lines += [f"{monomer} opt+esp\n\n", f"{charge} {mult}\n"]
    for el,x,y,z in atoms:
        lines += [f"{el:2s} {x:.6f} {y:.6f} {z:.6f}\n"]
    lines += ["\n"]
    out_inp.write_text("".join(lines), encoding="utf-8")
    print(f"[gaussian] wrote {out_inp}")

def write_qsub_r1(inp_path: Path, nproc:int, queue:str) -> Path:
    base = inp_path.with_suffix("")
    r1 = base.with_suffix(".r1")
    name = base.name
    r1.write_text("\n".join([
        "#!/bin/sh",
        "#$ -S /bin/sh",
        "#$ -cwd",
        "#$ -V",
        f"#$ -q {queue}",
        f"#$ -pe OpenMP {nproc}",
        f"#$ -N {name}",
        "hostname",
        "export g16root=/home/g03",
        "source $g16root/g16/bsd/g16.profile",
        "export GAUSS_SCRDIR=/scr/$JOB_ID",
        "mkdir -p /scr/$JOB_ID",
        f"g16 < {name}.inp > {name}.log",
        "rm -rf /scr/$JOB_ID",
        ""
    ]), encoding="utf-8")
    r1.chmod(0o755)
    print(f"[qsub] wrote {r1}")
    return r1

def resp_make_mol2_from_log(log_path: Path, out_mol2: Path, residue: str="MOL") -> None:
    # antechamber で二段 RESP（バージョンにより eq オプションの挙動が違うので try→fallback）
    out_mol2.parent.mkdir(parents=True, exist_ok=True)
    tried = []
    try:
        cmd = f"antechamber -i {log_path} -fi gout -o {out_mol2} -fo mol2 -c resp -eq 2 -rn {residue} -s 2"
        tried.append(cmd); run(cmd)
    except Exception:
        # fallback: BCC
        cmd2 = f"antechamber -i {log_path} -fi gout -o {out_mol2} -fo mol2 -c bcc -rn {residue} -s 2"
        tried.append(cmd2); run(cmd2)
    print(f"[mol2] wrote {out_mol2} via: {'  ||  '.join(tried)}")

def bcc_make_mol2_from_xyz(xyz_path: Path, out_mol2: Path, residue: str="MOL", net_charge:int=0) -> None:
    out_mol2.parent.mkdir(parents=True, exist_ok=True)
    run(f"antechamber -i {xyz_path} -fi xyz -o {out_mol2} -fo mol2 -c bcc -nc {net_charge} -rn {residue} -s 2")
    print(f"[mol2] wrote {out_mol2}")

# --- Amber single-point energy ----------------------------------------------
def write_tleap_and_run(mol2_path: Path, out_prefix: Path, frcmods: List[Path]) -> Tuple[Path,Path]:
    tleap_in = out_prefix.with_suffix(".tleap.in")
    prmtop = out_prefix.with_suffix(".prmtop")
    inpcrd = out_prefix.with_suffix(".inpcrd")
    lines = ["source leaprc.gaff2\n", f"MOL = loadmol2 {mol2_path.name}\n"]
    for frc in frcmods:
        if frc.exists():
            lines += [f"loadamberparams {frc.name}\n"]
    lines += [f"saveamberparm MOL {prmtop.name} {inpcrd.name}\n", "quit\n"]
    tleap_in.write_text("".join(lines), encoding="utf-8")

    # parmchk2 for missing types (not strictly required if frcmod provided)
    run(f"parmchk2 -s gaff2 -i {mol2_path.name} -f mol2 -o {out_prefix.name}.frcmod", cwd=mol2_path.parent, check=False)

    # run tleap
    run(f"tleap -f {tleap_in.name}", cwd=mol2_path.parent)
    return prmtop, inpcrd

def ensure_ff_calc_in(dst_dir: Path) -> Path:
    dst = dst_dir / "FF_calc.in"
    if dst.exists():
        return dst
    # resources にあればコピー、なければ最小入力を作る
    src = RES / "FF_calc.in"
    if src.exists():
        dst.write_text(src.read_text(), encoding="utf-8")
    else:
        dst.write_text("\n".join([
            "Single-point",
            " &cntrl",
            "  imin=1, maxcyc=1, ntb=0, igb=0, cut=999.0, ntpr=1,",
            " /",
            ""
        ]), encoding="utf-8")
    return dst

def run_sander_energy(workdir: Path, base: str) -> Path:
    prmtop = workdir / f"{base}.prmtop"
    inpcrd = workdir / f"{base}.inpcrd"
    out = workdir / f"{base}.out"
    ff = ensure_ff_calc_in(workdir)
    run(f"sander -O -i {ff.name} -o {out.name} -p {prmtop.name} -c {inpcrd.name} -r min.rst -ref {inpcrd.name}", cwd=workdir)
    print(f"[amber] energy -> {out}")
    return out

# --- main pipeline -----------------------------------------------------------
@dataclass
class Args:
    xyz: Path
    monomer: str
    mode: str              # "resp" or "bcc"
    out_mol2: Optional[Path]
    out_csv: Optional[Path]
    make_amber_ref: bool
    submit: bool
    finalize: bool
    queue: str
    nproc: int
    level: str
    net_charge: int

def parse_args() -> Args:
    import argparse
    ap = argparse.ArgumentParser(description="XYZ → mol2, CSV, Amber monomer energy (.out)")
    ap.add_argument("--in", dest="xyz", required=True, help="input XYZ (Element x y z)")
    ap.add_argument("--monomer", required=True, help="monomer name, e.g., PFA")
    ap.add_argument("--mode", choices=["resp","bcc"], default="resp", help="charge model")
    ap.add_argument("--out-mol2", default=None, help="output mol2 path (default: data/monomer/<monomer>_HF_esp.mol2 or _bcc.mol2)")
    ap.add_argument("--out-csv",  default=None, help="output CSV path (default: data/monomer/<monomer>.csv)")
    ap.add_argument("--amber-ref", action="store_true", help="also write data/amber_ref/<monomer>_HF_esp_gaff2_{p,t}.out")
    ap.add_argument("--submit", action="store_true", help="(RESP) write & qsub Gaussian, then exit")
    ap.add_argument("--finalize", action="store_true", help="(RESP) Gaussian .log 完了後に mol2 生成")
    ap.add_argument("--queue", default=GAUSS_QUEUE_DEFAULT)
    ap.add_argument("--nproc", type=int, default=GAUSS_NPROC_DEFAULT)
    ap.add_argument("--level", default="HF/6-31G*", help="Gaussian level (RESP)")
    ap.add_argument("--net-charge", type=int, default=0, help="total charge for BCC/RESP")
    a = ap.parse_args()

    xyz = Path(os.path.expanduser(a.xyz)).resolve()
    mon = a.monomer
    out_mol2 = Path(os.path.expanduser(a.out_mol2)).resolve() if a.out_mol2 else (
        MONO_DIR / (f"{mon}_HF_esp.mol2" if a.mode=="resp" else f"{mon}_bcc.mol2")
    )
    out_csv = Path(os.path.expanduser(a.out_csv)).resolve() if a.out_csv else (MONO_DIR / f"{mon}.csv")
    return Args(xyz=xyz, monomer=mon, mode=a.mode, out_mol2=out_mol2, out_csv=out_csv,
                make_amber_ref=a.amber_ref, submit=a.submit, finalize=a.finalize,
                queue=a.queue, nproc=a.nproc, level=a.level, net_charge=a.net_charge)

def main():
    args = parse_args()
    MONO_DIR.mkdir(parents=True, exist_ok=True)
    AMBER_REF.mkdir(parents=True, exist_ok=True)

    # 1) CSVは即作る
    write_csv_from_xyz(args.xyz, args.out_csv)

    # 2) mol2
    if args.mode == "bcc":
        if not which("antechamber"):
            raise SystemExit("antechamber が見つからない（AmberTools をロードして）")
        bcc_make_mol2_from_xyz(args.xyz, args.out_mol2, residue=args.monomer, net_charge=args.net_charge)

    else:  # RESP
        work = args.out_mol2.parent; work.mkdir(parents=True, exist_ok=True)
        inp = args.out_mol2.with_suffix(".inp")
        write_gaussian_inp_from_xyz(args.xyz, args.monomer, inp, level=args.level, charge=args.net_charge, mult=1)
        r1 = write_qsub_r1(inp, nproc=args.nproc, queue=args.queue)
        # copy xyz beside for trace
        (work / f"{args.monomer}.xyz").write_text(Path(args.xyz).read_text())

        if args.submit:
            run(["qsub", r1.name], cwd=work)
            print("[resp] submitted. 収束後に --finalize で再実行して mol2 を作る。")
            return

        log = inp.with_suffix(".log")
        if args.finalize:
            if not log.exists():
                raise SystemExit(f"Gaussian .log がまだ無い: {log}")
            if not which("antechamber"):
                raise SystemExit("antechamber が見つからない（AmberTools をロードして）")
            resp_make_mol2_from_log(log, args.out_mol2, residue=args.monomer)
        else:
            print("[resp] --submit か --finalize を指定して。何もしない。")
            return

    # 3) Amber 単分子エネルギー（p/t 両方の .out を互換名で作る）
    if args.make_amber_ref:
        if not which("tleap") or not which("sander"):
            raise SystemExit("tleap/sander が見つからない（AmberTools をロードして）")
        wd = args.out_mol2.parent
        # frcmod はあれば読む
        frcmods = [RES / "epsilon_p.frcmod", RES / "epsilon_t_3.frcmod"]
        # 共通の parm を作って sander を2回（p/t名）で保存（内容は同一でOK）
        base = f"{args.monomer}_HF_esp_gaff2" if args.mode=="resp" else f"{args.monomer}_bcc_gaff2"
        prmtop, inpcrd = write_tleap_and_run(args.out_mol2, wd / base, frcmods=[])

        # p/t それぞれの .out を data/amber_ref/ に配置（互換名 lower）
        out_p = AMBER_REF / f"{args.monomer.lower()}_HF_esp_gaff2_p.out"
        out_t = AMBER_REF / f"{args.monomer.lower()}_HF_esp_gaff2_t.out"
        # sander 実行（同じ構造でOK）
        tmp_out = run_sander_energy(wd, base)
        out_p.write_text(tmp_out.read_text())
        out_t.write_text(tmp_out.read_text())
        print(f"[amber_ref] wrote {out_p}")
        print(f"[amber_ref] wrote {out_t}")  ###ここのp,tの使い分けは必要ないので削除します

if __name__ == "__main__":
    main()
