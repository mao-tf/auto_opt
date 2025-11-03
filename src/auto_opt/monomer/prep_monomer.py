# path: src/auto_opt/monomer/prep_monomer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESP .dat を書かせる
python -m auto_opt.monomer.prep_monomer --in data/monomer/PFA.xyz --monomer PFA --mode resp --submit
.dat から RESP mol2 を生成（& Amber基準Eも作るなら --amber-ref)
python -m auto_opt.monomer.prep_monomer --in data/monomer/PFA.xyz --monomer PFA --mode resp --finalize --amber-ref
"""

from __future__ import annotations
import os, shlex, subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from auto_opt.utils import vdw_radius
import shutil

# Perl 汚染を除去するため、サブプロセスに渡す環境から特定の変数を削除
_CLEAN_PERL_VARS = ["PERL5OPT","PERL5LIB","PERL_LOCAL_LIB_ROOT","PERL_MB_OPT","PERL_MM_OPT"]


# ---------------- config / paths ----------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                 # ~/Working/auto_opt を想定
DATA = ROOT / "data"
MONO_DIR = DATA / "monomer"
AMBER_REF = DATA / "amber_ref"
RES = ROOT / "src" / "auto_opt" / "amber" / "resources"  # FF_calc.in など

GAUSS_QUEUE_DEFAULT = "gr1.q"
GAUSS_NPROC_DEFAULT = 40

def _clean_env():
    env = os.environ.copy()
    for k in _CLEAN_PERL_VARS:
        env.pop(k, None)
    return env

# ---------------- shell helpers -----------------
def run(cmd: str | List[str], cwd: Optional[Path]=None, check: bool=True) -> int:
    msg = cmd if isinstance(cmd, str) else " ".join(map(shlex.quote, cmd))
    print(f"[cmd] {msg}")
    r = subprocess.run(cmd if isinstance(cmd, list) else shlex.split(cmd),
                       cwd=str(cwd) if cwd else None,
                       env=_clean_env())
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")
    return r.returncode

def which(x: str) -> bool:
    # bash を起動せず、純粋に PATH から探す
    return shutil.which(x) is not None
    
# ---------------- io: XYZ/CSV -------------------
def read_xyz(xyz_path: Path) -> List[Tuple[str,float,float,float]]:
    rows: List[Tuple[str,float,float,float]] = []
    with open(xyz_path, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if len(t) == 4:
                try:
                    el, x, y, z = t[0], float(t[1]), float(t[2]), float(t[3])
                except ValueError:
                    continue
                rows.append((el, x, y, z))
    if not rows:
        raise ValueError(f"no atoms parsed from {xyz_path}")
    return rows

def write_csv_from_xyz(xyz_path: Path, out_csv: Path) -> None:
    rows = read_xyz(xyz_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("X,Y,Z,R\n")
        for (el, x, y, z) in rows:
            R = vdw_radius(el)
            f.write(f"{x:.6f},{y:.6f},{z:.6f},{R:.2f}\n")
    print(f"[csv] wrote {out_csv} (n={len(rows)})")

# ------------- Gaussian (ESP .dat) -------------
def write_gaussian_inp_for_gesp(xyz_path: Path, monomer: str, out_inp: Path,
                                level: str="HF/6-31G(d)",  # ユーザ指定例に合わせる
                                charge:int=0, mult:int=1, nproc:int=GAUSS_NPROC_DEFAULT) -> Path:
    """
    Pop=MK IOp(6/50=1) を用いて、ESP格子点を <base>.dat に出力する Gaussian 入力を作成。
    座標ブロックの直後に 1 行、ファイル名を記述する形式。
    """
    atoms = read_xyz(xyz_path)
    base = out_inp.with_suffix("")       # e.g., PFA_HF_esp
    esp_dat = base.with_suffix(".dat")   # e.g., PFA_HF_esp.dat

    lines: List[str] = []
    lines += [f"%mem=15GB\n", f"%nproc={nproc}\n"]
    # 例に合わせて SCF=Tight / Pop=MK / IOp(6/50=1) を指定
    lines += [f"#P {level} SCF=Tight Pop=MK IOp(6/50=1)\n\n"]
    lines += [f"{monomer}\n\n", f"{charge} {mult}\n"]
    for el,x,y,z in atoms:
        lines += [f"{el:2s}  {x: .6E}  {y: .6E}  {z: .6E}\n"]
    lines += ["\n", esp_dat.name, "\n"]  # 空行の後、出力ファイル名 1 行
    out_inp.write_text("".join(lines), encoding="utf-8")
    print(f"[gaussian] wrote {out_inp}  (ESP -> {esp_dat.name})")
    return esp_dat

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

# ------------- antechamber (RESP via GESP) -------------
def resp_mol2_from_gesp(gesp_dat: Path, xyz_path: Path, out_mol2: Path, residue: str="MOL") -> None:
    """
    ESP .dat (Gaussian 'gesp' フォーマット) を使って RESP フィットを実行し、mol2 を出力。
    antechamber に -fi gesp -c resp を直接渡す。
    注意: antechamber は構造情報を内部で必要とするので、-fi xyz で構造も渡す方法を取る。
    ここでは -i <gesp> -fi gesp に加え、-ic <xyz> -fc xyz を使う実装にする。
    （AmberTools の版により -ic/-fc が無い場合は、作業ディレクトリに XYZ を置く運用でもOK）
    """
    if not which("antechamber"):
        raise SystemExit("antechamber が見つからない（AmberTools をロードして）")
    out_mol2.parent.mkdir(parents=True, exist_ok=True)

    # antechamber のバージョン差を吸収：-ic/-fc が通らない場合は、フォールバックで -j 5 等の既定に任せる
    cmd_try = [
        f"antechamber -i {gesp_dat} -fi gesp -o {out_mol2} -fo mol2 -c resp -rn {residue} -s 2 -ic {xyz_path} -fc xyz",
        f"antechamber -i {gesp_dat} -fi gesp -o {out_mol2} -fo mol2 -c resp -rn {residue} -s 2"
    ]
    last_err = None
    for c in cmd_try:
        try:
            run(c)
            print(f"[mol2] wrote {out_mol2} via: {c}")
            return
        except Exception as e:
            last_err = e
            continue
    raise SystemExit(f"antechamber(resp from gesp) failed: {last_err}")

# ------------- Amber single-point energy -------------
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

    run(f"parmchk2 -s gaff2 -i {mol2_path.name} -f mol2 -o {out_prefix.name}.frcmod",
        cwd=mol2_path.parent, check=False)
    run(f"tleap -f {tleap_in.name}", cwd=mol2_path.parent)
    return prmtop, inpcrd

def ensure_ff_calc_in(dst_dir: Path) -> Path:
    dst = dst_dir / "FF_calc.in"
    if dst.exists():
        return dst
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

# ---------------- args -------------------------
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

def parse_args() -> Args:
    import argparse
    ap = argparse.ArgumentParser(description="XYZ → (RESP via GESP / BCC) mol2, CSV, Amber monomer energy")
    ap.add_argument("--in", dest="xyz", required=True, help="input XYZ (Element x y z)")
    ap.add_argument("--monomer", required=True, help="monomer name, e.g., PFA")
    ap.add_argument("--mode", choices=["resp","bcc"], default="resp", help="charge model")
    ap.add_argument("--out-mol2", default=None, help="output mol2 path (default: data/monomer/<monomer>_HF_esp.mol2 or _bcc.mol2)")
    ap.add_argument("--out-csv",  default=None, help="output CSV path (default: data/monomer/<monomer>.csv)")
    ap.add_argument("--amber-ref", action="store_true", help="also write data/amber_ref/<monomer>_..._{p,t}.out")
    ap.add_argument("--submit", action="store_true", help="(RESP) write & qsub Gaussian (ESP .dat), then exit")
    ap.add_argument("--finalize", action="store_true", help="(RESP) use GESP .dat to build mol2")
    ap.add_argument("--queue", default=GAUSS_QUEUE_DEFAULT)
    ap.add_argument("--nproc", type=int, default=GAUSS_NPROC_DEFAULT)
    ap.add_argument("--level", default="HF/6-31G(d)", help="Gaussian level for ESP .dat")
    a = ap.parse_args()

    xyz = Path(os.path.expanduser(a.xyz)).resolve()
    mon = a.monomer
    out_mol2 = Path(os.path.expanduser(a.out_mol2)).resolve() if a.out_mol2 else (
        MONO_DIR / (f"{mon}_HF_esp.mol2" if a.mode=="resp" else f"{mon}_bcc.mol2")
    )
    out_csv = Path(os.path.expanduser(a.out_csv)).resolve() if a.out_csv else (MONO_DIR / f"{mon}.csv")
    return Args(xyz=xyz, monomer=mon, mode=a.mode, out_mol2=out_mol2, out_csv=out_csv,
                make_amber_ref=a.amber_ref, submit=a.submit, finalize=a.finalize,
                queue=a.queue, nproc=a.nproc, level=a.level)

# ---------------- main -------------------------
def main():
    args = parse_args()
    MONO_DIR.mkdir(parents=True, exist_ok=True)
    AMBER_REF.mkdir(parents=True, exist_ok=True)

    # 1) CSV は即作る
    write_csv_from_xyz(args.xyz, args.out_csv)

    if args.mode == "bcc":
        # ---- BCC: そのまま antechamber で mol2 ----
        if not which("antechamber"):
            raise SystemExit("antechamber が見つからない（AmberTools をロードして）")
        run(f"antechamber -i {args.xyz} -fi xyz -o {args.out_mol2} -fo mol2 -c bcc -rn {args.monomer} -s 2")
        print(f"[mol2] wrote {args.out_mol2}")

    else:
        # ---- RESP via GESP: GaussianでESP.datを出して、それでRESP ----
        work = args.out_mol2.parent; work.mkdir(parents=True, exist_ok=True)
        base = work / f"{args.monomer}_HF_esp"
        inp = base.with_suffix(".inp")
        gesp_dat = write_gaussian_inp_for_gesp(args.xyz, args.monomer, inp, level=args.level,
                                               charge=0, mult=1, nproc=args.nproc)
        r1 = write_qsub_r1(inp, nproc=args.nproc, queue=args.queue)
        # 追跡用に XYZ を横に置く
        (work / f"{args.monomer}.xyz").write_text(Path(args.xyz).read_text())

        if args.submit:
            run(["qsub", r1.name], cwd=work)
            print("[resp-gesp] submitted. 収束後に --finalize で mol2 を作る。")
            return

        if args.finalize:
            if not gesp_dat.exists():
                # .dat ファイル名が変わっている・場所が異なるケースのため .dat を探索
                cand = list(work.glob(f"{args.monomer}_HF_esp*.dat"))
                if not cand:
                    raise SystemExit(f"GESP .dat が見つからない: {gesp_dat}  （work={work}）")
                gesp_dat = cand[0]
            resp_mol2_from_gesp(gesp_dat, work / f"{args.monomer}.xyz", args.out_mol2, residue=args.monomer)
        else:
            print("[resp-gesp] --submit か --finalize を指定して。何もしない。")
            return

    # 3) Amber 単分子エネルギー（任意）
    if args.make_amber_ref:
        if not which("tleap") or not which("sander"):
            raise SystemExit("tleap/sander が見つからない（AmberTools をロードして）")
        wd = args.out_mol2.parent
        tag = "HF_esp" if args.mode=="resp" else "bcc"
        base2 = f"{args.monomer}_{tag}_gaff2"
        prmtop, inpcrd = write_tleap_and_run(args.out_mol2, wd / base2, frcmods=[])
        tmp_out = run_sander_energy(wd, base2)
        out_p = AMBER_REF / f"{args.monomer.lower()}_{tag}_gaff2_p.out"
        out_t = AMBER_REF / f"{args.monomer.lower()}_{tag}_gaff2_t.out"
        out_p.write_text(tmp_out.read_text(), encoding="utf-8")
        out_t.write_text(tmp_out.read_text(), encoding="utf-8")
        print(f"[amber_ref] wrote {out_p}")
        print(f"[amber_ref] wrote {out_t}")

if __name__ == "__main__":
    main()
