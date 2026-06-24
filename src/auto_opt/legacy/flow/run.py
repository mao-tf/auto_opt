#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow.run — Run the whole pipeline in one shot:
  sweep.py → extract_init.py → amber/driver.py → gaussian/pipeline.py → gaussian/collect.py

You can skip phases or stop at a phase with flags.
"""

from __future__ import annotations
import argparse, subprocess, sys, shutil
from pathlib import Path

def run(cmd: list[str], cwd: str|None=None, check: bool=True) -> None:
    print("[$]", " ".join(cmd), "(cwd:", (cwd or "."), ")")
    subprocess.run(cmd, check=check, cwd=cwd)

def main():
    ap = argparse.ArgumentParser(description="auto_opt end-to-end runner")
    ap.add_argument("--out-dir", required=True, help="作業ディレクトリ（例: runs/PFA_test）")
    ap.add_argument("--monomer", required=True, help="モノマー名（例: PFA）")
    ap.add_argument("--monomer-path", default=None, help="monomerのXYZ（未指定なら data/monomer/<name>.xyz を探す）")

    # vdW sweep params
    ap.add_argument("--z-max", type=float, required=True)
    ap.add_argument("--z-step", type=float, default=0.1)
    ap.add_argument("--alpha-step", type=float, default=5)
    ap.add_argument("--theta-step", type=float, default=5)
    ap.add_argument("--eps-a", type=float, default=1e-3)
    ap.add_argument("--eps-b", type=float, default=1e-2)
    ap.add_argument("--vdw-backend", choices=["auto","c6","builtin"], default="auto")

    # amber driver params
    ap.add_argument("--amber-num-nodes", type=int, default=2)
    ap.add_argument("--amber-isTest", action="store_true", help="Amberを実行せず入出力だけ作る（デバッグ用）")

    # gaussian pipeline params
    ap.add_argument("--gaussian-no-throttle", action="true", default=False,
                    help="qstatによる投機抑制を無効化（即qsub）")

    # phase control
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-extract-init", action="store_true")
    ap.add_argument("--skip-amber", action="store_true")
    ap.add_argument("--skip-gaussian", action="store_true")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--stop-after", choices=["sweep","extract-init","amber","gaussian","collect"], default=None)

    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]      # .../src/auto_opt/flow -> up2 = project root
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # monomer xyz の解決
    monomer_xyz = args.monomer_path
    if not monomer_xyz:
        candidate = root / "data" / "monomer" / f"{args.monomer}.xyz"
        if candidate.exists():
            monomer_xyz = str(candidate)
        else:
            ap.error("--monomer-path が未指定で、既定の data/monomer/*.xyz も見つかりませんでした。")
    monomer_xyz = str(Path(monomer_xyz).resolve())

    # 1) sweep
    if not args.skip_sweep:
        run([
            sys.executable, "-m", "auto_opt.vdw.sweep",
            "--monomer-path", monomer_xyz,
            "--out-dir", str(out_dir),
            "--z-max", str(args.z_max),
            "--z-step", str(args.z_step),
            "--alpha-step", str(args.alpha_step),
            "--theta-step", str(args.theta_step),
            "--eps-a", str(args.eps_a),
            "--eps-b", str(args.eps_b),
            "--backend", args.vdw_backend,
        ])
    if args.stop-after == "sweep": return

    # 2) extract_init
    vdw_csv = out_dir / f"vdW_r_contact_{Path(monomer_xyz).stem}.csv"
    if not args.skip_extract_init:
        run([
            sys.executable, "-m", "auto_opt.vdw.extract_init",
            "--vdw-csv", str(vdw_csv),
            "--out", str(out_dir / "step1_init_params.csv"),
            "--round-ab", "1",
        ])
    if args.stop-after == "extract-init": return

    # 3) amber driver（ブロッキング; driverが完了まで回す想定）
    if not args.skip_amber:
        run([
            sys.executable, "-m", "auto_opt.amber.driver",
            "--auto-dir", str(out_dir),
            "--monomer-name", args.monomer,
            "--num-nodes", str(args.amber_num_nodes),
        ] + (["--isTest"] if args.amber_isTest else []))
    if args.stop-after == "amber": return

    # 4) gaussian pipeline（抽出＋投入をまとめて）
    if not args.skip_gaussian:
        run([
            sys.executable, "-m", "auto_opt.gaussian.pipeline",
            "--auto-dir", str(out_dir),
            "--monomer", args.monomer,
        ] + (["--no-throttle"] if args.gaussian_no_throttle else []))
    if args.stop-after == "gaussian": return

    # 5) collect（Gaussianログの集計）
    if not args.skip_collect:
        # collectのCLIが "--auto-dir" "--monomer" で動く設計前提
        run([
            sys.executable, "-m", "auto_opt.gaussian.collect",
            "--auto-dir", str(out_dir),
            "--monomer", args.monomer,
        ])

if __name__ == "__main__":
    main()
