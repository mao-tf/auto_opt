# auto_opt

Automated in-layer geometry optimization pipeline for organic semiconductor crystals.

Given a monomer XYZ file, this tool sweeps van der Waals contact geometries, optimizes lattice parameters with Amber force field, and refines the local minima with Gaussian DFT (counterpoise-corrected interaction energy).

## Supported Crystal Symmetries

| Symmetry | Description | Dimers | Energy formula |
|----------|-------------|--------|----------------|
| **glide** | Mirror + translation (映進対称) | a, b, t | `E = 2E1 + 2E2 + 4E3` |
| **screw** | Rotation + translation (螺旋軸対称) | a, b, t1, t3 | `E = 2E1 + 2E2 + 2E3 + 2E4` |

## Prerequisites

| Software | Purpose |
|----------|---------|
| Python ≥ 3.10 | Runtime |
| [Amber](https://ambermd.org/) (GAFF2, sander, tleap, antechamber) | Force field optimization |
| [Gaussian 16](https://gaussian.com/) | DFT single-point energy |
| SGE job scheduler (`qsub` / `qstat`) | HPC job submission |

## Installation

```bash
git clone https://github.com/<your-org>/auto_opt.git
cd auto_opt
pip install -e .
```

## Workflow

```
monomer.xyz
    │
    ▼  Step 0: Monomer preparation
    │  python -m auto_opt.monomer.prep_monomer ...
    │  → monomer.csv, monomer.mol2, monomer_gaff2.frcmod
    │
    ▼  Step 1: VdW contact sweep
    │  [glide]  python -m auto_opt.vdw.sweep_phi ...
    │           → vdW_r_contact_<monomer>.csv
    │  [screw]  python -m auto_opt.vdw.sweep_screw_phi ...
    │           → step1_init_params.csv  (go to Step 3)
    │
    ▼  Step 2: Initial point extraction  [glide only]
    │  python -m auto_opt.vdw.extract_init_phi ...
    │  → step1_init_params.csv
    │
    ▼  Step 3: Amber in-layer optimization
    │  [glide]  python -m auto_opt.amber.job_phi ...
    │  [screw]  python -m auto_opt.amber.job_screw_phi ...
    │  → step1.csv  (auto-calls Step 4 on completion)
    │
    ▼  Step 4: Local minima extraction  [auto-called by Step 3]
    │  python -m auto_opt.gaussian.extract_minima --symmetry glide|screw ...
    │  → filtered_step1.csv
    │
    ▼  Step 5: Gaussian DFT single-point
       [glide]  python -m auto_opt.gaussian.pipeline_phi ...
       [screw]  python -m auto_opt.gaussian.pipeline_screw_phi ...
       → *.inp, *.log
```

## Usage Examples

### Step 1 — VdW sweep (glide)

```bash
python -m auto_opt.vdw.sweep_phi \
    --monomer-path data/monomer/DNTT.xyz \
    --out-dir runs/DNTT_glide \
    --z-min 0.0 --z-max 3.0 --z-step 0.5 \
    --alpha-min 60 --alpha-max 70 --alpha-step 5 \
    --phi-min 0 --phi-max 10 --phi-step 5 \
    --theta-step 5
```

### Step 1 — VdW sweep (screw)

```bash
python -m auto_opt.vdw.sweep_screw_phi \
    --monomer-path data/monomer/DNTT.xyz \
    --out-dir runs/DNTT_screw \
    --z-min 0.0 --z-max 3.0 --z-step 0.5 \
    --alpha-min 60 --alpha-max 70 --alpha-step 5 \
    --beta-min 0 --beta-max 10 --beta-step 5 \
    --phi-min 0 --phi-max 10 --phi-step 5 \
    --select all          # or: --select a-stack  /  --select b-stack
```

### Step 2 — Initial point extraction (glide only)

```bash
python -m auto_opt.vdw.extract_init_phi \
    --vdw-csv runs/DNTT_glide/vdW_r_contact_DNTT.csv \
    --out     runs/DNTT_glide/step1_init_params.csv \
    --select  all         # or: --select a-stack b-stack
```

### Step 3 — Amber optimization

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

Step 4 (`extract_minima`) runs automatically when Step 3 completes.

### Step 4 — Local minima extraction (manual re-run)

```bash
python -m auto_opt.gaussian.extract_minima \
    --symmetry glide \          # or: screw
    --auto-dir runs/DNTT_glide
# → runs/DNTT_glide/filtered_step1.csv
```

### Step 5 — Gaussian DFT submission

```bash
# glide
python -m auto_opt.gaussian.pipeline_phi \
    --auto-dir runs/DNTT_glide \
    --monomer  DNTT

# screw
python -m auto_opt.gaussian.pipeline_screw_phi \
    --auto-dir runs/DNTT_screw \
    --monomer  DNTT \
    --E-threshold -10.0    # optional: submit only E <= -10 kcal/mol
```

## Repository Structure

```
auto_opt/
├── data/
│   └── monomer/          # monomer.xyz, monomer.csv, monomer.mol2, *.frcmod
├── docs/
│   └── spec_overview.md  # detailed design document (Japanese)
├── runs/                 # output directory (created at runtime)
└── src/auto_opt/
    ├── monomer/          # Step 0: monomer preparation
    ├── vdw/              # Step 1-2: VdW contact sweep & initial point extraction
    ├── amber/            # Step 3: Amber force field optimization
    ├── gaussian/         # Step 4-5: local minima extraction & DFT submission
    └── utils.py          # shared utilities (rotation matrices, VdW radii, etc.)
```

## DFT Settings

- Method: PBEPBE/6-311G** with Grimme D3BJ dispersion correction
- Counterpoise correction for BSSE
- Dimer interaction energy referenced to isolated monomer (Amber GAFF2)

## License

MIT
