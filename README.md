# auto_opt

Automated in-layer and interlayer geometry optimization pipeline for organic semiconductor crystals.

Given a monomer XYZ file, this tool sweeps van der Waals contact geometries, optimizes lattice parameters with Amber force field, and identifies local minima for subsequent Gaussian DFT refinement.

## Supported Crystal Symmetries

| Symmetry | Description | Dimers | Energy formula |
|----------|-------------|--------|----------------|
| **glide** | Mirror + translation | a, b, t | `E = 2E1 + 2E2 + 4E3` |
| **screw** | Rotation + translation | a, b, t1, t3 | `E = 2E1 + 2E2 + 2E3 + 2E4` |

## Prerequisites

| Software | Purpose |
|----------|---------|
| Python ≥ 3.8 | Runtime |
| [Amber](https://ambermd.org/) (GAFF2, sander, tleap, antechamber) | Force field optimization |
| [Gaussian 16](https://gaussian.com/) | DFT geometry optimization & ESP charges |
| SGE job scheduler (`qsub` / `qstat`) | HPC job submission |

## Installation

```bash
git clone https://github.com/mao-tf/auto_opt.git
cd auto_opt
pip install -e .
```

## Quick Start

The recommended way is to run all steps via `run.py` with a `run_config.yaml`:

```bash
python -m auto_opt.run --config /path/to/run_config.yaml
```

See `examples/run_config.yaml` for a template. Key fields:

```yaml
monomer: DNTT
symmetry: glide
auto_dir: /home/user/work/runs/DNTT_glide

data_dir:    /home/user/work/data
monomer_xyz: /home/user/work/data/monomer/DNTT_raw.xyz

charge: 0
mult:   1

parameters:
  z:     {min: -2.0, max: 2.0, step: 0.5}
  alpha: {min: 0,    max: 90,  step: 10}
  phi:   {min: -10,  max: 10,  step: 4}

vdw_select: all   # or: a-stack / b-stack

amber:
  num_nodes: 38
```

Run a specific range of steps:

```bash
# Monomer preparation + VdW sweep only
python -m auto_opt.run --config run_config.yaml --start-from monomer --stop-after vdw

# Amber optimization only (after VdW sweep)
python -m auto_opt.run --config run_config.yaml --start-from amber --stop-after amber

# Collect local minima
python -m auto_opt.run --config run_config.yaml --start-from collect
```

## Workflow

```
monomer_raw.xyz
    │
    ▼  Step 0: Monomer preparation
    │  prep_monomer.py
    │  → {monomer}.xyz (PCA-aligned), {monomer}.mol2 (RESP charges), {monomer}_gaff2.out
    │
    ▼  Step 1: VdW contact sweep
    │  [glide]  vdw/sweep_phi.py       → step1_init_params.csv
    │  [screw]  vdw/sweep_screw_phi.py → step1_init_params.csv
    │
    ▼  Step 2: Amber in-layer optimization
    │  [glide]  amber/job_phi.py
    │  [screw]  amber/job_screw_phi.py
    │  → filtered_step1.csv
    │
    ▼  Step 3: Stacking VdW sweep
    │  stacking/sweep_stacking_vdw.py
    │  → step1_init_params.csv (cx, cy, cz)
    │
    ▼  Step 4: Amber interlayer optimization
    │  stacking/job_stacking.py
    │
    ▼  Step 5: Result collection
       stacking/merge_results.py
       → stacking_results.csv
```

## Individual Script Usage

### VdW sweep (glide)

```bash
python -m auto_opt.vdw.sweep_phi \
    --monomer-path /path/to/DNTT.xyz \
    --out-dir runs/DNTT_glide \
    --z-min -2.0 --z-max 2.0 --z-step 0.5 \
    --alpha-min 0 --alpha-max 90 --alpha-step 10 \
    --phi-min -10 --phi-max 10 --phi-step 4 \
    --select all    # or: a-stack / b-stack
```

### VdW sweep (screw)

```bash
python -m auto_opt.vdw.sweep_screw_phi \
    --monomer-path /path/to/DNTT.xyz \
    --out-dir runs/DNTT_screw \
    --z-min -2.0 --z-max 2.0 --z-step 0.5 \
    --alpha-min 0 --alpha-max 90 --alpha-step 10 \
    --beta-min -20 --beta-max 20 --beta-step 5 \
    --phi-min -10 --phi-max 10 --phi-step 4
```

### Amber optimization

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

### Gaussian DFT submission

```bash
# glide
python -m auto_opt.gaussian.pipeline_phi \
    --auto-dir runs/DNTT_glide \
    --monomer  DNTT

# screw
python -m auto_opt.gaussian.pipeline_screw_phi \
    --auto-dir runs/DNTT_screw \
    --monomer  DNTT \
    --E-threshold -10.0
```

## Visualization (Streamlit UI)

```bash
streamlit run src/auto_opt/app.py
```

Provides interactive heatmaps, 3D molecular cluster viewer, and candidate selection for stacking calculations. Run locally; HPC results are loaded via the local work directory.

## Repository Structure

```
auto_opt/
├── src/auto_opt/
│   ├── run.py          # orchestrator (recommended entry point)
│   ├── app.py          # Streamlit visualization UI
│   ├── cluster.py      # SGE job management
│   ├── utils.py        # shared utilities
│   ├── monomer/        # Step 0: monomer preparation
│   ├── vdw/            # Step 1: VdW contact sweep
│   ├── amber/          # Step 2: Amber force field optimization
│   ├── stacking/       # Steps 3-5: interlayer optimization
│   ├── gaussian/       # DFT job submission
│   └── plot/           # visualization tools
├── examples/
│   ├── run_config.yaml # run_config.yaml template
│   └── auto_opt.yaml   # ~/.auto_opt.yaml template
└── docs/
    └── spec_overview.md
```

`data/`, `runs/`, and `legacy/` are excluded from git. Place working data outside the repository.

## Parameters

| Parameter | Description |
|-----------|-------------|
| `alpha` | Molecular rotation around z-axis (°) |
| `phi` | Tilt around x-axis / long-axis inclination (°) |
| `beta` | Additional tilt around x-axis (°), screw only |
| `a`, `b` | Unit cell constants (Å) |
| `z` | T-dimer stacking offset (Å) |

## DFT Settings

- Method: PBEPBE/6-311G** with Grimme D3BJ dispersion
- BSSE correction: Counterpoise method
- Reference energy: isolated monomer (Amber GAFF2)

## License

MIT
