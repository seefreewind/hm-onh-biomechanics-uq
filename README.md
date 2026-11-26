# Finite Element Modeling and Global Uncertainty Quantification of Myopic Optic Nerve Head Biomechanics

Official reproducible computational repository accompanying the manuscript:

> **Finite element modeling and global uncertainty quantification of myopic optic nerve head biomechanics for axial-length-stratified safe intraocular pressure estimation**

## Overview

This repository provides the complete, end-to-end computational pipeline for:

1. **Distortion-free parameterized 3D ONH meshing (`src/geometry/onh_mesh.py`)**: Six-domain conformal all-hexahedral (`hex8`) multi-block O-grid diffeomorphism supporting neural canal obliqueness up to 35° with automated scaled-Jacobian quality verification.
2. **FEBio 4.0 anisotropic hyperelastic solver interface (`src/febio/`)**: Automated `.feb` XML assembly implementing planar von Mises collagen fiber dispersion (`Mooney-Rivlin von Mises Fibers` in the peripapillary and posterior sclera; uncoupled Neo-Hookean / Mooney-Rivlin in the lamina cribrosa, neural tissues, and pia mater), adaptive step-size retry, and parallel campaign execution.
3. **Volume-weighted QoI extraction (`src/postprocess/qoi_extractor.py`)**: Element-volume-weighted 95th-percentile lamina cribrosa (LC) first principal tensile strain ($E_{1,95}$), effective strain ($\varepsilon_{\mathrm{eff},95}$), 3D maximum shear strain ($\gamma_{\max,95}$), and posterior bowing relative to Bruch's membrane opening ($u_{\mathrm{AP},95}$).
4. **Surrogate modeling & dual-track global UQ (`src/surrogate/`, `src/uq/`)**:
   - Anisotropic Matérn-5/2 automatic relevance determination (ARD) Gaussian process surrogates trained on 400 Maximin Latin hypercube simulations and validated on 80 unseen test simulations ($Q^2 = 0.9952\text{--}0.9968$, NRMSE $= 1.27\%\text{--}1.54\%$);
   - **Track A (Mechanistic Decoupling)**: 90,112-evaluation variance-based Sobol global sensitivity analysis ($S_1$, $S_T$, $S_{ij}$) with 1,000-resample bootstrap 95% confidence intervals;
   - **Track B (Population Remodeling & Inverse Safe IOP)**: 10,000-eye empirically correlated Gaussian Copula virtual cohort, axial-length-stratified exceedance analysis, and continuous inverse root-finding for biomechanically equivalent safe IOP boundaries $\mathrm{IOP}_{\mathrm{eq}}(\mathrm{AL})$.
5. **Publication figure & table reproduction (`src/visualization/`)**: Deterministic regeneration of Figures 1–6 (350 DPI PNG, vector PDF, editable SVG) and Tables 1–2 directly from the archived datasets.

## Physical Unit System & Reproducibility Standards

- **Consistent solver units**: `mm – N – MPa` (`1 mmHg = 0.000133322 MPa`, enforced in `src/common/units.py`).
- **Deterministic random seed**: `seed = 42` across LHS sampling, GP fitting, Sobol quasi-Monte Carlo sequences, and Gaussian Copula cohort synthesis.
- **Included precomputed outputs**: To allow immediate reproduction of surrogate validation, UQ analyses, and all publication figures without re-running 515 FEBio simulations (~1.5 GB of raw `.feb`/`.xplt` solver artifacts), this repository includes the exact LHS designs (`data/lhs_samples/`), extracted 515-case simulation results and baseline VTU fields (`data/processed/`), fitted GP surrogates and UQ tables (`data/uq_results/`), and summary tables (`results/tables/`).

## Repository Structure

```text
.
├── config/
│   ├── parameters.yaml          # 10D parameter bounds, baselines, and 10x10 Copula correlation matrix
│   └── sim_settings.yaml        # Mesh density, FEBio solver controls, and UQ settings
├── data/
│   ├── lhs_samples/             # Maximin LHS designs (20 smoke, 400 train, 80 unseen test)
│   ├── processed/               # Extracted FEBio QoI tables + 4 baseline 3D VTU fields
│   └── uq_results/              # Fitted ARD GP bundle, Sobol indices, 10,000-eye cohort, safe-IOP curves
├── results/
│   ├── figures/                 # Target output folder for regenerated Figures 1-6
│   └── tables/                  # Phase 1 verification, Phase 3 UQ, and manuscript Tables 1-2
├── src/
│   ├── common/                  # Unit system and hyperelastic modulus conversion helpers
│   ├── geometry/                # Parameterized 3D O-grid hexahedral mesh generator (Gmsh)
│   ├── febio/                   # FEBio 4.0 XML builder, runner, verification, and batch orchestration
│   ├── postprocess/             # Volume-weighted 95th-percentile QoI extractor and VTU exporter
│   ├── surrogate/               # Maximin LHS sampler and Matérn-5/2 ARD GP surrogate bundle
│   ├── uq/                      # Sobol sensitivity, Gaussian Copula cohort, and safe-IOP solver
│   └── visualization/           # Publication-grade Figure 1-6 and Table 1-2 generators
├── tests/                       # Pytest verification suite
├── pyproject.toml
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

Python 3.11+ is required. Install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To re-run finite element simulations from scratch (Phase 1 & Phase 2), install [FEBio 4.0](https://febio.org/) and ensure `febio4` is available on your `PATH` (or update the executable path in `config/sim_settings.yaml`). Reproducing Phase 3 (GP/Sobol/Copula/Safe IOP) and Phase 4 (Figures 1–6 & Tables 1–2) requires **Python only** and does not require FEBio.

## Quickstart & Reproduction

```bash
# 1. Run unit and verification tests (excluding live FEBio single-case test if FEBio is not installed)
PYTHONPATH=. pytest -v --ignore=tests/test_single_case_convergence.py --ignore=tests/test_phase4_deliverables.py

# 2. Reproduce Phase 3: GP validation, Sobol sensitivity, 10,000-eye Copula cohort & safe-IOP boundary
PYTHONPATH=. python src/uq/run_phase3_pipeline.py

# 3. Reproduce Phase 4: Publication Figures 1-6 and Tables 1-2
PYTHONPATH=. python src/visualization/generate_all.py
```

### Optional: Full Finite Element Re-Execution (Requires FEBio 4.0)

```bash
# Phase 1: Emmetropia/high-myopia baselines, 3-level mesh convergence, and 0-35 deg tilt sweep
PYTHONPATH=. python src/febio/phase1_verification.py

# Phase 2: 20 smoke-test runs + 400 training / 80 test LHS runs (supports resume)
PYTHONPATH=. python src/febio/batch_campaign.py --mode smoke --workers 3
PYTHONPATH=. python src/febio/batch_campaign.py --mode train_and_test --workers 3
```

## Citation

If you use this software or its simulation data, cite this project using `CITATION.cff`.

Project author: **HM-ONH Biomechanics UQ Project**. No affiliation is specified.

Repository: https://github.com/seefreewind/hm-onh-biomechanics-uq

Zenodo archival metadata are provided in `.zenodo.json`. Releases are intended for archival through the GitHub–Zenodo integration.

The virtual cohort is simulated and contains no patient-level clinical records. Equivalent IOP boundaries are model-based research estimates and require independent clinical validation.

The archive omits regenerated figures and internal manuscript-planning documents. `tests/test_phase4_deliverables.py` is an internal full-workspace check, so the quickstart excludes it along with the live FEBio convergence test.

## License

This project is released under the MIT License (see `LICENSE`).

## Release verification

The quickstart test command passed **10 tests** on Python 3.14.8. Scikit-learn reported 15 optimizer-bound convergence warnings in the analytical GP interface test. Live FEBio reruns and the internal full-workspace deliverable check were excluded as documented above. The archived numerical outputs were preserved from the supplied repository ZIP.
