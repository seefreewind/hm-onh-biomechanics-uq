"""
Phase 4 全部出版级图表 (Figure 1-6) 与汇总表 (Table 1-2) 一键生成入口脚本
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
"""

from __future__ import annotations

from pathlib import Path

from src.common.units import FORCE_UNIT, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT
from src.visualization.fig_part1 import (
    plot_figure1_architecture_and_verification,
    plot_figure2_baseline_biomechanics,
    plot_figure3_gp_validation_and_ard,
)
from src.visualization.fig_part2 import (
    generate_publication_tables,
    plot_figure4_sobol_sensitivity,
    plot_figure5_virtual_cohort_risk,
    plot_figure6_safe_iop_boundary,
)

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_all_figures_and_tables() -> None:
    fig_dir = PROJECT_ROOT / "results" / "figures"
    tbl_dir = PROJECT_ROOT / "results" / "tables"

    print("[Phase 4] Generating Figure 1 (Model Architecture, Fibers & Verification)...")
    plot_figure1_architecture_and_verification(fig_dir)

    print("[Phase 4] Generating Figure 2 (Baseline Biomechanics Comparison)...")
    plot_figure2_baseline_biomechanics(fig_dir)

    print("[Phase 4] Generating Figure 3 (GP Surrogate Validation & ARD)...")
    plot_figure3_gp_validation_and_ard(fig_dir)

    print("[Phase 4] Generating Figure 4 (Track A Sobol Global Sensitivity)...")
    plot_figure4_sobol_sensitivity(fig_dir)

    print("[Phase 4] Generating Figure 5 (Track B 10,000-Eye Copula Cohort Risk)...")
    plot_figure5_virtual_cohort_risk(fig_dir)

    print("[Phase 4] Generating Figure 6 (AL x IOP Surface & Safe IOP Curve)...")
    plot_figure6_safe_iop_boundary(fig_dir)

    print("[Phase 4] Generating Table 1 & Table 2 (CSV + Markdown)...")
    generate_publication_tables(tbl_dir)

    print("[Phase 4] All 6 main figures and 2 summary tables exported successfully.")


if __name__ == "__main__":
    build_all_figures_and_tables()
