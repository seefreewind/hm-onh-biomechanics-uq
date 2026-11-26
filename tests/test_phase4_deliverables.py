"""
Phase 4 出版级图表、汇总表与复现文档完整性测试
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
"""

from __future__ import annotations

from pathlib import Path

from src.common.units import FORCE_UNIT, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase4_figures_tables_and_docs_exist() -> None:
    fig_stems = [
        "Figure1_Model_Architecture_and_Verification",
        "Figure2_Baseline_Biomechanics_Comparison",
        "Figure3_GP_Surrogate_Validation_and_ARD",
        "Figure4_TrackA_Sobol_Sensitivity",
        "Figure5_TrackB_Virtual_Cohort_Risk",
        "Figure6_Safe_IOP_Boundary_and_Surface",
    ]
    fig_dir = PROJECT_ROOT / "results" / "figures"
    for stem in fig_stems:
        for ext in ("pdf", "png", "svg"):
            fpath = fig_dir / f"{stem}.{ext}"
            assert fpath.is_file(), f"Missing publication figure: {fpath}"
            assert fpath.stat().st_size > 10_000, f"Figure unexpectedly small: {fpath}"

    tbl_dir = PROJECT_ROOT / "results" / "tables"
    for tstem in (
        "Table1_Parameters_Sensitivity_and_GP",
        "Table2_Cohort_Risk_and_Safe_IOP_Boundaries",
    ):
        for ext in ("csv", "md"):
            tpath = tbl_dir / f"{tstem}.{ext}"
            assert tpath.is_file(), f"Missing publication table: {tpath}"
            assert tpath.stat().st_size > 500, f"Table unexpectedly small: {tpath}"

    for doc_rel in (
        "docs/PLAN.md",
        "docs/VERIFICATION.md",
        "docs/MANUSCRIPT_METHODS_RESULTS.md",
        "README.md",
    ):
        dpath = PROJECT_ROOT / doc_rel
        assert dpath.is_file(), f"Missing documentation file: {dpath}"
        assert dpath.stat().st_size > 1_000
