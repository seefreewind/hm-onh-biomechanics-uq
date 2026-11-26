"""
Phase 1 门禁测试 2：
验证 FEBio 4.x 单算例端到端真实求解收敛性、压力载荷方向物理正确性（筛板后向位移 > 0、眼压升高应变单调增大）
以及第 95 百分位体积加权指标无 NaN/奇异值。
"""

from __future__ import annotations

from pathlib import Path

from src.common.units import MMHG_TO_MPA, load_yaml_config
from src.febio.runner import run_single_fem_case

assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_single_case_febio_convergence_and_monotonic_iop(tmp_path: Path) -> None:
    cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    p_low = {p["key"]: float(p["baselines"]["emmetropia"]) for p in cfg["parameters"]}
    p_high = dict(p_low, IOP=25.0)

    res_low = run_single_fem_case(
        case_id="pytest_emme_iop15",
        params=p_low,
        run_dir=tmp_path / "emme_15",
        phase="pytest",
        mesh_level="coarse",
        export_full_field=False,
        cleanup_feb_on_success=True,
        omp_threads=1,
    )
    res_high = run_single_fem_case(
        case_id="pytest_emme_iop25",
        params=p_high,
        run_dir=tmp_path / "emme_25",
        phase="pytest",
        mesh_level="coarse",
        export_full_field=False,
        cleanup_feb_on_success=True,
        omp_threads=1,
    )

    assert res_low["converged"] is True
    assert res_high["converged"] is True

    q_low = res_low["qois"]
    q_high = res_high["qois"]

    # 校验正视眼生理应变范围与眼压升高单调性
    assert 0.002 < q_low["lc_eeff_p95"] < 0.06
    assert q_low["lc_ap_disp_p95"] > 0.0
    assert q_high["lc_e1_p95"] > q_low["lc_e1_p95"]
    assert q_high["lc_eeff_p95"] > q_low["lc_eeff_p95"]
    assert q_high["lc_gmax_p95"] > q_low["lc_gmax_p95"]
    assert q_high["lc_ap_disp_p95"] > q_low["lc_ap_disp_p95"]
