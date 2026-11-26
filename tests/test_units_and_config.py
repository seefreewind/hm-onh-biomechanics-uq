"""
Phase 0 强制门禁单元测试：
1. 物理单位体系 (mm - N - MPa) 与眼压转换常量 (1 mmHg = 0.000133322 MPa) 严格校验；
2. config/parameters.yaml 10 参数完备性、边界合法性与基线一致性校验；
3. Gaussian Copula 10x10 实证相关矩阵对称性与严格正定性 (Cholesky 可分解) 校验；
4. config/sim_settings.yaml 载荷与网格收敛门禁一致性校验。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.common.units import (
    DEG_TO_RAD,
    FORCE_UNIT,
    GLOBAL_SEED,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    N_PER_MM2_TO_MPA,
    STRESS_UNIT,
    deg_to_rad,
    load_yaml_config,
    mm_to_um,
    mmhg_to_mpa,
    mpa_to_mmhg,
    neohookean_from_youngs,
    um_to_mm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_YAML = PROJECT_ROOT / "config" / "parameters.yaml"
SIM_YAML = PROJECT_ROOT / "config" / "sim_settings.yaml"


def test_unit_constants_exact() -> None:
    assert LENGTH_UNIT == "mm"
    assert FORCE_UNIT == "N"
    assert STRESS_UNIT == "MPa"
    assert N_PER_MM2_TO_MPA == 1.0
    assert MMHG_TO_MPA == 0.000133322
    assert GLOBAL_SEED == 42

    # 关键生理压力点精确校验
    assert math.isclose(mmhg_to_mpa(1.0), 0.000133322, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(mmhg_to_mpa(15.0), 0.001999830, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mmhg_to_mpa(21.0), 0.002799762, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mmhg_to_mpa(11.0), 0.001466542, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mpa_to_mmhg(mmhg_to_mpa(28.5)), 28.5, rel_tol=1e-12)

    # 角度与微米换算校验
    assert math.isclose(deg_to_rad(35.0), 35.0 * math.pi / 180.0, rel_tol=1e-14)
    assert math.isclose(um_to_mm(260.0), 0.26, rel_tol=1e-14)
    assert math.isclose(mm_to_um(0.025), 25.0, rel_tol=1e-14)


def test_neohookean_conversion_consistency() -> None:
    props = neohookean_from_youngs(0.30, nu=0.49)
    # E = 4 * (1 + nu) * c1
    reconstructed_e = 4.0 * (1.0 + props["nu"]) * props["c1_mpa"]
    assert math.isclose(reconstructed_e, 0.30, rel_tol=1e-12)
    assert props["c2_mpa"] == 0.0
    assert props["k_mpa"] > 10.0 * props["mu_mpa"]

    with pytest.raises(ValueError):
        neohookean_from_youngs(-0.1)
    with pytest.raises(ValueError):
        neohookean_from_youngs(0.3, nu=0.5)


def test_parameters_yaml_integrity_and_spd_copula() -> None:
    cfg = load_yaml_config(PARAMS_YAML)

    assert cfg["metadata"]["random_seed"] == 42
    assert cfg["metadata"]["unit_system"]["mmhg_to_mpa"] == MMHG_TO_MPA

    params = cfg["parameters"]
    assert len(params) == 10, f"Expected exactly 10 parameters, got {len(params)}"

    expected_keys = [
        "AL",
        "t_ppS",
        "theta_canal",
        "t_LC",
        "D_canal",
        "c1_sclera",
        "ksi_fiber",
        "b_fiber",
        "E_LC",
        "IOP",
    ]
    actual_keys = [p["key"] for p in params]
    assert actual_keys == expected_keys
    assert [p["index"] for p in params] == list(range(10))

    for p in params:
        low, high = p["bounds"]
        assert low < high, f"Invalid bounds for {p['key']}: {p['bounds']}"
        for b_name, b_val in p["baselines"].items():
            assert low <= b_val <= high, (
                f"Baseline {b_name}={b_val} out of bounds [{low}, {high}] for {p['key']}"
            )
        assert len(p["references"]) >= 1

    # 专门校验 IOP 与倾斜角换算因子
    iop_param = params[9]
    assert iop_param["key"] == "IOP"
    assert iop_param["conversion_to_solver"] == MMHG_TO_MPA
    theta_param = params[2]
    assert math.isclose(theta_param["conversion_to_solver"], DEG_TO_RAD, rel_tol=1e-12)

    # 校验 10x10 Gaussian Copula 相关矩阵严格对称正定 (SPD)
    corr_order = cfg["copula_correlation"]["parameter_order"]
    assert corr_order == expected_keys
    corr = np.asarray(cfg["copula_correlation"]["matrix"], dtype=float)
    assert corr.shape == (10, 10)
    assert np.allclose(corr, corr.T, atol=1e-12), "Correlation matrix must be symmetric"
    assert np.allclose(np.diag(corr), 1.0, atol=1e-12), "Diagonal entries must equal 1.0"
    off_diag = corr[~np.eye(10, dtype=bool)]
    assert np.all((off_diag > -1.0) & (off_diag < 1.0))

    eigvals = np.linalg.eigvalsh(corr)
    assert eigvals.min() > 0.05, f"Correlation matrix not safely SPD, min eigval = {eigvals.min()}"
    chol = np.linalg.cholesky(corr)
    assert np.allclose(chol @ chol.T, corr, atol=1e-12)


def test_sim_settings_consistency() -> None:
    sim = load_yaml_config(SIM_YAML)
    assert sim["units"]["mmhg_to_mpa"] == MMHG_TO_MPA
    assert sim["reproducibility"]["seed"] == 42
    csfp_mmhg = sim["boundary_and_loads"]["csfp_mmhg"]
    csfp_mpa = sim["boundary_and_loads"]["csfp_mpa"]
    assert math.isclose(csfp_mpa, mmhg_to_mpa(csfp_mmhg), rel_tol=0.0, abs_tol=1e-12)
    assert sim["surrogate"]["acceptance_gates"]["min_q2_test"] >= 0.98
    assert sim["surrogate"]["acceptance_gates"]["max_nrmse_test"] <= 0.02
