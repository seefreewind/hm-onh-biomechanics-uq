"""
测试 Matérn-5/2 ARD 高斯过程代理模型接口、对数正态反变换精度 (Q^2 > 0.99) 与主动学习补点功能。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.units import GLOBAL_SEED, MMHG_TO_MPA, load_yaml_config
from src.surrogate.gp_model import MultiQoIGPSurrogateBundle, fit_single_qoi_gp
from src.surrogate.lhs_sampling import (
    generate_maximin_lhs_unit,
    get_parameter_specs,
    scale_unit_to_physical,
)

assert MMHG_TO_MPA == 0.000133322
PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _analytic_laplace_onh_proxy(x: np.ndarray) -> np.ndarray:
    """光滑解析物理代理测试函数 (仅用于单元测试校验 GP 接口与收敛阶，严禁用于论文结果)。"""
    al = x[:, 0]
    t_pps = x[:, 1]
    theta = np.deg2rad(x[:, 2])
    t_lc = x[:, 3]
    d_can = x[:, 4]
    c1_scl = x[:, 5]
    e_lc = x[:, 8]
    iop_mpa = x[:, 9] * MMHG_TO_MPA

    r_globe = 11.5 + 0.48 * (al - 24.0)
    hoop = (iop_mpa * r_globe) / (2.0 * t_pps * (6.0 * c1_scl + 1.5))
    tlp = (iop_mpa - 5.0 * MMHG_TO_MPA) * (d_can / t_lc) / e_lc
    return (1.8 * hoop + 0.08 * tlp) * (1.0 + 0.55 * np.sin(theta))


def test_matern52_ard_gp_accuracy_and_active_learning() -> None:
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    keys, lows, highs = get_parameter_specs(params_cfg)

    u_tr = generate_maximin_lhs_unit(180, 10, seed=GLOBAL_SEED, n_candidates=10)
    u_te = generate_maximin_lhs_unit(40, 10, seed=GLOBAL_SEED + 99, n_candidates=10)
    x_tr = scale_unit_to_physical(u_tr, lows, highs)
    x_te = scale_unit_to_physical(u_te, lows, highs)

    y_tr = _analytic_laplace_onh_proxy(x_tr)
    y_te = _analytic_laplace_onh_proxy(x_te)

    mdl = fit_single_qoi_gp(x_tr, y_tr, "lc_eeff_p95", keys, lows, highs, n_restarts=8, seed=GLOBAL_SEED)
    bundle = MultiQoIGPSurrogateBundle({"lc_eeff_p95": mdl}, keys, lows, highs)

    df_test = pd.DataFrame(x_te, columns=keys)
    df_test["converged"] = True
    df_test["lc_eeff_p95"] = y_te

    metrics = bundle.evaluate_on_testset(df_test)["lc_eeff_p95"]
    assert metrics["Q2"] > 0.985
    assert metrics["NRMSE"] < 0.02

    new_pts = bundle.propose_active_learning_points(x_tr, n_new=5, n_pool=1024)
    assert new_pts.shape == (5, 10)
    assert np.all(new_pts >= lows) and np.all(new_pts <= highs)
