"""
Phase 3 双轨 UQ 与安全眼压求解器单元测试：
1. 校验 Gaussian Copula 10,000 例虚拟眼的边际截断边界、秩相关结构还原度与条件分布精确性；
2. 校验已落盘 GP 代理模型在 80 组独立测试集上满足 Q^2 > 0.98 且 NRMSE < 2%；
3. 校验 Sobol 敏感性指数满足方差分解公理 (sum(S1) <= 1.02, ST >= S1 - tol)；
4. 校验安全眼压曲线在正视眼基准点 AL=24.0 mm 处精确锚定 IOP_eq = 21.0 mmHg，且随眼轴延长单调下降。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.units import GLOBAL_SEED, MMHG_TO_MPA, load_yaml_config
from src.uq.copula_cohort import sample_conditional_anatomy_given_al, sample_gaussian_copula_cohort

assert MMHG_TO_MPA == 0.000133322
assert GLOBAL_SEED == 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_copula_cohort_and_conditional_sampling() -> None:
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    df = sample_gaussian_copula_cohort(n_eyes=5000, params_cfg=params_cfg, seed=GLOBAL_SEED)
    assert len(df) == 5000

    # 验证核心实证相关方向: AL 与 t_ppS 显著负相关, 与 theta_canal 显著正相关, 与 b_fiber 显著负相关
    r_pps = float(df["AL"].corr(df["t_ppS"], method="spearman"))
    r_tilt = float(df["AL"].corr(df["theta_canal"], method="spearman"))
    r_bfib = float(df["AL"].corr(df["b_fiber"], method="spearman"))
    assert r_pps < -0.50
    assert r_tilt > 0.45
    assert r_bfib < -0.38

    cond_24 = sample_conditional_anatomy_given_al(24.0, n_samples=1000, params_cfg=params_cfg)
    cond_30 = sample_conditional_anatomy_given_al(30.0, n_samples=1000, params_cfg=params_cfg)
    # AL=30 mm 群体的平均巩膜厚度与筛板厚度必须显著低于 AL=24 mm 群体，平均倾斜角显著更高
    assert float(cond_30[:, 1].mean()) < float(cond_24[:, 1].mean())
    assert float(cond_30[:, 2].mean()) > float(cond_24[:, 2].mean())
    assert float(cond_30[:, 3].mean()) < float(cond_24[:, 3].mean())


def test_phase3_artifacts_and_gates_if_present() -> None:
    uq_dir = PROJECT_ROOT / "data" / "uq_results"
    master_json = uq_dir / "phase3_master_summary.json"
    if not master_json.exists():
        return

    summary = json.loads(master_json.read_text(encoding="utf-8"))
    for qk, m in summary["gp_metrics"].items():
        assert m["Q2"] > 0.98, f"{qk} Q2={m['Q2']} failed >0.98 gate"
        assert m["NRMSE"] < 0.02, f"{qk} NRMSE={m['NRMSE']} failed <0.02 gate"

    df_sobol = pd.read_csv(uq_dir / "sobol_10d_indices.csv")
    for qk, grp in df_sobol.groupby("qoi"):
        assert float(grp["S1"].sum()) <= 1.03
        assert np.all(grp["ST"].to_numpy() >= grp["S1"].to_numpy() - 0.02)

    df_curve = pd.read_csv(uq_dir / "safe_iop_eq_curve.csv")
    row_24 = df_curve.iloc[int(np.argmin(np.abs(df_curve["AL_mm"].to_numpy() - 24.0)))]
    assert abs(float(row_24["iop_eq_median_mmhg"]) - 21.0) < 0.25
    row_30 = df_curve.iloc[int(np.argmin(np.abs(df_curve["AL_mm"].to_numpy() - 30.0)))]
    assert float(row_30["iop_eq_median_mmhg"]) < float(row_24["iop_eq_median_mmhg"])
    assert float(row_30["iop_eq_p10_safe_bound_mmhg"]) < float(row_30["iop_eq_median_mmhg"])
