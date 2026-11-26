"""
轨道 B (人群映射)：基于 Gaussian Copula 的 10,000 例虚拟眼队列生成与眼轴分层高危应变统计模块
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
全局固定随机种子: seed = 42
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

from src.common.units import (
    FORCE_UNIT,
    GLOBAL_SEED,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    STRESS_UNIT,
    load_yaml_config,
)
from src.surrogate.gp_model import CORE_QOIS, MultiQoIGPSurrogateBundle

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322
assert GLOBAL_SEED == 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _inv_marginal_cdf(u: np.ndarray, dist_spec: dict) -> np.ndarray:
    """将 (0, 1) 均匀分位数精确映射为截断正态边际分布物理值。"""
    dtype = dist_spec["type"]
    if dtype != "truncnorm":
        raise ValueError(f"Unsupported marginal distribution type: {dtype}")
    mu = float(dist_spec["mean"])
    sd = float(dist_spec["std"])
    lo = float(dist_spec["low"])
    hi = float(dist_spec["high"])
    a = (lo - mu) / sd
    b = (hi - mu) / sd
    u_clipped = np.clip(u, 1e-12, 1.0 - 1e-12)
    return truncnorm.ppf(u_clipped, a=a, b=b, loc=mu, scale=sd)


def _forward_marginal_to_std_normal(val: float, dist_spec: dict) -> float:
    """
    将条件变量物理值 (如 AL = al_target) 映射回底层潜正态坐标 z0 = (val - mu) / sd，
    避免截断端点 (val == low 或 high) 处 truncnorm.cdf = 0/1 导致 norm.ppf 发散至 ±inf。
    """
    mu = float(dist_spec["mean"])
    sd = float(dist_spec["std"])
    return float((val - mu) / sd)


def sample_gaussian_copula_cohort(
    n_eyes: int = 10000,
    params_cfg: Optional[dict] = None,
    seed: int = GLOBAL_SEED,
) -> pd.DataFrame:
    """从 10x10 实证正定相关矩阵与截断正态边际分布联合抽样 N 例虚拟眼。"""
    if params_cfg is None:
        params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")

    p_specs = sorted(params_cfg["parameters"], key=lambda x: int(x["index"]))
    keys = [p["key"] for p in p_specs]
    corr = np.asarray(params_cfg["copula_correlation"]["matrix"], dtype=float)
    chol = np.linalg.cholesky(corr)

    rng = np.random.default_rng(seed)
    z_indep = rng.standard_normal(size=(n_eyes, len(keys)))
    z_corr = z_indep @ chol.T
    u_corr = norm.cdf(z_corr)

    x_mat = np.empty_like(u_corr)
    for j, p in enumerate(p_specs):
        x_mat[:, j] = _inv_marginal_cdf(u_corr[:, j], p["track_b_dist"])

    df = pd.DataFrame(x_mat, columns=keys)
    df.insert(0, "eye_id", np.arange(1, n_eyes + 1, dtype=np.int64))
    return df


def sample_conditional_anatomy_given_al(
    al_value: float,
    n_samples: int = 2000,
    params_cfg: Optional[dict] = None,
    seed: int = GLOBAL_SEED,
) -> np.ndarray:
    """
    给定特定眼轴长度 AL = al_value [mm]，利用多元高斯条件分布公式精确抽取其余 8 个解剖/材料参数 (index 1..8):
      Z_{1..8} | (Z_0 = z0) ~ N( R_{10} * z0,  R_{11} - R_{10} R_{01} )
    确保在任意 AL 下完美保留高度近视解剖/胶原重塑的条件共变规律与个体间异质性！
    返回形状 (n_samples, 9)，对应列 [AL, t_ppS, theta_canal, t_LC, D_canal, c1_sclera, ksi_fiber, b_fiber, E_LC]。
    """
    if params_cfg is None:
        params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")

    p_specs = sorted(params_cfg["parameters"], key=lambda x: int(x["index"]))
    corr9 = np.asarray(params_cfg["copula_correlation"]["matrix"], dtype=float)[:9, :9]

    z0 = _forward_marginal_to_std_normal(al_value, p_specs[0]["track_b_dist"])
    r_10 = corr9[1:, 0]  # (8,)
    r_11 = corr9[1:, 1:]  # (8, 8)
    cond_mean = r_10 * z0
    cond_cov = r_11 - np.outer(r_10, r_10)
    chol_cond = np.linalg.cholesky(cond_cov)

    rng = np.random.default_rng(seed)
    z_sub = cond_mean[None, :] + rng.standard_normal(size=(n_samples, 8)) @ chol_cond.T
    u_sub = norm.cdf(z_sub)

    out = np.empty((n_samples, 9), dtype=float)
    out[:, 0] = float(al_value)
    for k in range(1, 9):
        out[:, k] = _inv_marginal_cdf(u_sub[:, k - 1], p_specs[k]["track_b_dist"])
    return out


def run_copula_virtual_cohort_analysis(
    bundle: MultiQoIGPSurrogateBundle,
    reference_eeff_threshold: float,
    n_eyes: int = 10000,
    output_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成 10,000 例虚拟眼队列，利用 GP 代理预测全部力学指标，并按眼轴分层统计高危超标率。
    """
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    sim_cfg = load_yaml_config(PROJECT_ROOT / "config" / "sim_settings.yaml")

    df_cohort = sample_gaussian_copula_cohort(n_eyes=n_eyes, params_cfg=params_cfg, seed=GLOBAL_SEED)
    x_nat = df_cohort[bundle.param_keys].to_numpy(dtype=float)

    # 1. 自然静息眼压下的队列预测
    for qk in CORE_QOIS:
        mu_q, std_q = bundle.predict_qoi(qk, x_nat, return_std=True)
        df_cohort[qk] = mu_q
        df_cohort[f"{qk}_gp_std"] = std_q

    # 2. 标准化同眼压 (IOP = 15 mmHg 与 IOP = 21 mmHg) 下的内禀脆弱性预测
    for fixed_iop in (15, 21):
        x_fix = x_nat.copy()
        x_fix[:, 9] = float(fixed_iop)
        df_cohort[f"lc_eeff_p95_at_iop{fixed_iop}"] = bundle.predict_qoi("lc_eeff_p95", x_fix, return_std=False)
        df_cohort[f"lc_e1_p95_at_iop{fixed_iop}"] = bundle.predict_qoi("lc_e1_p95", x_fix, return_std=False)

    df_cohort["exceeds_ref_at_natural_iop"] = df_cohort["lc_eeff_p95"] > reference_eeff_threshold
    df_cohort["exceeds_ref_at_iop15"] = df_cohort["lc_eeff_p95_at_iop15"] > reference_eeff_threshold
    df_cohort["exceeds_ref_at_iop21"] = df_cohort["lc_eeff_p95_at_iop21"] > reference_eeff_threshold

    # 眼轴分层标注
    strata_cfg = sim_cfg["uq"]["copula_cohort"]["al_strata_mm"]
    strata_labels = []
    al_arr = df_cohort["AL"].to_numpy(dtype=float)
    for v in al_arr:
        assigned = "other"
        for s_idx, (sname, (slo, shi)) in enumerate(strata_cfg.items()):
            if (s_idx < len(strata_cfg) - 1 and slo <= v < shi) or (s_idx == len(strata_cfg) - 1 and slo <= v <= shi):
                assigned = sname
                break
        strata_labels.append(assigned)
    df_cohort["al_stratum"] = strata_labels

    # 分层统计汇总表
    strata_rows: List[Dict[str, Any]] = []
    for sname, (slo, shi) in strata_cfg.items():
        sub = df_cohort[df_cohort["al_stratum"] == sname]
        eeff_pct = sub["lc_eeff_p95"].to_numpy(dtype=float) * 100.0
        e1_pct = sub["lc_e1_p95"].to_numpy(dtype=float) * 100.0
        gmax_pct = sub["lc_gmax_p95"].to_numpy(dtype=float) * 100.0
        disp_um = sub["lc_ap_disp_p95"].to_numpy(dtype=float) * 1000.0

        strata_rows.append(
            {
                "stratum": sname,
                "al_range_mm": f"[{slo:.1f}, {shi:.1f}]",
                "n_eyes": int(len(sub)),
                "cohort_share_pct": float(len(sub)) / n_eyes * 100.0,
                "mean_AL_mm": float(sub["AL"].mean()),
                "mean_t_ppS_mm": float(sub["t_ppS"].mean()),
                "mean_theta_canal_deg": float(sub["theta_canal"].mean()),
                "mean_t_LC_mm": float(sub["t_LC"].mean()),
                "eeff_p95_median_pct": float(np.median(eeff_pct)),
                "eeff_p95_iqr_low_pct": float(np.percentile(eeff_pct, 25.0)),
                "eeff_p95_iqr_high_pct": float(np.percentile(eeff_pct, 75.0)),
                "eeff_p95_90th_pct": float(np.percentile(eeff_pct, 90.0)),
                "e1_p95_median_pct": float(np.median(e1_pct)),
                "gmax_p95_median_pct": float(np.median(gmax_pct)),
                "disp_p95_median_um": float(np.median(disp_um)),
                "high_risk_rate_natural_iop_pct": float(sub["exceeds_ref_at_natural_iop"].mean() * 100.0),
                "high_risk_rate_at_iop15_pct": float(sub["exceeds_ref_at_iop15"].mean() * 100.0),
                "high_risk_rate_at_iop21_pct": float(sub["exceeds_ref_at_iop21"].mean() * 100.0),
            }
        )

    df_strata = pd.DataFrame(strata_rows)

    out_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "data" / "uq_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_cohort.to_csv(out_dir / "virtual_cohort_10000.csv", index=False)
    df_strata.to_csv(out_dir / "cohort_al_strata_summary.csv", index=False)

    return df_cohort, df_strata
