"""
轨道 A (机理解耦)：基于 SALib 的全参数全局敏感性分析 (Sobol S1, ST, Sij 及 1000 次 Bootstrap 95% CI)
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
全局固定随机种子: seed = 42
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from src.common.units import (
    FORCE_UNIT,
    GLOBAL_SEED,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    STRESS_UNIT,
    load_yaml_config,
)
from src.surrogate.gp_model import CORE_QOIS, MultiQoIGPSurrogateBundle
from src.surrogate.lhs_sampling import get_parameter_specs

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322
assert GLOBAL_SEED == 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_sobol_global_sensitivity(
    bundle: MultiQoIGPSurrogateBundle,
    n_base: int = 4096,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    执行两层 Sobol 方差分解:
      1. 全 10 维参数空间 (含 IOP in [10, 40] mmHg)，量化载荷与解剖/刚度的全局主效应及二阶交互项 S_ij；
      2. 固定生理眼压 (IOP = 15 mmHg 与 21 mmHg) 下的 9 维内禀眼部脆弱性参数敏感性分解。
    """
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    param_keys, lows, highs = get_parameter_specs(params_cfg)

    problem_10d = {
        "num_vars": len(param_keys),
        "names": param_keys,
        "bounds": [[float(lo), float(hi)] for lo, hi in zip(lows, highs)],
    }

    # Saltelli 二阶抽样: 总评估点数 = N * (2D + 2) = 4096 * 22 = 90,112
    x_sobol_10d = sobol_sample.sample(
        problem_10d,
        N=int(n_base),
        calc_second_order=True,
        seed=GLOBAL_SEED,
    )

    out_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "data" / "uq_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    full_10d_summary_rows: List[Dict[str, Any]] = []
    s2_interaction_rows: List[Dict[str, Any]] = []
    results_dict: Dict[str, Any] = {"full_10d": {}, "fixed_iop_9d": {}}

    for q_idx, qk in enumerate(CORE_QOIS):
        y_pred = bundle.predict_qoi(qk, x_sobol_10d, return_std=False)
        si = sobol_analyze.analyze(
            problem_10d,
            y_pred,
            calc_second_order=True,
            num_resamples=int(n_bootstrap),
            conf_level=float(ci_level),
            seed=GLOBAL_SEED + q_idx,
        )

        s1 = np.asarray(si["S1"], dtype=float)
        s1_conf = np.asarray(si["S1_conf"], dtype=float)
        st = np.asarray(si["ST"], dtype=float)
        st_conf = np.asarray(si["ST_conf"], dtype=float)
        s2 = np.asarray(si["S2"], dtype=float)
        s2_conf = np.asarray(si["S2_conf"], dtype=float)

        results_dict["full_10d"][qk] = {
            "names": param_keys,
            "S1": s1.tolist(),
            "S1_conf": s1_conf.tolist(),
            "ST": st.tolist(),
            "ST_conf": st_conf.tolist(),
            "S2": np.nan_to_num(s2, nan=0.0).tolist(),
            "S2_conf": np.nan_to_num(s2_conf, nan=0.0).tolist(),
        }

        for i, pk in enumerate(param_keys):
            full_10d_summary_rows.append(
                {
                    "qoi": qk,
                    "parameter": pk,
                    "S1": float(s1[i]),
                    "S1_ci95_halfwidth": float(s1_conf[i]),
                    "S1_ci95_low": float(s1[i] - s1_conf[i]),
                    "S1_ci95_high": float(s1[i] + s1_conf[i]),
                    "ST": float(st[i]),
                    "ST_ci95_halfwidth": float(st_conf[i]),
                    "ST_ci95_low": float(st[i] - st_conf[i]),
                    "ST_ci95_high": float(st[i] + st_conf[i]),
                    "interaction_ST_minus_S1": float(st[i] - s1[i]),
                }
            )

        for i in range(len(param_keys)):
            for j in range(i + 1, len(param_keys)):
                val_ij = float(s2[i, j])
                conf_ij = float(s2_conf[i, j])
                if not np.isnan(val_ij):
                    s2_interaction_rows.append(
                        {
                            "qoi": qk,
                            "param_i": param_keys[i],
                            "param_j": param_keys[j],
                            "pair": f"{param_keys[i]} × {param_keys[j]}",
                            "S2": val_ij,
                            "S2_ci95_halfwidth": conf_ij,
                        }
                    )

    # 固定 IOP = 21.0 mmHg 下的 9 维解剖与本构内禀敏感性分析
    anat_keys = param_keys[:9]
    problem_9d = {
        "num_vars": 9,
        "names": anat_keys,
        "bounds": [[float(lows[i]), float(highs[i])] for i in range(9)],
    }
    x_sobol_9d = sobol_sample.sample(
        problem_9d,
        N=int(n_base),
        calc_second_order=True,
        seed=GLOBAL_SEED + 100,
    )
    x_9d_iop21 = np.column_stack([x_sobol_9d, np.full(len(x_sobol_9d), 21.0)])

    fixed_9d_rows: List[Dict[str, Any]] = []
    for q_idx, qk in enumerate(CORE_QOIS):
        y_pred9 = bundle.predict_qoi(qk, x_9d_iop21, return_std=False)
        si9 = sobol_analyze.analyze(
            problem_9d,
            y_pred9,
            calc_second_order=True,
            num_resamples=int(n_bootstrap),
            conf_level=float(ci_level),
            seed=GLOBAL_SEED + 200 + q_idx,
        )
        s1_9 = np.asarray(si9["S1"], dtype=float)
        s1c_9 = np.asarray(si9["S1_conf"], dtype=float)
        st_9 = np.asarray(si9["ST"], dtype=float)
        stc_9 = np.asarray(si9["ST_conf"], dtype=float)
        s2_9 = np.asarray(si9["S2"], dtype=float)

        results_dict["fixed_iop_9d"][qk] = {
            "iop_mmhg": 21.0,
            "names": anat_keys,
            "S1": s1_9.tolist(),
            "S1_conf": s1c_9.tolist(),
            "ST": st_9.tolist(),
            "ST_conf": stc_9.tolist(),
            "S2": np.nan_to_num(s2_9, nan=0.0).tolist(),
        }
        for i, pk in enumerate(anat_keys):
            fixed_9d_rows.append(
                {
                    "qoi": qk,
                    "iop_mmhg": 21.0,
                    "parameter": pk,
                    "S1": float(s1_9[i]),
                    "S1_ci95_halfwidth": float(s1c_9[i]),
                    "ST": float(st_9[i]),
                    "ST_ci95_halfwidth": float(stc_9[i]),
                    "interaction_ST_minus_S1": float(st_9[i] - s1_9[i]),
                }
            )

    pd.DataFrame(full_10d_summary_rows).to_csv(out_dir / "sobol_10d_indices.csv", index=False)
    pd.DataFrame(s2_interaction_rows).to_csv(out_dir / "sobol_10d_interactions_s2.csv", index=False)
    pd.DataFrame(fixed_9d_rows).to_csv(out_dir / "sobol_9d_fixed_iop21.csv", index=False)
    (out_dir / "sobol_results.json").write_text(json.dumps(results_dict, indent=2), encoding="utf-8")

    return results_dict
