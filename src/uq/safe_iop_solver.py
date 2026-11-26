"""
“眼轴 × 眼压”交互响应曲面与生物力学等效安全眼压曲线 IOP_eq(AL) 求解器
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
全局固定随机种子: seed = 42

科学目标 5 定义:
  以正视眼 (AL = 24.0 mm, IOP = 21.0 mmHg) 的筛板中位有效应变为安全耐受基准阈值 eps_ref，
  对任意眼轴长度 AL in [22.0, 32.0] mm 下的条件虚拟眼群体，反演求解使筛板有效应变等于 eps_ref 的
  等效眼压 IOP_eq(AL)，输出中位曲线 IOP_eq,50(AL)、四分位带 [P25, P75] 以及保护 90% 高危个体的
  第 90 百分位保护带下界 IOP_eq,P10(AL)。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.common.units import (
    FORCE_UNIT,
    GLOBAL_SEED,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    STRESS_UNIT,
    load_yaml_config,
)
from src.surrogate.gp_model import MultiQoIGPSurrogateBundle
from src.uq.copula_cohort import sample_conditional_anatomy_given_al

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322
assert GLOBAL_SEED == 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_safe_iop_curves_and_surface(
    bundle: MultiQoIGPSurrogateBundle,
    n_cond_eyes_per_al: int = 1500,
    n_al_grid: int = 81,
    n_iop_grid: int = 81,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    bench_cfg = params_cfg["safe_iop_benchmark"]
    ref_al = float(bench_cfg["reference_al_mm"])
    ref_iop = float(bench_cfg["reference_iop_mmhg"])

    # 1. 计算正视眼 AL = 24.0 mm, IOP = 21.0 mmHg 条件虚拟眼群体的中位有效应变基准 eps_ref
    cond_24 = sample_conditional_anatomy_given_al(
        al_value=ref_al,
        n_samples=n_cond_eyes_per_al,
        params_cfg=params_cfg,
        seed=GLOBAL_SEED + 777,
    )
    x_ref_24 = np.column_stack([cond_24, np.full(len(cond_24), ref_iop)])
    eeff_ref_samples = bundle.predict_qoi("lc_eeff_p95", x_ref_24, return_std=False)
    eps_ref = float(np.median(eeff_ref_samples))

    # 同时记录确定性正视基线参数在 21 mmHg 下的 GP 预测值供交叉验证
    emme_base = np.array([[float(p["baselines"]["emmetropia"]) for p in sorted(params_cfg["parameters"], key=lambda x: x["index"])]])
    emme_base[0, 0] = ref_al
    emme_base[0, 9] = ref_iop
    eps_emme_det = float(bundle.predict_qoi("lc_eeff_p95", emme_base, return_std=False)[0])

    # 2. 构建精细 AL x IOP 二维网格并向量化求解每个 AL 处 M 例条件虚拟眼的等效安全眼压 IOP_eq
    al_grid = np.linspace(22.0, 32.0, n_al_grid)
    # 搜索网格步长 0.1 mmHg (精确包含 21.0 mmHg)，下探至 4.0 mmHg 以平滑捕捉极端超长眼轴 P10 保护下界
    iop_search_grid = np.linspace(4.0, 40.0, 361)
    iop_surface_grid = np.linspace(10.0, 40.0, n_iop_grid)

    surface_median_pct = np.zeros((n_al_grid, n_iop_grid), dtype=float)
    surface_p90_pct = np.zeros((n_al_grid, n_iop_grid), dtype=float)
    surface_det_emme_anat_pct = np.zeros((n_al_grid, n_iop_grid), dtype=float)

    curve_rows = []

    # 固定随机种子使不同 AL 间采用共同随机数 (Common Random Numbers, CRN)，令分位曲线光滑无蒙特卡洛抖动！
    for i, al_val in enumerate(al_grid):
        anat_m = sample_conditional_anatomy_given_al(
            al_value=float(al_val),
            n_samples=n_cond_eyes_per_al,
            params_cfg=params_cfg,
            seed=GLOBAL_SEED + 777,  # CRN 锁定潜变量抽样确保随 AL 光滑可微!
        )

        # (a) 计算响应曲面在 iop_surface_grid [10, 40] 上的群体中位值与 P90 值
        n_m = len(anat_m)
        anat_rep = np.repeat(anat_m, len(iop_surface_grid), axis=0)
        iop_tile = np.tile(iop_surface_grid, n_m)
        x_surf_batch = np.column_stack([anat_rep, iop_tile])
        y_surf = bundle.predict_qoi("lc_eeff_p95", x_surf_batch, return_std=False).reshape(n_m, len(iop_surface_grid))

        surface_median_pct[i, :] = np.median(y_surf, axis=0) * 100.0
        surface_p90_pct[i, :] = np.percentile(y_surf, 90.0, axis=0) * 100.0

        # 纯几何眼轴效应对照曲面 (固定其余 8 个参数于正视眼基线)
        det_block = np.repeat(emme_base[:, :9], len(iop_surface_grid), axis=0)
        det_block[:, 0] = float(al_val)
        x_det_batch = np.column_stack([det_block, iop_surface_grid])
        surface_det_emme_anat_pct[i, :] = bundle.predict_qoi("lc_eeff_p95", x_det_batch, return_std=False) * 100.0

        # (b) 对每例条件虚拟眼在 iop_search_grid 上精确线性/单调逆插值求根 y(IOP_eq) == eps_ref
        anat_rep_s = np.repeat(anat_m, len(iop_search_grid), axis=0)
        iop_tile_s = np.tile(iop_search_grid, n_m)
        y_search = bundle.predict_qoi(
            "lc_eeff_p95",
            np.column_stack([anat_rep_s, iop_tile_s]),
            return_std=False,
        ).reshape(n_m, len(iop_search_grid))

        iop_eq_eyes = np.empty(n_m, dtype=float)
        for m_idx in range(n_m):
            ym = y_search[m_idx]
            # 确保单调累加以应对极低 IOP (< CSFP) 区可能的轻微非单调谷底: 从最低谷起取单调增分支
            min_pos = int(np.argmin(ym))
            ym_mono = np.maximum.accumulate(ym[min_pos:])
            iop_mono = iop_search_grid[min_pos:]
            iop_eq_eyes[m_idx] = float(np.interp(eps_ref, ym_mono, iop_mono))

        curve_rows.append(
            {
                "AL_mm": float(al_val),
                "iop_eq_median_mmhg": float(np.median(iop_eq_eyes)),
                "iop_eq_mean_mmhg": float(np.mean(iop_eq_eyes)),
                "iop_eq_p25_mmhg": float(np.percentile(iop_eq_eyes, 25.0)),
                "iop_eq_p75_mmhg": float(np.percentile(iop_eq_eyes, 75.0)),
                "iop_eq_p10_safe_bound_mmhg": float(np.percentile(iop_eq_eyes, 10.0)),
                "iop_eq_p90_mmhg": float(np.percentile(iop_eq_eyes, 90.0)),
                "iop_reduction_vs_21_median_mmhg": 21.0 - float(np.median(iop_eq_eyes)),
                "iop_reduction_vs_21_p10_mmhg": 21.0 - float(np.percentile(iop_eq_eyes, 10.0)),
            }
        )

    df_curve = pd.DataFrame(curve_rows)

    out_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "data" / "uq_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_curve.to_csv(out_dir / "safe_iop_eq_curve.csv", index=False)
    np.savez_compressed(
        out_dir / "al_iop_response_surfaces.npz",
        al_grid=al_grid,
        iop_grid=iop_surface_grid,
        surface_median_pct=surface_median_pct,
        surface_p90_pct=surface_p90_pct,
        surface_det_emme_anat_pct=surface_det_emme_anat_pct,
        eps_ref_pct=eps_ref * 100.0,
        eps_emme_det_pct=eps_emme_det * 100.0,
    )

    # 提取关键临床眼轴节点 (24, 26, 28, 30, 32 mm) 摘要
    key_als = [24.0, 26.0, 28.0, 30.0, 32.0]
    milestones = []
    for kal in key_als:
        idx_k = int(np.argmin(np.abs(al_grid - kal)))
        milestones.append(curve_rows[idx_k])

    meta = {
        "reference_al_mm": ref_al,
        "reference_iop_mmhg": ref_iop,
        "reference_iop_mpa": ref_iop * MMHG_TO_MPA,
        "eps_ref_cohort_median": eps_ref,
        "eps_ref_cohort_median_pct": eps_ref * 100.0,
        "eps_emmetropia_deterministic_pct": eps_emme_det * 100.0,
        "key_al_milestones": milestones,
    }
    (out_dir / "safe_iop_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "eps_ref": eps_ref,
        "df_curve": df_curve,
        "meta": meta,
    }
