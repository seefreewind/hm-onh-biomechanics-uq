"""
Phase 3 一键全流程编排器：
1. 训练 Matérn-5/2 ARD 高斯过程代理模型并在 80 组独立测试集上严格校验门禁 (Q^2 > 0.98, NRMSE < 2%)；
   若任一指标未达门禁，自动触发基于后验方差的主动学习补点并调用 FEBio 真实求解重训；
2. 轨道 A (机理解耦)：SALib Sobol 全局敏感性分析 (S1, ST, Sij + 1000 次 Bootstrap 95% CI)；
3. 求解正视眼 (AL=24 mm, IOP=21 mmHg) 基准应变阈值、AL × IOP 响应曲面与等效安全眼压曲线 IOP_eq(AL)；
4. 轨道 B (人群映射)：Gaussian Copula 10,000 例虚拟眼队列生成与不同眼轴分层高危超标率统计。

强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.common.units import FORCE_UNIT, GLOBAL_SEED, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT, load_yaml_config
from src.febio.runner import run_batch_fem_cases
from src.surrogate.gp_model import CORE_QOIS, train_and_validate_gp_bundle
from src.uq.copula_cohort import run_copula_virtual_cohort_analysis
from src.uq.safe_iop_solver import compute_safe_iop_curves_and_surface
from src.uq.sobol_analysis import run_sobol_global_sensitivity

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322
assert GLOBAL_SEED == 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_phase3_all() -> Dict[str, Any]:
    sim_cfg = load_yaml_config(PROJECT_ROOT / "config" / "sim_settings.yaml")
    proc_dir = PROJECT_ROOT / "data" / "processed"
    uq_dir = PROJECT_ROOT / "data" / "uq_results"
    table_dir = PROJECT_ROOT / "results" / "tables"
    uq_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    train_csv = proc_dir / "sim_results_train.csv"
    test_csv = proc_dir / "sim_results_test.csv"

    min_q2 = float(sim_cfg["surrogate"]["acceptance_gates"]["min_q2_test"])
    max_nrmse = float(sim_cfg["surrogate"]["acceptance_gates"]["max_nrmse_test"])
    al_cfg = sim_cfg["surrogate"]["active_learning"]

    print("[Phase 3] Step 1/4: Training Matérn-5/2 ARD GP surrogate models...")
    bundle, metrics = train_and_validate_gp_bundle(
        train_csv=train_csv,
        test_csv=test_csv,
        qoi_keys=CORE_QOIS,
        save_dir=uq_dir,
    )

    # 检查门禁并在必要时自动启动主动学习真实 FEBio 补点循环
    round_idx = 0
    cur_train_csv = train_csv
    while al_cfg["enabled_if_below_gate"] and round_idx < int(al_cfg["max_rounds"]):
        failed_qois = [
            qk
            for qk, m in metrics.items()
            if (m["Q2"] < min_q2 or m["NRMSE"] > max_nrmse)
        ]
        if not failed_qois:
            break

        round_idx += 1
        n_add = int(al_cfg["batch_size"])
        print(
            f"[Phase 3] Active Learning Round {round_idx}: {failed_qois} below gate; "
            f"sampling {n_add} high-variance infill points via FEBio..."
        )
        df_tr_cur = pd.read_csv(cur_train_csv)
        x_exist = df_tr_cur[bundle.param_keys].to_numpy(dtype=float)
        x_new = bundle.propose_active_learning_points(
            x_exist,
            n_new=n_add,
            seed=GLOBAL_SEED + 5000 + round_idx * 100,
        )

        al_runs_dir = PROJECT_ROOT / "data" / "sim_runs" / f"phase3_al_round{round_idx}"
        specs = []
        for k_i, row_vals in enumerate(x_new):
            cid = f"al_r{round_idx}_{k_i:03d}"
            p_dict = {pk: float(row_vals[j]) for j, pk in enumerate(bundle.param_keys)}
            specs.append(
                {
                    "case_id": cid,
                    "params": p_dict,
                    "run_dir": al_runs_dir / cid,
                    "phase": f"phase3_al_r{round_idx}",
                    "mesh_level": sim_cfg["mesh"]["production_level"],
                    "export_full_field": False,
                    "cleanup_feb_on_success": True,
                }
            )
        al_res = run_batch_fem_cases(specs, n_workers=3)
        new_rows = []
        for r in al_res:
            if r["converged"]:
                nr = {
                    "case_id": r["case_id"],
                    "converged": True,
                    "attempts": r["attempts"],
                    "schedule": r["schedule"],
                    "solver_seconds": r["solver_seconds"],
                }
                for pk in bundle.param_keys:
                    nr[pk] = r["params"][pk]
                for qk_all, qv_all in r["qois"].items():
                    if qk_all in df_tr_cur.columns:
                        nr[qk_all] = qv_all
                new_rows.append(nr)

        df_aug = pd.concat([df_tr_cur, pd.DataFrame(new_rows)], ignore_index=True)
        cur_train_csv = proc_dir / f"sim_results_train_aug_r{round_idx}.csv"
        df_aug.to_csv(cur_train_csv, index=False)

        bundle, metrics = train_and_validate_gp_bundle(
            train_csv=cur_train_csv,
            test_csv=test_csv,
            qoi_keys=CORE_QOIS,
            save_dir=uq_dir,
        )

    for qk, m in metrics.items():
        print(
            f"  [GP Test Gate | {qk:15s}] Q^2 = {m['Q2']:.5f} (gate > {min_q2}), "
            f"NRMSE = {m['NRMSE_pct']:.3f}% (gate < {max_nrmse*100:.1f}%), "
            f"95%CI_cov = {m['empirical_95ci_coverage']*100:.1f}%"
        )

    # 导出 GP 独立测试集验证汇总表
    gp_rows = []
    for qk, m in metrics.items():
        mdl = bundle.models[qk]
        row_g = {
            "qoi": qk,
            "train_5fold_cv_Q2": m["train_5fold_cv_Q2"],
            "test_Q2": m["Q2"],
            "test_NRMSE_pct": m["NRMSE_pct"],
            "test_RMSE": m["RMSE"],
            "test_MAE": m["MAE"],
            "test_max_abs_err": m["max_abs_err"],
            "empirical_95ci_coverage_pct": m["empirical_95ci_coverage"] * 100.0,
            "log_transform": bool(mdl.use_log_transform),
        }
        for p_i, pk in enumerate(bundle.param_keys):
            row_g[f"ard_ls_{pk}"] = float(mdl.ard_length_scales[p_i])
        gp_rows.append(row_g)
    pd.DataFrame(gp_rows).to_csv(table_dir / "phase3_gp_validation_and_ard.csv", index=False)

    print("[Phase 3] Step 2/4: Running SALib Sobol Global Sensitivity Analysis (N=4096, 1000 Bootstrap)...")
    sobol_cfg = sim_cfg["uq"]["sobol"]
    sobol_res = run_sobol_global_sensitivity(
        bundle=bundle,
        n_base=int(sobol_cfg["n_base_samples"]),
        n_bootstrap=int(sobol_cfg["bootstrap_resamples"]),
        ci_level=float(sobol_cfg["ci_level"]),
        output_dir=uq_dir,
    )

    print("[Phase 3] Step 3/4: Solving AL x IOP Response Surface and Equivalent Safe IOP Curve IOP_eq(AL)...")
    safe_iop_res = compute_safe_iop_curves_and_surface(
        bundle=bundle,
        n_cond_eyes_per_al=1500,
        n_al_grid=81,
        n_iop_grid=81,
        output_dir=uq_dir,
    )

    print("[Phase 3] Step 4/4: Generating 10,000 Virtual Eyes via Gaussian Copula & Stratifying Risk...")
    n_eyes = int(sim_cfg["uq"]["copula_cohort"]["n_virtual_eyes"])
    _, df_strata = run_copula_virtual_cohort_analysis(
        bundle=bundle,
        reference_eeff_threshold=float(safe_iop_res["eps_ref"]),
        n_eyes=n_eyes,
        output_dir=uq_dir,
    )
    df_strata.to_csv(table_dir / "phase3_cohort_strata_summary.csv", index=False)
    safe_iop_res["df_curve"].to_csv(table_dir / "phase3_safe_iop_curve.csv", index=False)

    summary_out = {
        "gp_metrics": metrics,
        "active_learning_rounds": round_idx,
        "safe_iop_meta": safe_iop_res["meta"],
        "cohort_strata": df_strata.to_dict(orient="records"),
        "sobol_eeff_10d": sobol_res["full_10d"]["lc_eeff_p95"],
        "sobol_eeff_9d_iop21": sobol_res["fixed_iop_9d"]["lc_eeff_p95"],
    }
    (uq_dir / "phase3_master_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    print("[Phase 3] ALL STEPS COMPLETED SUCCESSFULLY.")
    return summary_out


if __name__ == "__main__":
    run_phase3_all()
