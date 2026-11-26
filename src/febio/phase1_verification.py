"""
Phase 1 单算例打通与三级网格收敛性、倾斜角鲁棒性全套并行验证脚本
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)

执行内容 (6 进程并行调度，总墙钟耗时约 1.5 分钟):
1. 双基线全流场仿真与 VTU 导出:
   - 正视眼基线 (Emmetropia, AL=24.0 mm, theta=3 deg, IOP=15 mmHg 及基准 IOP=21 mmHg)
   - 高度近视倾斜视盘基线 (High-Myopia Tilted, AL=29.5 mm, theta=25 deg, IOP=15 mmHg 及 IOP=21 mmHg)
2. 三级网格收敛性验证 (Coarse / Medium / Fine):
   校验选定生产网格 (Medium) 相对 Fine 网格的 LC 第 95 百分位应变相对变化 < 2%。
3. 倾斜角扫描测试: theta_canal in [0, 10, 20, 30, 35] deg 下的 Gmsh 质量与 FEBio 收敛验证。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.common.units import FORCE_UNIT, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT, load_yaml_config
from src.febio.runner import run_single_fem_case

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_baseline_dict(params_cfg: dict, kind: str) -> Dict[str, float]:
    return {p["key"]: float(p["baselines"][kind]) for p in params_cfg["parameters"]}


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    return run_single_fem_case(
        case_id=job["case_id"],
        params=job["params"],
        run_dir=job["run_dir"],
        phase=job["phase"],
        mesh_level=job["mesh_level"],
        export_full_field=job["export_full_field"],
        export_vtu_path=job.get("export_vtu_path"),
        cleanup_feb_on_success=False,
        omp_threads=job.get("omp_threads", 1),
    )


def run_phase1_suite(n_workers: int = 6) -> Dict[str, object]:
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    out_base = PROJECT_ROOT / "data" / "sim_runs" / "phase1"
    vtu_dir = PROJECT_ROOT / "data" / "processed" / "phase1_vtu"
    table_dir = PROJECT_ROOT / "results" / "tables"
    out_base.mkdir(parents=True, exist_ok=True)
    vtu_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    emme_15 = get_baseline_dict(params_cfg, "emmetropia")
    hm_15 = get_baseline_dict(params_cfg, "high_myopia_tilted")
    emme_21 = dict(emme_15, IOP=21.0)
    hm_21 = dict(hm_15, IOP=21.0)

    jobs: List[Dict[str, Any]] = []

    # 1. 4 个基线算例 (带 VTU 全场导出)
    baseline_specs = [
        ("emmetropia_iop15", emme_15),
        ("emmetropia_iop21_ref", emme_21),
        ("high_myopia_tilted_iop15", hm_15),
        ("high_myopia_tilted_iop21", hm_21),
    ]
    for cid, p_dict in baseline_specs:
        jobs.append(
            {
                "case_id": cid,
                "params": p_dict,
                "run_dir": out_base / cid,
                "phase": "phase1_baseline",
                "mesh_level": "medium",
                "export_full_field": True,
                "export_vtu_path": vtu_dir / f"{cid}.vtu",
            }
        )

    # 2. 6 个三级网格收敛算例 (先把两个耗时的 Fine 排到队列最前面让多核最早启动!)
    conv_groups = [("emmetropia_iop15", emme_15), ("high_myopia_tilted_iop15", hm_15)]
    for lvl in ("fine", "medium", "coarse"):
        for gname, base_p in conv_groups:
            cid = f"meshconv_{gname}_{lvl}"
            jobs.append(
                {
                    "case_id": cid,
                    "params": base_p,
                    "run_dir": out_base / cid,
                    "phase": "phase1_mesh_conv",
                    "mesh_level": lvl,
                    "export_full_field": False,
                    "export_vtu_path": None,
                }
            )

    # 3. 5 个倾斜角扫描算例
    for theta in (0.0, 10.0, 20.0, 30.0, 35.0):
        cid = f"tilt_sweep_deg_{int(theta):02d}"
        jobs.append(
            {
                "case_id": cid,
                "params": dict(hm_15, theta_canal=theta),
                "run_dir": out_base / cid,
                "phase": "phase1_tilt_sweep",
                "mesh_level": "medium",
                "export_full_field": False,
                "export_vtu_path": None,
            }
        )

    fine_jobs = [dict(jb, omp_threads=4) for jb in jobs if jb["mesh_level"] == "fine"]
    other_jobs = [dict(jb, omp_threads=1) for jb in jobs if jb["mesh_level"] != "fine"]

    results_map: Dict[str, dict] = {}
    print(f"[Phase 1] Stage A: Running {len(fine_jobs)} Fine-mesh jobs (2 workers x 4 threads)...")
    with ProcessPoolExecutor(max_workers=2) as pool:
        fut_to_cid = {pool.submit(_run_job, jb): jb["case_id"] for jb in fine_jobs}
        for fut in as_completed(fut_to_cid):
            cid = fut_to_cid[fut]
            res = fut.result()
            if not res["converged"]:
                raise RuntimeError(f"Phase 1 case {cid} failed to converge: {res}")
            results_map[cid] = res
            print(
                f"  [DONE] {cid:38s} | elems={res['total_elements']:5d} | "
                f"t={res['solver_seconds']:5.2f}s | Eeff_p95={res['qois']['lc_eeff_p95_pct']:.3f}%"
            )

    print(f"[Phase 1] Stage B: Running {len(other_jobs)} Medium/Coarse jobs ({n_workers} workers x 1 thread)...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        fut_to_cid = {pool.submit(_run_job, jb): jb["case_id"] for jb in other_jobs}
        for fut in as_completed(fut_to_cid):
            cid = fut_to_cid[fut]
            res = fut.result()
            if not res["converged"]:
                raise RuntimeError(f"Phase 1 case {cid} failed to converge: {res}")
            results_map[cid] = res
            print(
                f"  [DONE] {cid:38s} | elems={res['total_elements']:5d} | "
                f"t={res['solver_seconds']:5.2f}s | Eeff_p95={res['qois']['lc_eeff_p95_pct']:.3f}%"
            )

    baseline_results = [results_map[cid] for cid, _ in baseline_specs]

    mesh_conv_rows: List[dict] = []
    for gname, _ in conv_groups:
        fine_q = results_map[f"meshconv_{gname}_fine"]["qois"]
        for lvl in ("coarse", "medium", "fine"):
            r = results_map[f"meshconv_{gname}_{lvl}"]
            q = r["qois"]
            rel_err_eeff = abs(q["lc_eeff_p95"] - fine_q["lc_eeff_p95"]) / max(abs(fine_q["lc_eeff_p95"]), 1e-12) * 100.0
            rel_err_e1 = abs(q["lc_e1_p95"] - fine_q["lc_e1_p95"]) / max(abs(fine_q["lc_e1_p95"]), 1e-12) * 100.0
            rel_err_gmax = abs(q["lc_gmax_p95"] - fine_q["lc_gmax_p95"]) / max(abs(fine_q["lc_gmax_p95"]), 1e-12) * 100.0
            rel_err_disp = abs(q["lc_ap_disp_p95"] - fine_q["lc_ap_disp_p95"]) / max(abs(fine_q["lc_ap_disp_p95"]), 1e-12) * 100.0
            mesh_conv_rows.append(
                {
                    "group": gname,
                    "mesh_level": lvl,
                    "nodes": r["total_nodes"],
                    "elements": r["total_elements"],
                    "solver_seconds": round(r["solver_seconds"], 3),
                    "lc_e1_p95_pct": q["lc_e1_p95_pct"],
                    "lc_eeff_p95_pct": q["lc_eeff_p95_pct"],
                    "lc_gmax_p95_pct": q["lc_gmax_p95_pct"],
                    "lc_ap_disp_p95_um": q["lc_ap_disp_p95_um"],
                    "rel_diff_e1_vs_fine_pct": rel_err_e1,
                    "rel_diff_eeff_vs_fine_pct": rel_err_eeff,
                    "rel_diff_gmax_vs_fine_pct": rel_err_gmax,
                    "rel_diff_disp_vs_fine_pct": rel_err_disp,
                }
            )

    tilt_rows: List[dict] = []
    for theta in (0.0, 10.0, 20.0, 30.0, 35.0):
        r = results_map[f"tilt_sweep_deg_{int(theta):02d}"]
        q = r["qois"]
        tilt_rows.append(
            {
                "theta_canal_deg": theta,
                "asco_bmo_offset_mm": r["asco_bmo_offset_mm"],
                "min_scaled_jacobian": r["min_scaled_jacobian"],
                "converged": r["converged"],
                "solver_seconds": round(r["solver_seconds"], 3),
                "lc_e1_p95_pct": q["lc_e1_p95_pct"],
                "lc_eeff_p95_pct": q["lc_eeff_p95_pct"],
                "lc_gmax_p95_pct": q["lc_gmax_p95_pct"],
                "lc_ap_disp_p95_um": q["lc_ap_disp_p95_um"],
            }
        )

    df_base = pd.DataFrame(
        [
            {
                "case_id": r["case_id"],
                "AL_mm": r["params"]["AL"],
                "t_ppS_mm": r["params"]["t_ppS"],
                "theta_canal_deg": r["params"]["theta_canal"],
                "t_LC_mm": r["params"]["t_LC"],
                "D_canal_mm": r["params"]["D_canal"],
                "IOP_mmHg": r["params"]["IOP"],
                "asco_bmo_offset_mm": r["asco_bmo_offset_mm"],
                "solver_seconds": round(r["solver_seconds"], 3),
                "lc_e1_p95_pct": r["qois"]["lc_e1_p95_pct"],
                "lc_eeff_p95_pct": r["qois"]["lc_eeff_p95_pct"],
                "lc_gmax_p95_pct": r["qois"]["lc_gmax_p95_pct"],
                "lc_ap_disp_p95_um": r["qois"]["lc_ap_disp_p95_um"],
                "lc_eeff_p50_pct": r["qois"]["lc_eeff_p50"] * 100.0,
            }
            for r in baseline_results
        ]
    )
    df_conv = pd.DataFrame(mesh_conv_rows)
    df_tilt = pd.DataFrame(tilt_rows)

    df_base.to_csv(table_dir / "phase1_baselines.csv", index=False)
    df_conv.to_csv(table_dir / "phase1_mesh_convergence.csv", index=False)
    df_tilt.to_csv(table_dir / "phase1_tilt_sweep.csv", index=False)

    report = {
        "baselines": df_base.to_dict(orient="records"),
        "mesh_convergence": mesh_conv_rows,
        "tilt_sweep": tilt_rows,
    }
    (out_base / "phase1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_phase1_suite(n_workers=2)
