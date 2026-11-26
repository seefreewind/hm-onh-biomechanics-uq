"""
Phase 2 批量有限元仿真调度与结果汇总流水线
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)

支持模式:
  - smoke          : 运行 20 组 LHS 试算例 (Smoke Test)，输出收敛率与单算例墙钟吞吐评估
  - train_and_test : 运行完整 400 训练集 + 80 独立测试集并行仿真，输出真实 QoI 数据表
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.common.units import FORCE_UNIT, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT, load_yaml_config
from src.febio.runner import run_batch_fem_cases
from src.surrogate.lhs_sampling import generate_all_lhs_datasets, get_parameter_specs

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_QOIS = ("lc_e1_p95", "lc_eeff_p95", "lc_gmax_p95", "lc_ap_disp_p95")
AUX_QOIS = ("lc_e1_p50", "lc_eeff_p50", "lc_gmax_p50", "lc_ap_disp_p50", "lc_e3_p05")


def execute_lhs_split(
    split_name: str,
    lhs_csv_path: Path,
    n_workers: int = 2,
    mesh_level: str = "medium",
) -> Dict[str, Any]:
    if not lhs_csv_path.exists():
        generate_all_lhs_datasets()

    df_in = pd.read_csv(lhs_csv_path)
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    param_keys, _, _ = get_parameter_specs(params_cfg)

    runs_root = PROJECT_ROOT / "data" / "sim_runs" / f"phase2_{split_name}"
    runs_root.mkdir(parents=True, exist_ok=True)

    task_specs: List[Dict[str, Any]] = []
    for _, row in df_in.iterrows():
        cid = str(row["case_id"])
        p_dict = {k: float(row[k]) for k in param_keys}
        task_specs.append(
            {
                "case_id": cid,
                "params": p_dict,
                "run_dir": runs_root / cid,
                "phase": f"phase2_{split_name}",
                "mesh_level": mesh_level,
                "export_full_field": False,
                "cleanup_feb_on_success": True,
            }
        )

    t_wall0 = time.perf_counter()
    results = run_batch_fem_cases(task_specs, n_workers=n_workers)
    wall_seconds = time.perf_counter() - t_wall0

    out_rows: List[Dict[str, Any]] = []
    n_converged = 0
    n_retried = 0

    for res in results:
        row_out: Dict[str, Any] = {
            "case_id": res["case_id"],
            "converged": bool(res["converged"]),
            "attempts": int(res["attempts"]),
            "schedule": res["schedule"],
            "solver_seconds": round(float(res["solver_seconds"]), 3),
        }
        for k in param_keys:
            row_out[k] = float(res["params"][k])

        if res["converged"]:
            n_converged += 1
            if int(res["attempts"]) > 1:
                n_retried += 1
            q = res["qois"]
            for qk in CORE_QOIS + AUX_QOIS:
                row_out[qk] = float(q[qk])
        else:
            for qk in CORE_QOIS + AUX_QOIS:
                row_out[qk] = float("nan")
            row_out["error_reason"] = res.get("error_reason", "FAILED")

        out_rows.append(row_out)

    df_out = pd.DataFrame(out_rows)
    proc_dir = PROJECT_ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    out_csv = proc_dir / f"sim_results_{split_name}.csv"
    df_out.to_csv(out_csv, index=False)

    mean_solver_s = float(df_out["solver_seconds"].mean())
    eff_sec_per_case = wall_seconds / max(len(df_out), 1)
    stats = {
        "split": split_name,
        "n_total": len(df_out),
        "n_converged": n_converged,
        "convergence_rate": n_converged / max(len(df_out), 1),
        "n_retried_and_recovered": n_retried,
        "n_workers": n_workers,
        "wall_seconds": round(wall_seconds, 2),
        "mean_single_solver_seconds": round(mean_solver_s, 2),
        "effective_wall_seconds_per_case": round(eff_sec_per_case, 2),
        "estimated_480_cases_minutes": round(480.0 * eff_sec_per_case / 60.0, 1),
        "output_csv": str(out_csv),
    }
    (proc_dir / f"batch_stats_{split_name}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(f"[Batch {split_name}] Completed: {json.dumps(stats, ensure_ascii=False)}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 FEBio batch campaign")
    parser.add_argument(
        "--mode",
        choices=("smoke", "train", "test", "train_and_test"),
        default="smoke",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--mesh-level", type=str, default="medium")
    args = parser.parse_args()

    lhs_dir = PROJECT_ROOT / "data" / "lhs_samples"
    if args.mode == "smoke":
        execute_lhs_split("smoke", lhs_dir / "lhs_smoke_20.csv", n_workers=args.workers, mesh_level=args.mesh_level)
    elif args.mode == "train":
        execute_lhs_split("train", lhs_dir / "lhs_train_400.csv", n_workers=args.workers, mesh_level=args.mesh_level)
    elif args.mode == "test":
        execute_lhs_split("test", lhs_dir / "lhs_test_80.csv", n_workers=args.workers, mesh_level=args.mesh_level)
    elif args.mode == "train_and_test":
        execute_lhs_split("train", lhs_dir / "lhs_train_400.csv", n_workers=args.workers, mesh_level=args.mesh_level)
        execute_lhs_split("test", lhs_dir / "lhs_test_80.csv", n_workers=args.workers, mesh_level=args.mesh_level)


if __name__ == "__main__":
    main()
