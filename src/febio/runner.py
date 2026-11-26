"""
FEBio 4.x 单算例自适应步长重试执行器与多进程批量并行调度器
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)

强制工程规范 2:
  严禁静默跳过或插值伪造未收敛算例！
  任何一次尝试若未正常收敛 (NORMAL TERMINATION)，立即将失败详情追加写入 `logs/failed_cases.csv`，
  并按 `primary -> retry_1 -> retry_2` 三阶自适应步长与刚度重构策略自动重试。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.units import FORCE_UNIT, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT, load_yaml_config
from src.febio.feb_builder import write_feb_file
from src.geometry.onh_mesh import generate_onh_mesh
from src.postprocess.qoi_extractor import check_febio_normal_termination, extract_case_qois

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEDULES_ORDER = ("primary", "retry_1", "retry_2")


def _append_failed_case_log(
    failed_csv_path: Path,
    case_id: str,
    phase: str,
    attempt: int,
    schedule_name: str,
    return_code: int,
    term_status: str,
    error_reason: str,
    params: Dict[str, float],
) -> None:
    failed_csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failed_csv_path.exists() or failed_csv_path.stat().st_size == 0
    with failed_csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "case_id",
                    "phase",
                    "attempt",
                    "solver_schedule",
                    "return_code",
                    "termination_status",
                    "error_reason",
                    "timestamp",
                    "parameters_json",
                ]
            )
        writer.writerow(
            [
                case_id,
                phase,
                attempt,
                schedule_name,
                return_code,
                term_status,
                error_reason,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(params, sort_keys=True),
            ]
        )


def resolve_febio_executable(sim_cfg: dict) -> Path:
    exe_str = sim_cfg["febio"]["executable"]
    p = Path(exe_str)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if p.exists():
        return p
    which_p = shutil.which(exe_str)
    if which_p:
        return Path(which_p)
    raise FileNotFoundError(f"FEBio executable not found at {p} or in PATH")


def run_single_fem_case(
    case_id: str,
    params: Dict[str, float],
    run_dir: str | Path,
    phase: str = "phase1",
    mesh_level: Optional[str] = None,
    export_full_field: bool = False,
    export_vtu_path: Optional[str | Path] = None,
    cleanup_feb_on_success: bool = False,
    omp_threads: int = 1,
) -> Dict[str, Any]:
    """
    执行单个参数化 3D ONH 有限元算例:
      Gmsh 网格生成 + 雅可比门禁 -> .feb 生成 -> FEBio 求解 (含三阶自适应重试) -> 体积加权第 95 百分位 QoI 提取。
    """
    t0 = time.perf_counter()
    sim_cfg = load_yaml_config(PROJECT_ROOT / "config" / "sim_settings.yaml")
    febio_exe = resolve_febio_executable(sim_cfg)
    failed_csv = PROJECT_ROOT / sim_cfg["febio"]["failed_cases_log"]
    timeout_sec = int(sim_cfg["febio"]["timeout_seconds"])

    rdir = Path(run_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    # 1. 生成三维网格并校验无负雅可比
    t_mesh0 = time.perf_counter()
    mesh = generate_onh_mesh(params, sim_cfg=sim_cfg, mesh_level=mesh_level, use_cache=True)
    mesh_sec = time.perf_counter() - t_mesh0

    env = os.environ.copy()
    n_thr = str(max(1, int(omp_threads)))
    env["OMP_NUM_THREADS"] = n_thr
    env["VECLIB_MAXIMUM_THREADS"] = n_thr

    converged = False
    used_schedule = ""
    attempts = 0
    last_reason = ""
    solver_sec = 0.0

    for attempt_idx, sched_name in enumerate(SCHEDULES_ORDER, start=1):
        attempts = attempt_idx
        used_schedule = sched_name

        # 清理上一轮残留输出记录文件
        for stale in rdir.glob("*.csv"):
            if not stale.name.startswith("._"):
                stale.unlink(missing_ok=True)

        feb_path = rdir / "model.feb"
        log_path = rdir / "model.log"
        write_feb_file(
            mesh=mesh,
            params=params,
            feb_path=feb_path,
            sim_cfg=sim_cfg,
            schedule_name=sched_name,
            export_full_field=export_full_field,
        )

        cmd = [
            str(febio_exe),
            "-i",
            str(feb_path),
            "-o",
            str(log_path),
            "-silent",
            "-nosplash",
        ]
        t_sol0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(rdir),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            ret_code = int(proc.returncode)
        except subprocess.TimeoutExpired:
            ret_code = -999
            proc = None
        solver_sec += time.perf_counter() - t_sol0

        ok, reason = check_febio_normal_termination(log_path)
        if ret_code == 0 and ok:
            converged = True
            last_reason = "NORMAL_TERMINATION"
            break

        stderr_tail = ""
        if proc is not None and proc.stderr:
            stderr_tail = proc.stderr.strip().splitlines()[-1][:120] if proc.stderr.strip() else ""
        full_reason = f"{reason} | stderr={stderr_tail}".strip(" |")
        last_reason = full_reason

        _append_failed_case_log(
            failed_csv_path=failed_csv,
            case_id=case_id,
            phase=phase,
            attempt=attempt_idx,
            schedule_name=sched_name,
            return_code=ret_code,
            term_status="FAILED",
            error_reason=full_reason,
            params=params,
        )

    if not converged:
        return {
            "case_id": case_id,
            "converged": False,
            "attempts": attempts,
            "schedule": used_schedule,
            "error_reason": last_reason,
            "mesh_seconds": mesh_sec,
            "solver_seconds": solver_sec,
            "total_seconds": time.perf_counter() - t0,
            "params": dict(params),
        }

    qois = extract_case_qois(
        run_dir=rdir,
        mesh=mesh,
        params=params,
        export_vtu_path=export_vtu_path,
    )

    if cleanup_feb_on_success:
        (rdir / "model.feb").unlink(missing_ok=True)

    total_sec = time.perf_counter() - t0
    summary = {
        "case_id": case_id,
        "converged": True,
        "attempts": attempts,
        "schedule": used_schedule,
        "mesh_level": mesh.mesh_level,
        "total_nodes": mesh.quality.total_nodes,
        "total_elements": mesh.quality.total_elements,
        "min_scaled_jacobian": mesh.quality.min_scaled_jacobian,
        "asco_bmo_offset_mm": mesh.quality.asco_bmo_offset_mm,
        "posterior_inner_radius_mm": mesh.quality.posterior_inner_radius_mm,
        "mesh_seconds": mesh_sec,
        "solver_seconds": solver_sec,
        "total_seconds": total_sec,
        "params": dict(params),
        "qois": qois,
    }
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _batch_worker(task_spec: Dict[str, Any]) -> Dict[str, Any]:
    return run_single_fem_case(
        case_id=task_spec["case_id"],
        params=task_spec["params"],
        run_dir=task_spec["run_dir"],
        phase=task_spec["phase"],
        mesh_level=task_spec.get("mesh_level"),
        export_full_field=task_spec.get("export_full_field", False),
        cleanup_feb_on_success=task_spec.get("cleanup_feb_on_success", True),
        omp_threads=1,
    )


def run_batch_fem_cases(
    cases: List[Dict[str, Any]],
    n_workers: int = 6,
) -> List[Dict[str, Any]]:
    """
    多进程并行批量调度器（每个子进程固定 OMP_NUM_THREADS=1 防止线程争用）。
    支持断点续跑：若目标 run_dir/summary.json 已存在且 converged==True，则直接复用真实结果。
    """
    results_by_id: Dict[str, Dict[str, Any]] = {}
    pending_tasks: List[Dict[str, Any]] = []

    for spec in cases:
        cid = spec["case_id"]
        summary_file = Path(spec["run_dir"]) / "summary.json"
        if summary_file.exists() and not spec.get("force_rerun", False):
            cached = json.loads(summary_file.read_text(encoding="utf-8"))
            if cached.get("converged") is True:
                results_by_id[cid] = cached
                continue
        pending_tasks.append(spec)

    if pending_tasks:
        n_pend = len(pending_tasks)
        done_cnt = 0
        t_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as pool:
            fut_map = {pool.submit(_batch_worker, sp): sp["case_id"] for sp in pending_tasks}
            for fut in as_completed(fut_map):
                cid = fut_map[fut]
                results_by_id[cid] = fut.result()
                done_cnt += 1
                if done_cnt % 25 == 0 or done_cnt == n_pend:
                    el = time.perf_counter() - t_start
                    rate = el / done_cnt
                    rem_min = (n_pend - done_cnt) * rate / 60.0
                    print(
                        f"  [Progress] {done_cnt}/{n_pend} done | elapsed={el/60.0:.1f}m | est_rem={rem_min:.1f}m",
                        flush=True,
                    )

    return [results_by_id[sp["case_id"]] for sp in cases]
