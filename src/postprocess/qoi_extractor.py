"""
FEBio 输出解析、单元体积加权第 95 百分位力学指标 (QoI) 提取与三维 VTU 场导出模块
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)

强制科学规范 3:
  所有筛板 (LC) 应变与后移指标均采用按单元体积加权的第 95 百分位 (volume-weighted 95th percentile),
  彻底消除孤立边界节点或局部角点网格奇异值对高斯过程代理模型的污染。同时输出第 50 百分位中位值供对照。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import meshio
import numpy as np

from src.common.units import (
    DEG_TO_RAD,
    FORCE_UNIT,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    MM_TO_UM,
    STRESS_UNIT,
    deg_to_rad,
)
from src.febio.feb_builder import DOMAIN_ORDER
from src.geometry.onh_mesh import ONHMeshData

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322
assert DEG_TO_RAD == math.pi / 180.0


def volume_weighted_percentile(values: np.ndarray, weights: np.ndarray, q_percent: float) -> float:
    """
    计算严格连续线性插值的体积加权百分位数 (q_percent in [0, 100])。
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if len(v) == 0 or len(v) != len(w):
        raise ValueError(f"Invalid value/weight lengths: {len(v)} vs {len(w)}")
    if np.any(w <= 0.0):
        raise ValueError("Element volumes/weights must be strictly positive")

    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    # 中点累计权重分位映射: p_i = (cum_w_i - 0.5 * w_i) / total_w
    mid_p = (cum_w - 0.5 * w_sorted) / total_w
    target_p = np.clip(q_percent / 100.0, 0.0, 1.0)
    return float(np.interp(target_p, mid_p, v_sorted))


def check_febio_normal_termination(log_path: str | Path) -> Tuple[bool, str]:
    """检查 FEBio 主日志文件是否包含 N O R M A L   T E R M I N A T I O N。"""
    p = Path(log_path)
    if not p.exists():
        return False, "missing_log_file"
    text = p.read_text(encoding="utf-8", errors="replace")
    if "N O R M A L   T E R M I N A T I O N" in text:
        return True, "NORMAL_TERMINATION"
    if "E R R O R   T E R M I N A T I O N" in text:
        # 抓取最后几行错误提示方便诊断
        err_lines = [ln.strip() for ln in text.splitlines() if ("ERROR" in ln or "Fatal" in ln or "Negative jacobian" in ln)]
        reason = "; ".join(err_lines[-3:]) if err_lines else "ERROR_TERMINATION"
        return False, reason[:240]
    return False, "incomplete_or_unknown_termination"


def parse_febio_data_record(record_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    解析 FEBio <logfile> 输出的最后一个时间步数据块。
    FEBio 数据记录格式:
      *Step  = ...
      *Time  = 1
      *Data  = E1;E2;E3;effective strain;max shear strain
      id,val1,val2,...
    返回:
      ids: (N_items,) int64
      vals: (N_items, N_cols) float64
    """
    p = Path(record_path)
    if not p.exists():
        raise FileNotFoundError(f"FEBio data record not found: {p}")

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: List[List[str]] = []
    cur_block: List[str] = []

    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if s.startswith("*Step"):
            if cur_block:
                blocks.append(cur_block)
                cur_block = []
        elif s.startswith("*") or s.startswith("#"):
            continue
        else:
            cur_block.append(s)
    if cur_block:
        blocks.append(cur_block)

    if not blocks:
        raise ValueError(f"No numerical data rows found in {p}")

    last_block = blocks[-1]
    ids = np.empty(len(last_block), dtype=np.int64)
    first_parts = [tok.strip() for tok in last_block[0].replace(",", " ").split() if tok.strip()]
    n_cols = len(first_parts) - 1
    vals = np.empty((len(last_block), n_cols), dtype=float)

    for i, row_str in enumerate(last_block):
        parts = [tok.strip() for tok in row_str.split(",") if tok.strip()]
        if len(parts) != n_cols + 1:
            parts = [tok.strip() for tok in row_str.replace(",", " ").split() if tok.strip()]
        ids[i] = int(parts[0])
        vals[i] = [float(x) for x in parts[1:]]

    return ids, vals


def extract_case_qois(
    run_dir: str | Path,
    mesh: ONHMeshData,
    params: Dict[str, float],
    export_vtu_path: Optional[str | Path] = None,
) -> Dict[str, float]:
    """
    从已收敛的 FEBio 算例目录提取筛板 (LC) 第 95 百分位（及第 50 百分位）力学指标，
    并按需导出完整三维变形与应变场 VTU 文件供 PyVista 云图绘制。
    """
    rdir = Path(run_dir)
    lc_vols = mesh.elem_volumes["LC"]

    # 1. 解析 LC 单元应变: E1, E2, E3, effective strain, max shear strain
    _, lc_strains = parse_febio_data_record(rdir / "lc_elem_strains.csv")
    if len(lc_strains) != len(lc_vols):
        raise ValueError(f"LC element count mismatch: {len(lc_strains)} vs {len(lc_vols)}")

    e1 = lc_strains[:, 0]
    e3 = lc_strains[:, 2]
    e_eff = lc_strains[:, 3]
    g_max = lc_strains[:, 4]

    # 2. 解析节点位移并计算 LC 相对于 BMO 参考平面的轴向后移量 [mm]
    lc_nids, lc_u = parse_febio_data_record(rdir / "lc_nodal_disp.csv")
    bmo_nids, bmo_u = parse_febio_data_record(rdir / "bmo_nodal_disp.csv")

    max_nid = len(mesh.nodes)
    u_lookup = np.zeros((max_nid + 1, 3), dtype=float)
    u_lookup[lc_nids] = lc_u
    u_lookup[bmo_nids] = bmo_u

    u_bmo_mean = np.mean(bmo_u, axis=0)

    # 沿倾斜视神经管后向单位轴 e_post = (sin(theta), 0, -cos(theta)) 投影后移量
    theta_rad = deg_to_rad(float(params["theta_canal"]))
    e_post = np.array([math.sin(theta_rad), 0.0, -math.cos(theta_rad)], dtype=float)

    lc_conn = mesh.domains["LC"]
    lc_elem_u = np.mean(u_lookup[lc_conn], axis=1)  # (N_lc_elem, 3) [mm]
    lc_rel_post_disp_mm = (lc_elem_u - u_bmo_mean) @ e_post
    lc_abs_post_disp_mm = lc_elem_u @ e_post

    qois = {
        # 4 个核心体积加权第 95 百分位 QoI (严格按物理单位存储: 应变无量纲, 位移 mm)
        "lc_e1_p95": volume_weighted_percentile(e1, lc_vols, 95.0),
        "lc_eeff_p95": volume_weighted_percentile(e_eff, lc_vols, 95.0),
        "lc_gmax_p95": volume_weighted_percentile(g_max, lc_vols, 95.0),
        "lc_ap_disp_p95": volume_weighted_percentile(lc_rel_post_disp_mm, lc_vols, 95.0),
        # 辅助第 50 百分位中位值与压缩第 5 百分位主应变 (供生理机制深度剖析与基准双校核)
        "lc_e1_p50": volume_weighted_percentile(e1, lc_vols, 50.0),
        "lc_eeff_p50": volume_weighted_percentile(e_eff, lc_vols, 50.0),
        "lc_gmax_p50": volume_weighted_percentile(g_max, lc_vols, 50.0),
        "lc_ap_disp_p50": volume_weighted_percentile(lc_rel_post_disp_mm, lc_vols, 50.0),
        "lc_e3_p05": volume_weighted_percentile(e3, lc_vols, 5.0),
        # 常用展示百分比 / 微米派生键 (仅作可读辅助，GP 训练统一读取无量纲/mm 原值)
        "lc_e1_p95_pct": volume_weighted_percentile(e1, lc_vols, 95.0) * 100.0,
        "lc_eeff_p95_pct": volume_weighted_percentile(e_eff, lc_vols, 95.0) * 100.0,
        "lc_gmax_p95_pct": volume_weighted_percentile(g_max, lc_vols, 95.0) * 100.0,
        "lc_ap_disp_p95_um": volume_weighted_percentile(lc_rel_post_disp_mm, lc_vols, 95.0) * MM_TO_UM,
        "lc_abs_post_disp_p95_um": volume_weighted_percentile(lc_abs_post_disp_mm, lc_vols, 95.0) * MM_TO_UM,
    }

    if export_vtu_path is not None and (rdir / "full_nodal_disp.csv").exists():
        _export_full_vtu(rdir, mesh, Path(export_vtu_path))

    return qois


def _export_full_vtu(run_dir: Path, mesh: ONHMeshData, vtu_path: Path) -> None:
    """将全模型变形前后几何、各解剖域标签、应变/应力张量与环向纤维向量导出为 VTU 文件。"""
    nids, all_u = parse_febio_data_record(run_dir / "full_nodal_disp.csv")
    u_full = np.zeros_like(mesh.nodes)
    u_full[nids - 1] = all_u

    conn_list = []
    dom_ids = []
    e1_list, eeff_list, gmax_list, s_eff_list = [], [], [], []
    fib_a0_list = []

    for d_idx, dname in enumerate(DOMAIN_ORDER, start=1):
        conn0 = mesh.domains[dname] - 1  # 0-based for VTK/meshio
        conn_list.append(conn0)
        dom_ids.append(np.full(len(conn0), d_idx, dtype=np.int32))

        _, d_vals = parse_febio_data_record(run_dir / f"full_elem_{dname}.csv")
        e1_list.append(d_vals[:, 0] * 100.0)
        eeff_list.append(d_vals[:, 3] * 100.0)
        gmax_list.append(d_vals[:, 4] * 100.0)
        s_eff_list.append(d_vals[:, 8] * 1000.0)  # MPa -> kPa 便于云图显示

        if dname in mesh.mat_axes:
            fib_a0_list.append(mesh.mat_axes[dname][0])
        else:
            fib_a0_list.append(np.zeros((len(conn0), 3), dtype=float))

    cells = [("hexahedron", np.vstack(conn_list))]
    point_data = {
        "disp_mm": u_full,
        "disp_um": u_full * MM_TO_UM,
        "disp_mag_um": np.linalg.norm(u_full, axis=1) * MM_TO_UM,
    }
    cell_data = {
        "domain_id": [np.concatenate(dom_ids)],
        "E1_pct": [np.concatenate(e1_list)],
        "E_eff_pct": [np.concatenate(eeff_list)],
        "gamma_max_pct": [np.concatenate(gmax_list)],
        "von_mises_stress_kPa": [np.concatenate(s_eff_list)],
        "fiber_a0": [np.vstack(fib_a0_list)],
    }

    vtu_path.parent.mkdir(parents=True, exist_ok=True)
    m_out = meshio.Mesh(
        points=mesh.nodes,
        cells=cells,
        point_data=point_data,
        cell_data=cell_data,
    )
    m_out.write(str(vtu_path))
