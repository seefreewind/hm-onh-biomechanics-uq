"""
Phase 1 基线验证云图、三级网格收敛性与视神经管倾斜角响应绘图脚本
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import pyvista as pv

from src.common.units import FORCE_UNIT, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _extract_2d_polys_from_pv_polydata(
    poly: pv.PolyData,
    scalar_name: str,
    axes_idx: Tuple[int, int],
    center_shift: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[List[np.ndarray], np.ndarray]:
    pts = np.asarray(poly.points, dtype=float)
    faces = poly.faces
    scalars = np.asarray(poly.cell_data[scalar_name], dtype=float)

    polys_2d: List[np.ndarray] = []
    valid_scalars: List[float] = []
    i = 0
    cell_idx = 0
    n_len = len(faces)
    ax0, ax1 = axes_idx
    cx, cy = center_shift

    while i < n_len:
        nv = int(faces[i])
        v_ids = faces[i + 1 : i + 1 + nv]
        xy = np.column_stack([pts[v_ids, ax0] - cx, pts[v_ids, ax1] - cy])
        polys_2d.append(xy)
        valid_scalars.append(float(scalars[cell_idx]))
        i += nv + 1
        cell_idx += 1

    return polys_2d, np.asarray(valid_scalars, dtype=float)


def _extract_enface_hex_top_quads(
    ug: pv.UnstructuredGrid,
    scalar_name: str,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    提取三维六面体网格前表面层单元的俯视 (X-Y) 四边形投影，完整展示圆盘形筛板与 360° 环向纤维。
    针对同一 (X,Y) 柱上的多层厚度单元，取每个径向—周向柱的最大值代表局部峰值特征，或取最前层几何投影。
    """
    pts = np.asarray(ug.points, dtype=float)
    # VTK UnstructuredGrid cells array: [8, n0..n7, 8, ...]
    cells_raw = ug.cells.reshape(-1, 9)[:, 1:]
    vals = np.asarray(ug.cell_data[scalar_name], dtype=float)
    fibs = np.asarray(ug.cell_data["fiber_a0"], dtype=float)

    # 按单元中心 Z 选取最靠近筛板中前层的一层单元避免多层重叠遮挡
    centers = pts[cells_raw].mean(axis=1)
    z_med = float(np.percentile(centers[:, 2], 75.0))
    z_span = max(float(np.ptp(centers[:, 2])), 1e-3)
    layer_mask = np.abs(centers[:, 2] - z_med) <= 0.22 * z_span
    if np.sum(layer_mask) < 20:
        layer_mask = np.ones(len(cells_raw), dtype=bool)

    sel_cells = cells_raw[layer_mask]
    sel_vals = vals[layer_mask]
    sel_fibs = fibs[layer_mask]
    sel_centers = centers[layer_mask]

    polys_xy: List[np.ndarray] = []
    for c_nodes in sel_cells:
        # 取底面 (0,1,2,3) 的 X-Y 投影多边形
        quad_xy = pts[c_nodes[:4], :2]
        polys_xy.append(quad_xy)

    return polys_xy, sel_vals, sel_centers[:, :2], sel_fibs[:, :2]


def generate_phase1_verification_figure() -> Path:
    vtu_dir = PROJECT_ROOT / "data" / "processed" / "phase1_vtu"
    table_dir = PROJECT_ROOT / "results" / "tables"
    fig_dir = PROJECT_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    emme_grid = pv.read(str(vtu_dir / "emmetropia_iop15.vtu"))
    hm_grid = pv.read(str(vtu_dir / "high_myopia_tilted_iop15.vtu"))

    df_conv = pd.read_csv(table_dir / "phase1_mesh_convergence.csv")
    df_tilt = pd.read_csv(table_dir / "phase1_tilt_sweep.csv")
    df_base = pd.read_csv(table_dir / "phase1_baselines.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.2,
            "figure.dpi": 150,
        }
    )

    fig = plt.figure(figsize=(15.6, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 1.0], hspace=0.36, wspace=0.36)

    def _get_onh_subgrid(ug: pv.UnstructuredGrid, radius_mm: float = 2.6) -> pv.UnstructuredGrid:
        warped = ug.copy(deep=True)
        warped.points = warped.points + 5.0 * np.asarray(warped.point_data["disp_mm"])
        centers = warped.cell_centers().points
        mask = np.hypot(centers[:, 0], centers[:, 1]) <= radius_mm
        return warped.extract_cells(mask)

    emme_onh = _get_onh_subgrid(emme_grid, radius_mm=2.6)
    hm_onh = _get_onh_subgrid(hm_grid, radius_mm=2.9)

    vmax_eeff = max(
        float(np.percentile(emme_onh.cell_data["E_eff_pct"], 98.5)),
        float(np.percentile(hm_onh.cell_data["E_eff_pct"], 98.5)),
    )
    vmax_eeff = max(round(vmax_eeff, 2), 1.5)

    # ------------------------------------------------------------------
    # Panel A & B: 鼻—颞向中心子午切面 (Y = 0, 5x 变形放大)
    # ------------------------------------------------------------------
    for col_idx, (title_str, sub_ug) in enumerate(
        [
            (
                "A. Emmetropia (AL=24 mm, Tilt=3°, 15 mmHg)\n"
                "Y=0 Cross-Section (5× Deformed, Cyan=LC)",
                emme_onh,
            ),
            (
                "B. High Myopia (AL=29.5 mm, Tilt=25°, 15 mmHg)\n"
                "Y=0 Cross-Section (5× Deformed, Cyan=LC)",
                hm_onh,
            ),
        ]
    ):
        ax = fig.add_subplot(gs[0, col_idx])
        slc = sub_ug.slice(normal=[0.0, 1.0, 0.0], origin=[0.0, 0.0, 0.0])
        z_zero = float(np.percentile(slc.points[:, 2], 98.0))
        polys, vals = _extract_2d_polys_from_pv_polydata(
            slc, "E_eff_pct", axes_idx=(0, 2), center_shift=(0.0, z_zero)
        )
        pc = PolyCollection(
            polys,
            array=vals,
            cmap="inferno",
            clim=(0.0, vmax_eeff),
            edgecolors=(0.15, 0.15, 0.15, 0.18),
            linewidths=0.25,
        )
        ax.add_collection(pc)

        lc_cells = sub_ug.extract_cells(np.asarray(sub_ug.cell_data["domain_id"]) == 2)
        lc_slc = lc_cells.slice(normal=[0.0, 1.0, 0.0], origin=[0.0, 0.0, 0.0])
        lc_bnd = lc_slc.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        if lc_bnd.n_points > 0:
            lines_arr = lc_bnd.lines
            k = 0
            while k < len(lines_arr):
                npt = int(lines_arr[k])
                ids = lines_arr[k + 1 : k + 1 + npt]
                ax.plot(
                    lc_bnd.points[ids, 0],
                    lc_bnd.points[ids, 2] - z_zero,
                    color="#00f5d4",
                    lw=1.25,
                    alpha=0.95,
                )
                k += npt + 1

        ax.set_xlim(-2.6, 2.8)
        ax.set_ylim(-2.35, 0.28)
        ax.set_aspect("equal")
        ax.set_xlabel("Nasal  ←  X [mm]  →  Temporal")
        ax.set_ylabel("Relative Depth ΔZ [mm]")
        ax.set_title(title_str, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.3)
        cbar = fig.colorbar(pc, ax=ax, fraction=0.040, pad=0.03, shrink=0.72)
        cbar.set_label("Effective Strain E_eff [%]")

    # ------------------------------------------------------------------
    # Panel C: 高度近视 筛板完整圆盘俯视第一主应变 E1 + 360° 视盘周巩膜环向胶原纤维矢量场
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    hm_lc = hm_grid.extract_cells(np.asarray(hm_grid.cell_data["domain_id"]) == 2)
    lc_polys, lc_e1_vals, lc_centers_xy, _ = _extract_enface_hex_top_quads(hm_lc, "E1_pct")
    cx0, cy0 = float(np.mean(lc_centers_xy[:, 0])), float(np.mean(lc_centers_xy[:, 1]))
    lc_polys_shifted = [p - np.array([cx0, cy0]) for p in lc_polys]

    vmax_e1 = max(float(np.percentile(lc_e1_vals, 99.0)), 1.0)
    pc_c = PolyCollection(
        lc_polys_shifted,
        array=lc_e1_vals,
        cmap="viridis",
        clim=(float(np.min(lc_e1_vals)) * 0.9, vmax_e1),
        edgecolors=(1.0, 1.0, 1.0, 0.22),
        linewidths=0.30,
    )
    ax_c.add_collection(pc_c)

    hm_pps = hm_grid.extract_cells(np.asarray(hm_grid.cell_data["domain_id"]) == 5)
    _, _, pps_xy, pps_fib_xy = _extract_enface_hex_top_quads(hm_pps, "E1_pct")
    # 归一化 2D 投影切向量并均匀抽样绘制 360° 环向纤维保护环
    norms_2d = np.linalg.norm(pps_fib_xy, axis=1, keepdims=True)
    pps_fib_dir = pps_fib_xy / np.maximum(norms_2d, 1e-12)
    stride = max(1, len(pps_xy) // 72)
    ax_c.quiver(
        pps_xy[::stride, 0] - cx0,
        pps_xy[::stride, 1] - cy0,
        pps_fib_dir[::stride, 0],
        pps_fib_dir[::stride, 1],
        color="#d90429",
        scale=16,
        width=0.0045,
        headwidth=3.2,
        alpha=0.82,
        label="PPS Circumferential Fibers",
    )

    ax_c.set_xlim(-2.35, 2.35)
    ax_c.set_ylim(-2.35, 2.35)
    ax_c.set_aspect("equal")
    ax_c.set_xlabel("Nasal  ←  ΔX [mm]  →  Temporal")
    ax_c.set_ylabel("Inferior  ←  ΔY [mm]  →  Superior")
    ax_c.set_title(
        "C. High-Myopia En-Face LC Tensile Strain\n& 360° PPS Circumferential Fiber Ring",
        fontweight="bold",
        pad=8,
    )
    ax_c.legend(loc="upper right", fontsize=7.8, framealpha=0.9)
    ax_c.grid(True, linestyle=":", alpha=0.3)
    cbar_c = fig.colorbar(pc_c, ax=ax_c, fraction=0.040, pad=0.03)
    cbar_c.set_label("LC 1st Principal Strain E1 [%]")

    # ------------------------------------------------------------------
    # Panel D: 三级网格收敛性验证 (Coarse / Medium / Fine)
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    for grp, label_str, marker, col in (
        ("emmetropia_iop15", "Emmetropia (15 mmHg)", "o", "#1d3557"),
        ("high_myopia_tilted_iop15", "High Myopia Tilted (15 mmHg)", "s", "#e63946"),
    ):
        sub_df = df_conv[df_conv["group"] == grp].copy()
        fine_val = float(sub_df[sub_df["mesh_level"] == "fine"]["lc_eeff_p95_pct"].iloc[0])
        ax_d.plot(
            sub_df["elements"],
            sub_df["lc_eeff_p95_pct"],
            marker=marker,
            lw=2.0,
            color=col,
            label=label_str,
        )
        ax_d.axhspan(fine_val * 0.98, fine_val * 1.02, color=col, alpha=0.11)
        for _, row in sub_df.iterrows():
            ax_d.annotate(
                f"{row['mesh_level']}\n({row['rel_diff_eeff_vs_fine_pct']:.2f}%)",
                (row["elements"], row["lc_eeff_p95_pct"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7.8,
            )

    ax_d.set_ylim(0.85, 5.75)
    ax_d.set_xlim(1500, 24800)
    ax_d.set_xlabel("Total Hexahedral Elements (hex8)")
    ax_d.set_ylabel("LC 95th-Percentile E_eff [%]")
    ax_d.set_title("D. 3-Level Mesh Convergence (< 2% Gate Band)", fontweight="bold", pad=8)
    ax_d.legend(loc="center right", fontsize=8.0)
    ax_d.grid(True, linestyle="--", alpha=0.35)

    # ------------------------------------------------------------------
    # Panel E: 视神经管倾斜角扫描响应 (theta_canal in [0, 35] deg)
    # ------------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.plot(
        df_tilt["theta_canal_deg"],
        df_tilt["lc_eeff_p95_pct"],
        "o-",
        color="#d62828",
        lw=2.0,
        label="E_eff (95th %)",
    )
    ax_e.plot(
        df_tilt["theta_canal_deg"],
        df_tilt["lc_gmax_p95_pct"],
        "^:",
        color="#2a9d8f",
        lw=1.8,
        label="Max Shear γ_max (95th %)",
    )
    ax_e.plot(
        df_tilt["theta_canal_deg"],
        df_tilt["lc_e1_p95_pct"],
        "s--",
        color="#003049",
        lw=1.8,
        label="Tensile E1 (95th %)",
    )
    ax_e.set_xlabel("Neural Canal Tilt Angle θ_canal [deg]")
    ax_e.set_ylabel("LC 95th-Percentile Strain [%]")
    ax_e.set_title("E. Optic Canal Tilt (0°–35°) vs LC Strain", fontweight="bold", pad=8)
    ax_e.legend(loc="upper left", fontsize=8.0)
    ax_e.grid(True, linestyle="--", alpha=0.35)

    # ------------------------------------------------------------------
    # Panel F: 双基线对比柱状图 + 活体 OCT 文献区间
    # ------------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    labels_f = ["Emme\n15 mmHg", "Emme\n21 mmHg", "High Myopia\n15 mmHg", "High Myopia\n21 mmHg"]
    x_pos = np.arange(len(labels_f))
    w_bar = 0.26

    ax_f.bar(x_pos - w_bar, df_base["lc_e1_p95_pct"], width=w_bar, color="#457b9d", label="E1 (95th)")
    ax_f.bar(x_pos, df_base["lc_eeff_p95_pct"], width=w_bar, color="#e63946", label="E_eff (95th)")
    ax_f.bar(x_pos + w_bar, df_base["lc_gmax_p95_pct"], width=w_bar, color="#2a9d8f", label="γ_max (95th)")
    ax_f.axhspan(1.0, 6.0, color="#ffb703", alpha=0.14, label="In-vivo OCT Strain Range")

    ax_f.set_xticks(x_pos)
    ax_f.set_xticklabels(labels_f, fontsize=8.4)
    ax_f.set_ylabel("LC 95th-Percentile Strain [%]")
    ax_f.set_title("F. Baselines vs In-Vivo OCT Literature Range", fontweight="bold", pad=8)
    ax_f.legend(loc="upper left", fontsize=7.8)
    ax_f.grid(True, axis="y", linestyle="--", alpha=0.35)

    png_out = fig_dir / "phase1_baseline_and_verification.png"
    svg_out = fig_dir / "phase1_baseline_and_verification.svg"
    fig.savefig(png_out, dpi=350, bbox_inches="tight")
    fig.savefig(svg_out, bbox_inches="tight")
    plt.close(fig)
    return png_out


if __name__ == "__main__":
    out_p = generate_phase1_verification_figure()
    print(f"Saved Phase 1 verification figure to: {out_p}")
