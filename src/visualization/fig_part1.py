"""
出版级论文主图生成器 (Part 1: Figure 1 - Figure 3)
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
遵循科研出版级 Matplotlib 样式规范 (350 DPI PNG + Vector PDF + Editable SVG, TrueType 字体嵌入)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.special import i0 as bessel_i0

from src.common.units import FORCE_UNIT, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT
from src.surrogate.gp_model import CORE_QOIS, MultiQoIGPSurrogateBundle

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_sec": "#3775BA",
    "teal": "#2A9D8F",
    "green_mid": "#52B788",
    "red_strong": "#B64342",
    "red_light": "#E9A6A1",
    "gold": "#E9C46A",
    "slate": "#4D4D4D",
    "gray_light": "#D9D9D9",
    "purple": "#6D597A",
}


def apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.6,
            "axes.linewidth": 1.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_pub_fig(fig: plt.Figure, out_dir: Path, stem: str) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / f"{stem}.pdf",
        out_dir / f"{stem}.png",
        out_dir / f"{stem}.svg",
    ]
    fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths[1], dpi=350, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return paths


def _slice_polys_2d(
    poly: pv.PolyData,
    scalar_name: str,
    axes_idx: Tuple[int, int],
    shift: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[List[np.ndarray], np.ndarray]:
    pts = np.asarray(poly.points, dtype=float)
    faces = poly.faces
    scalars = np.asarray(poly.cell_data[scalar_name], dtype=float)
    polys: List[np.ndarray] = []
    vals: List[float] = []
    i = 0
    c_idx = 0
    ax0, ax1 = axes_idx
    sx, sy = shift
    while i < len(faces):
        nv = int(faces[i])
        vids = faces[i + 1 : i + 1 + nv]
        polys.append(np.column_stack([pts[vids, ax0] - sx, pts[vids, ax1] - sy]))
        vals.append(float(scalars[c_idx]))
        i += nv + 1
        c_idx += 1
    return polys, np.asarray(vals, dtype=float)


def _enface_hex_layer(
    ug: pv.UnstructuredGrid,
    scalar_name: str,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(ug.points, dtype=float)
    cells_raw = ug.cells.reshape(-1, 9)[:, 1:]
    vals = np.asarray(ug.cell_data[scalar_name], dtype=float)
    fibs = np.asarray(ug.cell_data["fiber_a0"], dtype=float)
    centers = pts[cells_raw].mean(axis=1)
    z_target = float(np.percentile(centers[:, 2], 70.0))
    z_span = max(float(np.ptp(centers[:, 2])), 1e-3)
    mask = np.abs(centers[:, 2] - z_target) <= 0.22 * z_span
    if np.sum(mask) < 20:
        mask = np.ones(len(cells_raw), dtype=bool)
    sel_c = cells_raw[mask]
    polys = [pts[c[:4], :2] for c in sel_c]
    return polys, vals[mask], centers[mask, :2], fibs[mask, :2]


# ==============================================================================
# Figure 1: 参数化三维几何、环向纤维微结构、三级网格收敛与倾斜角验证
# ==============================================================================
def plot_figure1_architecture_and_verification(fig_dir: Path) -> List[Path]:
    apply_pub_style()
    vtu_dir = PROJECT_ROOT / "data" / "processed" / "phase1_vtu"
    tbl_dir = PROJECT_ROOT / "results" / "tables"

    hm_ug = pv.read(str(vtu_dir / "high_myopia_tilted_iop15.vtu"))
    df_conv = pd.read_csv(tbl_dir / "phase1_mesh_convergence.csv")
    df_tilt = pd.read_csv(tbl_dir / "phase1_tilt_sweep.csv")

    fig = plt.figure(figsize=(15.2, 9.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0], hspace=0.34, wspace=0.26)

    # (a) 6 大解剖域子午切面拓扑分区与载荷示意图 (Y = 0)
    ax_a = fig.add_subplot(gs[0, 0])
    centers = hm_ug.cell_centers().points
    onh_sub = hm_ug.extract_cells(np.hypot(centers[:, 0], centers[:, 1]) <= 3.0)
    slc_a = onh_sub.slice(normal=[0.0, 1.0, 0.0], origin=[0.0, 0.0, 0.0])
    z_top_ref = float(np.percentile(slc_a.points[:, 2], 98.0))
    polys_a, dom_vals = _slice_polys_2d(slc_a, "domain_id", axes_idx=(0, 2), shift=(0.0, z_top_ref))

    dom_colors = {
        1: "#A8DADC",  # PLNT
        2: "#E63946",  # LC
        3: "#F4A261",  # RLON
        4: "#6D597A",  # PIA
        5: "#1D3557",  # PPS
        6: "#457B9D",  # PSCL
    }
    face_cols = [dom_colors[int(round(v))] for v in dom_vals]
    pc_a = PolyCollection(
        polys_a,
        facecolors=face_cols,
        edgecolors=(1.0, 1.0, 1.0, 0.35),
        linewidths=0.35,
    )
    ax_a.add_collection(pc_a)

    # 标注 IOP 与 CSFP 方向箭头及倾斜角轴线
    ax_a.annotate(
        "Vitreous IOP\n(10–40 mmHg)",
        xy=(-0.2, -0.18),
        xytext=(-1.95, 0.16),
        fontsize=8.5,
        fontweight="bold",
        color="#0F4D92",
        arrowprops=dict(arrowstyle="->", color="#0F4D92", lw=1.6),
    )
    ax_a.annotate(
        "Subarachnoid CSFP\n(11 mmHg)",
        xy=(0.95, -2.06),
        xytext=(1.92, -1.42),
        fontsize=8.5,
        fontweight="bold",
        color="#6D597A",
        arrowprops=dict(arrowstyle="->", color="#6D597A", lw=1.6),
    )
    # 绘出倾斜轴虚线
    th_r = math.radians(25.0)
    t_line = np.linspace(-0.2, 2.2, 50)
    ax_a.plot(
        0.1 + t_line * math.sin(th_r),
        -0.25 - t_line * math.cos(th_r),
        "--",
        color="#FFD166",
        lw=1.8,
        label="Tilted Canal Axis (θ_canal)",
    )

    legend_handles = [
        Patch(facecolor=dom_colors[2], edgecolor="k", lw=0.5, label="LC (Lamina Cribrosa)"),
        Patch(facecolor=dom_colors[5], edgecolor="k", lw=0.5, label="PPS (Circumf. Fibers)"),
        Patch(facecolor=dom_colors[1], edgecolor="k", lw=0.5, label="PLNT (Prelaminar)"),
        Patch(facecolor=dom_colors[3], edgecolor="k", lw=0.5, label="RLON (Retrolaminar)"),
        Patch(facecolor=dom_colors[4], edgecolor="k", lw=0.5, label="PIA (Pia Mater)"),
        Patch(facecolor=dom_colors[6], edgecolor="k", lw=0.5, label="PSCL (Post. Sclera)"),
    ]
    ax_a.legend(handles=legend_handles, loc="lower left", fontsize=7.6, frameon=True, facecolor="white", framealpha=0.92)
    ax_a.set_xlim(-2.7, 3.30)
    ax_a.set_ylim(-2.45, 0.45)
    ax_a.set_aspect("equal")
    ax_a.set_xlabel("Nasal  ←  X [mm]  →  Temporal")
    ax_a.set_ylabel("Relative Posterior Depth ΔZ [mm]")
    ax_a.set_title("a  Parameterized 3D Tilted ONH Multi-Domain Hex8 Architecture", loc="left", fontweight="bold")

    # (b) O-Grid 共形网格俯视图 + Von Mises 环向胶原纤维角度概率密度
    gs_b = gs[0, 1].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    hm_lc = hm_ug.extract_cells(np.asarray(hm_ug.cell_data["domain_id"]) == 2)
    hm_pps = hm_ug.extract_cells(np.asarray(hm_ug.cell_data["domain_id"]) == 5)
    lc_p, _, lc_cxy, _ = _enface_hex_layer(hm_lc, "domain_id")
    pps_p, _, pps_cxy, pps_fxy = _enface_hex_layer(hm_pps, "domain_id")
    cx, cy = float(np.mean(lc_cxy[:, 0])), float(np.mean(lc_cxy[:, 1]))

    ax_b1.add_collection(
        PolyCollection(
            [p - np.array([cx, cy]) for p in lc_p],
            facecolors="#FCEAEA",
            edgecolors="#B64342",
            linewidths=0.45,
        )
    )
    ax_b1.add_collection(
        PolyCollection(
            [p - np.array([cx, cy]) for p in pps_p],
            facecolors="#EBF2FA",
            edgecolors="#3775BA",
            linewidths=0.30,
        )
    )
    fnorm = np.linalg.norm(pps_fxy, axis=1, keepdims=True)
    fdir = pps_fxy / np.maximum(fnorm, 1e-12)
    stride = max(1, len(pps_cxy) // 64)
    ax_b1.quiver(
        pps_cxy[::stride, 0] - cx,
        pps_cxy[::stride, 1] - cy,
        fdir[::stride, 0],
        fdir[::stride, 1],
        color="#0F4D92",
        scale=15,
        width=0.006,
        headwidth=3.0,
    )
    ax_b1.set_xlim(-2.25, 2.25)
    ax_b1.set_ylim(-2.25, 2.25)
    ax_b1.set_aspect("equal")
    ax_b1.set_xlabel("ΔX [mm]")
    ax_b1.set_ylabel("ΔY [mm]")
    ax_b1.set_title("b  O-Grid LC & PPS Fiber Ring", loc="left", fontweight="bold")

    ax_b2 = fig.add_subplot(gs_b[0, 1])
    theta_deg = np.linspace(-90.0, 90.0, 361)
    theta_rad = np.deg2rad(theta_deg)
    for kf_val, col_k, ls_k, lbl_k in (
        (0.4, PALETTE["slate"], ":", r"$b_{\mathrm{fib}} = 0.4$ (Dispersed)"),
        (1.1, PALETTE["red_strong"], "--", r"$b_{\mathrm{fib}} = 1.1$ (High Myopia)"),
        (2.5, PALETTE["blue_main"], "-", r"$b_{\mathrm{fib}} = 2.5$ (Emmetropia)"),
        (4.2, PALETTE["teal"], "-.", r"$b_{\mathrm{fib}} = 4.2$ (Highly Aligned)"),
    ):
        pdf_vm = np.exp(kf_val * np.cos(2.0 * theta_rad)) / (math.pi * float(bessel_i0(kf_val)))
        ax_b2.plot(theta_deg, pdf_vm, color=col_k, ls=ls_k, lw=2.0, label=lbl_k)
    ax_b2.set_xlabel("Deviation from Circumf. Axis [deg]")
    ax_b2.set_ylabel("Von Mises Density $R(\\theta)$ [$\\mathrm{rad}^{-1}$]")
    ax_b2.set_title("Fiber Dispersion Law", fontweight="bold")
    ax_b2.legend(loc="upper right", frameon=False, fontsize=7.2, labelspacing=0.30)
    ax_b2.set_xlim(-90, 90)
    ax_b2.set_ylim(0.0, 2.08)

    # (c) 三级网格收敛性验证
    ax_c = fig.add_subplot(gs[1, 0])
    for grp, lbl_g, mk, col_g in (
        ("emmetropia_iop15", "Emmetropia Baseline (15 mmHg)", "o", PALETTE["blue_main"]),
        ("high_myopia_tilted_iop15", "High-Myopia Tilted Baseline (15 mmHg)", "s", PALETTE["red_strong"]),
    ):
        sub = df_conv[df_conv["group"] == grp].copy()
        fine_v = float(sub[sub["mesh_level"] == "fine"]["lc_eeff_p95_pct"].iloc[0])
        ax_c.plot(sub["elements"], sub["lc_eeff_p95_pct"], marker=mk, ms=7, lw=2.2, color=col_g, label=lbl_g)
        ax_c.axhspan(fine_v * 0.98, fine_v * 1.02, color=col_g, alpha=0.12)
        for _, r in sub.iterrows():
            ax_c.annotate(
                f"{r['mesh_level'].capitalize()}\n({r['rel_diff_eeff_vs_fine_pct']:.2f}%)",
                (r["elements"], r["lc_eeff_p95_pct"]),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=8.2,
            )
    ax_c.set_xlim(1800, 24800)
    ax_c.set_ylim(0.85, 5.75)
    ax_c.set_xlabel("Total Hexahedral Elements (hex8)")
    ax_c.set_ylabel("LC 95th-Percentile Effective Strain [%]")
    ax_c.set_title("c  Three-Level Mesh Convergence (< 2% Acceptance Band Shaded)", loc="left", fontweight="bold")
    ax_c.legend(loc="center right", frameon=False)
    ax_c.grid(True, linestyle=":", alpha=0.35)

    # (d) 视神经管倾斜角鲁棒性与网格质量双轴验证
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(df_tilt["theta_canal_deg"], df_tilt["lc_eeff_p95_pct"], "o-", color=PALETTE["red_strong"], lw=2.2, label=r"LC $\varepsilon_{\mathrm{eff},95}$")
    ax_d.plot(df_tilt["theta_canal_deg"], df_tilt["lc_gmax_p95_pct"], "^--", color=PALETTE["teal"], lw=2.0, label=r"LC $\gamma_{\max,95}$")
    ax_d.plot(df_tilt["theta_canal_deg"], df_tilt["lc_e1_p95_pct"], "s-.", color=PALETTE["blue_main"], lw=2.0, label=r"LC $E_{1,95}$")
    ax_d.set_xlabel(r"Neural Canal Tilt Angle $\theta_{\mathrm{canal}}$ [deg]")
    ax_d.set_ylabel("LC 95th-Percentile Strain [%]")
    ax_d.set_ylim(0.8, 6.5)
    ax_d.grid(True, linestyle=":", alpha=0.35)

    ax_d_r = ax_d.twinx()
    ax_d_r.spines["top"].set_visible(False)
    ax_d_r.plot(
        df_tilt["theta_canal_deg"],
        df_tilt["min_scaled_jacobian"],
        "D:",
        color=PALETTE["slate"],
        lw=1.6,
        alpha=0.85,
        label="Min Scaled Jacobian",
    )
    ax_d_r.axhline(0.20, color=PALETTE["slate"], ls="--", lw=1.0, alpha=0.5)
    ax_d_r.text(
        34.6,
        0.175,
        "Quality gate (0.20)",
        color=PALETTE["slate"],
        fontsize=7.8,
        ha="right",
        va="top",
        style="italic",
    )
    ax_d_r.set_ylabel("Minimum Scaled Jacobian Quality", color=PALETTE["slate"])
    ax_d_r.set_ylim(0.0, 1.08)

    h1, l1 = ax_d.get_legend_handles_labels()
    h2, l2 = ax_d_r.get_legend_handles_labels()
    ax_d.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=7.9,
        columnspacing=1.1,
    )
    ax_d.set_title("d  Mesh Robustness & Strain Amplification Across Canal Tilt (0°–35°)", loc="left", fontweight="bold")

    return save_pub_fig(fig, fig_dir, "Figure1_Model_Architecture_and_Verification")


# ==============================================================================
# Figure 2: 正视眼与高度近视倾斜视盘基线三维力学场对比
# ==============================================================================
def plot_figure2_baseline_biomechanics(fig_dir: Path) -> List[Path]:
    apply_pub_style()
    vtu_dir = PROJECT_ROOT / "data" / "processed" / "phase1_vtu"
    tbl_dir = PROJECT_ROOT / "results" / "tables"

    emme_ug = pv.read(str(vtu_dir / "emmetropia_iop15.vtu"))
    hm_ug = pv.read(str(vtu_dir / "high_myopia_tilted_iop15.vtu"))
    df_base = pd.read_csv(tbl_dir / "phase1_baselines.csv")

    fig = plt.figure(figsize=(15.8, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.36, wspace=0.36)

    def _warped_onh(ug: pv.UnstructuredGrid, r_lim: float = 2.7) -> pv.UnstructuredGrid:
        w = ug.copy(deep=True)
        w.points = w.points + 5.0 * np.asarray(w.point_data["disp_mm"])
        c = w.cell_centers().points
        return w.extract_cells(np.hypot(c[:, 0], c[:, 1]) <= r_lim)

    em_w = _warped_onh(emme_ug, 2.6)
    hm_w = _warped_onh(hm_ug, 2.85)
    vmax_eff = float(np.percentile(hm_w.cell_data["E_eff_pct"], 99.0))

    # (a, b) 正视 vs 高度近视 Y=0 子午切面有效应变云图
    for idx_p, (panel_lbl, sub_w) in enumerate(
        [
            (r"a  Emmetropia Meridional $\varepsilon_{\mathrm{eff}}$ (15 mmHg)", em_w),
            (r"b  High-Myopia Meridional $\varepsilon_{\mathrm{eff}}$ (15 mmHg)", hm_w),
        ]
    ):
        ax = fig.add_subplot(gs[0, idx_p])
        slc = sub_w.slice(normal=[0.0, 1.0, 0.0], origin=[0.0, 0.0, 0.0])
        z0 = float(np.percentile(slc.points[:, 2], 98.0))
        polys, vals = _slice_polys_2d(slc, "E_eff_pct", axes_idx=(0, 2), shift=(0.0, z0))
        pc = PolyCollection(
            polys,
            array=vals,
            cmap="magma",
            clim=(0.0, vmax_eff),
            edgecolors=(0.15, 0.15, 0.15, 0.16),
            linewidths=0.22,
        )
        ax.add_collection(pc)

        lc_only = sub_w.extract_cells(np.asarray(sub_w.cell_data["domain_id"]) == 2)
        lc_slc = lc_only.slice(normal=[0.0, 1.0, 0.0], origin=[0.0, 0.0, 0.0])
        lc_edges = lc_slc.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False)
        if lc_edges.n_points > 0:
            lines_a = lc_edges.lines
            k = 0
            while k < len(lines_a):
                npt = int(lines_a[k])
                ids = lines_a[k + 1 : k + 1 + npt]
                ax.plot(lc_edges.points[ids, 0], lc_edges.points[ids, 2] - z0, color="#00F5D4", lw=1.3)
                k += npt + 1

        ax.text(
            0.97,
            0.06,
            "5× Deformed",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.0,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.86, edgecolor="#CCCCCC"),
        )
        ax.set_xlim(-2.55, 2.75)
        ax.set_ylim(-2.35, 0.25)
        ax.set_aspect("equal")
        ax.set_xlabel("Nasal  ←  X [mm]  →  Temporal")
        ax.set_ylabel("Relative Depth ΔZ [mm]")
        ax.set_title(panel_lbl, loc="left", fontweight="bold", fontsize=9.8, pad=10)
        cb = fig.colorbar(pc, ax=ax, fraction=0.040, pad=0.03, shrink=0.68, extend="max")
        cb.set_label(r"$\varepsilon_{\mathrm{eff}}$ [%]")

    # (c, d) 放在左下和中间下: 正视眼 vs 高度近视眼 筛板俯视第一主拉应变 E1 (统一色标直观对比!)
    em_lc = emme_ug.extract_cells(np.asarray(emme_ug.cell_data["domain_id"]) == 2)
    hm_lc = hm_ug.extract_cells(np.asarray(hm_ug.cell_data["domain_id"]) == 2)
    p_em, v_em, c_em, _ = _enface_hex_layer(em_lc, "E1_pct")
    p_hm, v_hm, c_hm, _ = _enface_hex_layer(hm_lc, "E1_pct")
    vmax_e1_shared = max(float(np.max(v_hm)), 2.3)

    for col_p, (lbl_p, polys_l, vals_l, c_xy) in enumerate(
        [
            (r"c  Emmetropia En-Face LC Tensile Strain $E_1$ (15 mmHg)", p_em, v_em, c_em),
            (r"d  High-Myopia En-Face LC Tensile Strain $E_1$ (15 mmHg)", p_hm, v_hm, c_hm),
        ]
    ):
        ax = fig.add_subplot(gs[1, col_p])
        cx, cy = float(np.mean(c_xy[:, 0])), float(np.mean(c_xy[:, 1]))
        pc_l = PolyCollection(
            [p - np.array([cx, cy]) for p in polys_l],
            array=vals_l,
            cmap="viridis",
            clim=(0.0, vmax_e1_shared),
            edgecolors=(1.0, 1.0, 1.0, 0.22),
            linewidths=0.30,
        )
        ax.add_collection(pc_l)
        ax.set_xlim(-1.28, 1.28)
        ax.set_ylim(-1.28, 1.28)
        ax.set_aspect("equal")
        ax.set_xlabel("Nasal  ←  ΔX [mm]  →  Temporal")
        ax.set_ylabel("Inferior  ←  ΔY [mm]  →  Superior")
        ax.set_title(lbl_p, loc="left", fontweight="bold", fontsize=10.0)
        cb_l = fig.colorbar(pc_l, ax=ax, fraction=0.042, pad=0.03, shrink=0.82, extend="max")
        cb_l.set_label(r"LC $E_1$ [%]")

    # (e) 右上: 四工况三大第 95 百分位应变指标对比柱状图
    ax_e = fig.add_subplot(gs[0, 2])
    cond_labels = ["Emme\n15 mmHg", "Emme\n21 mmHg", "HM Tilted\n15 mmHg", "HM Tilted\n21 mmHg"]
    x_idx = np.arange(len(cond_labels))
    bw = 0.25
    b1 = ax_e.bar(x_idx - bw, df_base["lc_e1_p95_pct"], bw, color=PALETTE["blue_main"], edgecolor="black", lw=0.8, label=r"Tensile $E_{1,95}$")
    b2 = ax_e.bar(x_idx, df_base["lc_eeff_p95_pct"], bw, color=PALETTE["red_strong"], edgecolor="black", lw=0.8, label=r"Effective $\varepsilon_{\mathrm{eff},95}$")
    b3 = ax_e.bar(x_idx + bw, df_base["lc_gmax_p95_pct"], bw, color=PALETTE["teal"], edgecolor="black", lw=0.8, label=r"3D Max Shear $\gamma_{\max,95}$")
    ax_e.axhspan(1.0, 6.0, color=PALETTE["gold"], alpha=0.16, zorder=0, label="In vivo OCT range")
    for bar_item in b2:
        yv = bar_item.get_height()
        ax_e.text(bar_item.get_x() + bar_item.get_width() / 2.0, yv + 0.14, f"{yv:.2f}%", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax_e.set_xticks(x_idx)
    ax_e.set_xticklabels(cond_labels)
    ax_e.set_ylim(0.0, 8.2)
    ax_e.set_ylabel("LC 95th-Percentile Strain [%]")
    ax_e.set_title("e  95th-Percentile LC Strain Amplification", loc="left", fontweight="bold", fontsize=10.0)
    ax_e.legend(loc="upper left", frameon=False, fontsize=8.0)
    ax_e.grid(True, axis="y", linestyle=":", alpha=0.35)

    # (f) 右下: 相对 BMO 参考平面的筛板轴向后移量对比
    ax_f = fig.add_subplot(gs[1, 2])
    cols_f = [PALETTE["blue_sec"], PALETTE["blue_main"], PALETTE["red_light"], PALETTE["red_strong"]]
    bars_f = ax_f.bar(x_idx, df_base["lc_ap_disp_p95_um"], width=0.52, color=cols_f, edgecolor="black", lw=1.0)
    for bf in bars_f:
        yv = bf.get_height()
        ax_f.text(bf.get_x() + bf.get_width() / 2.0, yv + 0.30, f"{yv:.2f} μm", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_f.set_xticks(x_idx)
    ax_f.set_xticklabels(cond_labels)
    ax_f.set_ylim(0.0, 16.5)
    ax_f.set_ylabel("LC Posterior Displacement vs BMO [μm]")
    ax_f.set_title("f  LC Posterior Bowing Relative to BMO Plane", loc="left", fontweight="bold", fontsize=10.0)
    ax_f.grid(True, axis="y", linestyle=":", alpha=0.35)

    return save_pub_fig(fig, fig_dir, "Figure2_Baseline_Biomechanics_Comparison")


# ==============================================================================
# Figure 3: Matérn-5/2 ARD 高斯过程代理模型独立测试集验证 (N=80) 与 ARD 相关性权重
# ==============================================================================
def plot_figure3_gp_validation_and_ard(fig_dir: Path) -> List[Path]:
    apply_pub_style()
    proc_dir = PROJECT_ROOT / "data" / "processed"
    uq_dir = PROJECT_ROOT / "data" / "uq_results"
    tbl_dir = PROJECT_ROOT / "results" / "tables"

    bundle = MultiQoIGPSurrogateBundle.load(uq_dir / "gp_surrogate_bundle.pkl")
    df_test = pd.read_csv(proc_dir / "sim_results_test.csv")
    df_ard = pd.read_csv(tbl_dir / "phase3_gp_validation_and_ard.csv")
    metrics = json.loads((uq_dir / "gp_validation_metrics.json").read_text(encoding="utf-8"))

    x_te = df_test[bundle.param_keys].to_numpy(dtype=float)

    fig = plt.figure(figsize=(16.4, 9.8))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.44)

    qoi_specs = [
        ("lc_e1_p95", r"a  LC 1st Principal Strain $E_{1,95}$ [%]", 100.0, PALETTE["blue_main"]),
        ("lc_eeff_p95", r"b  LC Effective Strain $\varepsilon_{\mathrm{eff},95}$ [%]", 100.0, PALETTE["red_strong"]),
        ("lc_gmax_p95", r"c  LC 3D Max Shear Strain $\gamma_{\max,95}$ [%]", 100.0, PALETTE["teal"]),
        ("lc_ap_disp_p95", r"d  LC Posterior Displacement $u_{\mathrm{AP},95}$ [μm]", 1000.0, PALETTE["purple"]),
    ]

    positions = [(0, 0), (0, 1), (0, 2), (1, 0)]
    std_residuals_eeff = None

    for (r_i, c_i), (qk, title_q, scale_f, col_q) in zip(positions, qoi_specs):
        ax = fig.add_subplot(gs[r_i, c_i])
        yt = df_test[qk].to_numpy(dtype=float) * scale_f
        yp, ystd = bundle.predict_qoi(qk, x_te, return_std=True)
        yp = yp * scale_f
        ystd = ystd * scale_f

        if qk == "lc_eeff_p95":
            std_residuals_eeff = (yt - yp) / np.maximum(ystd, 1e-9)

        lo_v = min(float(yt.min()), float(yp.min())) * 0.92
        hi_v = max(float(yt.max()), float(yp.max())) * 1.06
        ax.plot([lo_v, hi_v], [lo_v, hi_v], "--", color="#272727", lw=1.4, label="1:1 Exact Parity")
        ax.errorbar(
            yt,
            yp,
            yerr=1.96 * ystd,
            fmt="o",
            ms=4.8,
            color=col_q,
            ecolor=col_q,
            elinewidth=0.9,
            alpha=0.78,
            label="Test Set ± 95% GP CI",
        )
        m = metrics[qk]
        stat_txt = (
            f"$Q^2 = {m['Q2']:.4f}$\n"
            f"$\\mathrm{{NRMSE}} = {m['NRMSE_pct']:.2f}\\%$\n"
            f"$N_{{\\mathrm{{test}}}} = 80$"
        )
        ax.text(
            0.05,
            0.93,
            stat_txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F9FA", edgecolor="#CCCCCC"),
        )
        ax.set_xlim(lo_v, hi_v)
        ax.set_ylim(lo_v, hi_v)
        ax.set_xlabel("FEBio 4.0 Ground Truth")
        ax.set_ylabel("GP Surrogate Prediction")
        ax.set_title(title_q, loc="left", fontweight="bold", fontsize=10.2)
        ax.legend(loc="lower right", frameon=False, fontsize=8.0)
        ax.grid(True, linestyle=":", alpha=0.35)

    # (e) 独立测试集标准化残差 vs 眼内压 & 眼轴长度 无偏性诊断
    ax_e = fig.add_subplot(gs[1, 1])
    sc_e = ax_e.scatter(
        df_test["IOP"],
        std_residuals_eeff,
        c=df_test["AL"],
        cmap="coolwarm",
        s=42,
        edgecolors="k",
        linewidths=0.45,
        alpha=0.9,
    )
    ax_e.axhline(0.0, color="#272727", lw=1.3)
    ax_e.axhline(1.96, color=PALETTE["red_strong"], ls="--", lw=1.1, label="±1.96σ Bounds")
    ax_e.axhline(-1.96, color=PALETTE["red_strong"], ls="--", lw=1.1)
    ax_e.set_ylim(-3.2, 3.4)
    ax_e.set_xlabel("Test Case Intraocular Pressure IOP [mmHg]")
    ax_e.set_ylabel(r"Standardized Residual $(y - \hat{y}) / \sigma_{\mathrm{GP}}$")
    ax_e.set_title("e  Unbiased Test Residuals Across IOP & AL", loc="left", fontweight="bold", fontsize=10.2)
    cb_e = fig.colorbar(sc_e, ax=ax_e, fraction=0.045, pad=0.03)
    cb_e.set_label("Axial Length AL [mm]")
    ax_e.legend(loc="upper right", frameon=False, fontsize=8.0)
    ax_e.grid(True, linestyle=":", alpha=0.35)

    # (f) ARD 逆长度尺度 (1 / length_scale) 相关性热图
    ax_f = fig.add_subplot(gs[1, 2])
    p_keys = bundle.param_keys
    pretty_p = [
        "AL",
        r"$t_{\mathrm{PPS}}$",
        r"$\theta_{\mathrm{canal}}$",
        r"$t_{\mathrm{LC}}$",
        r"$D_{\mathrm{canal}}$",
        r"$c_{1,\mathrm{scl}}$",
        r"$\xi_{\mathrm{fib}}$",
        r"$b_{\mathrm{fib}}$",
        r"$E_{\mathrm{LC}}$",
        "IOP",
    ]
    q_labels = [r"$E_{1,95}$", r"$\varepsilon_{\mathrm{eff},95}$", r"$\gamma_{\max,95}$", r"$u_{\mathrm{AP},95}$"]

    inv_ls_mat = np.zeros((len(CORE_QOIS), len(p_keys)), dtype=float)
    for qi, qk in enumerate(CORE_QOIS):
        row_q = df_ard[df_ard["qoi"] == qk].iloc[0]
        inv_ls = np.array([1.0 / float(row_q[f"ard_ls_{pk}"]) for pk in p_keys])
        inv_ls_mat[qi, :] = inv_ls / np.max(inv_ls)  # 行归一化至 [0, 1] 展示相对相关性强度

    im_f = ax_f.imshow(inv_ls_mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax_f.set_xticks(np.arange(len(pretty_p)))
    ax_f.set_xticklabels(pretty_p, rotation=45, ha="right", fontsize=7.8)
    ax_f.set_yticks(np.arange(len(q_labels)))
    ax_f.set_yticklabels(q_labels, fontsize=8.8)
    for i in range(inv_ls_mat.shape[0]):
        for j in range(inv_ls_mat.shape[1]):
            val = inv_ls_mat[i, j]
            txt_c = "white" if val > 0.58 else "#1D1D1D"
            ax_f.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.3, color=txt_c)
    ax_f.spines["top"].set_visible(True)
    ax_f.spines["right"].set_visible(True)
    cb_f = fig.colorbar(im_f, ax=ax_f, fraction=0.045, pad=0.03)
    cb_f.set_label(r"Normalized ARD Relevance ($1 / \ell_j$)")
    ax_f.set_title("f  Matérn-5/2 ARD Feature Relevance Matrix", loc="left", fontweight="bold", fontsize=10.2)

    return save_pub_fig(fig, fig_dir, "Figure3_GP_Surrogate_Validation_and_ARD")
