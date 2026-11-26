"""
出版级论文主图与汇总表生成器 (Part 2: Figure 4 - Figure 6 & Table 1 - Table 2)
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.units import FORCE_UNIT, LENGTH_UNIT, MMHG_TO_MPA, STRESS_UNIT, load_yaml_config
from src.visualization.fig_part1 import PALETTE, apply_pub_style, save_pub_fig

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PRETTY_PARAM_MAP = {
    "AL": "AL (Axial Length)",
    "t_ppS": r"$t_{\mathrm{PPS}}$ (PPS Thickness)",
    "theta_canal": r"$\theta_{\mathrm{canal}}$ (Canal Tilt)",
    "t_LC": r"$t_{\mathrm{LC}}$ (LC Thickness)",
    "D_canal": r"$D_{\mathrm{canal}}$ (Canal Diam.)",
    "c1_sclera": r"$c_{1,\mathrm{scl}}$ (Matrix Stiff.)",
    "ksi_fiber": r"$\xi_{\mathrm{fib}}$ (Fiber Modulus)",
    "b_fiber": r"$b_{\mathrm{fib}}$ (Ring Align.)",
    "E_LC": r"$E_{\mathrm{LC}}$ (LC Modulus)",
    "IOP": "IOP (Intraocular Press.)",
}

SHORT_PARAM_MAP = {
    "AL": "AL",
    "t_ppS": r"$t_{\mathrm{PPS}}$",
    "theta_canal": r"$\theta_{\mathrm{canal}}$",
    "t_LC": r"$t_{\mathrm{LC}}$",
    "D_canal": r"$D_{\mathrm{canal}}$",
    "c1_sclera": r"$c_{1,\mathrm{scl}}$",
    "ksi_fiber": r"$\xi_{\mathrm{fib}}$",
    "b_fiber": r"$b_{\mathrm{fib}}$",
    "E_LC": r"$E_{\mathrm{LC}}$",
    "IOP": "IOP",
}


# ==============================================================================
# Figure 4: 轨道 A (机理解耦) — SALib Sobol 全局敏感性分析 (S1, ST, Sij + 1000 Bootstrap)
# ==============================================================================
def plot_figure4_sobol_sensitivity(fig_dir: Path) -> List[Path]:
    apply_pub_style()
    uq_dir = PROJECT_ROOT / "data" / "uq_results"

    df_10d = pd.read_csv(uq_dir / "sobol_10d_indices.csv")
    df_9d = pd.read_csv(uq_dir / "sobol_9d_fixed_iop21.csv")
    sobol_json = json.loads((uq_dir / "sobol_results.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(15.6, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.35, wspace=0.28)

    # (a) 全 10 维参数空间 Sobol 指数 (lc_eeff_p95, 按 ST 排序横向条形图 + 1000 次 Bootstrap 95% CI)
    ax_a = fig.add_subplot(gs[0, 0])
    sub_eff10 = df_10d[df_10d["qoi"] == "lc_eeff_p95"].sort_values("ST", ascending=True)
    y_pos = np.arange(len(sub_eff10))
    bh = 0.36
    lbls_10 = [PRETTY_PARAM_MAP[k] for k in sub_eff10["parameter"]]

    ax_a.barh(
        y_pos + bh / 2,
        sub_eff10["ST"],
        height=bh,
        xerr=sub_eff10["ST_ci95_halfwidth"],
        color=PALETTE["blue_main"],
        edgecolor="black",
        lw=0.7,
        capsize=2.5,
        label=r"Total-Order $S_T$ ($\pm 95\%$ CI)",
    )
    ax_a.barh(
        y_pos - bh / 2,
        sub_eff10["S1"],
        height=bh,
        xerr=sub_eff10["S1_ci95_halfwidth"],
        color=PALETTE["teal"],
        edgecolor="black",
        lw=0.7,
        capsize=2.5,
        label=r"First-Order $S_1$ ($\pm 95\%$ CI)",
    )
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(lbls_10, fontsize=8.8)
    ax_a.set_xlim(0.0, 0.72)
    ax_a.set_xlabel("Global Sobol Variance Fraction (Full 10D Space)")
    ax_a.set_title("a  Full 10D Sobol Indices for LC Effective Strain (N = 90,112)", loc="left", fontweight="bold")
    ax_a.legend(loc="lower right", frameon=False)
    ax_a.grid(True, axis="x", linestyle=":", alpha=0.35)

    # (b) 固定 IOP = 21 mmHg 下 9 维内禀脆弱性参数双通路机制对比 (拉应变 E1 vs 有效应变 E_eff vs 后移量 Disp)
    ax_b = fig.add_subplot(gs[0, 1])
    anat_order = ["theta_canal", "ksi_fiber", "t_ppS", "E_LC", "t_LC", "D_canal", "c1_sclera", "b_fiber", "AL"]
    x9 = np.arange(len(anat_order))
    bw = 0.26

    for offset, qk, lbl_q, col_q in (
        (-bw, "lc_e1_p95", r"Tensile Strain $E_{1,95}$ ($S_T$)", PALETTE["red_strong"]),
        (0.0, "lc_eeff_p95", r"Effective Strain $\varepsilon_{\mathrm{eff},95}$ ($S_T$)", PALETTE["blue_main"]),
        (+bw, "lc_ap_disp_p95", r"Posterior Bowing $u_{\mathrm{AP},95}$ ($S_T$)", PALETTE["gold"]),
    ):
        sub_q = df_9d[df_9d["qoi"] == qk].set_index("parameter").loc[anat_order]
        ax_b.bar(
            x9 + offset,
            sub_q["ST"],
            width=bw,
            yerr=sub_q["ST_ci95_halfwidth"],
            color=col_q,
            edgecolor="black",
            lw=0.7,
            capsize=2.2,
            label=lbl_q,
        )

    ax_b.set_xticks(x9)
    ax_b.set_xticklabels([SHORT_PARAM_MAP[k] for k in anat_order], rotation=28, ha="right")
    ax_b.set_ylim(0.0, 0.48)
    ax_b.set_ylabel(r"Intrinsic Total-Order Index $S_T$ (at IOP = 21 mmHg)")
    ax_b.set_title("b  Intrinsic Ocular Vulnerability: Strain vs Displacement Pathways", loc="left", fontweight="bold")
    ax_b.legend(loc="upper right", frameon=False, fontsize=8.4)
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.35)

    # (c) 交互效应总量 (S_T - S_1) 四大 QoI 对比图
    ax_c = fig.add_subplot(gs[1, 0])
    p10_order = ["IOP", "D_canal", "theta_canal", "E_LC", "t_LC", "t_ppS", "ksi_fiber", "c1_sclera", "b_fiber", "AL"]
    x10 = np.arange(len(p10_order))
    bw_c = 0.26
    for offset, qk, lbl_q, col_q in (
        (-bw_c, "lc_e1_p95", r"$E_{1,95}$ Interaction ($S_T - S_1$)", PALETTE["blue_sec"]),
        (0.0, "lc_eeff_p95", r"$\varepsilon_{\mathrm{eff},95}$ Interaction ($S_T - S_1$)", PALETTE["red_strong"]),
        (+bw_c, "lc_ap_disp_p95", r"$u_{\mathrm{AP},95}$ Interaction ($S_T - S_1$)", PALETTE["purple"]),
    ):
        sub_q = df_10d[df_10d["qoi"] == qk].set_index("parameter").loc[p10_order]
        vals_int = np.maximum(sub_q["interaction_ST_minus_S1"].to_numpy(), 0.0)
        ax_c.bar(x10 + offset, vals_int, width=bw_c, color=col_q, edgecolor="black", lw=0.65, label=lbl_q)

    ax_c.set_xticks(x10)
    ax_c.set_xticklabels([SHORT_PARAM_MAP[k] for k in p10_order], rotation=28, ha="right")
    ax_c.set_ylabel(r"Higher-Order Interaction Variance ($S_T - S_1$)")
    ax_c.set_title("c  Synergistic Parameter Coupling Strength Across QoIs", loc="left", fontweight="bold")
    ax_c.legend(loc="upper right", frameon=False, fontsize=8.4)
    ax_c.grid(True, axis="y", linestyle=":", alpha=0.35)

    # (d) 二阶交互矩阵 S_ij 热图 (取高交互性的 lc_ap_disp_p95 与 lc_e1_p95 双三角合成矩阵!)
    ax_d = fig.add_subplot(gs[1, 1])
    names_10 = sobol_json["full_10d"]["lc_e1_p95"]["names"]
    s2_e1 = np.maximum(np.asarray(sobol_json["full_10d"]["lc_e1_p95"]["S2"], dtype=float), 0.0)
    s2_disp = np.maximum(np.asarray(sobol_json["full_10d"]["lc_ap_disp_p95"]["S2"], dtype=float), 0.0)

    # 上三角展示 E1 的 S_ij (x100%), 下三角展示后移量 Disp 的 S_ij (x100%)
    mat_comb = np.zeros((10, 10), dtype=float)
    for i in range(10):
        for j in range(i + 1, 10):
            mat_comb[i, j] = s2_e1[i, j] * 100.0
            mat_comb[j, i] = s2_disp[i, j] * 100.0

    vmax_d = max(float(mat_comb.max()), 2.5)
    im_d = ax_d.imshow(mat_comb, cmap="OrRd", vmin=0.0, vmax=vmax_d)
    short_10 = [SHORT_PARAM_MAP[k] for k in names_10]
    ax_d.set_xticks(np.arange(10))
    ax_d.set_xticklabels(short_10, rotation=45, ha="right", fontsize=7.8)
    ax_d.set_yticks(np.arange(10))
    ax_d.set_yticklabels(short_10, fontsize=8.2)
    ax_d.plot([-0.5, 9.5], [-0.5, 9.5], color="#4D4D4D", lw=1.2)
    for ii in range(10):
        for jj in range(10):
            if ii != jj and mat_comb[ii, jj] >= 0.5:
                v_cell = mat_comb[ii, jj]
                txt_col = "white" if v_cell >= 0.55 * vmax_d else "#1D1D1D"
                ax_d.text(jj, ii, f"{v_cell:.1f}%", ha="center", va="center", fontsize=7.0, fontweight="bold", color=txt_col)
    ax_d.text(5.1, 0.55, r"Upper: $E_{1,95}$ $S_{ij}$ [%]", fontsize=8.1, fontweight="bold", color="#7F0000")
    ax_d.text(0.25, 7.45, r"Lower: $u_{\mathrm{AP},95}$ $S_{ij}$ [%]", fontsize=8.1, fontweight="bold", color="#7F0000")
    ax_d.spines["top"].set_visible(True)
    ax_d.spines["right"].set_visible(True)
    cb_d = fig.colorbar(im_d, ax=ax_d, fraction=0.045, pad=0.03)
    cb_d.set_label(r"Second-Order Sobol $S_{ij}$ [%]")
    ax_d.set_title(r"d  Pairwise Second-Order Sobol Interaction Matrix $S_{ij}$", loc="left", fontweight="bold")

    return save_pub_fig(fig, fig_dir, "Figure4_TrackA_Sobol_Sensitivity")


# ==============================================================================
# Figure 5: 轨道 B (人群映射) — 10,000 例 Gaussian Copula 虚拟眼共变与眼轴分层高危风险
# ==============================================================================
def plot_figure5_virtual_cohort_risk(fig_dir: Path) -> List[Path]:
    apply_pub_style()
    uq_dir = PROJECT_ROOT / "data" / "uq_results"

    df_coh = pd.read_csv(uq_dir / "virtual_cohort_10000.csv")
    df_str = pd.read_csv(uq_dir / "cohort_al_strata_summary.csv")
    safe_meta = json.loads((uq_dir / "safe_iop_summary.json").read_text(encoding="utf-8"))
    eps_ref_pct = float(safe_meta["eps_ref_cohort_median_pct"])

    fig = plt.figure(figsize=(15.6, 9.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.26)

    # (a) 10,000 例虚拟眼实证解剖/微结构共变 2x2 子面板
    gs_a = gs[0, 0].subgridspec(2, 2, hspace=0.38, wspace=0.34)
    cov_pairs = [
        ("t_ppS", r"$t_{\mathrm{PPS}}$ [mm]", "#0F4D92"),
        ("theta_canal", r"$\theta_{\mathrm{canal}}$ [deg]", "#B64342"),
        ("t_LC", r"$t_{\mathrm{LC}}$ [mm]", "#2A9D8F"),
        ("b_fiber", r"$b_{\mathrm{fib}}$ [-]", "#6D597A"),
    ]
    for idx_sub, (col_name, y_lbl, hex_c) in enumerate(cov_pairs):
        ax_s = fig.add_subplot(gs_a[idx_sub // 2, idx_sub % 2])
        hb = ax_s.hexbin(df_coh["AL"], df_coh[col_name], gridsize=26, cmap="Blues" if idx_sub % 2 == 0 else "Reds", mincnt=1)
        r_sp = float(df_coh["AL"].corr(df_coh[col_name], method="spearman"))
        ax_s.text(
            0.05,
            0.90,
            f"ρ = {r_sp:+.2f}",
            transform=ax_s.transAxes,
            fontsize=8.2,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
        )
        if idx_sub // 2 == 1:
            ax_s.set_xlabel("AL [mm]", fontsize=8.8)
        ax_s.set_ylabel(y_lbl, fontsize=8.8)
        if idx_sub == 0:
            ax_s.set_title("a  Gaussian Copula Anatomical Couplings (N = 10,000)", loc="left", fontweight="bold", fontsize=10.2)

    # (b) 四大眼轴分层下筛板有效应变小提琴 + 箱线 + 散点分布 (Raincloud-style)
    ax_b = fig.add_subplot(gs[0, 1])
    strata_keys = [
        "emmetropia_low_myopia",
        "moderate_myopia",
        "high_myopia",
        "extreme_pathologic_myopia",
    ]
    strata_titles = [
        "Emme / Low\n[22.0, 24.5)",
        "Moderate\n[24.5, 26.5)",
        "High Myopia\n[26.5, 29.0)",
        "Pathologic HM\n[29.0, 32.0]",
    ]
    strata_cols = [PALETTE["teal"], PALETTE["blue_sec"], "#E76F51", PALETTE["red_strong"]]

    data_groups = [df_coh[df_coh["al_stratum"] == sk]["lc_eeff_p95"].to_numpy() * 100.0 for sk in strata_keys]
    pos_b = np.arange(1, 5)

    vp = ax_b.violinplot(data_groups, positions=pos_b, widths=0.68, showmeans=False, showextrema=False)
    for body, col_v in zip(vp["bodies"], strata_cols):
        body.set_facecolor(col_v)
        body.set_edgecolor("black")
        body.set_alpha(0.55)

    bp = ax_b.boxplot(
        data_groups,
        positions=pos_b,
        widths=0.20,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", lw=2.0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.9)

    ax_b.axhline(
        eps_ref_pct,
        color="#B64342",
        ls="--",
        lw=1.8,
        label=rf"Safety Benchmark $\varepsilon_{{\mathrm{{ref}}}} = {eps_ref_pct:.2f}\%$ (24 mm at 21 mmHg)",
    )
    ax_b.set_xticks(pos_b)
    ax_b.set_xticklabels(strata_titles)
    y_max_5b = max(9.8, float(max(np.max(g) for g in data_groups)) * 1.06)
    ax_b.set_ylim(0.2, y_max_5b)
    ax_b.set_ylabel(r"LC 95th-Percentile Effective Strain $\varepsilon_{\mathrm{eff},95}$ [%]")
    ax_b.set_title("b  Strain Distribution Across Clinical Axial-Length Strata", loc="left", fontweight="bold")
    ax_b.legend(loc="upper left", frameon=False, fontsize=8.4)
    ax_b.grid(True, axis="y", linestyle=":", alpha=0.35)

    # (c) 超标概率互补累积分布函数 (CCDF: P(E_eff > threshold))
    ax_c = fig.add_subplot(gs[1, 0])
    e_grid = np.linspace(0.5, 8.0, 300)
    for arr_g, st_lbl, col_v in zip(data_groups, strata_titles, strata_cols):
        ccdf = np.array([np.mean(arr_g > ev) * 100.0 for ev in e_grid])
        ax_c.plot(e_grid, ccdf, lw=2.3, color=col_v, label=st_lbl.replace("\n", " "))
    ax_c.axvline(eps_ref_pct, color="#272727", ls="--", lw=1.5, label=rf"Benchmark $\varepsilon_{{\mathrm{{ref}}}}$ ({eps_ref_pct:.2f}%)")
    ax_c.set_xlim(0.5, 7.8)
    ax_c.set_ylim(0.0, 102.0)
    ax_c.set_xlabel(r"LC Effective Strain Threshold $\varepsilon$ [%]")
    ax_c.set_ylabel(r"Exceedance Probability $P(\varepsilon_{\mathrm{eff},95} > \varepsilon)$ [%]")
    ax_c.set_title("c  Complementary Cumulative Risk Curves by Axial Length", loc="left", fontweight="bold")
    ax_c.legend(loc="upper right", frameon=False, fontsize=8.2)
    ax_c.grid(True, linestyle=":", alpha=0.35)

    # (d) 不同眼压制度下的高危应变超标率对比柱状图
    ax_d = fig.add_subplot(gs[1, 1])
    xd = np.arange(len(strata_keys))
    bw_d = 0.25
    r_iop15 = df_str["high_risk_rate_at_iop15_pct"].to_numpy()
    r_nat = df_str["high_risk_rate_natural_iop_pct"].to_numpy()
    r_iop21 = df_str["high_risk_rate_at_iop21_pct"].to_numpy()

    b_15 = ax_d.bar(xd - bw_d, r_iop15, bw_d, color=PALETTE["teal"], edgecolor="black", lw=0.8, label="Standardized IOP = 15 mmHg")
    b_na = ax_d.bar(xd, r_nat, bw_d, color=PALETTE["blue_main"], edgecolor="black", lw=0.8, label="Natural Resting Cohort IOP")
    b_21 = ax_d.bar(xd + bw_d, r_iop21, bw_d, color=PALETTE["red_strong"], edgecolor="black", lw=0.8, label="Clinical Upper-Normal 21 mmHg")

    for b_grp in (b_15, b_na, b_21):
        for b_it in b_grp:
            hv = b_it.get_height()
            ax_d.text(
                b_it.get_x() + b_it.get_width() / 2.0,
                hv + 1.2,
                f"{hv:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
            )

    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(strata_titles)
    ax_d.set_ylim(0.0, 112.0)
    ax_d.set_ylabel("Proportion Exceeding Safety Benchmark [%]")
    ax_d.set_title("d  High-Risk Exceedance Rate Under Three IOP Regimes", loc="left", fontweight="bold")
    ax_d.legend(loc="upper left", frameon=False, fontsize=8.2)
    ax_d.grid(True, axis="y", linestyle=":", alpha=0.35)

    return save_pub_fig(fig, fig_dir, "Figure5_TrackB_Virtual_Cohort_Risk")


# ==============================================================================
# Figure 6: “眼轴 × 眼压”交互响应曲面与生物力学等效安全眼压曲线 IOP_eq(AL) 主图
# ==============================================================================
def plot_figure6_safe_iop_boundary(fig_dir: Path) -> List[Path]:
    apply_pub_style()
    uq_dir = PROJECT_ROOT / "data" / "uq_results"

    surf_npz = np.load(uq_dir / "al_iop_response_surfaces.npz")
    df_curve = pd.read_csv(uq_dir / "safe_iop_eq_curve.csv")
    safe_meta = json.loads((uq_dir / "safe_iop_summary.json").read_text(encoding="utf-8"))

    al_grid = surf_npz["al_grid"]
    iop_grid = surf_npz["iop_grid"]
    surf_med = surf_npz["surface_median_pct"]   # (N_al, N_iop)
    surf_p90 = surf_npz["surface_p90_pct"]      # (N_al, N_iop)
    eps_ref_pct = float(safe_meta["eps_ref_cohort_median_pct"])

    fig = plt.figure(figsize=(15.8, 9.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.12], hspace=0.35, wspace=0.26)

    AL_M, IOP_M = np.meshgrid(al_grid, iop_grid, indexing="ij")

    # (a) 群体中位有效应变 AL x IOP 响应曲面与等应变线
    ax_a = fig.add_subplot(gs[0, 0])
    cf_a = ax_a.contourf(AL_M, IOP_M, surf_med, levels=np.linspace(0.8, 9.5, 24), cmap="RdYlBu_r", extend="both")
    cs_a = ax_a.contour(AL_M, IOP_M, surf_med, levels=[1.5, eps_ref_pct, 3.5, 5.0, 7.0], colors="black", linewidths=[0.9, 2.2, 0.9, 0.9, 0.9])
    ax_a.clabel(cs_a, fmt="%.2f%%", fontsize=8.2)
    ax_a.plot(24.0, 21.0, "*", ms=14, color="#FFD700", markeredgecolor="black", markeredgewidth=1.1, label="Emmetropic Benchmark (24 mm, 21 mmHg)")
    ax_a.set_xlabel("Axial Length AL [mm]")
    ax_a.set_ylabel("Intraocular Pressure IOP [mmHg]")
    ax_a.set_title(r"a  AL × IOP Response Surface: Cohort Median LC $\varepsilon_{\mathrm{eff},95}$ [%]", loc="left", fontweight="bold")
    ax_a.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=8.0)
    cb_a = fig.colorbar(cf_a, ax=ax_a, fraction=0.045, pad=0.03, ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    cb_a.set_label(r"Median LC $\varepsilon_{\mathrm{eff},95}$ [%]")

    # (b) 群体第 90 百分位高易感人群 AL x IOP 响应曲面
    ax_b = fig.add_subplot(gs[0, 1])
    cf_b = ax_b.contourf(AL_M, IOP_M, surf_p90, levels=np.linspace(1.2, 12.0, 24), cmap="magma_r", extend="both")
    cs_b = ax_b.contour(AL_M, IOP_M, surf_p90, levels=[eps_ref_pct, 4.0, 6.0, 8.0, 10.0], colors="white", linewidths=[2.2, 0.9, 0.9, 0.9, 0.9])
    ax_b.clabel(cs_b, fmt="%.2f%%", fontsize=8.2)
    ax_b.axhline(21.0, color="white", ls=":", lw=1.3, label="Conventional 21 mmHg Cutoff")
    ax_b.set_xlabel("Axial Length AL [mm]")
    ax_b.set_ylabel("Intraocular Pressure IOP [mmHg]")
    ax_b.set_title("b  AL × IOP Response Surface: 90th-Percentile High-Risk Tail [%]", loc="left", fontweight="bold")
    ax_b.legend(loc="upper right", frameon=True, facecolor="#272727", labelcolor="white", framealpha=0.85, fontsize=8.0)
    cb_b = fig.colorbar(cf_b, ax=ax_b, fraction=0.045, pad=0.03, ticks=[2, 4, 6, 8, 10, 11])
    cb_b.set_label(r"90th-Percentile LC $\varepsilon_{\mathrm{eff},95}$ [%]")

    # (c) 跨整行主图: 生物力学等效安全眼压曲线 IOP_eq(AL) 与第 90 百分位个体保护带
    ax_c = fig.add_subplot(gs[1, :])
    al_x = df_curve["AL_mm"].to_numpy()
    med_iop = df_curve["iop_eq_median_mmhg"].to_numpy()
    p25_iop = df_curve["iop_eq_p25_mmhg"].to_numpy()
    p75_iop = df_curve["iop_eq_p75_mmhg"].to_numpy()
    p10_iop = df_curve["iop_eq_p10_safe_bound_mmhg"].to_numpy()
    p90_iop = df_curve["iop_eq_p90_mmhg"].to_numpy()

    # 背景高危区与安全区淡色分层渲染
    ax_c.fill_between(al_x, p10_iop, 28.0, color="#F6CFCB", alpha=0.22, label="Biomechanical Overload Zone (for High-Susceptibility Eyes)")
    ax_c.fill_between(al_x, 2.0, p10_iop, color="#DDF3DE", alpha=0.42, label="90%-Protected Safe IOP Target Zone (≤ P10 Bound)")
    ax_c.fill_between(al_x, p10_iop, p90_iop, color=PALETTE["blue_sec"], alpha=0.14, label="Cohort 10th–90th Percentile Band")
    ax_c.fill_between(al_x, p25_iop, p75_iop, color=PALETTE["blue_main"], alpha=0.26, label="Cohort Interquartile Band [P25, P75]")

    ax_c.axhline(21.0, color="#4D4D4D", ls="--", lw=1.8, label="Traditional Static Upper-Normal IOP (21.0 mmHg)")
    ax_c.plot(al_x, med_iop, color=PALETTE["blue_main"], lw=3.0, label=r"Biomechanical Equivalent Safe IOP Curve ($\mathrm{IOP}_{\mathrm{eq},50}(\mathrm{AL})$)")
    ax_c.plot(al_x, p10_iop, color=PALETTE["red_strong"], lw=2.5, ls="-.", label=r"90%-Protection Lower Bound ($\mathrm{IOP}_{\mathrm{eq},P10}(\mathrm{AL})$, 10th percentile)")

    # 精确标注 5 个临床关键眼轴里程碑点 (24, 26, 28, 30, 32 mm)，与 Table 2 保持两位小数完全一致
    for m_item in safe_meta["key_al_milestones"]:
        kal = float(m_item["AL_mm"])
        v_med = float(m_item["iop_eq_median_mmhg"])
        v_p10 = float(m_item["iop_eq_p10_safe_bound_mmhg"])
        ax_c.plot(kal, v_med, "o", ms=7.5, color=PALETTE["blue_main"], markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax_c.plot(kal, v_p10, "s", ms=6.8, color=PALETTE["red_strong"], markeredgecolor="white", markeredgewidth=1.1, zorder=5)
        med_xy = (-32, 9) if kal >= 31.5 else (6, 8)
        p10_xy = (-34, -18) if kal >= 31.5 else (2, -18)
        ax_c.annotate(
            f"{v_med:.2f} mmHg",
            (kal, v_med),
            textcoords="offset points",
            xytext=med_xy,
            fontsize=8.4,
            fontweight="bold",
            color=PALETTE["blue_main"],
        )
        ax_c.annotate(
            f"P10: {v_p10:.2f}",
            (kal, v_p10),
            textcoords="offset points",
            xytext=p10_xy,
            fontsize=7.9,
            fontweight="bold",
            color=PALETTE["red_strong"],
        )

    ax_c.set_xlim(22.0, 32.35)
    ax_c.set_ylim(2.0, 27.5)
    ax_c.set_xticks(np.arange(22.0, 33.0, 1.0))
    ax_c.set_xlabel("Axial Length AL [mm]")
    ax_c.set_ylabel("Biomechanically Equivalent Safe IOP [mmHg]")
    ax_c.set_title(
        rf"c  Biomechanical Equivalent Safe IOP Boundary $\mathrm{{IOP}}_{{\mathrm{{eq}}}}(\mathrm{{AL}})$ Anchored to Emmetropic 21 mmHg Benchmark ($\varepsilon_{{\mathrm{{ref}}}} = {eps_ref_pct:.2f}\%$)",
        loc="left",
        fontweight="bold",
    )
    ax_c.legend(loc="upper right", ncol=2, frameon=True, facecolor="white", framealpha=0.92, fontsize=8.3)
    ax_c.grid(True, linestyle=":", alpha=0.40)

    return save_pub_fig(fig, fig_dir, "Figure6_Safe_IOP_Boundary_and_Surface")


# ==============================================================================
# Table 1 & Table 2: 生成出版级论文汇总表 (CSV + GitHub/LaTeX-Ready Markdown)
# ==============================================================================
def generate_publication_tables(table_dir: Path) -> Dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    uq_dir = PROJECT_ROOT / "data" / "uq_results"

    df_sobol10 = pd.read_csv(uq_dir / "sobol_10d_indices.csv")
    df_sobol9 = pd.read_csv(uq_dir / "sobol_9d_fixed_iop21.csv")
    df_gp = pd.read_csv(table_dir / "phase3_gp_validation_and_ard.csv")
    df_strata = pd.read_csv(table_dir / "phase3_cohort_strata_summary.csv")
    safe_meta = json.loads((uq_dir / "safe_iop_summary.json").read_text(encoding="utf-8"))

    # ---------------- Table 1: 10 维输入参数、Copula 相关系数与 Sobol 敏感性及 GP 精度汇总 ----------------
    corr_al = params_cfg["copula_correlation"]["matrix"][0]
    s10_eff = df_sobol10[df_sobol10["qoi"] == "lc_eeff_p95"].set_index("parameter")
    s9_eff = df_sobol9[df_sobol9["qoi"] == "lc_eeff_p95"].set_index("parameter")
    s9_e1 = df_sobol9[df_sobol9["qoi"] == "lc_e1_p95"].set_index("parameter")

    t1_rows = []
    for p in sorted(params_cfg["parameters"], key=lambda x: int(x["index"])):
        idx = int(p["index"])
        pk = p["key"]
        tb = p["track_b_dist"]
        lo, hi = p["bounds"]
        r10 = s10_eff.loc[pk]
        st_9_eff = f"{s9_eff.loc[pk, 'ST']:.3f} ± {s9_eff.loc[pk, 'ST_ci95_halfwidth']:.3f}" if pk in s9_eff.index else "Fixed (21 mmHg)"
        st_9_e1 = f"{s9_e1.loc[pk, 'ST']:.3f} ± {s9_e1.loc[pk, 'ST_ci95_halfwidth']:.3f}" if pk in s9_e1.index else "Fixed (21 mmHg)"

        if pk == "IOP":
            unit_str = "mmHg (×0.000133322 MPa)"
        elif pk == "theta_canal":
            unit_str = "deg (rad in solver)"
        else:
            unit_str = p["solver_unit"]

        t1_rows.append(
            {
                "Parameter": pk,
                "Description": p["name_en"],
                "Solver_Unit": unit_str,
                "LHS_Bounds": f"[{lo}, {hi}]",
                "Emmetropia_Baseline": p["baselines"]["emmetropia"],
                "High_Myopia_Baseline": p["baselines"]["high_myopia_tilted"],
                "Cohort_TruncNorm_Mean_SD": f"{tb['mean']} ± {tb['std']}",
                "Copula_r_with_AL": f"{corr_al[idx]:+.2f}",
                "Sobol_10D_ST_Eeff": f"{r10['ST']:.3f} ± {r10['ST_ci95_halfwidth']:.3f}",
                "Intrinsic_9D_ST_Eeff_IOP21": st_9_eff,
                "Intrinsic_9D_ST_E1_IOP21": st_9_e1,
            }
        )

    df_t1 = pd.DataFrame(t1_rows)
    t1_csv = table_dir / "Table1_Parameters_Sensitivity_and_GP.csv"
    t1_md = table_dir / "Table1_Parameters_Sensitivity_and_GP.md"
    df_t1.to_csv(t1_csv, index=False)

    # 构建 Table 1 Markdown (含 Panel A 参数与敏感性 + Panel B GP 独立测试集精度)
    md1_lines = [
        "# Table 1. Parameter Space Specification, Empirical Copula Couplings, Global Sobol Indices, and GP Surrogate Validation",
        "",
        "### Panel A: 10-Dimensional Ocular Biomechanical Input Space and Variance-Based Sensitivity Indices (Mean ± 95% Bootstrap CI)",
        "",
        "| Parameter | Physical Description | Unit | LHS / Sobol Bounds | Emmetropia | High Myopia | Cohort Mean ± SD | Copula $r(AL)$ | 10D $S_T$ ($\\varepsilon_{\\text{eff}}$) | Intrinsic 9D $S_T$ ($\\varepsilon_{\\text{eff}}$) | Intrinsic 9D $S_T$ ($E_1$) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in t1_rows:
        md1_lines.append(
            f"| `{r['Parameter']}` | {r['Description']} | {r['Solver_Unit']} | `{r['LHS_Bounds']}` | "
            f"`{r['Emmetropia_Baseline']}` | `{r['High_Myopia_Baseline']}` | `{r['Cohort_TruncNorm_Mean_SD']}` | "
            f"`{r['Copula_r_with_AL']}` | `{r['Sobol_10D_ST_Eeff']}` | `{r['Intrinsic_9D_ST_Eeff_IOP21']}` | `{r['Intrinsic_9D_ST_E1_IOP21']}` |"
        )

    md1_lines.extend(
        [
            "",
            "### Panel B: Matérn-5/2 ARD Gaussian Process Surrogate Performance on Unseen Test Set ($N_{\\text{train}}=400, N_{\\text{test}}=80$)",
            "",
            "| Quantity of Interest (QoI) | Aggregation | 5-Fold Train CV $Q^2$ | Unseen Test $Q^2$ | Test NRMSE (%) | Test RMSE | Test MAE | Log-Space GP |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    q_disp_names = {
        "lc_e1_p95": "LC 1st Principal Tensile Strain $E_1$",
        "lc_eeff_p95": "LC Effective Strain $\\varepsilon_{\\text{eff}}$",
        "lc_gmax_p95": "LC Maximum Shear Strain $\\gamma_{\\max}$",
        "lc_ap_disp_p95": "LC Posterior Displacement vs BMO",
    }
    for _, gr in df_gp.iterrows():
        qk = gr["qoi"]
        is_disp = qk == "lc_ap_disp_p95"
        rmse_s = f"{gr['test_RMSE']*1000.0:.3f} μm" if is_disp else f"{gr['test_RMSE']*100.0:.4f}%"
        mae_s = f"{gr['test_MAE']*1000.0:.3f} μm" if is_disp else f"{gr['test_MAE']*100.0:.4f}%"
        md1_lines.append(
            f"| {q_disp_names[qk]} (`{qk}`) | 95th %ile (Vol.-Weighted) | `{gr['train_5fold_cv_Q2']:.4f}` | "
            f"**`{gr['test_Q2']:.4f}`** | **`{gr['test_NRMSE_pct']:.2f}%`** | `{rmse_s}` | `{mae_s}` | "
            f"{'Yes' if gr['log_transform'] else 'No'} |"
        )
    t1_md.write_text("\n".join(md1_lines) + "\n", encoding="utf-8")

    # ---------------- Table 2: 10,000 虚拟眼分层高危比例与临床安全眼压边界速查表 ----------------
    t2_csv = table_dir / "Table2_Cohort_Risk_and_Safe_IOP_Boundaries.csv"
    t2_md = table_dir / "Table2_Cohort_Risk_and_Safe_IOP_Boundaries.md"

    # 导出结构化合并 CSV (含队列分层块与关键眼轴安全阈值块)
    milestones_df = pd.DataFrame(safe_meta["key_al_milestones"])
    milestones_df.to_csv(t2_csv, index=False)

    stratum_pretty = {
        "emmetropia_low_myopia": "Emmetropia / Low Myopia",
        "moderate_myopia": "Moderate Myopia",
        "high_myopia": "High Myopia",
        "extreme_pathologic_myopia": "Extreme / Pathologic Myopia",
    }

    md2_lines = [
        "# Table 2. Population Risk Stratification (10,000 Virtual Eyes) and Biomechanical Equivalent Safe IOP Boundaries",
        "",
        f"### Panel A: Virtual Cohort Biomechanical Strain and High-Risk Exceedance Rates Across Axial-Length Strata ($\\varepsilon_{{\\text{{ref}}}} = {safe_meta['eps_ref_cohort_median_pct']:.2f}\\%$)",
        "",
        "| Clinical Stratum | $AL$ Range (mm) | $N$ (%) | Mean $t_{\\text{ppS}}$ (mm) | Mean $\\theta_{\\text{canal}}$ (deg) | Median $\\varepsilon_{\\text{eff,p95}}$ [IQR] (%) | 90th %ile $\\varepsilon_{\\text{eff,p95}}$ (%) | Exceedance @ Resting IOP | Exceedance @ 15 mmHg | Exceedance @ 21 mmHg |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, sr in df_strata.iterrows():
        md2_lines.append(
            f"| **{stratum_pretty[sr['stratum']]}** | `{sr['al_range_mm']}` | `{int(sr['n_eyes'])}` (`{sr['cohort_share_pct']:.1f}%`) | "
            f"`{sr['mean_t_ppS_mm']:.2f}` | `{sr['mean_theta_canal_deg']:.1f}°` | "
            f"`{sr['eeff_p95_median_pct']:.2f}%` [`{sr['eeff_p95_iqr_low_pct']:.2f}%–{sr['eeff_p95_iqr_high_pct']:.2f}%`] | "
            f"`{sr['eeff_p95_90th_pct']:.2f}%` | **`{sr['high_risk_rate_natural_iop_pct']:.1f}%`** | "
            f"`{sr['high_risk_rate_at_iop15_pct']:.1f}%` | `{sr['high_risk_rate_at_iop21_pct']:.1f}%` |"
        )

    md2_lines.extend(
        [
            "",
            "### Panel B: Biomechanical Equivalent Safe Intraocular Pressure Lookup Table by Axial Length",
            "",
            "| Axial Length $AL$ (mm) | Median Equivalent Safe IOP $\\text{IOP}_{\\text{eq,50}}$ (mmHg) | Interquartile Range `[P25, P75]` (mmHg) | 90% Individual Protection Bound $\\text{IOP}_{\\text{eq,P10}}$ (mmHg) | Required Reduction vs 21 mmHg (Median / P10) |",
            "| :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for m in safe_meta["key_al_milestones"]:
        md2_lines.append(
            f"| **`{m['AL_mm']:.1f}`** | **`{m['iop_eq_median_mmhg']:.2f}`** | "
            f"`[{m['iop_eq_p25_mmhg']:.2f}, {m['iop_eq_p75_mmhg']:.2f}]` | "
            f"**`{m['iop_eq_p10_safe_bound_mmhg']:.2f}`** | "
            f"`{-m['iop_reduction_vs_21_median_mmhg']:+.2f}` / `{-m['iop_reduction_vs_21_p10_mmhg']:+.2f} mmHg` |"
        )
    t2_md.write_text("\n".join(md2_lines) + "\n", encoding="utf-8")

    return {
        "Table1_csv": t1_csv,
        "Table1_md": t1_md,
        "Table2_csv": t2_csv,
        "Table2_md": t2_md,
    }
