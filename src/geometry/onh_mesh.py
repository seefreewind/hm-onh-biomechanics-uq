"""
参数化 3D 后段眼球—倾斜视神经乳头 (ONH) 有限元网格生成器 (基于 Python gmsh API)
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)

核心几何与拓扑特性:
1. 采用 Gmsh 三维结构化 O-Grid 六面体 (hex8) 分块共形剖分，涵盖 6 大解剖域:
   - LC   : 筛板 (Lamina Cribrosa)
   - PLNT : 筛板前神经组织层 (Prelaminar Neural Tissue)
   - RLON : 筛板后视神经实质 (Retrolaminar Optic Nerve)
   - PIA  : 软脑膜鞘 (Pia Mater)
   - PPS  : 视盘周巩膜环 (Peripapillary Sclera, 富集环向 Von Mises 胶原纤维)
   - PSCL : 后部/周边巩膜壳 (Posterior Sclera, 延伸至眼球赤道部)
2. 解析光滑同胚映射精确实现 5 个几何参数:
   - AL          : 眼轴长度 [mm] -> 调控后段眼球内壁曲率半径 R_in(AL)
   - t_ppS       : 视盘周巩膜厚度 [mm]
   - theta_canal : 视神经管倾斜角 [deg -> rad] (同步耦合 ASCO-BMO 颞侧偏心)
   - t_LC        : 筛板厚度 [mm]
   - D_canal     : 前巩膜管口直径 [mm] (含 1.12 生理后口锥度扩展)
3. 内置全单元 8 高斯点雅可比行列式 det(J) 与归一化雅可比 (Scaled Jacobian) 自检门禁，
   确保无任何负雅可比或畸变单元，并精确生成视盘周环向胶原纤维局部正交标架 (a0, d0)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gmsh
import numpy as np
from scipy.interpolate import PchipInterpolator

from src.common.units import (
    ANGLE_INPUT_UNIT,
    ANGLE_SOLVER_UNIT,
    DEG_TO_RAD,
    FORCE_UNIT,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    STRESS_UNIT,
    deg_to_rad,
    load_yaml_config,
)

# 显式声明模块级强制单位常量
UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322
assert ANGLE_INPUT_UNIT == "deg" and ANGLE_SOLVER_UNIT == "rad"

# 标准 8 节点六面体参考坐标 [-1, 1]^3 与 2x2x2 高斯积分点
_HEX8_REF_CORNERS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
    ],
    dtype=float,
)

_GP = 1.0 / math.sqrt(3.0)
_HEX8_GAUSS_PTS = _HEX8_REF_CORNERS * _GP


def _hex8_dshape(xi: float, eta: float, zeta: float) -> np.ndarray:
    """计算 hex8 在自然坐标 (xi, eta, zeta) 处的形函数偏导矩阵 dN/d(xi,eta,zeta), 形状 (8, 3)。"""
    s = _HEX8_REF_CORNERS
    d_xi = 0.125 * s[:, 0] * (1.0 + s[:, 1] * eta) * (1.0 + s[:, 2] * zeta)
    d_eta = 0.125 * (1.0 + s[:, 0] * xi) * s[:, 1] * (1.0 + s[:, 2] * zeta)
    d_zeta = 0.125 * (1.0 + s[:, 0] * xi) * (1.0 + s[:, 1] * eta) * s[:, 2]
    return np.column_stack([d_xi, d_eta, d_zeta])


_HEX8_DSHAPE_GAUSS = np.stack([_hex8_dshape(*pt) for pt in _HEX8_GAUSS_PTS], axis=0)  # (8_gp, 8_node, 3)
_HEX8_DSHAPE_CORNERS = np.stack([_hex8_dshape(*pt) for pt in _HEX8_REF_CORNERS], axis=0)  # (8_corner, 8_node, 3)

# hex8 的 6 个边界面局部节点索引 (右手定则外法向指向单元外部!)
# 底面 zeta=-1: (0,3,2,1) -> (x3-x0)x(x1-x0) = (+y)x(+x) = -z (向外!)
# 顶面 zeta=+1: (4,5,6,7) -> (x5-x4)x(x7-x4) = (+x)x(+y) = +z (向外!)
_HEX8_OUTWARD_FACES = (
    (0, 3, 2, 1),  # bottom (-zeta)
    (4, 5, 6, 7),  # top (+zeta)
    (0, 1, 5, 4),  # front (-eta)
    (1, 2, 6, 5),  # right (+xi)
    (2, 3, 7, 6),  # back (+eta)
    (3, 0, 4, 7),  # left (-xi)
)


@dataclass
class MeshQualityReport:
    total_nodes: int
    total_elements: int
    domain_element_counts: Dict[str, int]
    min_gauss_jacobian_det: float
    min_corner_jacobian_det: float
    min_scaled_jacobian: float
    mean_scaled_jacobian: float
    negative_jacobian_count: int
    asco_bmo_offset_mm: float
    posterior_inner_radius_mm: float
    lc_volume_mm3: float


@dataclass
class ONHMeshData:
    nodes: np.ndarray                                # (N_nodes, 3), 单位 mm, 对应 1..N_nodes
    canonical_coords: np.ndarray                     # (N_nodes, 3), 参考柱坐标 (xi_x, xi_y, zeta)
    domains: Dict[str, np.ndarray]                   # domain_name -> (N_elem, 8) 1-based node IDs
    elem_volumes: Dict[str, np.ndarray]              # domain_name -> (N_elem,) 单元体积 mm^3
    mat_axes: Dict[str, Tuple[np.ndarray, np.ndarray]]  # domain -> (a0 (N,3), d0 (N,3))
    surfaces: Dict[str, np.ndarray]                  # surf_name -> (N_quad, 4) 1-based outward quad4
    node_sets: Dict[str, np.ndarray]                 # set_name -> (N_set,) 1-based node IDs
    quality: MeshQualityReport
    params: Dict[str, float] = field(default_factory=dict)
    mesh_level: str = "medium"


def _smooth_step_c1(u: np.ndarray) -> np.ndarray:
    """C1 光滑余弦过渡函数 S(u) = 0.5*(1 - cos(pi*u)), u 截断至 [0, 1]。"""
    uc = np.clip(u, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(math.pi * uc))


def _build_canonical_ogrid_gmsh(level_cfg: dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    使用 Python gmsh.model.geo 构建标准参考空间 (xi_x, xi_y, zeta) 下的三维全六面体共形 O-Grid 网格。
    参考径向分界点:
      rho_0 = 1.00 (视神经管边界)
      rho_1 = 1.12 (软脑膜外边界)
      rho_2 = 2.20 (视盘周巩膜外边界)
      rho_3 = 3.60 (厚度过渡区外边界)
      rho_4 = 8.00 (眼球赤道边界)
    参考深度分层:
      Layer A: zeta in [0.00, 0.35] (内层: PLNT + PPS_ant + PSCL_ant)
      Layer B: zeta in [0.35, 1.00] (筛板层: LC + PPS_post + PSCL_post)
      Layer C: zeta in [1.00, 2.00] (筛板后层: RLON + PIA, 仅在 rho <= rho_1 内拉伸)
    """
    nc = int(level_cfg["n_circum_quarter"])
    nr_lc = int(level_cfg["n_rad_lc_ring"])
    nr_pia = int(level_cfg["n_rad_pia"])
    nr_pps = int(level_cfg["n_rad_pps"])
    nr_trans = int(level_cfg["n_rad_trans"])
    nr_pscl = int(level_cfg["n_rad_pscl"])
    nz_ant = int(level_cfg["n_z_ant"])
    nz_lc = int(level_cfg["n_z_lc"])
    nz_retro = int(level_cfg["n_z_retro"])

    radii = [1.00, 1.12, 2.20, 3.60, 8.00]
    n_rads = [nr_lc, nr_pia, nr_pps, nr_trans, nr_pscl]
    a_core = 0.42  # 中心正方形半宽 (对角半径 0.594 < 1.00)

    need_finalize = False
    if not gmsh.isInitialized():
        gmsh.initialize()
        need_finalize = True

    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Geometry.AutoCoherence", 1)
    gmsh.model.add("canonical_onh_ogrid")

    try:
        geo = gmsh.model.geo
        center_pt = geo.addPoint(0.0, 0.0, 0.0)

        # 4 个象限对角方向角: 0:+X+Y (pi/4), 1:-X+Y (3pi/4), 2:-X-Y (5pi/4), 3:+X-Y (7pi/4)
        angles = [0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi]
        sq_coords = [
            (+a_core, +a_core),
            (-a_core, +a_core),
            (-a_core, -a_core),
            (+a_core, -a_core),
        ]
        sq_pts = [geo.addPoint(x, y, 0.0) for (x, y) in sq_coords]

        # 中心正方形 4 条边 (逆时针)
        sq_lines = [geo.addLine(sq_pts[i], sq_pts[(i + 1) % 4]) for i in range(4)]
        sq_loop = geo.addCurveLoop(sq_lines)
        sq_surf = geo.addPlaneSurface([sq_loop])

        # 各同心圆环上的 4 个角点与 4 段圆弧
        ring_pts: List[List[int]] = []
        ring_arcs: List[List[int]] = []
        for r_val in radii:
            pts_k = [
                geo.addPoint(r_val * math.cos(ang), r_val * math.sin(ang), 0.0)
                for ang in angles
            ]
            arcs_k = [
                geo.addCircleArc(pts_k[i], center_pt, pts_k[(i + 1) % 4])
                for i in range(4)
            ]
            ring_pts.append(pts_k)
            ring_arcs.append(arcs_k)

        # 径向连接线与扇区曲面: ring_idx = 0..4, sector = 0..3
        # sector 0: 東 (+X, -pi/4 -> +pi/4, 对应角点 3 -> 0)
        # 为保持统一直接用 i -> (i+1)%4:
        #   i=0: 北 (+Y, pi/4 -> 3pi/4)
        #   i=1: 西 (-X, 3pi/4 -> 5pi/4)
        #   i=2: 南 (-Y, 5pi/4 -> 7pi/4)
        #   i=3: 东 (+X, 7pi/4 -> pi/4)
        radial_lines: List[List[int]] = []
        ring_surfs: List[List[int]] = []

        for k in range(len(radii)):
            inner_pts = sq_pts if k == 0 else ring_pts[k - 1]
            inner_curves = sq_lines if k == 0 else ring_arcs[k - 1]
            outer_pts = ring_pts[k]
            outer_curves = ring_arcs[k]

            rad_k = [geo.addLine(inner_pts[i], outer_pts[i]) for i in range(4)]
            radial_lines.append(rad_k)

            surfs_k = []
            for i in range(4):
                # 逆时针回路: inner_curve[i] -> rad_k[(i+1)%4] -> -outer_curve[i] -> -rad_k[i]
                loop_ki = geo.addCurveLoop(
                    [
                        inner_curves[i],
                        rad_k[(i + 1) % 4],
                        -outer_curves[i],
                        -rad_k[i],
                    ]
                )
                s_ki = geo.addPlaneSurface([loop_ki])
                surfs_k.append(s_ki)
            ring_surfs.append(surfs_k)

        # 设置二维 Transfinite 节点数与四边形重组 (Recombine)
        for c in sq_lines:
            geo.mesh.setTransfiniteCurve(c, nc + 1)
        geo.mesh.setTransfiniteSurface(sq_surf, cornerTags=sq_pts)
        geo.mesh.setRecombine(2, sq_surf)

        for k in range(len(radii)):
            nr_k = n_rads[k]
            for i in range(4):
                geo.mesh.setTransfiniteCurve(ring_arcs[k][i], nc + 1)
                # 最外环 PSCL 适度径向渐进加密靠近视盘侧
                prog = 1.12 if k == 4 else 1.0
                geo.mesh.setTransfiniteCurve(
                    radial_lines[k][i], nr_k + 1, meshType="Progression", coef=prog
                )
            inner_pts = sq_pts if k == 0 else ring_pts[k - 1]
            outer_pts = ring_pts[k]
            for i in range(4):
                corners = [
                    inner_pts[i],
                    inner_pts[(i + 1) % 4],
                    outer_pts[(i + 1) % 4],
                    outer_pts[i],
                ]
                geo.mesh.setTransfiniteSurface(ring_surfs[k][i], cornerTags=corners)
                geo.mesh.setRecombine(2, ring_surfs[k][i])

        # ------------------------------------------------------------------
        # 三层三维共形拉伸 (Extrude)
        # 全部 21 个二维面列表顺序: [sq_surf] + ring_surfs[0] + ring_surfs[1] + ring_surfs[2] + ring_surfs[3] + ring_surfs[4]
        # 索引映射:
        #   0      : 中心正方形 (视神经管核)
        #   1..4   : Ring 0 (视神经管外环)
        #   5..8   : Ring 1 (软脑膜环 / 视盘周巩膜内缘法兰)
        #   9..12  : Ring 2 (视盘周巩膜主环)
        #   13..16 : Ring 3 (巩膜过渡环)
        #   17..20 : Ring 4 (后部巩膜至赤道环)
        # ------------------------------------------------------------------
        base_surfs_21 = [sq_surf] + [s for grp in ring_surfs for s in grp]

        # Layer A: zeta 0.00 -> 0.35
        out_a = geo.extrude(
            [(2, s) for s in base_surfs_21],
            0.0,
            0.0,
            0.35,
            numElements=[nz_ant],
            recombine=True,
        )
        # 对 4 边形面，geo.extrude 每个面精确返回 6 项: [top_surf, vol, side1, side2, side3, side4]
        assert len(out_a) == 6 * len(base_surfs_21)
        tops_a = [out_a[6 * idx][1] for idx in range(len(base_surfs_21))]
        vols_a = [out_a[6 * idx + 1][1] for idx in range(len(base_surfs_21))]

        # Layer B: zeta 0.35 -> 1.00
        out_b = geo.extrude(
            [(2, s) for s in tops_a],
            0.0,
            0.0,
            0.65,
            numElements=[nz_lc],
            recombine=True,
        )
        assert len(out_b) == 6 * len(tops_a)
        tops_b = [out_b[6 * idx][1] for idx in range(len(tops_a))]
        vols_b = [out_b[6 * idx + 1][1] for idx in range(len(tops_a))]

        # Layer C: zeta 1.00 -> 2.00 (仅拉伸索引 0..8: 视神经管核 + Ring 0 -> RLON, Ring 1 -> PIA)
        retro_base_surfs = tops_b[:9]
        out_c = geo.extrude(
            [(2, s) for s in retro_base_surfs],
            0.0,
            0.0,
            1.00,
            numElements=[nz_retro],
            recombine=True,
        )
        assert len(out_c) == 6 * len(retro_base_surfs)
        vols_c = [out_c[6 * idx + 1][1] for idx in range(len(retro_base_surfs))]

        geo.removeAllDuplicates()
        geo.synchronize()

        # 定义 6 个三维物理组 (Physical Groups)
        phys_groups = {
            "PLNT": vols_a[0:5],
            "LC": vols_b[0:5],
            "RLON": vols_c[0:5],
            "PIA": vols_c[5:9],
            "PPS": vols_a[5:13] + vols_b[5:13],
            "PSCL": vols_a[13:21] + vols_b[13:21],
        }
        phys_tags = {}
        for p_idx, (name, v_list) in enumerate(phys_groups.items(), start=1):
            gmsh.model.addPhysicalGroup(3, v_list, tag=p_idx, name=name)
            phys_tags[name] = p_idx

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.removeDuplicateNodes()

        # 提取全部唯一节点并重排为紧凑 1..N_nodes 编号
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords_raw = np.asarray(node_coords, dtype=float).reshape(-1, 3)

        sort_order = np.argsort(node_tags)
        sorted_tags = node_tags[sort_order]
        canonical_coords = coords_raw[sort_order]

        tag_to_compact = np.zeros(int(sorted_tags[-1]) + 1, dtype=np.int64)
        tag_to_compact[sorted_tags] = np.arange(1, len(sorted_tags) + 1, dtype=np.int64)

        domains: Dict[str, np.ndarray] = {}
        for name, p_tag in phys_tags.items():
            ent_tags = gmsh.model.getEntitiesForPhysicalGroup(3, p_tag)
            conn_blocks = []
            for v_tag in ent_tags:
                elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(3, v_tag)
                for etype, enodes in zip(elem_types, elem_node_tags):
                    # Gmsh type 5 === 8-node hexahedron (hex8)
                    if int(etype) != 5:
                        raise RuntimeError(f"Expected pure hex8 (type 5) in {name}, got Gmsh type {etype}")
                    block = tag_to_compact[np.asarray(enodes, dtype=np.int64)].reshape(-1, 8)
                    conn_blocks.append(block)
            domains[name] = np.vstack(conn_blocks)

        return canonical_coords, domains
    finally:
        gmsh.model.remove()
        if need_finalize:
            gmsh.finalize()


def _map_canonical_to_physical(
    canonical_coords: np.ndarray,
    params: Dict[str, float],
    anat_cfg: dict,
) -> Tuple[np.ndarray, float, float]:
    """
    将参考坐标 (xi_x, xi_y, zeta) 通过 C1 解析同胚映射为三维物理坐标 (X, Y, Z) [单位: mm]。
    坐标系约定:
      +X : 颞侧 (Temporal, 高度近视视盘倾斜与 ASCO-BMO 偏心主方向)
      +Y : 上方 (Superior)
      -Z : 眼球后极部外向 (Posterior pole at X=0, Y=0, Z = -(R_in + w); 赤道平面位于 Z ≈ 0)
    """
    al = float(params["AL"])
    t_pps = float(params["t_ppS"])
    theta_deg = float(params["theta_canal"])
    theta_rad = deg_to_rad(theta_deg)
    t_lc = float(params["t_LC"])
    d_canal = float(params["D_canal"])
    r_canal = 0.5 * d_canal

    r_base = float(anat_cfg["posterior_base_inner_radius_mm"])
    r_slope = float(anat_cfg["posterior_radius_al_slope"])
    r_in = r_base + r_slope * (al - 24.0)

    t_periph = float(anat_cfg["peripheral_sclera_thickness_mm"])
    f_pps = float(anat_cfg["peripapillary_ring_outer_radius_factor"])
    f_trans = float(anat_cfg["transition_zone_outer_radius_factor"])
    f_tilt_in = float(anat_cfg["tilt_blend_inner_factor"])
    f_tilt_out = float(anat_cfg["tilt_blend_outer_factor"])
    exp_ratio = float(anat_cfg["posterior_canal_expansion_ratio"])
    eta_ant = float(anat_cfg["lc_anterior_insertion_fraction"])
    d_cup = float(anat_cfg["optic_cup_depth_mm"])
    d_lc_cup = float(anat_cfg["lc_anterior_cup_depth_mm"])
    l_rlon = float(anat_cfg["retrolaminar_nerve_length_mm"])
    t_pia = float(anat_cfg["pia_mater_thickness_mm"])

    xi_x = canonical_coords[:, 0]
    xi_y = canonical_coords[:, 1]
    zeta = canonical_coords[:, 2]
    rho_hat = np.hypot(xi_x, xi_y)

    # 先计算 z_clamp = min(zeta, 1.0) 处的球壳内壁至外壁坐标 (Layer C 在 zeta=1.0 基底上沿倾斜视神经轴延伸)
    z_shell = np.clip(zeta, 0.0, 1.0)

    # 1. 壁厚分布 t_wall(rho_hat) [mm]
    blend_periph = _smooth_step_c1((rho_hat - f_pps) / (f_trans - f_pps))
    t_wall = t_pps + (t_periph - t_pps) * blend_periph

    # 2. 三控制面物理深度 w_0, w_035, w_1 [mm]
    inside_canal = rho_hat <= 1.0 + 1e-9
    rho_clip1 = np.clip(rho_hat, 0.0, 1.0)

    w0 = np.where(inside_canal, d_cup * (1.0 - rho_clip1**2) ** 2, 0.0)
    w035 = np.where(
        inside_canal,
        eta_ant * t_pps + d_lc_cup * (1.0 - rho_clip1**2),
        eta_ant * t_wall,
    )

    w_flange = eta_ant * t_pps + t_lc
    blend_flange = _smooth_step_c1((rho_hat - 1.0) / (f_pps - 1.0))
    w1_outer = np.where(
        rho_hat <= f_pps,
        w_flange + (t_pps - w_flange) * blend_flange,
        t_wall,
    )
    w1 = np.where(inside_canal, w035 + t_lc, w1_outer)

    # 按 z_shell 分段线性插值获得壳内深度 w_shell [mm]
    in_layer_a = z_shell <= 0.35
    alpha_a = np.clip(z_shell / 0.35, 0.0, 1.0)
    alpha_b = np.clip((z_shell - 0.35) / 0.65, 0.0, 1.0)
    w_shell = np.where(
        in_layer_a,
        w0 + alpha_a * (w035 - w0),
        w035 + alpha_b * (w1 - w035),
    )

    # 3. 物理弧长半径 s_arc(rho_hat, z_shell): 采用严格单调 PCHIP 插值连接各解剖环边界
    # 为保留巩膜管 1.12 后口锥度扩展，对 z_shell 的线性依赖做两点精确插值 (z_shell=0 与 z_shell=1)
    r_eq_arc = 0.5 * math.pi * r_in
    rho_knots = np.array([0.0, 1.00, 1.12, 2.20, 3.60, 8.00], dtype=float)

    r0_ant = r_canal
    r0_post = r_canal * exp_ratio
    knots_ant = np.array([0.0, r0_ant, r0_ant + t_pia, f_pps * r_canal, f_trans * r_canal, r_eq_arc])
    knots_post = np.array([0.0, r0_post, r0_post + t_pia, f_pps * r_canal, f_trans * r_canal, r_eq_arc])

    pchip_ant = PchipInterpolator(rho_knots, knots_ant)
    pchip_post = PchipInterpolator(rho_knots, knots_post)

    s_ant = pchip_ant(rho_hat)
    s_post = pchip_post(rho_hat)
    s_arc = (1.0 - z_shell) * s_ant + z_shell * s_post

    # 转换回切平面二维坐标 (sx, sy), 在原点 rho_hat=0 处用导数极限消除 0/0
    scale_xy = np.where(rho_hat > 1e-12, s_arc / np.maximum(rho_hat, 1e-12), float(pchip_ant.derivative()(0.0)))
    sx = xi_x * scale_xy
    sy = xi_y * scale_xy

    # 4. 视神经管倾斜与 ASCO-BMO 颞侧偏心平移 Delta_sx
    # 在 rho_hat <= f_tilt_in (含全部 LC/PLNT/RLON/PIA) 内 B=1.0, 在 [f_tilt_in, f_tilt_out] 内 C1 衰减至 0
    tilt_decay = 1.0 - _smooth_step_c1((rho_hat - f_tilt_in) / (f_tilt_out - f_tilt_in))
    tilt_weight = np.where(rho_hat <= f_tilt_in, 1.0, np.where(rho_hat >= f_tilt_out, 0.0, tilt_decay))

    tan_theta = math.tan(theta_rad)
    sx_tilted = sx + tilt_weight * w_shell * tan_theta
    sy_tilted = sy

    # 5. 方位等距球面投影至三维后半球壳 (X, Y, Z)
    s_tot = np.hypot(sx_tilted, sy_tilted)
    psi = np.clip(s_tot / r_in, 0.0, 0.5 * math.pi)
    r_sph = r_in + w_shell

    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)
    dir_x = np.where(s_tot > 1e-12, sx_tilted / np.maximum(s_tot, 1e-12), 1.0)
    dir_y = np.where(s_tot > 1e-12, sy_tilted / np.maximum(s_tot, 1e-12), 0.0)

    x_phys = r_sph * sin_psi * dir_x
    y_phys = r_sph * sin_psi * dir_y
    z_phys = -r_sph * cos_psi

    # 6. Layer C (zeta > 1.0): 筛板后视神经与软脑膜沿倾斜轴 (sin(theta), 0, -cos(theta)) 延伸
    is_retro = zeta > 1.0 + 1e-9
    tau_retro = np.maximum(zeta - 1.0, 0.0)
    x_phys = np.where(is_retro, x_phys + tau_retro * l_rlon * math.sin(theta_rad), x_phys)
    z_phys = np.where(is_retro, z_phys - tau_retro * l_rlon * math.cos(theta_rad), z_phys)

    asco_bmo_offset_mm = w_flange * tan_theta
    nodes_phys = np.column_stack([x_phys, y_phys, z_phys])
    return nodes_phys, asco_bmo_offset_mm, r_in


def _orient_and_evaluate_hex8(
    nodes: np.ndarray,
    conn_1based: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """
    校验并确保所有 hex8 单元为正定向，计算:
      - 精确 8 高斯点积分体积 volumes (N_elem,) [mm^3]
      - 最小高斯点雅可比行列式 min_gauss_det
      - 最小角点雅可比行列式 min_corner_det
      - 每个单元最小归一化雅可比 scaled_jac (N_elem,) in [-1, 1]
    """
    # 取单元节点物理坐标: (N_elem, 8, 3)
    el_xyz = nodes[conn_1based - 1].copy()

    # 先检查中心点 (xi=eta=zeta=0) 雅可比符号，如 Gmsh 局部绕序相反则交换顶底面修正右手系
    d0 = _hex8_dshape(0.0, 0.0, 0.0)  # (8, 3)
    j_center = np.einsum("nid,ij->njd", el_xyz, d0)  # dx/dxi: (N_elem, 3_xi, 3_xyz)
    det_center = np.linalg.det(j_center)
    flipped = det_center < 0.0
    conn_fixed = conn_1based.copy()
    if np.any(flipped):
        conn_fixed[flipped] = conn_fixed[flipped][:, [4, 5, 6, 7, 0, 1, 2, 3]]
        el_xyz = nodes[conn_fixed - 1]

    # 8 高斯点雅可比矩阵: J[n, g, a, b] = sum_i dN_i/dxi_a * x_{n,i,b}
    j_gauss = np.einsum("gik,nid->ngkd", _HEX8_DSHAPE_GAUSS, el_xyz)  # (N_elem, 8_gp, 3, 3)
    det_gauss = np.linalg.det(j_gauss)  # (N_elem, 8_gp), 高斯权重全为 1.0
    volumes = np.sum(det_gauss, axis=1)

    # 8 角点归一化雅可比 (Scaled Jacobian = det(J) / (||J_0|| * ||J_1|| * ||J_2||))
    j_corners = np.einsum("cik,nid->nckd", _HEX8_DSHAPE_CORNERS, el_xyz)
    det_corners = np.linalg.det(j_corners)
    col_norms = np.linalg.norm(j_corners, axis=-1)  # (N_elem, 8_corner, 3)
    denom = np.prod(col_norms, axis=-1)
    scaled_jac_corners = det_corners / np.maximum(denom, 1e-30)
    min_scaled_per_elem = np.min(scaled_jac_corners, axis=1)

    return (
        conn_fixed,
        volumes,
        float(np.min(det_gauss)),
        float(np.min(det_corners)),
        min_scaled_per_elem,
    )


def _compute_circumferential_mat_axes(
    nodes: np.ndarray,
    canonical_coords: np.ndarray,
    conn_1based: np.ndarray,
    params: Dict[str, float],
    anat_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为巩膜单元 (PPS / PSCL) 计算精确局部正交材料标架 (a0, d0):
      a0 : 绕视神经管轴线的单位环向切向量 (Preferred circumferential collagen fiber axis)
      d0 : 沿巩膜壳子午方向的单位切向量 (与 a0 严格正交, a0 x d0 = 壳外法向 en)
    """
    el_xyz = nodes[conn_1based - 1]         # (N_elem, 8, 3)
    el_can = canonical_coords[conn_1based - 1]  # (N_elem, 8, 3)

    # 用参考空间周向微扰 d_phi = 1e-5 推送至物理空间求精确切向量 dPhi/dphi
    can_center = np.mean(el_can, axis=1)    # (N_elem, 3)
    d_phi = 1.0e-5
    cos_dp = math.cos(d_phi)
    sin_dp = math.sin(d_phi)

    can_plus = can_center.copy()
    can_plus[:, 0] = can_center[:, 0] * cos_dp - can_center[:, 1] * sin_dp
    can_plus[:, 1] = can_center[:, 0] * sin_dp + can_center[:, 1] * cos_dp

    can_minus = can_center.copy()
    can_minus[:, 0] = can_center[:, 0] * cos_dp + can_center[:, 1] * sin_dp
    can_minus[:, 1] = -can_center[:, 0] * sin_dp + can_center[:, 1] * cos_dp

    xyz_plus, _, _ = _map_canonical_to_physical(can_plus, params, anat_cfg)
    xyz_minus, _, _ = _map_canonical_to_physical(can_minus, params, anat_cfg)
    t_circum = xyz_plus - xyz_minus  # (N_elem, 3)

    # 计算壳厚度方向法向量 en (由内侧 zeta 面指向外侧 zeta 面)
    # 注意单元可能经过翻转修正，统一用参考 zeta 坐标高低区分内外侧节点
    n_el = len(conn_1based)
    en = np.zeros((n_el, 3), dtype=float)
    for idx in range(n_el):
        zc = el_can[idx, :, 2]
        z_mid = 0.5 * (zc.min() + zc.max())
        bot_mask = zc < z_mid
        top_mask = ~bot_mask
        v_thk = el_xyz[idx, top_mask].mean(axis=0) - el_xyz[idx, bot_mask].mean(axis=0)
        en[idx] = v_thk / np.linalg.norm(v_thk)

    # Gram-Schmidt 投影使 a0 严格位于壳切平面内且单位化
    proj = np.sum(t_circum * en, axis=1, keepdims=True)
    a0 = t_circum - proj * en
    a0 /= np.linalg.norm(a0, axis=1, keepdims=True)

    # d0 = en x a0 -> 保证 a0 . d0 = 0 且 a0 x d0 = en
    d0 = np.cross(en, a0)
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)

    return a0, d0


def _extract_boundary_surfaces_and_sets(
    nodes: np.ndarray,
    canonical_coords: np.ndarray,
    domains: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    通过三维全网格边界面哈希表精确提取无内部裂缝的外表面，并分类为:
      - iop_surface  : 玻璃体腔内表面 (zeta == 0, 外法向指向玻璃体腔, FEBio t = -IOP*N 向外扩张眼球并向后推压筛板)
      - csfp_surface : 蛛网膜下腔表面 (PIA 外侧面 rho_hat == 1.12 且 zeta >= 1, 以及筛板后远端面 zeta == 2.0,
                       外法向指向固体外部蛛网膜腔, FEBio t = -CSFP*N 向内/向前施加脑脊液压力)
    同时校验内部界面 100% 共形无裂缝！
    """
    all_hex = np.vstack([domains[k] for k in ("PLNT", "LC", "RLON", "PIA", "PPS", "PSCL")])

    # 统计每个四边面出现次数；内部共形界面必出现 2 次，真实外边界面仅出现 1 次
    face_map: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}
    face_counts: Dict[Tuple[int, int, int, int], int] = {}

    for elem in all_hex:
        for lf in _HEX8_OUTWARD_FACES:
            quad = (int(elem[lf[0]]), int(elem[lf[1]]), int(elem[lf[2]]), int(elem[lf[3]]))
            key = tuple(sorted(quad))
            face_counts[key] = face_counts.get(key, 0) + 1
            face_map[key] = quad

    boundary_quads = [face_map[k] for k, cnt in face_counts.items() if cnt == 1]
    invalid_counts = [cnt for cnt in face_counts.values() if cnt not in (1, 2)]
    if invalid_counts:
        raise RuntimeError(f"Non-manifold hex8 faces detected: {invalid_counts[:5]}")

    iop_quads: List[Tuple[int, int, int, int]] = []
    csfp_quads: List[Tuple[int, int, int, int]] = []
    other_ext_count = 0

    for q in boundary_quads:
        idx0 = np.array(q, dtype=np.int64) - 1
        c_can = canonical_coords[idx0]
        z_vals = c_can[:, 2]
        rho_vals = np.hypot(c_can[:, 0], c_can[:, 1])

        # 校验实空间面外法向是否与固体外法向一致 (因为可能翻转过单元顺序，直接用几何外方向显式校正每个 quad4 绕序!)
        pts = nodes[idx0]
        n_vec = np.cross(pts[1] - pts[0], pts[3] - pts[0])  # FEBio quad4 d_r x d_s 在角点近似方向

        if np.all(np.abs(z_vals - 0.0) < 1e-6):
            # 玻璃体腔内表面: 固体位于球壳外侧, 玻璃体腔球心在 (0, 0, 0)
            # 固体外法向必须指向球心 (0, 0, 0), 即 dot(n_vec, -centroid) > 0
            centroid = pts.mean(axis=0)
            if np.dot(n_vec, -centroid) < 0.0:
                q = (q[0], q[3], q[2], q[1])
            iop_quads.append(q)
        elif np.all(np.abs(z_vals - 2.0) < 1e-6):
            # 筛板后视神经远端截面 (zeta == 2.0): 固体在 +Z 侧, 外部在 -Z (后方) 侧 -> 外法向 n_z < 0
            if n_vec[2] > 0.0:
                q = (q[0], q[3], q[2], q[1])
            csfp_quads.append(q)
        elif np.all(z_vals >= 1.0 - 1e-6) and np.all(np.abs(rho_vals - 1.12) < 1e-4):
            # PIA 外周蛛网膜下腔表面: 固体在轴心侧, 外部在径向外侧
            # 用同一深度中心点做径向外判据: 在参考空间径向外方向
            c_xy = c_can[:, :2].mean(axis=0)
            # 推算实空间外向
            center_approx = np.array([pts[:, 0].mean() - c_xy[0] * 0.05, pts[:, 1].mean() - c_xy[1] * 0.05, pts[:, 2].mean()])
            if np.dot(n_vec[:2], c_xy) < 0.0:
                q = (q[0], q[3], q[2], q[1])
            csfp_quads.append(q)
        else:
            other_ext_count += 1

    if len(iop_quads) == 0 or len(csfp_quads) == 0:
        raise RuntimeError(f"Empty boundary surface extracted: IOP={len(iop_quads)}, CSFP={len(csfp_quads)}")

    # 提取关键节点集 (NodeSets)
    rho_all = np.hypot(canonical_coords[:, 0], canonical_coords[:, 1])
    z_all = canonical_coords[:, 2]

    equator_nodes = np.where(np.abs(rho_all - 8.00) < 1e-4)[0] + 1
    bmo_ring_nodes = np.where((np.abs(rho_all - 1.00) < 1e-4) & (np.abs(z_all - 0.0) < 1e-5))[0] + 1
    lc_nodes = np.unique(domains["LC"])
    lc_ant_center_nodes = np.where((rho_all < 0.25) & (np.abs(z_all - 0.35) < 1e-5))[0] + 1

    surfaces = {
        "iop_surface": np.asarray(iop_quads, dtype=np.int64),
        "csfp_surface": np.asarray(csfp_quads, dtype=np.int64),
    }
    node_sets = {
        "equator_nodes": equator_nodes.astype(np.int64),
        "bmo_ring_nodes": bmo_ring_nodes.astype(np.int64),
        "lc_nodes": lc_nodes.astype(np.int64),
        "lc_ant_center_nodes": lc_ant_center_nodes.astype(np.int64),
    }
    return surfaces, node_sets


# 缓存同一网格级别下的参考拓扑，避免批量 480 算例重复调用 Gmsh 剖分相同参考拓扑
_CANONICAL_CACHE: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}


def generate_onh_mesh(
    params: Dict[str, float],
    sim_cfg: Optional[dict] = None,
    mesh_level: Optional[str] = None,
    use_cache: bool = True,
) -> ONHMeshData:
    """
    根据 10 维输入参数字典生成完整的参数化 3D 后段眼球—ONH 六面体网格并执行严格雅可比门禁校验。
    """
    if sim_cfg is None:
        root = Path(__file__).resolve().parents[2]
        sim_cfg = load_yaml_config(root / "config" / "sim_settings.yaml")

    lvl = mesh_level or sim_cfg["mesh"]["production_level"]
    level_cfg = sim_cfg["mesh"]["levels"][lvl]
    anat_cfg = sim_cfg["anatomy_constants"]
    q_gates = sim_cfg["mesh"]["quality_gates"]

    if use_cache and lvl in _CANONICAL_CACHE:
        can_coords_ref, domains_ref = _CANONICAL_CACHE[lvl]
        can_coords = can_coords_ref.copy()
        domains_raw = {k: v.copy() for k, v in domains_ref.items()}
    else:
        can_coords, domains_raw = _build_canonical_ogrid_gmsh(level_cfg)
        if use_cache:
            _CANONICAL_CACHE[lvl] = (can_coords.copy(), {k: v.copy() for k, v in domains_raw.items()})

    nodes_phys, asco_bmo_offset_mm, r_in = _map_canonical_to_physical(can_coords, params, anat_cfg)

    domains_oriented: Dict[str, np.ndarray] = {}
    elem_volumes: Dict[str, np.ndarray] = {}
    min_gauss_det = float("inf")
    min_corner_det = float("inf")
    all_scaled_jacs: List[np.ndarray] = []
    neg_jac_count = 0

    for d_name, conn in domains_raw.items():
        conn_ok, vols, g_min, c_min, s_jac = _orient_and_evaluate_hex8(nodes_phys, conn)
        domains_oriented[d_name] = conn_ok
        elem_volumes[d_name] = vols
        min_gauss_det = min(min_gauss_det, g_min)
        min_corner_det = min(min_corner_det, c_min)
        all_scaled_jacs.append(s_jac)
        neg_jac_count += int(np.sum(s_jac <= 0.0))

    scaled_jac_cat = np.concatenate(all_scaled_jacs)
    min_sj = float(np.min(scaled_jac_cat))
    mean_sj = float(np.mean(scaled_jac_cat))

    if neg_jac_count > 0 or min_gauss_det <= q_gates["min_jacobian_det_exclusive"]:
        raise ValueError(
            f"Mesh Jacobian gate FAILED: neg_count={neg_jac_count}, "
            f"min_gauss_det={min_gauss_det:.3e}, min_SJ={min_sj:.3f}"
        )
    if min_sj < q_gates["min_scaled_jacobian"]:
        raise ValueError(
            f"Mesh Scaled Jacobian {min_sj:.3f} below gate {q_gates['min_scaled_jacobian']}"
        )

    mat_axes: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for scl_dom in ("PPS", "PSCL"):
        mat_axes[scl_dom] = _compute_circumferential_mat_axes(
            nodes_phys, can_coords, domains_oriented[scl_dom], params, anat_cfg
        )

    surfaces, node_sets = _extract_boundary_surfaces_and_sets(nodes_phys, can_coords, domains_oriented)

    total_elems = sum(len(v) for v in domains_oriented.values())
    quality = MeshQualityReport(
        total_nodes=len(nodes_phys),
        total_elements=total_elems,
        domain_element_counts={k: int(len(v)) for k, v in domains_oriented.items()},
        min_gauss_jacobian_det=min_gauss_det,
        min_corner_jacobian_det=min_corner_det,
        min_scaled_jacobian=min_sj,
        mean_scaled_jacobian=mean_sj,
        negative_jacobian_count=neg_jac_count,
        asco_bmo_offset_mm=asco_bmo_offset_mm,
        posterior_inner_radius_mm=r_in,
        lc_volume_mm3=float(np.sum(elem_volumes["LC"])),
    )

    return ONHMeshData(
        nodes=nodes_phys,
        canonical_coords=can_coords,
        domains=domains_oriented,
        elem_volumes=elem_volumes,
        mat_axes=mat_axes,
        surfaces=surfaces,
        node_sets=node_sets,
        quality=quality,
        params=dict(params),
        mesh_level=lvl,
    )
