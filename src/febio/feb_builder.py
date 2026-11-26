"""
FEBio 4.0 (.feb) XML 输入文件自动生成器
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)

本构与载荷实现细节:
1. 视盘周巩膜 (PPS) 与后部巩膜 (PSCL):
   采用 FEBio 原生眼巩膜本构 `Mooney-Rivlin von Mises Fibers` (Girard & Gouget, JBME 2009 / BMMB 2012):
   - 基质: Mooney-Rivlin (c1 = c1_sclera [MPa], c2 = 0, 体积模量 k = 100 MPa)
   - 纤维: 切平面半圆 Von Mises 连续角度分布 (集中度 kf = b_fiber [PPS] / 0.15 [PSCL], tp = 0 rad),
           指数解卷曲纤维律 sigma_n = c3 * (exp(c4 * (lam_tilde - 1)) - 1), 其中初始纤维模量匹配 4*ksi_fiber
   - 局部材料轴: 通过 `<ElementData type="mat_axis">` 逐单元写入环向切向量 a0 与子午切向量 d0
2. 筛板 (LC)、筛板前神经组织 (PLNT)、筛板后视神经 (RLON)、软脑膜 (PIA):
   采用近似不可压缩超弹性 `Mooney-Rivlin` (c2 = 0, 即近似不可压缩 Neo-Hookean)
3. 边界与压力载荷:
   - 赤道边界节点集 `equator_nodes` 固定三向位移 (u_x = u_y = u_z = 0)
   - 玻璃体腔内表面 `iop_surface` 施加 IOP [mmHg * 0.000133322 -> MPa]
   - 蛛网膜下腔外表面 `csfp_surface` 施加 CSFP [11.0 mmHg * 0.000133322 -> MPa]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.common.units import (
    FORCE_UNIT,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    STRESS_UNIT,
    load_yaml_config,
    mmhg_to_mpa,
    neohookean_from_youngs,
)
from src.geometry.onh_mesh import ONHMeshData

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322

DOMAIN_ORDER = ("PLNT", "LC", "RLON", "PIA", "PPS", "PSCL")


def write_feb_file(
    mesh: ONHMeshData,
    params: Dict[str, float],
    feb_path: str | Path,
    sim_cfg: Optional[dict] = None,
    schedule_name: str = "primary",
    export_full_field: bool = False,
) -> Dict[str, float]:
    """
    写出符合 FEBio 4.0 规范的 .feb 文件，并返回写入求解器的精确物理单位参数摘要。
    """
    if sim_cfg is None:
        root = Path(__file__).resolve().parents[2]
        sim_cfg = load_yaml_config(root / "config" / "sim_settings.yaml")

    feb_file = Path(feb_path)
    feb_file.parent.mkdir(parents=True, exist_ok=True)

    sched = sim_cfg["febio"]["solver_schedule"][schedule_name]
    lin_solver = sim_cfg["febio"].get("linear_solver", "accelerate")
    const_cfg = sim_cfg["constitutive_constants"]
    load_cfg = sim_cfg["boundary_and_loads"]

    # 严格单位转换: mmHg -> MPa
    iop_mmhg = float(params["IOP"])
    iop_mpa = mmhg_to_mpa(iop_mmhg)
    csfp_mmhg = float(load_cfg["csfp_mmhg"])
    csfp_mpa = mmhg_to_mpa(csfp_mmhg)

    # 材料参数计算 (全部为 MPa 或无量纲)
    c1_scl = float(params["c1_sclera"])
    c2_scl = float(const_cfg["sclera"]["c2_mpa"])
    k_scl = float(const_cfg["sclera"]["bulk_modulus_mpa"])
    c4_fib = float(const_cfg["sclera"]["fiber_c4"])
    lam_max = float(const_cfg["sclera"]["fiber_lam_max"])
    gipt = int(const_cfg["sclera"]["fiber_gipt"])
    vmc = int(const_cfg["sclera"]["fiber_vmc"])

    # 匹配初始纤维网络切线刚度 4 * ksi_fiber = c3 * c4
    ksi_fib = float(params["ksi_fiber"])
    c3_fib = (4.0 * ksi_fib) / c4_fib
    # 超过 lam_max 后的切线模量连续衔接: c5 = c3 * c4 * exp(c4 * (lam_max - 1))
    c5_fib = c3_fib * c4_fib * float(np.exp(c4_fib * (lam_max - 1.0)))

    kf_pps = float(params["b_fiber"])
    kf_pscl = float(const_cfg["sclera"]["peripheral_kf"])

    lc_mat = neohookean_from_youngs(
        float(params["E_LC"]),
        nu=float(const_cfg["lamina_cribrosa"]["poisson_ratio"]),
    )
    plnt_mat = neohookean_from_youngs(
        float(const_cfg["prelaminar_neural"]["youngs_modulus_mpa"]),
        nu=float(const_cfg["prelaminar_neural"]["poisson_ratio"]),
    )
    rlon_mat = neohookean_from_youngs(
        float(const_cfg["retrolaminar_neural"]["youngs_modulus_mpa"]),
        nu=float(const_cfg["retrolaminar_neural"]["poisson_ratio"]),
    )
    c1_pia = float(const_cfg["pia_mater"]["c1_mpa"])
    c2_pia = float(const_cfg["pia_mater"]["c2_mpa"])
    k_pia = float(const_cfg["pia_mater"]["bulk_modulus_mpa"])

    max_retries = max(10, int(sched["time_steps"]) // 2)

    lines: List[str] = []
    a = lines.append

    a('<?xml version="1.0" encoding="ISO-8859-1"?>')
    a('<febio_spec version="4.0">')
    a('  <Module type="solid"/>')

    # 1. Control
    a("  <Control>")
    a("    <analysis>STATIC</analysis>")
    a(f'    <time_steps>{int(sched["time_steps"])}</time_steps>')
    a(f'    <step_size>{float(sched["step_size"]):.6g}</step_size>')
    a("    <plot_zero_state>0</plot_zero_state>")
    a("    <plot_level>PLOT_NEVER</plot_level>")
    a("    <output_level>OUTPUT_FINAL</output_level>")
    a('    <time_stepper type="default">')
    a(f"      <max_retries>{max_retries}</max_retries>")
    a(f'      <opt_iter>{int(sched["opt_iter"])}</opt_iter>')
    a(f'      <dtmin>{float(sched["dtmin"]):.6g}</dtmin>')
    a(f'      <dtmax>{float(sched["dtmax"]):.6g}</dtmax>')
    a("      <aggressiveness>0</aggressiveness>")
    a("      <cutback>0.5</cutback>")
    a("    </time_stepper>")
    a('    <solver type="solid">')
    a("      <symmetric_stiffness>symmetric</symmetric_stiffness>")
    a(f'      <max_refs>{int(sched["max_refs"])}</max_refs>')
    a(f'      <diverge_reform>{int(sched["diverge_reform"])}</diverge_reform>')
    a(f'      <reform_each_time_step>{int(sched.get("reform_each_time_step", 1))}</reform_each_time_step>')
    a('      <qn_method type="BFGS">')
    a(f'        <max_ups>{int(sched["max_ups"])}</max_ups>')
    a("      </qn_method>")
    a(f'      <linear_solver type="{lin_solver}"/>')
    a("    </solver>")
    a("  </Control>")

    # 2. Material (顺序: 1 PPS, 2 PSCL, 3 LC, 4 PLNT, 5 RLON, 6 PIA)
    a("  <Material>")
    a('    <material id="1" name="mat_PPS" type="Mooney-Rivlin von Mises Fibers">')
    a(f"      <c1>{c1_scl:.8g}</c1>")
    a(f"      <c2>{c2_scl:.8g}</c2>")
    a(f"      <c3>{c3_fib:.8g}</c3>")
    a(f"      <c4>{c4_fib:.8g}</c4>")
    a(f"      <c5>{c5_fib:.8g}</c5>")
    a(f"      <lam_max>{lam_max:.8g}</lam_max>")
    a(f"      <kf>{kf_pps:.8g}</kf>")
    a("      <tp>0.0</tp>")
    a(f"      <gipt>{gipt}</gipt>")
    a(f"      <vmc>{vmc}</vmc>")
    a("      <var_n>3.0</var_n>")
    a(f"      <k>{k_scl:.8g}</k>")
    a("    </material>")

    a('    <material id="2" name="mat_PSCL" type="Mooney-Rivlin von Mises Fibers">')
    a(f"      <c1>{c1_scl:.8g}</c1>")
    a(f"      <c2>{c2_scl:.8g}</c2>")
    a(f"      <c3>{c3_fib:.8g}</c3>")
    a(f"      <c4>{c4_fib:.8g}</c4>")
    a(f"      <c5>{c5_fib:.8g}</c5>")
    a(f"      <lam_max>{lam_max:.8g}</lam_max>")
    a(f"      <kf>{kf_pscl:.8g}</kf>")
    a("      <tp>0.0</tp>")
    a(f"      <gipt>{gipt}</gipt>")
    a(f"      <vmc>{vmc}</vmc>")
    a("      <var_n>3.0</var_n>")
    a(f"      <k>{k_scl:.8g}</k>")
    a("    </material>")

    for mid, mname, mdict in (
        (3, "mat_LC", lc_mat),
        (4, "mat_PLNT", plnt_mat),
        (5, "mat_RLON", rlon_mat),
    ):
        a(f'    <material id="{mid}" name="{mname}" type="Mooney-Rivlin">')
        a(f'      <c1>{mdict["c1_mpa"]:.8g}</c1>')
        a("      <c2>0.0</c2>")
        a(f'      <k>{mdict["k_mpa"]:.8g}</k>')
        a("    </material>")

    a('    <material id="6" name="mat_PIA" type="Mooney-Rivlin">')
    a(f"      <c1>{c1_pia:.8g}</c1>")
    a(f"      <c2>{c2_pia:.8g}</c2>")
    a(f"      <k>{k_pia:.8g}</k>")
    a("    </material>")
    a("  </Material>")

    # 3. Mesh
    a("  <Mesh>")
    a('    <Nodes name="all_nodes">')
    for nid, (x, y, z) in enumerate(mesh.nodes, start=1):
        a(f'      <node id="{nid}">{x:.8f},{y:.8f},{z:.8f}</node>')
    a("    </Nodes>")

    global_eid = 1
    domain_elem_id_ranges: Dict[str, Tuple[int, int]] = {}
    for dname in DOMAIN_ORDER:
        conn = mesh.domains[dname]
        start_eid = global_eid
        a(f'    <Elements type="hex8" name="{dname}">')
        for row in conn:
            n_str = ",".join(str(int(v)) for v in row)
            a(f'      <elem id="{global_eid}">{n_str}</elem>')
            global_eid += 1
        a("    </Elements>")
        domain_elem_id_ranges[dname] = (start_eid, global_eid - 1)

    # NodeSets (FEBio 4.0 逗号分隔叶子节点格式)
    for ns_name, ns_arr in mesh.node_sets.items():
        ns_csv = ",".join(str(int(v)) for v in ns_arr)
        a(f'    <NodeSet name="{ns_name}">{ns_csv}</NodeSet>')

    # Surfaces
    fid = 1
    for s_name, s_quads in mesh.surfaces.items():
        a(f'    <Surface name="{s_name}">')
        for q in s_quads:
            q_csv = ",".join(str(int(v)) for v in q)
            a(f'      <quad4 id="{fid}">{q_csv}</quad4>')
            fid += 1
        a("    </Surface>")
    a("  </Mesh>")

    # 4. MeshDomains
    dom_to_mat = {
        "PLNT": "mat_PLNT",
        "LC": "mat_LC",
        "RLON": "mat_RLON",
        "PIA": "mat_PIA",
        "PPS": "mat_PPS",
        "PSCL": "mat_PSCL",
    }
    a("  <MeshDomains>")
    for dname in DOMAIN_ORDER:
        a(f'    <SolidDomain name="{dname}" mat="{dom_to_mat[dname]}"/>')
    a("  </MeshDomains>")

    # 5. MeshData: 视盘周与后部巩膜局部环向胶原纤维材料正交轴 (a0, d0)
    a("  <MeshData>")
    for scl_dom in ("PPS", "PSCL"):
        a0_arr, d0_arr = mesh.mat_axes[scl_dom]
        a(f'    <ElementData type="mat_axis" elem_set="{scl_dom}">')
        for lid, (va, vd) in enumerate(zip(a0_arr, d0_arr), start=1):
            a(f'      <e lid="{lid}">')
            a(f"        <a>{va[0]:.8f},{va[1]:.8f},{va[2]:.8f}</a>")
            a(f"        <d>{vd[0]:.8f},{vd[1]:.8f},{vd[2]:.8f}</d>")
            a("      </e>")
        a("    </ElementData>")
    a("  </MeshData>")

    # 6. Boundary
    a("  <Boundary>")
    a('    <bc name="equator_clamped" type="zero displacement" node_set="equator_nodes">')
    a("      <x_dof>1</x_dof>")
    a("      <y_dof>1</y_dof>")
    a("      <z_dof>1</z_dof>")
    a("    </bc>")
    a("  </Boundary>")

    # 7. Loads
    a("  <Loads>")
    a('    <surface_load name="vitreous_iop" type="pressure" surface="iop_surface">')
    a(f'      <pressure lc="1">{iop_mpa:.9f}</pressure>')
    a("      <symmetric_stiffness>1</symmetric_stiffness>")
    a("      <linear>0</linear>")
    a("    </surface_load>")
    a('    <surface_load name="subarachnoid_csfp" type="pressure" surface="csfp_surface">')
    a(f'      <pressure lc="1">{csfp_mpa:.9f}</pressure>')
    a("      <symmetric_stiffness>1</symmetric_stiffness>")
    a("      <linear>0</linear>")
    a("    </surface_load>")
    a("  </Loads>")

    # 8. LoadData
    a("  <LoadData>")
    a('    <load_controller id="1" type="loadcurve">')
    a("      <interpolate>SMOOTH</interpolate>")
    a("      <extend>CONSTANT</extend>")
    a("      <points>")
    a("        <pt>0.0,0.0</pt>")
    a("        <pt>1.0,1.0</pt>")
    a("      </points>")
    a("    </load_controller>")
    a("  </LoadData>")

    # 9. Output: 精确提取最终收敛态 LC 应变场与节点位移场
    a("  <Output>")
    a("    <logfile>")
    a('      <element_data data="E1;E2;E3;effective strain;max shear strain" elem_set="LC" file="lc_elem_strains.csv" delim=","/>')
    a('      <node_data data="ux;uy;uz" node_set="lc_nodes" file="lc_nodal_disp.csv" delim=","/>')
    a('      <node_data data="ux;uy;uz" node_set="bmo_ring_nodes" file="bmo_nodal_disp.csv" delim=","/>')
    if export_full_field:
        for dname in DOMAIN_ORDER:
            a(
                f'      <element_data data="E1;E2;E3;effective strain;max shear strain;sx;sy;sz;effective stress" '
                f'elem_set="{dname}" file="full_elem_{dname}.csv" delim=","/>'
            )
        a('      <node_data data="ux;uy;uz" node_set="all_nodes" file="full_nodal_disp.csv" delim=","/>')
    a("    </logfile>")
    a("  </Output>")
    a("</febio_spec>")

    feb_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "iop_mmhg": iop_mmhg,
        "iop_mpa": iop_mpa,
        "csfp_mmhg": csfp_mmhg,
        "csfp_mpa": csfp_mpa,
        "c1_sclera_mpa": c1_scl,
        "c3_fiber_mpa": c3_fib,
        "c4_fiber": c4_fib,
        "kf_pps": kf_pps,
        "c1_lc_mpa": lc_mat["c1_mpa"],
        "k_lc_mpa": lc_mat["k_mpa"],
    }
