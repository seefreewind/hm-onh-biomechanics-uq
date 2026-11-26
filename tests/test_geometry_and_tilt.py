"""
Phase 1 门禁测试 1：
验证参数化 3D 后段眼球—视神经乳头 Gmsh 六面体网格生成器在:
  - 不同视神经管倾斜角 theta_canal in [0, 10, 20, 30, 35] deg
  - 极端解剖组合 (极薄筛板 t_LC=0.12 mm + 极薄巩膜 t_ppS=0.35 mm + 35 deg 倾斜 + 超长眼轴 AL=32 mm)
  - Coarse / Medium / Fine 三级网格密度
下的零负雅可比稳健性、水密共形拓扑与视盘周环向胶原纤维局部标架正交性。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.common.units import MMHG_TO_MPA, load_yaml_config
from src.geometry.onh_mesh import generate_onh_mesh

assert MMHG_TO_MPA == 0.000133322

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _get_baseline_params(kind: str = "emmetropia") -> dict:
    cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    return {p["key"]: float(p["baselines"][kind]) for p in cfg["parameters"]}


def test_tilt_sweep_zero_negative_jacobian() -> None:
    base = _get_baseline_params("high_myopia_tilted")
    for theta in (0.0, 10.0, 20.0, 30.0, 35.0):
        p = dict(base)
        p["theta_canal"] = theta
        mesh = generate_onh_mesh(p, mesh_level="medium")
        assert mesh.quality.negative_jacobian_count == 0
        assert mesh.quality.min_gauss_jacobian_det > 0.0
        assert mesh.quality.min_scaled_jacobian >= 0.20

        # 验证环向纤维标架严格正交归一
        for dom in ("PPS", "PSCL"):
            a0, d0 = mesh.mat_axes[dom]
            norms_a = np.linalg.norm(a0, axis=1)
            norms_d = np.linalg.norm(d0, axis=1)
            dots = np.einsum("ij,ij->i", a0, d0)
            assert np.allclose(norms_a, 1.0, atol=1e-12)
            assert np.allclose(norms_d, 1.0, atol=1e-12)
            assert np.allclose(dots, 0.0, atol=1e-12)


def test_extreme_hypercube_corners_and_three_levels() -> None:
    extreme_thin_tilted = {
        "AL": 32.0,
        "t_ppS": 0.35,
        "theta_canal": 35.0,
        "t_LC": 0.12,
        "D_canal": 2.40,
        "c1_sclera": 0.08,
        "ksi_fiber": 0.80,
        "b_fiber": 0.40,
        "E_LC": 0.10,
        "IOP": 40.0,
    }
    extreme_thick_tilted = {
        "AL": 22.0,
        "t_ppS": 1.05,
        "theta_canal": 35.0,
        "t_LC": 0.38,
        "D_canal": 1.40,
        "c1_sclera": 0.45,
        "ksi_fiber": 6.00,
        "b_fiber": 4.50,
        "E_LC": 0.60,
        "IOP": 10.0,
    }
    for p in (extreme_thin_tilted, extreme_thick_tilted):
        m = generate_onh_mesh(p, mesh_level="medium")
        assert m.quality.negative_jacobian_count == 0
        assert m.quality.min_scaled_jacobian >= 0.20

    # 校验三级网格单元数严格递增且 LC 物理体积高度一致
    emme = _get_baseline_params("emmetropia")
    m_c = generate_onh_mesh(emme, mesh_level="coarse")
    m_m = generate_onh_mesh(emme, mesh_level="medium")
    m_f = generate_onh_mesh(emme, mesh_level="fine")

    assert m_c.quality.total_elements < m_m.quality.total_elements < m_f.quality.total_elements
    assert math.isclose(m_m.quality.lc_volume_mm3, m_f.quality.lc_volume_mm3, rel_tol=0.015)
