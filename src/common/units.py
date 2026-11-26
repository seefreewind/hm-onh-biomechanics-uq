"""
统一物理单位制常量与转换模块 (强制规范 1)
体系锁定: 长度 [mm] - 力 [N] - 应力/压力/模量 [MPa]
推导校验: 1 MPa = 1e6 N/m^2 = 1 N / (1e3 mm)^2 * 1e6 = 1 N / mm^2
眼压换算: 1 mmHg = 133.322 Pa = 0.000133322 MPa
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import yaml

# ==============================================================================
# 核心锁定物理与单位常量 (mm - N - MPa)
# ==============================================================================
LENGTH_UNIT: str = "mm"
FORCE_UNIT: str = "N"
STRESS_UNIT: str = "MPa"
PRESSURE_UNIT: str = "MPa"
ANGLE_INPUT_UNIT: str = "deg"
ANGLE_SOLVER_UNIT: str = "rad"

# 1 N / mm^2 === 1 MPa (自洽无量纲比例因子 = 1.0)
N_PER_MM2_TO_MPA: float = 1.0

# 强制眼压/脑脊液压力换算常量
MMHG_TO_MPA: float = 0.000133322
MPA_TO_MMHG: float = 1.0 / MMHG_TO_MPA

# 常用辅助长度与应力换算常量
UM_TO_MM: float = 1.0e-3
MM_TO_UM: float = 1.0e3
KPA_TO_MPA: float = 1.0e-3
MPA_TO_KPA: float = 1.0e3
DEG_TO_RAD: float = math.pi / 180.0
RAD_TO_DEG: float = 180.0 / math.pi

GLOBAL_SEED: int = 42


def mmhg_to_mpa(pressure_mmhg: float) -> float:
    """将压力从临床单位 mmHg 严格转换为有限元求解器单位 MPa。"""
    return float(pressure_mmhg) * MMHG_TO_MPA


def mpa_to_mmhg(pressure_mpa: float) -> float:
    """将压力从有限元求解器单位 MPa 转换为临床单位 mmHg。"""
    return float(pressure_mpa) * MPA_TO_MMHG


def deg_to_rad(angle_deg: float) -> float:
    """将角度从度 (deg) 转换为弧度 (rad)。"""
    return float(angle_deg) * DEG_TO_RAD


def um_to_mm(length_um: float) -> float:
    """将微米 (um) 转换为毫米 (mm)。"""
    return float(length_um) * UM_TO_MM


def mm_to_um(length_mm: float) -> float:
    """将毫米 (mm) 转换为微米 (um)。"""
    return float(length_mm) * MM_TO_UM


def neohookean_from_youngs(youngs_mpa: float, nu: float = 0.49) -> Dict[str, float]:
    """
    将等效杨氏模量 E (MPa) 与泊松比 nu 转换为近似不可压缩 Neo-Hookean / Mooney-Rivlin (c2=0) 参数:
      剪切模量 mu = E / (2*(1 + nu))
      Mooney-Rivlin c1 = mu / 2 = E / (4*(1 + nu))
      体积模量 K = E / (3*(1 - 2*nu))
    """
    if youngs_mpa <= 0.0:
        raise ValueError(f"Young's modulus must be positive, got {youngs_mpa}")
    if not (0.0 <= nu < 0.5):
        raise ValueError(f"Poisson ratio must be in [0, 0.5), got {nu}")
    mu = youngs_mpa / (2.0 * (1.0 + nu))
    c1 = 0.5 * mu
    bulk_k = youngs_mpa / (3.0 * (1.0 - 2.0 * nu))
    return {"E_mpa": youngs_mpa, "nu": nu, "mu_mpa": mu, "c1_mpa": c1, "c2_mpa": 0.0, "k_mpa": bulk_k}


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """读取 YAML 配置文件并校验非空。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root structure in {p}")
    return data
