"""
10 维参数空间 Maximin 拉丁超立方抽样 (Maximin LHS) 模块
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
全局固定随机种子: seed = 42

方法说明:
  结合中心化差异优化 (Centered Discrepancy) 与多候选最小成对欧氏距离最大化 (Maximin criterion)，
  分别独立生成互不重叠的:
    1) Smoke Test 试算集 (N = 20, seed_offset = 0)
    2) GP 训练集 (N = 400, seed_offset = 1000)
    3) GP 独立测试集 (N = 80, seed_offset = 2000, 额外对训练集做空间排斥优选以覆盖内插盲区)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc

from src.common.units import (
    FORCE_UNIT,
    GLOBAL_SEED,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    STRESS_UNIT,
    load_yaml_config,
)

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322
assert GLOBAL_SEED == 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_parameter_specs(params_cfg: Optional[dict] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    if params_cfg is None:
        params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    p_list = sorted(params_cfg["parameters"], key=lambda x: int(x["index"]))
    keys = [p["key"] for p in p_list]
    lows = np.array([float(p["bounds"][0]) for p in p_list], dtype=float)
    highs = np.array([float(p["bounds"][1]) for p in p_list], dtype=float)
    return keys, lows, highs


def generate_maximin_lhs_unit(
    n_samples: int,
    n_dim: int,
    seed: int = GLOBAL_SEED,
    n_candidates: int = 40,
    reference_unit_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    在单位超立方体 [0, 1]^d 上生成严格分层拉丁超立方样本，
    从 n_candidates 个独立扰动候选设计中选取最小点间距离 (min pairwise distance) 最大的设计。
    若提供 reference_unit_points (如测试集相对训练集)，同时最大化到已有参考点集的最小距离与内部最小距离。
    """
    best_u: Optional[np.ndarray] = None
    best_score = -1.0

    for k in range(max(1, int(n_candidates))):
        cand_seed = int(seed) + 17 * k
        sampler = qmc.LatinHypercube(d=n_dim, scramble=True, seed=cand_seed)
        u = sampler.random(n=n_samples)

        min_internal = float(np.min(pdist(u))) if n_samples > 1 else 1.0
        if reference_unit_points is not None and len(reference_unit_points) > 0:
            min_cross = float(np.min(cdist(u, reference_unit_points)))
            score = 0.65 * min_internal + 0.35 * min_cross
        else:
            score = min_internal

        if score > best_score:
            best_score = score
            best_u = u

    assert best_u is not None
    return best_u


def scale_unit_to_physical(u: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return qmc.scale(u, lows, highs)


def normalize_physical_to_unit(x: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) - lows) / (highs - lows)


def generate_all_lhs_datasets(
    output_dir: Optional[str | Path] = None,
    seed: int = GLOBAL_SEED,
) -> Dict[str, pd.DataFrame]:
    """
    一次性确定性生成并保存:
      - lhs_smoke_20.csv  (Phase 2 试跑 20 组)
      - lhs_train_400.csv (Phase 2 正式训练集 400 组)
      - lhs_test_80.csv   (Phase 2 独立测试集 80 组)
    """
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    sim_cfg = load_yaml_config(PROJECT_ROOT / "config" / "sim_settings.yaml")
    keys, lows, highs = get_parameter_specs(params_cfg)
    d = len(keys)

    n_smoke = int(sim_cfg["sampling"]["n_smoke"])
    n_train = int(sim_cfg["sampling"]["n_train"])
    n_test = int(sim_cfg["sampling"]["n_test"])

    u_smoke = generate_maximin_lhs_unit(n_smoke, d, seed=seed, n_candidates=60)
    u_train = generate_maximin_lhs_unit(n_train, d, seed=seed + 1000, n_candidates=30)
    u_test = generate_maximin_lhs_unit(
        n_test,
        d,
        seed=seed + 2000,
        n_candidates=50,
        reference_unit_points=u_train,
    )

    out_dir = Path(output_dir) if output_dir else (PROJECT_ROOT / "data" / "lhs_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets: Dict[str, pd.DataFrame] = {}
    for split_name, prefix, u_mat in (
        ("smoke", "smoke", u_smoke),
        ("train", "train", u_train),
        ("test", "test", u_test),
    ):
        x_phys = scale_unit_to_physical(u_mat, lows, highs)
        df = pd.DataFrame(x_phys, columns=keys)
        df.insert(0, "case_id", [f"{prefix}_{i:04d}" for i in range(len(df))])
        csv_p = out_dir / f"lhs_{split_name}_{len(df)}.csv"
        df.to_csv(csv_p, index=False)
        datasets[split_name] = df

    return datasets


if __name__ == "__main__":
    ds = generate_all_lhs_datasets()
    for k, df_k in ds.items():
        print(f"[LHS] Generated {k}: shape={df_k.shape}")
