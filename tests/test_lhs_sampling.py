"""
测试 Maximin LHS 抽样器的分层均匀性 (Latin Hypercube property)、物理边界包含性与可复现性 (seed=42)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.units import GLOBAL_SEED, MMHG_TO_MPA, load_yaml_config
from src.surrogate.lhs_sampling import (
    generate_all_lhs_datasets,
    generate_maximin_lhs_unit,
    get_parameter_specs,
)

assert MMHG_TO_MPA == 0.000133322
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_maximin_lhs_stratification_and_reproducibility(tmp_path: Path) -> None:
    u1 = generate_maximin_lhs_unit(20, 10, seed=GLOBAL_SEED, n_candidates=25)
    u2 = generate_maximin_lhs_unit(20, 10, seed=GLOBAL_SEED, n_candidates=25)
    assert np.allclose(u1, u2, atol=0.0)

    # 严格校验拉丁超立方性质：每一维在 20 个等宽区间 [k/20, (k+1)/20) 内恰好各含 1 个样本
    for dim in range(10):
        bins = np.floor(u1[:, dim] * 20).astype(int)
        assert sorted(bins.tolist()) == list(range(20))

    ds = generate_all_lhs_datasets(output_dir=tmp_path)
    assert len(ds["smoke"]) == 20
    assert len(ds["train"]) == 400
    assert len(ds["test"]) == 80

    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    keys, lows, highs = get_parameter_specs(params_cfg)
    for split_df in ds.values():
        arr = split_df[keys].to_numpy(dtype=float)
        assert np.all(arr >= lows - 1e-12)
        assert np.all(arr <= highs + 1e-12)
