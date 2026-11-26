"""
Matérn-5/2 ARD 高斯过程 (GP) 代理模型训练、交叉验证选型、独立测试集验证与主动学习补点模块
强制单位体系: mm - N - MPa (1 mmHg = 0.000133322 MPa)
全局随机种子: seed = 42

验收门禁:
  独立测试集 (N_test = 80) 预测决定系数 Q^2 > 0.98，范围归一化均方根误差 NRMSE < 2%。
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

from src.common.units import (
    FORCE_UNIT,
    GLOBAL_SEED,
    LENGTH_UNIT,
    MMHG_TO_MPA,
    STRESS_UNIT,
    load_yaml_config,
)
from src.surrogate.lhs_sampling import get_parameter_specs, normalize_physical_to_unit, scale_unit_to_physical

UNIT_SYSTEM = f"{LENGTH_UNIT}-{FORCE_UNIT}-{STRESS_UNIT}"
assert MMHG_TO_MPA == 0.000133322
assert GLOBAL_SEED == 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORE_QOIS = ("lc_e1_p95", "lc_eeff_p95", "lc_gmax_p95", "lc_ap_disp_p95")


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    q2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    y_range = float(np.max(yt) - np.min(yt))
    nrmse_range = rmse / max(y_range, 1e-30)
    nrmse_std = rmse / max(float(np.std(yt)), 1e-30)
    return {
        "Q2": q2,
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE": nrmse_range,
        "NRMSE_pct": nrmse_range * 100.0,
        "NRMSE_std": nrmse_std,
        "max_abs_err": float(np.max(np.abs(yt - yp))),
    }


@dataclass
class SingleQoIGPSurrogate:
    qoi_name: str
    param_keys: List[str]
    lows: np.ndarray
    highs: np.ndarray
    use_log_transform: bool
    log_shift: float
    gpr: GaussianProcessRegressor
    ard_length_scales: np.ndarray
    train_cv_q2: float

    def predict(self, x_phys: np.ndarray, return_std: bool = False, batch_size: int = 32768) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        x_arr = np.atleast_2d(np.asarray(x_phys, dtype=float))
        u = normalize_physical_to_unit(x_arr, self.lows, self.highs)
        n_pts = len(u)

        if return_std:
            z_mu = np.empty(n_pts, dtype=float)
            z_std = np.empty(n_pts, dtype=float)
            for s in range(0, n_pts, batch_size):
                e = min(s + batch_size, n_pts)
                z_mu[s:e], z_std[s:e] = self.gpr.predict(u[s:e], return_std=True)
            if self.use_log_transform:
                # 对数正态变换的一阶精确中位/均值还原与标准差传播
                y_mu = np.exp(z_mu + 0.5 * (z_std**2)) - self.log_shift
                y_std = (y_mu + self.log_shift) * np.sqrt(np.maximum(np.expm1(z_std**2), 0.0))
                return y_mu, y_std
            return z_mu, z_std

        z_mu = np.empty(n_pts, dtype=float)
        for s in range(0, n_pts, batch_size):
            e = min(s + batch_size, n_pts)
            z_mu[s:e] = self.gpr.predict(u[s:e], return_std=False)
        if self.use_log_transform:
            return np.exp(z_mu) - self.log_shift
        return z_mu


def _build_matern52_ard_gpr(n_dim: int, n_restarts: int = 20, seed: int = GLOBAL_SEED) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-4, 1e5)) * Matern(
        length_scale=np.full(n_dim, 0.8, dtype=float),
        length_scale_bounds=(1e-2, 1e5),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-12, 1e-2))
    return GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        normalize_y=True,
        random_state=seed,
    )


def fit_single_qoi_gp(
    x_train_phys: np.ndarray,
    y_train: np.ndarray,
    qoi_name: str,
    param_keys: List[str],
    lows: np.ndarray,
    highs: np.ndarray,
    n_restarts: int = 20,
    seed: int = GLOBAL_SEED,
) -> SingleQoIGPSurrogate:
    """
    仅在训练集内部通过 5 折交叉验证比较原空间 vs 对数变换空间的泛化 Q^2（严禁触碰独立测试集），
    然后用最优表达在全量训练集上拟合 Matérn-5/2 ARD 高斯过程。
    """
    u_train = normalize_physical_to_unit(x_train_phys, lows, highs)
    y_arr = np.asarray(y_train, dtype=float).ravel()
    n_dim = u_train.shape[1]

    # 判断是否可构造严格正值对数变换 (针对乘性非线性应变场)
    can_log = bool(np.min(y_arr) > 0.0)
    log_shift = 0.0

    kf = KFold(n_splits=min(5, len(y_arr)), shuffle=True, random_state=seed)
    cv_preds_raw = np.zeros_like(y_arr)
    for tr_idx, va_idx in kf.split(u_train):
        m_raw = _build_matern52_ard_gpr(n_dim, n_restarts=max(5, n_restarts // 3), seed=seed)
        m_raw.fit(u_train[tr_idx], y_arr[tr_idx])
        cv_preds_raw[va_idx] = m_raw.predict(u_train[va_idx])
    q2_cv_raw = compute_regression_metrics(y_arr, cv_preds_raw)["Q2"]

    use_log = False
    best_cv_q2 = q2_cv_raw

    if can_log:
        cv_preds_log = np.zeros_like(y_arr)
        z_all = np.log(y_arr + log_shift)
        for tr_idx, va_idx in kf.split(u_train):
            m_log = _build_matern52_ard_gpr(n_dim, n_restarts=max(5, n_restarts // 3), seed=seed)
            m_log.fit(u_train[tr_idx], z_all[tr_idx])
            z_p, z_s = m_log.predict(u_train[va_idx], return_std=True)
            cv_preds_log[va_idx] = np.exp(z_p + 0.5 * (z_s**2)) - log_shift
        q2_cv_log = compute_regression_metrics(y_arr, cv_preds_log)["Q2"]
        if q2_cv_log > q2_cv_raw + 1e-4:
            use_log = True
            best_cv_q2 = q2_cv_log

    gpr_final = _build_matern52_ard_gpr(n_dim, n_restarts=n_restarts, seed=seed)
    target_z = np.log(y_arr + log_shift) if use_log else y_arr
    gpr_final.fit(u_train, target_z)

    # 提取训练后 ARD 核长度尺度: kernel_ = (Constant * Matern) + White
    matern_k = gpr_final.kernel_.k1.k2
    ard_ls = np.asarray(matern_k.length_scale, dtype=float).copy()

    return SingleQoIGPSurrogate(
        qoi_name=qoi_name,
        param_keys=list(param_keys),
        lows=lows.copy(),
        highs=highs.copy(),
        use_log_transform=use_log,
        log_shift=log_shift,
        gpr=gpr_final,
        ard_length_scales=ard_ls,
        train_cv_q2=best_cv_q2,
    )


class MultiQoIGPSurrogateBundle:
    """管理全部 QoI 的 Matérn-5/2 ARD GP 模型束、独立测试集门禁校验与持久化。"""

    def __init__(self, models: Dict[str, SingleQoIGPSurrogate], param_keys: List[str], lows: np.ndarray, highs: np.ndarray):
        self.models = models
        self.param_keys = param_keys
        self.lows = lows
        self.highs = highs

    def predict_qoi(self, qoi_name: str, x_phys: np.ndarray, return_std: bool = False):
        return self.models[qoi_name].predict(x_phys, return_std=return_std)

    def predict_all(self, x_phys: np.ndarray) -> Dict[str, np.ndarray]:
        return {qk: mdl.predict(x_phys, return_std=False) for qk, mdl in self.models.items()}

    def evaluate_on_testset(self, df_test: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        df_ok = df_test[df_test["converged"] == True].copy()
        x_test = df_ok[self.param_keys].to_numpy(dtype=float)
        report: Dict[str, Dict[str, float]] = {}
        for qk, mdl in self.models.items():
            yt = df_ok[qk].to_numpy(dtype=float)
            yp, ystd = mdl.predict(x_test, return_std=True)
            m = compute_regression_metrics(yt, yp)
            m["train_5fold_cv_Q2"] = mdl.train_cv_q2
            m["use_log_transform"] = float(mdl.use_log_transform)
            # 校验 95% 预测置信区间覆盖率
            cov95 = float(np.mean(np.abs(yt - yp) <= 1.96 * np.maximum(ystd, 1e-12)))
            m["empirical_95ci_coverage"] = cov95
            report[qk] = m
        return report

    def propose_active_learning_points(
        self,
        existing_x_phys: np.ndarray,
        n_new: int = 20,
        n_pool: int = 16384,
        seed: int = GLOBAL_SEED + 5000,
    ) -> np.ndarray:
        """
        基于各 QoI 标准化预测方差最大化 + 空间排斥准则，从 Sobol 候选池中贪心挑选高不确定度补点。
        """
        u_exist = normalize_physical_to_unit(existing_x_phys, self.lows, self.highs)
        sob = qmc.Sobol(d=len(self.param_keys), scramble=True, seed=seed)
        u_pool = sob.random(n=n_pool)
        x_pool = scale_unit_to_physical(u_pool, self.lows, self.highs)

        norm_stds = []
        for mdl in self.models.values():
            mu_p, std_p = mdl.predict(x_pool, return_std=True)
            scale_ref = max(float(np.ptp(mu_p)), 1e-9)
            norm_stds.append(std_p / scale_ref)
        joint_unc = np.max(np.vstack(norm_stds), axis=0)

        selected_u: List[np.ndarray] = []
        selected_idx: List[int] = []
        cur_ref = u_exist.copy()

        for _ in range(n_new):
            d_min = np.min(cdist(u_pool, cur_ref), axis=1)
            score = joint_unc * (d_min**0.5)
            for idx_used in selected_idx:
                score[idx_used] = -1.0
            best_i = int(np.argmax(score))
            selected_idx.append(best_i)
            selected_u.append(u_pool[best_i])
            cur_ref = np.vstack([cur_ref, u_pool[best_i : best_i + 1]])

        return x_pool[selected_idx]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "MultiQoIGPSurrogateBundle":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def train_and_validate_gp_bundle(
    train_csv: str | Path,
    test_csv: str | Path,
    qoi_keys: Tuple[str, ...] = CORE_QOIS,
    save_dir: Optional[str | Path] = None,
) -> Tuple[MultiQoIGPSurrogateBundle, Dict[str, Dict[str, float]]]:
    params_cfg = load_yaml_config(PROJECT_ROOT / "config" / "parameters.yaml")
    sim_cfg = load_yaml_config(PROJECT_ROOT / "config" / "sim_settings.yaml")
    param_keys, lows, highs = get_parameter_specs(params_cfg)
    n_restarts = int(sim_cfg["surrogate"]["n_restarts_optimizer"])

    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)
    df_tr_ok = df_tr[df_tr["converged"] == True].copy()

    x_train = df_tr_ok[param_keys].to_numpy(dtype=float)
    models: Dict[str, SingleQoIGPSurrogate] = {}

    for idx, qk in enumerate(qoi_keys):
        y_tr = df_tr_ok[qk].to_numpy(dtype=float)
        models[qk] = fit_single_qoi_gp(
            x_train_phys=x_train,
            y_train=y_tr,
            qoi_name=qk,
            param_keys=param_keys,
            lows=lows,
            highs=highs,
            n_restarts=n_restarts,
            seed=GLOBAL_SEED + idx,
        )

    bundle = MultiQoIGPSurrogateBundle(models=models, param_keys=param_keys, lows=lows, highs=highs)
    metrics = bundle.evaluate_on_testset(df_te)

    if save_dir is not None:
        sdir = Path(save_dir)
        sdir.mkdir(parents=True, exist_ok=True)
        bundle.save(sdir / "gp_surrogate_bundle.pkl")
        (sdir / "gp_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return bundle, metrics
