"""Evaluation utilities: discrimination, calibration, bootstrap CIs, trial power."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .calibration import expected_calibration_error


def safe_auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def bootstrap_ci(y, p, metric=safe_auc, n: int = 1000, seed: int = 0, groups: Optional[np.ndarray] = None):
    """Percentile CI; resamples accounts (clusters) when ``groups`` is given."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    if groups is not None:
        g = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(list).tolist()
        for _ in range(n):
            idx = np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
            vals.append(metric(y[idx], p[idx]))
    else:
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            vals.append(metric(y[idx], p[idx]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def binary_report(y, p, groups=None, ci: bool = True) -> Dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    r = {
        "n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()),
        "auc": safe_auc(y, p), "pr_auc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece": expected_calibration_error(y, p), "mean_pred": float(p.mean()),
    }
    if ci:
        lo, hi = bootstrap_ci(y, p, groups=groups)
        r["auc_ci95"] = [lo, hi]
    return r


def segment_report(df: pd.DataFrame, y_col: str, p_col: str, seg_col: str, min_n: int = 20) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby(seg_col, observed=True):
        if len(g) < min_n:
            continue
        rows.append({seg_col: s, "n": len(g), "keep_rate": g[y_col].mean(), "mean_pred": g[p_col].mean(),
                     "auc": safe_auc(g[y_col], g[p_col]), "ece": expected_calibration_error(g[y_col].values, g[p_col].values)})
    return pd.DataFrame(rows)


def broken_flag_recall(y, p, threshold: float) -> Dict[str, float]:
    """Share of broken PTPs flagged (p < threshold) and the false-alarm rate on kept PTPs."""
    y, p = np.asarray(y), np.asarray(p)
    flag = p < threshold
    return {"threshold": float(threshold), "broken_flagged_share": float(flag[y == 0].mean()),
            "kept_false_alarm_share": float(flag[y == 1].mean()), "flag_rate": float(flag.mean()),
            "precision_broken": float((y[flag] == 0).mean()) if flag.any() else float("nan")}


def threshold_for_false_alarm(y, p, max_kept_flagged: float = 0.20) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(np.quantile(p[y == 1], max_kept_flagged))


def two_proportion_sample_size(p0: float, uplift_abs: float, alpha: float = 0.05, power: float = 0.8,
                               icc: float = 0.0, cluster_size: float = 1.0) -> int:
    """Per-arm sample size for a two-sided two-proportion test, with design effect."""
    p1 = p0 + uplift_abs
    pbar = (p0 + p1) / 2
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    n = ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / uplift_abs ** 2
    deff = 1 + (cluster_size - 1) * icc
    return int(np.ceil(n * deff))
