"""Report figures (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

PALETTE = ["#1f3b73", "#2e86ab", "#a23b72", "#f18f01", "#3b8f4f", "#6c757d"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def roc(T: pd.DataFrame, out: Path, metrics: dict):
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for col, lab, c in (("p_keep", "Final model", PALETTE[0]), ("p_casemix", "Account + history only", PALETTE[3]),
                        ("hist_keep_rate_smoothed", "Past keep-rate heuristic", PALETTE[5])):
        fpr, tpr, _ = roc_curve(T.kept, T[col])
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, color=c, lw=1.8, label=f"{lab} (AUC {roc_auc_score(T.kept, T[col]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="Incumbent fixed rule (0.500)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title("Keep prediction, test split")
    ax.legend(loc="lower right", fontsize=7)
    _save(fig, out / "roc_test.png")


def calibration(T: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for ax, col, title in ((axes[0], "p_raw", "Before calibration"), (axes[1], "p_keep", "After hierarchical calibration")):
        for i, (pf, g) in enumerate(T.groupby("acct_portfolio", observed=True)):
            q = pd.qcut(g[col], 4, duplicates="drop")
            s = g.groupby(q, observed=True).agg(pred=(col, "mean"), obs=("kept", "mean"))
            ax.plot(s.pred, s.obs, "o-", color=PALETTE[i % len(PALETTE)], lw=1.2, ms=3, label=str(pf))
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
        ax.set_title(title); ax.set_xlabel("Predicted keep probability"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed keep rate"); axes[1].legend(fontsize=7, loc="upper left")
    _save(fig, out / "calibration_by_portfolio.png")


def streaming_curve(metrics: dict, out: Path):
    c = pd.DataFrame(metrics["streaming_test"]["auc_by_fraction_heard"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(c.fraction_of_call_heard, c.auc, "o-", color=PALETTE[0], lw=1.8)
    sr = metrics["streaming_test"]["stopping_rule"]
    ax.axhline(sr["auc_at_decision"], color=PALETTE[3], ls="--", lw=1.2,
               label=f"Stopping rule: AUC {sr['auc_at_decision']:.3f} at median {sr['median_fraction_heard']:.0%} heard")
    ax.set_xlabel("Fraction of call heard"); ax.set_ylabel("AUC (test)"); ax.set_title("Live score quality during the call")
    ax.legend(fontsize=7, loc="lower right"); ax.set_ylim(0.5, max(0.85, c.auc.max() + 0.02))
    _save(fig, out / "streaming_auc_by_progress.png")


def families(fam: pd.Series, out: Path):
    s = (fam / fam.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.barh(s.index, s.values, color=PALETTE[1])
    for i, v in enumerate(s.values):
        ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlabel("Share of mean |TreeSHAP| attribution"); ax.set_title("What drives the credibility score")
    _save(fig, out / "signal_family_attribution.png")


def by_type(T: pd.DataFrame, out: Path):
    order = ["genuine_feasible", "third_party", "genuine_infeasible", "escape", "repeat_promiser", "agent_recorded_or_pushed"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    data = [T.loc[T.ptp_type == t, "p_keep"].values for t in order]
    ax.boxplot([d if len(d) else [np.nan] for d in data], labels=[o.replace("_", "\n") for o in order], showfliers=False)
    obs = [T.loc[T.ptp_type == t, "kept"].mean() for t in order]
    ax.plot(range(1, len(order) + 1), obs, "D", color=PALETTE[3], label="Observed keep rate")
    ax.set_ylabel("Calibrated keep probability"); ax.set_title("Score by predicted PTP type (test)"); ax.legend(fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    _save(fig, out / "score_by_type.png")


def agents(integ: pd.DataFrame, out: Path):
    d = integ[integ.channel != "voice_bot"].sort_values("ptp_quality_ratio")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(d))
    ax.bar(x, d.raw_ptps, color="#d0d7e2", label="Raw PTPs")
    ax.bar(x, d.quality_weighted_ptps, color=PALETTE[0], label="Quality-weighted PTPs")
    for i, f in enumerate(d.integrity_flag):
        if f:
            ax.text(i, d.raw_ptps.iloc[i] + 3, "flag", ha="center", fontsize=6, color=PALETTE[2], rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(d.agent_id, rotation=90, fontsize=6)
    ax.set_ylabel("PTPs"); ax.set_title("Raw vs quality-weighted PTPs per agent"); ax.legend(fontsize=7)
    _save(fig, out / "agent_quality_weighted_ptps.png")


def make_all(df, T, metrics, fam, integ, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    roc(T, out, metrics)
    calibration(T, out)
    streaming_curve(metrics, out)
    families(fam, out)
    by_type(T, out)
    agents(integ, out)
