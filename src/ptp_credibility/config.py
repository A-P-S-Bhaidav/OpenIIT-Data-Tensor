"""Central configuration for the PTP credibility system.

All tunable business rules (label definition, grace windows, operating
thresholds, parking windows) live here so that they can be reviewed and
signed off by the business and compliance teams in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = REPO_ROOT
    ps1_dir: Path = REPO_ROOT / "ps1_fake_ptp"
    shared_dir: Path = REPO_ROOT / "shared"
    artifacts: Path = REPO_ROOT / "artifacts"
    reports: Path = REPO_ROOT / "reports"
    figures: Path = REPO_ROOT / "reports" / "figures"

    def ensure(self) -> "Paths":
        for p in (self.artifacts, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class LabelConfig:
    """Business definition of a 'kept' promise (see docs/LABEL_DEFINITION.md).

    A PTP is KEPT when cumulative payments on the account, from the moment the
    promise was captured until ``grace_days`` after the promised date, reach at
    least ``min_fraction`` of the promised amount. Payments through any channel
    count (UPI link, UPI app, NACH re-presentation, branch, NEFT, BBPS, field
    cash). When a newer PTP is captured on the same account before a payment
    lands, the payment is attributed to the newer promise only.
    """

    grace_days: int = 3
    min_fraction: float = 0.90
    partial_fraction: float = 0.25
    sensitivity_grace_days: tuple = (0, 1, 3, 5, 7)
    sensitivity_fractions: tuple = (0.5, 0.75, 0.9, 1.0)


@dataclass(frozen=True)
class PostCallConfig:
    """Post-call update window. Events are only used if they occur before the
    decision horizon and before any payment (so the label never leaks)."""

    horizon_hours: float = 30.0


@dataclass(frozen=True)
class PolicyConfig:
    """Operating thresholds for the disposition engine (calibrated scale)."""

    high_credibility: float = 0.55
    low_credibility: float = 0.25
    default_parking_days_cap: int = 15
    weak_promise_parking_cap_days: int = 2
    escape_follow_up_hours: int = 24
    repeat_parking_cap_days: int = 3
    hardship_route: str = "restructuring_desk_or_human"


@dataclass(frozen=True)
class ModelConfig:
    random_state: int = 20260406
    n_folds: int = 5
    xgb_params: dict = field(
        default_factory=lambda: dict(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            reg_alpha=0.0,
            gamma=0.0,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=8,
        )
    )


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    label: LabelConfig = field(default_factory=LabelConfig)
    post_call: PostCallConfig = field(default_factory=PostCallConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


CONFIG = Config()
