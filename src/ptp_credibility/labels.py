"""Outcome labelling: when is a promise to pay 'kept'?

The definition is deliberately explicit and parameterised (see
``LabelConfig``) because late, partial and other-channel payments blur the
outcome. The function returns a rich outcome frame rather than a single bit so
that downstream metrics can report kept / late / partial / broken separately
and run sensitivity analysis on the definition.

Attribution rule
----------------
For PTP *i* on account *a*, captured at ``t_i`` with promised date ``d_i``:

* the attribution window is ``[t_i, d_i + grace_days + 1 day)``;
* the window is truncated at the capture time of the next PTP on the same
  account, because a newer promise supersedes the old one (otherwise one
  payment would be double counted across overlapping promises);
* every payment channel counts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, LabelConfig


def build_outcomes(ptps: pd.DataFrame, payments: pd.DataFrame,
                   cfg: LabelConfig = CONFIG.label) -> pd.DataFrame:
    p = ptps[["ptp_id", "account_id", "captured_ts", "promised_date", "promised_amount"]].copy()
    p = p.sort_values(["account_id", "captured_ts"])
    p["next_capture_ts"] = p.groupby("account_id")["captured_ts"].shift(-1)

    m = p.merge(payments[["account_id", "payment_ts", "amount", "channel"]], on="account_id", how="left")
    m = m[m["payment_ts"].notna() & (m["payment_ts"] >= m["captured_ts"])]
    m = m[(m["next_capture_ts"].isna()) | (m["payment_ts"] < m["next_capture_ts"])]
    m["lag_days"] = (m["payment_ts"].dt.normalize() - m["promised_date"]).dt.days

    out = p.set_index("ptp_id")

    def paid_within(grace: int) -> pd.Series:
        return m[m["lag_days"] <= grace].groupby("ptp_id")["amount"].sum()

    out["paid_by_due"] = paid_within(0).reindex(out.index).fillna(0.0)
    out["paid_in_window"] = paid_within(cfg.grace_days).reindex(out.index).fillna(0.0)
    out["paid_30d"] = paid_within(30).reindex(out.index).fillna(0.0)
    first = m.sort_values("payment_ts").groupby("ptp_id").first()
    out["first_payment_ts"] = first["payment_ts"].reindex(out.index)
    out["first_payment_lag_days"] = first["lag_days"].reindex(out.index)
    out["first_payment_channel"] = first["channel"].reindex(out.index)

    frac = out["paid_in_window"] / out["promised_amount"].clip(lower=1.0)
    out["paid_fraction"] = frac.clip(upper=5.0)
    out["kept"] = (frac >= cfg.min_fraction).astype(int)
    out["kept_on_time"] = (out["paid_by_due"] / out["promised_amount"] >= cfg.min_fraction).astype(int)
    out["partial"] = ((frac >= cfg.partial_fraction) & (frac < cfg.min_fraction)).astype(int)
    out["any_payment"] = (out["paid_in_window"] > 0).astype(int)
    out["recovered_amount"] = np.minimum(out["paid_in_window"], out["promised_amount"])

    status = np.select(
        [out["kept_on_time"] == 1, out["kept"] == 1, out["partial"] == 1, out["paid_30d"] > 0],
        ["kept_on_time", "kept_late_within_grace", "partial", "paid_after_window"],
        default="broken",
    )
    out["outcome_status"] = status
    # time at which the outcome becomes known (used for point-in-time features)
    out["resolution_ts"] = out["promised_date"] + pd.Timedelta(days=cfg.grace_days + 1)
    return out.reset_index()


def label_sensitivity(ptps: pd.DataFrame, payments: pd.DataFrame,
                      cfg: LabelConfig = CONFIG.label) -> pd.DataFrame:
    """Keep-rate under alternative business definitions (grace x fraction)."""
    rows = []
    for g in cfg.sensitivity_grace_days:
        for f in cfg.sensitivity_fractions:
            o = build_outcomes(ptps, payments, LabelConfig(grace_days=g, min_fraction=f))
            rows.append({"grace_days": g, "min_fraction": f, "keep_rate": o["kept"].mean()})
    return pd.DataFrame(rows)
