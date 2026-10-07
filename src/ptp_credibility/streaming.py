"""Live, in-call scoring: prefix model and the "heard enough" stopping rule.

The in-call model is trained on every prefix of every training call (each
call down-weighted by its length so long calls do not dominate). Information
that is not yet known mid-call is masked (see ``models.mask_for_prefix``).

Knowing when it has heard enough
--------------------------------
A second, light model (``SufficiencyModel``) predicts, from the current
prefix state, how much the score is still expected to move before the call
ends: E[|p_final - p_t|]. The service raises its in-call prompt as soon as

* a promise is on the table (borrower has stated terms or the agent is
  confirming), and
* the expected remaining movement is below ``delta`` (default 0.05),

which leaves the agent or bot time to firm up the promise while the borrower
is still on the line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .models import KeepModel, _to_frame

SUFFICIENCY_EXTRA = ["prefix_turns", "confirm_reached", "_p_now"]


@dataclass
class SufficiencyModel:
    features: List[str] = field(default_factory=list)
    categories: dict = field(default_factory=dict)
    model: Optional[xgb.XGBRegressor] = None
    delta: float = 0.05

    def fit(self, prefix: pd.DataFrame, p_now: np.ndarray, p_final: np.ndarray, w: np.ndarray) -> "SufficiencyModel":
        X = _to_frame(prefix.assign(_p_now=p_now), self.features + SUFFICIENCY_EXTRA, self.categories)
        y = np.abs(p_final - p_now)
        self.model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                      colsample_bytree=0.7, enable_categorical=True, n_jobs=8,
                                      objective="reg:squarederror").fit(X, y, sample_weight=w)
        return self

    def predict(self, prefix: pd.DataFrame, p_now: np.ndarray) -> np.ndarray:
        X = _to_frame(prefix.assign(_p_now=p_now), self.features + SUFFICIENCY_EXTRA, self.categories)
        return np.clip(self.model.predict(X), 0, 1)


def promise_on_table(prefix: pd.DataFrame) -> np.ndarray:
    return ((prefix["confirm_reached"].fillna(0) > 0) | (prefix["borrower_stated_date"].fillna(0) > 0) |
            (prefix["borrower_stated_amount"].fillna(0) > 0)).values


def decision_points(prefix: pd.DataFrame, p: np.ndarray, expected_move: np.ndarray, delta: float) -> pd.DataFrame:
    """First prefix per call at which the stopping rule fires (fallback: end of call)."""
    d = prefix[["call_id", "prefix_turns", "prefix_frac"]].copy()
    d["p"] = p
    d["fire"] = promise_on_table(prefix) & (expected_move < delta)
    d = d.sort_values(["call_id", "prefix_turns"])
    first = d[d.fire].groupby("call_id").head(1).set_index("call_id")
    last = d.groupby("call_id").tail(1).set_index("call_id")
    out = last[["p", "prefix_turns", "prefix_frac"]].rename(columns={"p": "p_final", "prefix_turns": "n_turns"})
    out["p_decision"] = first["p"].reindex(out.index).fillna(out["p_final"])
    out["decision_turn"] = first["prefix_turns"].reindex(out.index).fillna(out["n_turns"])
    out["fired_before_end"] = first["p"].reindex(out.index).notna() & (out["decision_turn"] < out["n_turns"])
    out["turns_remaining"] = out["n_turns"] - out["decision_turn"]
    out["frac_heard"] = out["decision_turn"] / out["n_turns"]
    return out.reset_index()
