"""Model definitions: keep model, PTP-type model, post-call update model.

Design choices (all evaluated in ``scripts/train.py``; see reports/):

* **Keep model** - seed-bagged gradient-boosted trees on interpretable,
  point-in-time features, followed by hierarchical per-portfolio Platt
  calibration fitted on grouped out-of-fold predictions. TreeSHAP
  contributions (native in XGBoost) are aggregated into reason codes.
* **Type model** - six-way classifier trained on the 200 annotated PTPs,
  augmented with *outcome-aware self-training*: for un-annotated training
  PTPs the posterior P(type | features, observed outcome) is used as a
  pseudo-label when confident. Outcomes are never used at inference.
  Decisions use a cost matrix that makes "hardship classified as a harsh
  type" five times as expensive as any other error.
* **Post-call model** - a logistic update of the in-call log-odds with
  post-call engagement signals (link opened, confirmation reply, inbound
  call), learnt only on PTPs still unresolved at the decision horizon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from .calibration import HierarchicalPlatt, logit, sigmoid
from .features import CATEGORICAL

# --------------------------------------------------------------------------- feature sets
NON_FEATURES = {
    "ptp_id", "account_id", "attempt_id", "agent_id", "channel", "source", "captured_ts", "promised_date",
    "promised_amount", "answerer_recorded", "payment_link_sent", "split", "kept", "partial", "any_payment",
    "paid_fraction", "recovered_amount", "outcome_status", "first_payment_ts", "resolution_ts",
    "annotated_type", "annotator_confidence", "call_id", "prefix_turns", "prefix_frac", "confirm_reached",
    "customer_text", "agent_text",
}
# Promise terms are only known once the promise is being confirmed.
TERMS_PREFIX = "terms_"
TERMS_ALWAYS_AVAILABLE = {"terms_capture_hour", "terms_hours_to_shift_end", "terms_capture_dom"}
# Known only after the call has ended / disposition is saved.
END_OF_CALL_ONLY = ["call_talk_s", "call_hangup_customer", "call_disposition_third_party", "ptp_answerer",
                    "ptp_link_sent", "remark_copies_before"]
VOICE_PREFIX = "voice_"
POST_CALL_FEATURES = ["post_payment_link_opened", "post_confirmation_reply_yes", "post_confirmation_reply_no",
                      "post_inbound_call_from_customer", "post_confirmation_unanswered", "post_hours_to_link_open"]

TYPE_CLASSES = ["genuine_feasible", "genuine_infeasible", "escape", "third_party", "repeat_promiser",
                "agent_recorded_or_pushed"]
HARSH_TYPES = {"escape", "repeat_promiser", "agent_recorded_or_pushed"}

TYPE_FEATURES = [
    "cust_hedge", "cust_conditional", "cust_vague_date", "cust_specific_date", "borrower_stated_amount",
    "borrower_silent_on_terms", "agent_proposed_date_first", "cust_third_party", "cust_self_identify",
    "cust_hardship_any", "cust_hardship_kinds", "cust_partial_offer", "cust_closing_escape",
    "cust_prior_break_excuse", "agent_mentions_prior_break", "hist_total_broken", "hist_broken_share",
    "acct_prev_ptp_broken", "hist_total_prev_ptps", "agent_pressure", "agent_push_after_deferral",
    "cust_firm_commitment", "cust_link_refuse", "cust_link_accept", "cust_passive_agree",
    "confirm_day_matches_borrower", "confirm_amount_matches_borrower", "terms_salary_aligned",
    "terms_salary_promise_before_credit", "terms_salary_gap_days", "terms_days_to_promise", "voice_long_pauses",
    "voice_max_response_pause_to_ask_ms", "answerer_not_self", "call_disposition_third_party",
    "agent_third_party_message", "cust_refusal", "cust_pay_now", "negotiation_turns", "pressure_then_agree",
    "agree_and_close_same_turn", "cust_self_agency", "acct_ability_to_pay", "terms_amount_to_overdue",
    "agent_part_payment_ask", "cust_hardship_salary", "cust_hardship_business", "cust_hardship_medical",
    "cust_hardship_job_loss", "cust_hardship_no_funds", "field_met_family", "field_remark_third_party",
    "ptp_is_field", "field_photo_reused", "field_dwell_s", "agent_silent_terms_rate",
]


def candidate_features(df: pd.DataFrame, voice: bool = True) -> List[str]:
    cols = [c for c in df.columns if c not in NON_FEATURES and not c.startswith("post_")]
    if not voice:
        cols = [c for c in cols if not c.startswith(VOICE_PREFIX)]
    return cols


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["answerer_not_self"] = df["ptp_answerer"].astype(str).ne("self").astype(float) \
        if "ptp_answerer" in df else 0.0
    return df


def mask_for_prefix(df: pd.DataFrame) -> pd.DataFrame:
    """Hide information that is not yet available at a mid-call prefix."""
    df = df.copy()
    not_confirmed = df["confirm_reached"].fillna(0).values == 0
    not_final = df["prefix_frac"].values < 1.0
    for c in df.columns:
        if c.startswith(TERMS_PREFIX) and c not in TERMS_ALWAYS_AVAILABLE:
            df.loc[not_confirmed, c] = np.nan
    for c in END_OF_CALL_ONLY:
        if c in df:
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                df.loc[not_final, c] = np.nan
            else:
                df.loc[not_final, c] = np.nan
    if "answerer_not_self" in df:
        df.loc[not_final, "answerer_not_self"] = np.nan
    return df


def _to_frame(df: pd.DataFrame, cols: Sequence[str], categories: Dict[str, list]) -> pd.DataFrame:
    X = df.reindex(columns=list(cols)).copy()
    for c in cols:
        if c in categories:
            X[c] = pd.Categorical(X[c].astype(object), categories=categories[c])
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


# --------------------------------------------------------------------------- keep model
@dataclass
class KeepModel:
    params: dict
    features: List[str] = field(default_factory=list)
    n_seeds: int = 5
    calibrator: Optional[HierarchicalPlatt] = None
    categories: Dict[str, list] = field(default_factory=dict)
    boosters: List[xgb.XGBClassifier] = field(default_factory=list)
    segment_col: str = "acct_portfolio"

    def _frame(self, df: pd.DataFrame) -> pd.DataFrame:
        return _to_frame(df, self.features, self.categories)

    def _fit_raw(self, df: pd.DataFrame, y: np.ndarray, w: Optional[np.ndarray] = None) -> List[xgb.XGBClassifier]:
        X = self._frame(df)
        models = []
        for s in range(self.n_seeds):
            m = xgb.XGBClassifier(**{**self.params, "random_state": 1000 + s, "enable_categorical": True})
            m.fit(X, y, sample_weight=w)
            models.append(m)
        return models

    @staticmethod
    def _predict_raw(models, X) -> np.ndarray:
        return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    def fit(self, df: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_folds: int = 5,
            w: Optional[np.ndarray] = None) -> "KeepModel":
        self.categories = {c: sorted(df[c].dropna().astype(str).unique().tolist())
                           for c in self.features if c in CATEGORICAL or df[c].dtype == object
                           or isinstance(df[c].dtype, pd.CategoricalDtype)}
        y = np.asarray(y)
        oof = np.zeros(len(df))
        for tr, va in GroupKFold(n_folds).split(df, y, groups):
            ms = self._fit_raw(df.iloc[tr], y[tr], None if w is None else w[tr])
            oof[va] = self._predict_raw(ms, self._frame(df.iloc[va]))
        self.oof_raw_ = oof
        self.calibrator = HierarchicalPlatt().fit(oof, y, df[self.segment_col].astype(str).values)
        self.boosters = self._fit_raw(df, y, w)
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_raw(self.boosters, self._frame(df))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(df)
        return self.calibrator.transform(raw, df[self.segment_col].astype(str).values)

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """TreeSHAP contributions in log-odds space (seed-averaged); last column is bias."""
        X = self._frame(df)
        d = xgb.DMatrix(X, enable_categorical=True)
        c = np.mean([m.get_booster().predict(d, pred_contribs=True) for m in self.boosters], axis=0)
        return pd.DataFrame(c, columns=self.features + ["__bias__"], index=df.index)


# --------------------------------------------------------------------------- type model
COST_HARDSHIP_TO_HARSH = 5.0


def type_cost_matrix(classes: Sequence[str]) -> np.ndarray:
    """cost[true, pred]."""
    k = len(classes)
    C = np.ones((k, k)) - np.eye(k)
    gi = list(classes).index("genuine_infeasible")
    for j, c in enumerate(classes):
        if c in HARSH_TYPES:
            C[gi, j] = COST_HARDSHIP_TO_HARSH
    return C


@dataclass
class TypeModel:
    features: List[str] = field(default_factory=lambda: list(TYPE_FEATURES))
    pseudo_threshold: float = 0.7
    pseudo_weight: float = 0.5
    classes: List[str] = field(default_factory=lambda: list(TYPE_CLASSES))
    params: dict = field(default_factory=lambda: dict(n_estimators=300, learning_rate=0.05, max_depth=2,
                                                      subsample=0.8, colsample_bytree=0.6, n_jobs=8,
                                                      random_state=7))
    cost_sensitive: bool = True
    model: Optional[xgb.XGBClassifier] = None

    def _X(self, df):
        return df.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce").astype(float)

    def fit(self, gold: pd.DataFrame, y_gold: np.ndarray, w_gold: np.ndarray,
            pool: Optional[pd.DataFrame] = None, kept_pool: Optional[np.ndarray] = None,
            kept_gold: Optional[np.ndarray] = None) -> "TypeModel":
        idx = {c: i for i, c in enumerate(self.classes)}
        yg = np.array([idx[v] for v in y_gold])
        m = xgb.XGBClassifier(**self.params).fit(self._X(gold), yg, sample_weight=w_gold)
        self.n_pseudo_ = 0
        if pool is not None and len(pool) and kept_gold is not None:
            k = len(self.classes)
            keep_rate = np.array([(kept_gold[yg == j].sum() + 1) / ((yg == j).sum() + 2) for j in range(k)])
            pu = m.predict_proba(self._X(pool))
            lik = np.where(np.asarray(kept_pool)[:, None] == 1, keep_rate[None, :], 1 - keep_rate[None, :])
            post = pu * lik
            post /= post.sum(1, keepdims=True)
            sel = post.max(1) >= self.pseudo_threshold
            X2 = pd.concat([self._X(gold), self._X(pool)[sel]])
            y2 = np.r_[yg, post[sel].argmax(1)]
            w2 = np.r_[w_gold, self.pseudo_weight * post[sel].max(1)]
            m = xgb.XGBClassifier(**self.params).fit(X2, y2, sample_weight=w2)
            self.n_pseudo_ = int(sel.sum())
        self.model = m
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(df))

    def decide(self, proba: np.ndarray) -> np.ndarray:
        if not self.cost_sensitive:
            return np.array(self.classes)[proba.argmax(1)]
        risk = proba @ type_cost_matrix(self.classes)  # expected cost of each decision
        return np.array(self.classes)[risk.argmin(1)]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.decide(self.predict_proba(df))


# --------------------------------------------------------------------------- post-call model
@dataclass
class PostCallModel:
    features: List[str] = field(default_factory=lambda: list(POST_CALL_FEATURES))
    model: Optional[LogisticRegression] = None

    def _X(self, logit_in_call: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        P = df.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce").astype(float).copy()
        P["post_hours_to_link_open"] = np.log1p(P["post_hours_to_link_open"].fillna(48.0))
        P = P.fillna(0.0)
        return np.column_stack([logit_in_call, P.values])

    def fit(self, p_in_call: np.ndarray, df: pd.DataFrame, y: np.ndarray) -> "PostCallModel":
        self.model = LogisticRegression(C=1.0, max_iter=2000).fit(self._X(logit(p_in_call), df), y)
        return self

    def predict(self, p_in_call: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(logit(p_in_call), df))[:, 1]

    def coefficients(self) -> Dict[str, float]:
        names = ["in_call_log_odds"] + self.features
        return dict(zip(names, self.model.coef_[0].round(4).tolist()))
