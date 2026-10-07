"""Feature assembly.

Every feature is computed *point-in-time*: only information that would have
been available to the live system at the moment the PTP was captured is used.
In particular:

* prior-PTP outcomes are used only once they are resolved
  (``resolution_ts <= captured_ts``);
* agent track-record features use only resolved PTPs from *training-split*
  accounts, so no validation or test label ever enters a feature;
* post-call signals are cut at the decision horizon and at the first payment,
  so that a payment cannot leak into its own predictor.

Feature groups (prefixes) map one-to-one to the "Key signals" section of the
problem statement and drive the reason codes shown to agents and auditors.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import CONFIG
from .conversation import ConversationState, Turn
from .data import Tables

BUREAU_ORD = {"300-549": 0, "550-649": 1, "650-699": 2, "700-749": 3, "750-900": 4}
BUCKET_ORD = {"X": 0, "1-30": 1, "31-60": 2, "61-90": 3, "90+": 4}
SHIFT_END_HOUR = 18.0


# ----------------------------------------------------------------------------- transcripts
def iter_calls(transcripts: pd.DataFrame) -> Iterable[Tuple[str, List[Turn]]]:
    cols = ["speaker", "text", "pause_before_ms", "start_ms", "end_ms", "asr_confidence"]
    for call_id, g in transcripts.groupby("call_id", sort=False):
        turns = [Turn(*row) for row in g[cols].itertuples(index=False, name=None)]
        yield call_id, turns


def transcript_features(transcripts: pd.DataFrame, prefixes: bool = False) -> pd.DataFrame:
    """End-of-call features per call, or features after every turn if ``prefixes``."""
    rows = []
    for call_id, turns in iter_calls(transcripts):
        st = ConversationState()
        n = len(turns)
        for i, t in enumerate(turns):
            st.update(t)
            if prefixes:
                f = st.features()
                f.update(call_id=call_id, prefix_turns=i + 1, prefix_frac=(i + 1) / n,
                         confirm_reached=float(st.first_confirm_turn is not None),
                         customer_text=st.customer_text, agent_text=st.agent_text_joined)
                rows.append(f)
        if not prefixes:
            f = st.features()
            f.update(call_id=call_id, customer_text=st.customer_text, agent_text=st.agent_text_joined)
            rows.append(f)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- context
def _account_features(t: Tables) -> pd.DataFrame:
    a = t.accounts.copy()
    lt = t.lenders.set_index("lender_id")["lender_type"]
    out = pd.DataFrame({"account_id": a.account_id})
    out["acct_portfolio"] = a.portfolio.values
    out["acct_income_type"] = a.income_type.values
    out["acct_language"] = a.preferred_language.values
    out["acct_lender_type"] = a.lender_id.map(lt).values
    out["acct_bucket_ord"] = a.bucket_start.map(BUCKET_ORD).values
    out["acct_dpd_start"] = a.dpd_start.values
    out["acct_emi"] = a.emi_amount.values
    out["acct_overdue_start"] = a.overdue_start.values
    out["acct_outstanding"] = a.outstanding.values
    out["acct_salary_day"] = a.salary_credit_day.values
    out["acct_bureau_ord"] = a.bureau_score_band.map(BUREAU_ORD).values
    out["acct_other_loans"] = a.other_active_loans.values
    out["acct_paid_other_lenders_30d"] = a.paid_other_lenders_30d.astype(str).str.lower().eq("true").astype(float).values
    out["acct_bounce_reason"] = a.last_bounce_reason.values
    out["acct_ability_to_pay"] = a.ability_to_pay_estimate.values
    out["acct_prev_ptp_count"] = a.prev_ptp_count.values
    out["acct_prev_ptp_broken"] = a.prev_ptp_broken.values
    return out


def _ptp_terms(p: pd.DataFrame, acct: pd.DataFrame) -> pd.DataFrame:
    """Features of the promise itself (date and amount)."""
    emi = p.account_id.map(acct.set_index("account_id")["acct_emi"]).clip(lower=1).values
    out = pd.DataFrame({"ptp_id": p.ptp_id.values})
    days = (p.promised_date.values - p.captured_ts.dt.normalize().values) / np.timedelta64(1, "D")
    out["terms_days_to_promise"] = days
    out["terms_promise_dom"] = p.promised_date.dt.day.values
    out["terms_promise_dow"] = p.promised_date.dt.dayofweek.values
    out["terms_promise_is_month_end"] = (p.promised_date.dt.is_month_end | (p.promised_date.dt.day >= 28)).astype(float).values
    out["terms_amount"] = p.promised_amount.values
    out["terms_log_amount"] = np.log1p(p.promised_amount.values)
    out["terms_amount_to_overdue"] = (p.promised_amount / p.overdue_at_capture.clip(lower=1)).values
    out["terms_partial_promise"] = (p.promised_amount < p.overdue_at_capture - 1).astype(float).values
    out["terms_amount_to_emi"] = p.promised_amount.values / emi
    out["terms_overdue_emis"] = p.overdue_at_capture.values / emi
    out["terms_capture_hour"] = (p.captured_ts.dt.hour + p.captured_ts.dt.minute / 60).values
    out["terms_hours_to_shift_end"] = SHIFT_END_HOUR - out["terms_capture_hour"]
    out["terms_capture_dom"] = p.captured_ts.dt.day.values
    return out


SALARY_ALIGNED_MAX_GAP_DAYS = 2  # promise within two days after the credit (see docs/MODEL_CARD.md)


def _credit_date(year: int, month: int, day: int) -> pd.Timestamp:
    last = pd.Timestamp(year=year, month=month, day=1).days_in_month
    return pd.Timestamp(year=year, month=month, day=min(day, last))


def salary_alignment(promised_dates, salary_day) -> Dict[str, np.ndarray]:
    """Calendar-exact position of the promised date relative to the salary credit.

    ``salary_gap_days``    days since the most recent credit on or before the promised date
    ``days_to_next_credit`` days from the promised date to the next credit
    ``salary_aligned``     promise falls 0-2 days after a credit
    ``salary_promise_before_credit`` promise falls 1-5 days before the next credit
    """
    pdts = pd.to_datetime(pd.Series(promised_dates)).reset_index(drop=True)
    sd = pd.Series(np.asarray(salary_day, dtype=float)).reset_index(drop=True)
    n = len(pdts)
    gap = np.full(n, np.nan)
    to_next = np.full(n, np.nan)
    for i in range(n):
        d, s = pdts.iloc[i], sd.iloc[i]
        if pd.isna(d) or pd.isna(s):
            continue
        s = int(s)
        cur = _credit_date(d.year, d.month, s)
        prev_m = d - pd.DateOffset(months=1)
        last = cur if cur <= d else _credit_date(prev_m.year, prev_m.month, s)
        nxt_m = d + pd.offsets.MonthBegin(1)
        nxt = cur if cur > d else _credit_date(nxt_m.year, nxt_m.month, s)
        gap[i] = (d.normalize() - last.normalize()).days
        to_next[i] = (nxt.normalize() - d.normalize()).days
    known = ~np.isnan(gap)
    aligned = np.where(known, gap <= SALARY_ALIGNED_MAX_GAP_DAYS, np.nan)
    before = np.where(known, (to_next >= 1) & (to_next <= 5), np.nan)
    return {"salary_gap_days": gap, "salary_days_to_next_credit": to_next, "salary_aligned": aligned.astype(float),
            "salary_promise_before_credit": before.astype(float), "salary_day_known": known.astype(float)}


def _history_features(p: pd.DataFrame, outcomes: pd.DataFrame, t: Tables) -> pd.DataFrame:
    o = outcomes.set_index("ptp_id")
    base = p[["ptp_id", "account_id", "captured_ts"]].copy()
    base["kept"] = base.ptp_id.map(o["kept"])
    base["resolution_ts"] = base.ptp_id.map(o["resolution_ts"])
    rows = {}
    for acc, g in base.sort_values("captured_ts").groupby("account_id"):
        g = g.reset_index(drop=True)
        for i, r in g.iterrows():
            prior = g.iloc[:i]
            resolved = prior[prior.resolution_ts <= r.captured_ts]
            last = prior.iloc[-1] if len(prior) else None
            rows[r.ptp_id] = {
                "hist_inperiod_ptps": float(len(prior)),
                "hist_inperiod_resolved": float(len(resolved)),
                "hist_inperiod_broken": float((resolved.kept == 0).sum()),
                "hist_inperiod_kept": float((resolved.kept == 1).sum()),
                "hist_days_since_last_ptp": (r.captured_ts - last.captured_ts).total_seconds() / 86400 if last is not None else np.nan,
                "hist_open_ptp_at_capture": float(len(prior) - len(resolved) > 0),
            }
    h = pd.DataFrame.from_dict(rows, orient="index")
    h.index.name = "ptp_id"
    h = h.reset_index()
    return h


def _payment_history(p: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    m = p[["ptp_id", "account_id", "captured_ts"]].merge(payments[["account_id", "payment_ts", "amount"]], on="account_id", how="left")
    m = m[m.payment_ts < m.captured_ts]
    g = m.groupby("ptp_id")
    out = pd.DataFrame({"ptp_id": p.ptp_id})
    out["pay_prior_count"] = out.ptp_id.map(g.size()).fillna(0)
    out["pay_prior_amount"] = out.ptp_id.map(g.amount.sum()).fillna(0)
    last = g.payment_ts.max()
    out["pay_days_since_last"] = (p.set_index("ptp_id").captured_ts - last).dt.total_seconds().reindex(out.ptp_id).values / 86400
    return out


def _dial_features(p: pd.DataFrame, t: Tables) -> pd.DataFrame:
    da = t.dial_attempts
    out = pd.DataFrame({"ptp_id": p.ptp_id})
    # attempt that produced the PTP
    this = da.dropna(subset=["ptp_id"]).set_index("ptp_id")
    out["call_talk_s"] = out.ptp_id.map(this.talk_duration_s)
    out["call_hangup_customer"] = out.ptp_id.map(this.hangup_by.eq("customer").astype(float))
    out["call_disposition_third_party"] = out.ptp_id.map(this.disposition.eq("third_party_ptp").astype(float))
    # prior contact history (point-in-time)
    m = p[["ptp_id", "account_id", "captured_ts"]].merge(
        da[["account_id", "attempt_ts", "disposition", "network_response"]], on="account_id", how="left")
    m = m[m.attempt_ts < m.captured_ts - pd.Timedelta(minutes=1)]
    m30 = m[m.attempt_ts >= m.captured_ts - pd.Timedelta(days=30)]
    g = m30.groupby("ptp_id")
    out["dial_attempts_30d"] = out.ptp_id.map(g.size()).fillna(0)
    out["dial_answered_30d"] = out.ptp_id.map(g.network_response.apply(lambda s: (s == "answered").sum())).fillna(0)
    for disp in ["rpc_refused", "rpc_hardship", "rpc_hung_up", "rpc_call_back", "rpc_dispute", "third_party_contact"]:
        out[f"dial_{disp}_30d"] = out.ptp_id.map(g.disposition.apply(lambda s, d=disp: (s == d).sum())).fillna(0)
    out["dial_answer_rate_30d"] = out.dial_answered_30d / out.dial_attempts_30d.clip(lower=1)
    return out


def _agent_features(p: pd.DataFrame, outcomes: pd.DataFrame, t: Tables, train_accounts: set,
                    silent_flag: pd.Series) -> pd.DataFrame:
    """Agent track record (point-in-time, empirical-Bayes shrunk)."""
    o = outcomes.set_index("ptp_id")
    d = p[["ptp_id", "account_id", "agent_id", "captured_ts"]].copy()
    d["kept"] = d.ptp_id.map(o.kept)
    d["resolution_ts"] = d.ptp_id.map(o.resolution_ts)
    d["is_train"] = d.account_id.isin(train_accounts)
    d["silent"] = d.ptp_id.map(silent_flag)
    prior_rate = d.loc[d.is_train, "kept"].mean()
    k = 20.0  # shrinkage strength (pseudo-PTPs)
    rows = {}
    for agent, g in d.sort_values("captured_ts").groupby("agent_id"):
        lab = g[g.is_train]
        for _, r in g.iterrows():
            res = lab[lab.resolution_ts <= r.captured_ts]
            n, kept = len(res), res.kept.sum()
            prior = g[g.captured_ts < r.captured_ts]
            same_day = prior[prior.captured_ts.dt.date == r.captured_ts.date()]
            sil = prior.silent.dropna()
            rows[r.ptp_id] = {
                "agent_keep_rate_shrunk": (kept + k * prior_rate) / (n + k),
                "agent_resolved_n": float(n),
                "agent_ptps_same_day_before": float(len(same_day)),
                "agent_silent_terms_rate": float((sil.sum() + 1) / (len(sil) + 10)) if len(sil) else 0.1,
            }
    a = pd.DataFrame.from_dict(rows, orient="index")
    a.index.name = "ptp_id"
    return a.reset_index()


def _remark_copy_features(p: pd.DataFrame, t: Tables) -> pd.DataFrame:
    """Copied remarks: the same free-text remark logged by an agent across many accounts."""
    da = t.dial_attempts.dropna(subset=["ptp_id"])[["ptp_id", "agent_id", "attempt_ts", "remark", "account_id"]]
    fv = t.field_visits.dropna(subset=["ptp_id"])[["ptp_id", "agent_id", "checkin_ts", "remark", "account_id"]].rename(
        columns={"checkin_ts": "attempt_ts"})
    r = pd.concat([da, fv]).sort_values("attempt_ts")
    r["remark_norm"] = r.remark.fillna("").str.lower().str.replace(r"\d+", "#", regex=True).str.strip()
    r["remark_copies_before"] = r.groupby(["agent_id", "remark_norm"]).cumcount()
    return r[["ptp_id", "remark_copies_before"]]


def _field_features(p: pd.DataFrame, t: Tables) -> pd.DataFrame:
    fv = t.field_visits
    v = fv.dropna(subset=["ptp_id"]).set_index("ptp_id")
    photo_counts = fv.photo_hash.value_counts()
    out = pd.DataFrame({"ptp_id": p.ptp_id})
    out["field_dwell_s"] = out.ptp_id.map(v.dwell_s)
    out["field_gps_accuracy_m"] = out.ptp_id.map(v.gps_accuracy_m)
    out["field_met_family"] = out.ptp_id.map(v.outcome.eq("met_family").astype(float))
    out["field_photo_reused"] = out.ptp_id.map((v.photo_hash.map(photo_counts) > 1).astype(float))
    out["field_minutes_travel"] = out.ptp_id.map((v.checkin_ts - v.start_ts).dt.total_seconds() / 60)
    visits_day = fv.groupby(["agent_id", "visit_date"]).size()
    out["field_agent_visits_that_day"] = out.ptp_id.map(
        v.apply(lambda r: visits_day.get((r.agent_id, r.visit_date), np.nan), axis=1))
    rem = v.remark.fillna("").str.lower()
    out["field_remark_third_party"] = out.ptp_id.map(rem.str.contains(
        r"father|mother|wife|papa|appa|amma|hendathi|bhai|brother|beta|family|message de|spouse", regex=True).astype(float))
    # Integrity: a PTP logged on a visit whose remark says nobody was met is not a borrower promise
    out["field_remark_no_contact"] = out.ptp_id.map(rem.str.contains(
        r"lock|band tha|address nahi|not found|sikkilla|no one|koi nahi|yaaru illa|not traceable", regex=True).astype(float))
    out["field_remark_agent_paraphrased_date"] = out.ptp_id.map(rem.str.contains(
        r"bola \d+|promised by|tak karega|olage kodtini", regex=True).astype(float))
    out["field_remark_address_correction"] = out.ptp_id.map(rem.str.contains(";", regex=False).astype(float))
    return out


def post_call_features(p: pd.DataFrame, events: pd.DataFrame, outcomes: pd.DataFrame,
                       horizon_hours: float = CONFIG.post_call.horizon_hours) -> pd.DataFrame:
    o = outcomes.set_index("ptp_id")
    e = events.merge(p[["ptp_id", "captured_ts", "promised_date"]], on="ptp_id")
    e["first_payment_ts"] = e.ptp_id.map(o.first_payment_ts)
    cutoff = e.captured_ts + pd.Timedelta(hours=horizon_hours)
    ok = (e.event_ts <= cutoff) & ((e.first_payment_ts.isna()) | (e.event_ts < e.first_payment_ts))
    e = e[ok]
    e["hours"] = (e.event_ts - e.captured_ts).dt.total_seconds() / 3600
    out = pd.DataFrame({"ptp_id": p.ptp_id})
    for ev in ["payment_link_opened", "confirmation_reply_yes", "confirmation_reply_no", "inbound_call_from_customer",
               "whatsapp_confirmation_sent"]:
        s = e[e.event_type == ev].groupby("ptp_id").hours.min()
        out[f"post_{ev}"] = out.ptp_id.map(s.notna()).fillna(False).astype(float)
        if ev == "payment_link_opened":
            out["post_hours_to_link_open"] = out.ptp_id.map(s)
    out["post_confirmation_unanswered"] = ((out.post_whatsapp_confirmation_sent == 1) &
                                           (out.post_confirmation_reply_yes == 0) &
                                           (out.post_confirmation_reply_no == 0)).astype(float)
    # resolved before the horizon: such PTPs are excluded from post-call evaluation
    fp = p.ptp_id.map(o.first_payment_ts)
    out["post_resolved_before_horizon"] = (fp.notna() & (fp <= p.captured_ts + pd.Timedelta(hours=horizon_hours))).astype(float).values
    return out


def _stated_salary_history(t: Tables, call_feats: pd.DataFrame, p: pd.DataFrame) -> pd.Series:
    """Most recent salary day stated by the borrower in any call up to (and including) the PTP call."""
    if "stated_salary_day" not in call_feats:
        return pd.Series(dtype=float)
    s = call_feats[["call_id", "stated_salary_day"]].dropna()
    da = t.dial_attempts[["attempt_id", "account_id", "attempt_ts"]]
    s = s.merge(da, left_on="call_id", right_on="attempt_id")
    m = p[["ptp_id", "account_id", "captured_ts"]].merge(s, on="account_id")
    m = m[m.attempt_ts <= m.captured_ts + pd.Timedelta(minutes=10)]
    return m.sort_values("attempt_ts").groupby("ptp_id").stated_salary_day.last()


# ----------------------------------------------------------------------------- assembly
CATEGORICAL = ["acct_portfolio", "acct_income_type", "acct_language", "acct_lender_type", "acct_bounce_reason",
               "ptp_channel", "ptp_answerer"]


def build_context(t: Tables, outcomes: pd.DataFrame, call_feats: pd.DataFrame) -> pd.DataFrame:
    p = t.ptps.copy()
    train_accounts = set(t.splits.loc[t.splits.split == "train", "account_id"])
    acct = _account_features(t)
    df = p[["ptp_id", "account_id", "attempt_id", "agent_id", "channel", "source", "captured_ts",
            "promised_date", "promised_amount", "answerer_recorded", "payment_link_sent"]].copy()
    df = df.merge(t.splits, on="account_id", how="left").merge(acct, on="account_id", how="left")
    df["ptp_channel"] = df.channel
    df["ptp_answerer"] = df.answerer_recorded
    df["ptp_link_sent"] = df.payment_link_sent.astype(float)
    df["ptp_is_field"] = (df.source == "field_visit").astype(float)
    df["ptp_is_bot"] = (df.channel == "voice_bot").astype(float)

    df = df.merge(_ptp_terms(p, acct), on="ptp_id", how="left")
    df = df.merge(_history_features(p, outcomes, t), on="ptp_id", how="left")
    df = df.merge(_payment_history(p, t.payments), on="ptp_id", how="left")
    df = df.merge(_dial_features(p, t), on="ptp_id", how="left")
    df = df.merge(_field_features(p, t), on="ptp_id", how="left")
    df = df.merge(_remark_copy_features(p, t), on="ptp_id", how="left")

    # combined PTP track record (pre-period + resolved in-period), Beta(1,1) smoothed
    n_prev = df.acct_prev_ptp_count + df.hist_inperiod_resolved
    n_broken = df.acct_prev_ptp_broken + df.hist_inperiod_broken
    df["hist_total_prev_ptps"] = n_prev
    df["hist_total_broken"] = n_broken
    df["hist_keep_rate_smoothed"] = (n_prev - n_broken + 1) / (n_prev + 2)
    df["hist_broken_share"] = n_broken / n_prev.clip(lower=1)

    # transcript features joined via attempt_id (field PTPs have none -> NaN)
    cf = call_feats.drop(columns=[c for c in ("customer_text", "agent_text") if c in call_feats.columns])
    df = df.merge(cf.rename(columns={"call_id": "attempt_id"}), on="attempt_id", how="left")
    df["has_transcript"] = df.n_turns.notna().astype(float)

    # salary day: account record, else the most recent day the borrower stated in this or any earlier call
    stated_hist = _stated_salary_history(t, call_feats, p)
    df["hist_stated_salary_day"] = df.ptp_id.map(stated_hist)
    sal_day = df.acct_salary_day.where(df.acct_salary_day.notna(), df.get("stated_salary_day"))
    sal_day = sal_day.where(sal_day.notna(), df.hist_stated_salary_day)
    df["salary_day_source"] = np.select([df.acct_salary_day.notna(), sal_day.notna()], [2.0, 1.0], 0.0)
    for k, v in salary_alignment(df.promised_date.values, sal_day.values).items():
        df[f"terms_{k}"] = v

    silent = df.set_index("ptp_id")["borrower_silent_on_terms"]
    df = df.merge(_agent_features(p, outcomes, t, train_accounts, silent), on="ptp_id", how="left")
    df = df.merge(outcomes[["ptp_id", "kept", "partial", "any_payment", "paid_fraction", "recovered_amount",
                            "outcome_status", "first_payment_ts", "resolution_ts"]], on="ptp_id", how="left")
    df = df.merge(t.annotations, on="ptp_id", how="left")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df
