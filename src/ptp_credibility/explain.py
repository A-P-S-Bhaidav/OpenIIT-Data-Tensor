"""Reason codes: turn TreeSHAP contributions into auditable, plain-language flags.

Each model feature belongs to a *signal family* taken directly from the
problem statement. The service reports the top contributors in each
direction with a stable reason code, so an agent sees "Borrower never stated
an amount" while an auditor sees ``RC_TERMS_NO_AMOUNT`` with its log-odds
contribution and the raw feature value.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# feature -> (reason code, family, text when it LOWERS credibility, text when it RAISES credibility)
REASONS: Dict[str, Tuple[str, str, str, str]] = {
    "borrower_stated_amount": ("RC_TERMS_AMOUNT", "what_was_said", "Borrower never stated an amount; only the agent did", "Borrower stated the amount in their own words"),
    "borrower_silent_on_terms": ("RC_TERMS_SILENT", "what_was_said", "Borrower never stated a date or an amount", "Borrower engaged with date and amount"),
    "borrower_stated_specific_date": ("RC_TERMS_DATE", "what_was_said", "No specific date from the borrower", "Borrower named a specific date"),
    "borrower_only_vague_date": ("RC_VAGUE_DATE", "what_was_said", "Only a vague date (next week / month end)", "Date was not vague"),
    "cust_vague_date": ("RC_VAGUE_DATE", "what_was_said", "Vague timing language used", "No vague timing language"),
    "cust_specific_date": ("RC_TERMS_DATE", "what_was_said", "Few specific date mentions", "Specific date mentioned"),
    "cust_hedge": ("RC_HEDGING", "what_was_said", "Hedging language (dekhta hoon / try karunga / nodtini)", "No hedging language"),
    "cust_hedge_rate": ("RC_HEDGING", "what_was_said", "High share of hedged replies", "Low share of hedged replies"),
    "cust_conditional": ("RC_CONDITIONAL", "what_was_said", "Promise is conditional (salary aayi toh / if client pays)", "Promise is unconditional"),
    "cust_firm_commitment": ("RC_FIRM", "what_was_said", "No firm commitment language", "Firm commitment (pakka / definitely)"),
    "cust_full_amount": ("RC_FULL_AMOUNT", "what_was_said", "No commitment to the full amount", "Borrower referred to paying the full amount"),
    "cust_partial_offer": ("RC_PARTIAL_OFFER", "what_was_said", "No concrete part-payment offer", "Borrower offered a concrete part payment"),
    "cust_pay_now": ("RC_PAY_NOW", "what_was_said", "No offer to pay now", "Borrower offered to pay part now"),
    "cust_link_accept": ("RC_LINK", "what_was_said", "Did not accept a payment link", "Accepted the payment link"),
    "cust_link_refuse": ("RC_LINK_REFUSED", "what_was_said", "Refused the payment link (will pay elsewhere)", "Did not refuse the link"),
    "cust_closing_escape": ("RC_EXIT", "conversation", "Borrower was trying to end the call (busy / baad mein)", "Borrower stayed engaged"),
    "cust_passive_agree": ("RC_PASSIVE", "conversation", "Agreement was passive (ok ok / theek hai)", "Agreement was active"),
    "cust_prior_break_excuse": ("RC_PRIOR_EXCUSE", "history", "Borrower explained away an earlier broken promise", "No prior-break excuse"),
    "cust_third_party": ("RC_THIRD_PARTY", "conversation", "Promise given by someone other than the borrower", "Borrower spoke for themselves"),
    "cust_self_agency": ("RC_SELF_AGENCY", "what_was_said", "No ownership language", "Borrower took ownership (main khud karunga)"),
    "agent_proposed_date_first": ("RC_AGENT_DATE", "conversation", "Agent proposed the date before the borrower did", "Borrower proposed the date"),
    "agent_proposes_date": ("RC_AGENT_DATE", "conversation", "Agent suggested the date", "Agent did not lead the date"),
    "agent_push_after_deferral": ("RC_PUSHED", "conversation", "Agent kept pushing after the borrower deferred", "No pushing after deferral"),
    "agent_pressure": ("RC_PRESSURE", "conversation", "Promise followed pressure language", "No pressure language"),
    "pressure_then_agree": ("RC_PRESSURE_AGREE", "conversation", "Borrower agreed right after pressure", "Agreement did not follow pressure"),
    "agree_and_close_same_turn": ("RC_AGREE_EXIT", "conversation", "Agreed and tried to hang up in the same breath", "Agreement was not an exit line"),
    "negotiation_turns": ("RC_NEGOTIATION", "conversation", "No negotiation on date or amount", "Real negotiation on date or amount"),
    "confirm_amount_matches_borrower": ("RC_CONFIRM_MISMATCH", "conversation", "Recorded amount differs from what the borrower said", "Recorded amount matches the borrower's words"),
    "confirm_day_matches_borrower": ("RC_CONFIRM_MISMATCH", "conversation", "Recorded date differs from what the borrower said", "Recorded date matches the borrower's words"),
    "agent_consecutive_max": ("RC_AGENT_MONOLOGUE", "conversation", "Agent spoke several turns without borrower reply", "Balanced turn-taking"),
    "agent_talk_share": ("RC_AGENT_MONOLOGUE", "conversation", "Agent dominated the conversation", "Borrower spoke a fair share"),
    "voice_long_pauses": ("RC_HESITATION", "how_it_was_said", "Long hesitations before answering", "Answered without long hesitation"),
    "voice_max_pause_ms": ("RC_HESITATION", "how_it_was_said", "Long hesitation observed", "No long hesitation"),
    "voice_mean_pause_ms": ("RC_HESITATION", "how_it_was_said", "Slow responses overall", "Prompt responses"),
    "voice_max_response_pause_to_ask_ms": ("RC_HESITATION", "how_it_was_said", "Long pause before answering the date / amount ask", "Answered the ask promptly"),
    "voice_response_pause_to_ask_ms": ("RC_HESITATION", "how_it_was_said", "Slow answers to payment asks", "Quick answers to payment asks"),
    "terms_salary_aligned": ("RC_SALARY_ALIGN", "account", "Promised date does not follow a salary / income credit", "Promised date falls just after salary credit"),
    "terms_salary_gap_days": ("RC_SALARY_ALIGN", "account", "Promised date is far from the salary credit", "Promised date is close after salary credit"),
    "terms_salary_promise_before_credit": ("RC_BEFORE_SALARY", "account", "Promised date is just before the next salary", "Date not squeezed before salary"),
    "terms_days_to_promise": ("RC_HORIZON", "account", "Promised date is very soon relative to cash flow", "Promise horizon is realistic"),
    "terms_promise_is_month_end": ("RC_MONTH_END", "account", "Month-end date", "Not a month-end date"),
    "terms_promise_dom": ("RC_DOM", "account", "Day of month is unfavourable", "Day of month is favourable"),
    "terms_amount_to_overdue": ("RC_AMOUNT_RATIO", "account", "Promised amount is large relative to overdue", "Promised amount is manageable"),
    "terms_amount_to_emi": ("RC_AMOUNT_RATIO", "account", "Promised amount is several EMIs", "Promised amount is near one EMI"),
    "acct_ability_to_pay": ("RC_ABILITY", "account", "Low ability-to-pay estimate", "Good ability-to-pay estimate"),
    "acct_bureau_ord": ("RC_BUREAU", "account", "Weak bureau band", "Stronger bureau band"),
    "hist_keep_rate_smoothed": ("RC_HISTORY", "history", "Poor past PTP keep rate", "Good past PTP keep rate"),
    "hist_total_broken": ("RC_HISTORY", "history", "Several past broken promises", "Few past broken promises"),
    "acct_prev_ptp_broken": ("RC_HISTORY", "history", "Broken promises before this cycle", "No earlier broken promises"),
    "agent_keep_rate_shrunk": ("RC_AGENT_TRACK", "agent", "Capturing agent's PTPs convert below peers", "Capturing agent's PTPs convert well"),
    "agent_silent_terms_rate": ("RC_AGENT_TRACK", "agent", "Agent often logs PTPs without borrower terms", "Agent rarely logs PTPs without borrower terms"),
    "field_photo_reused": ("RC_FIELD_PHOTO", "agent", "Visit photo reused from another visit", "Unique visit photo"),
    "field_met_family": ("RC_FIELD_FAMILY", "conversation", "Field PTP given by a family member", "Field PTP given by the borrower"),
}

FAMILY_LABEL = {
    "what_was_said": "What was said",
    "conversation": "How the conversation went",
    "how_it_was_said": "How it was said (voice)",
    "account": "Account and promise terms",
    "history": "Account history",
    "agent": "Agent signals",
    "other": "Other",
}


def reason_codes(contrib_row: pd.Series, feature_row: pd.Series, top_k: int = 3) -> Dict[str, List[dict]]:
    c = contrib_row.drop(labels=["__bias__"], errors="ignore")
    c = c[c.index.isin(REASONS.keys())]
    out = {"lowers": [], "raises": []}
    for direction, series in (("lowers", c.sort_values()), ("raises", c.sort_values(ascending=False))):
        seen = set()
        for feat, val in series.items():
            if (direction == "lowers" and val >= -0.02) or (direction == "raises" and val <= 0.02):
                break
            code, fam, neg, pos = REASONS[feat]
            if code in seen:
                continue
            seen.add(code)
            raw = feature_row.get(feat, np.nan)
            out[direction].append({
                "code": code, "family": FAMILY_LABEL.get(fam, fam), "feature": feat,
                "text": neg if direction == "lowers" else pos,
                "log_odds": round(float(val), 3),
                "value": None if pd.isna(raw) else (float(raw) if isinstance(raw, (int, float, np.floating, np.integer)) else str(raw)),
            })
            if len(out[direction]) >= top_k:
                break
    return out


def family_attribution(contrib: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute contribution per signal family (global explanation)."""
    fam = {f: REASONS[f][1] if f in REASONS else
           ("how_it_was_said" if f.startswith("voice_") else
            "account" if f.startswith(("acct_", "terms_")) else
            "history" if f.startswith(("hist_", "pay_", "dial_")) else
            "agent" if f.startswith(("agent_keep", "agent_resolved", "agent_ptps", "agent_silent", "remark", "field_")) else
            "conversation" if f.startswith(("agent_", "n_", "call_", "confirm", "negotiation")) else
            "what_was_said" if f.startswith(("cust_", "borrower_")) else "other")
           for f in contrib.columns if f != "__bias__"}
    a = contrib.drop(columns="__bias__").abs().mean()
    return a.groupby(a.index.map(fam)).sum().sort_values(ascending=False).rename(index=FAMILY_LABEL)
