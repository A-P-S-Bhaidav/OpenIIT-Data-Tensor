"""Real-time credibility engine used by the API, the voice bot and the agent desktop.

A ``CallSession`` wraps one live call. After every transcript turn it:

1. updates the incremental ``ConversationState`` (O(1));
2. assembles the feature row (context from the feature store, live
   conversation features, promise terms once they are on the table);
3. scores the calibrated keep probability with the streaming model;
4. estimates the likely PTP type and the hardship flag;
5. decides whether the evidence is sufficient to act ("heard enough");
6. returns in-call prompts and reason codes.

At call end it produces the final score with the end-of-call model, the
disposition (parking window, reminders, confirmation call, integrity review)
and a contact-hour compliant schedule. Post-call events update the score.

Voice-derived features are DPDP-sensitive: they are only used when the
session was opened with ``voice_consent=True``; otherwise they are dropped
(the model handles them as missing) and the audit record says so.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from ..config import CONFIG
from ..conversation import ConversationState, Turn
from ..explain import reason_codes
from ..features import salary_alignment
from ..models import END_OF_CALL_ONLY, TERMS_ALWAYS_AVAILABLE, TERMS_PREFIX
from ..policy import (POLICY_VERSION, disposition, hardship_flag, in_call_prompts, materialise_schedule)

ARTIFACT = CONFIG.paths.artifacts / "ptp_credibility.joblib"
SCORED = CONFIG.paths.artifacts / "scored_ptps.pkl"


class ContextStore:
    """Stand-in for the CN feature store: point-in-time context per PTP / account."""

    def __init__(self, path: Path = SCORED):
        self.df = pd.read_pickle(path).set_index("ptp_id")

    def context(self, ptp_id: str) -> Dict:
        if ptp_id not in self.df.index:
            raise KeyError(f"Unknown ptp_id {ptp_id}")
        return self.df.loc[ptp_id].to_dict()


@dataclass
class CallSession:
    engine: "CredibilityEngine"
    context: Dict
    voice_consent: bool = True
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: ConversationState = field(default_factory=ConversationState)
    terms_set: bool = False
    history: List[Dict] = field(default_factory=list)
    acted: bool = False
    closed: bool = False

    # ------------------------------------------------------------------ inputs
    def set_terms(self, promised_date: str, promised_amount: float) -> None:
        """Called when the agent desktop / bot captures the promise terms."""
        pdte = pd.Timestamp(promised_date)
        cap = pd.Timestamp(self.context["captured_ts"])
        self.context["promised_date"] = pdte
        self.context["promised_amount"] = float(promised_amount)
        self.context["terms_days_to_promise"] = (pdte - cap.normalize()).days
        self.context["terms_promise_dom"] = pdte.day
        self.context["terms_amount"] = float(promised_amount)
        self.terms_set = True

    def _row(self, final: bool) -> pd.DataFrame:
        f = dict(self.context)
        conv = self.state.features()
        f.update(conv)
        terms_known = final or self.terms_set or self.state.first_confirm_turn is not None
        sal_day = f.get("acct_salary_day")
        if sal_day is None or (isinstance(sal_day, float) and np.isnan(sal_day)):
            sal_day = conv.get("stated_salary_day")
        if terms_known:
            al = salary_alignment([f.get("promised_date")],
                                  np.array([sal_day if sal_day is not None else np.nan], dtype=float))
            for k, v in al.items():
                f[f"terms_{k}"] = float(v[0])
        else:
            for k in list(f):
                if k.startswith(TERMS_PREFIX) and k not in TERMS_ALWAYS_AVAILABLE:
                    f[k] = np.nan
        if not final:
            for k in END_OF_CALL_ONLY + ["answerer_not_self"]:
                f[k] = np.nan
        if not self.voice_consent:
            for k in list(f):
                if k.startswith("voice_"):
                    f[k] = np.nan
        f["prefix_turns"] = self.state.n_turns
        f["confirm_reached"] = float(self.state.first_confirm_turn is not None)
        return pd.DataFrame([f])

    # ------------------------------------------------------------------ live update
    def add_turn(self, speaker: str, text: str, pause_before_ms: float = 0.0, start_ms: float = 0.0,
                 end_ms: float = 0.0, asr_confidence: float = 1.0) -> Dict:
        t0 = time.perf_counter()
        self.state.update(Turn(speaker, text, pause_before_ms, start_ms, end_ms, asr_confidence))
        row = self._row(final=False)
        e = self.engine
        p = float(e.stream.predict(row)[0])
        tp = e.type.predict_proba(row)[0]
        type_proba = dict(zip(e.type.classes, tp.round(4).tolist()))
        ptype = str(e.type.decide(tp[None, :])[0])
        f = row.iloc[0].to_dict()
        hs = hardship_flag(f, type_proba)
        exp_move = float(e.suff.predict(row, np.array([p]))[0])
        on_table = bool(f.get("confirm_reached") or f.get("borrower_stated_date") or f.get("borrower_stated_amount"))
        act_now = on_table and exp_move < e.suff.delta
        prompts = []
        if act_now or (on_table and self.state.first_confirm_turn is not None):
            ctx = self._prompt_ctx()
            prompts = [vars(x) for x in in_call_prompts(p, ptype, f, ctx, hs)]
            self.acted = True
        reasons = reason_codes(e.stream.contributions(row).iloc[0], row.iloc[0]) if e.explain_live else {}
        out = {
            "session_id": self.session_id, "turn": self.state.n_turns, "p_keep": round(p, 4),
            "credibility_band": e.band(p), "likely_type": ptype, "type_proba": type_proba,
            "hardship_flag": hs, "promise_on_table": on_table, "expected_score_movement": round(exp_move, 4),
            "act_now": act_now, "prompts": prompts, "reasons": reasons,
            "voice_features_used": self.voice_consent,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        self.history.append({k: out[k] for k in ("turn", "p_keep", "likely_type", "act_now", "latency_ms")})
        return out

    def _prompt_ctx(self) -> Dict:
        c = self.context
        pdte = c.get("promised_date")
        sal = c.get("acct_salary_day")
        if sal is None or (isinstance(sal, float) and np.isnan(sal)):
            sal = self.state.stated_salary_day
        return {"language": str(c.get("acct_language", "english")),
                "captured_ts": pd.Timestamp(c.get("captured_ts")).to_pydatetime(),
                "salary_day": sal, "promised_amount": c.get("promised_amount"),
                "promised_date_str": pd.Timestamp(pdte).strftime("%d %B") if pdte is not None and not pd.isna(pdte) else "the promised date",
                "ability_to_pay": c.get("acct_ability_to_pay")}

    # ------------------------------------------------------------------ end of call
    def end_call(self) -> Dict:
        t0 = time.perf_counter()
        self.closed = True
        row = self._row(final=True)
        e = self.engine
        p = float(e.keep.predict(row)[0])
        tp = e.type.predict_proba(row)[0]
        type_proba = dict(zip(e.type.classes, tp.round(4).tolist()))
        ptype = str(e.type.decide(tp[None, :])[0])
        f = row.iloc[0].to_dict()
        hs = hardship_flag(f, type_proba)
        cap = pd.Timestamp(self.context["captured_ts"]).to_pydatetime()
        dtp = float(f.get("terms_days_to_promise") or 7)
        d = disposition(p, ptype, hs, dtp, cap)
        pdte = pd.Timestamp(self.context.get("promised_date", cap)).to_pydatetime()
        sched = materialise_schedule(d, cap, pdte)
        reasons = reason_codes(e.keep.contributions(row).iloc[0], row.iloc[0], top_k=4)
        self.final = {"p_keep": p, "row": row}
        return {
            "session_id": self.session_id, "p_keep": round(p, 4), "credibility_band": e.band(p),
            "likely_type": ptype, "type_proba": type_proba, "hardship_flag": hs,
            "disposition": d.to_dict(), "schedule": sched, "reasons": reasons,
            "quality_weight": round(p, 4), "policy_version": POLICY_VERSION, "model_versions": e.versions,
            "voice_features_used": self.voice_consent,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "audit": {"turns": self.state.n_turns, "trajectory": self.history},
        }

    def post_call_event(self, event_type: str, hours_after_call: float) -> Dict:
        if not getattr(self, "final", None):
            raise RuntimeError("end_call must be called before post-call events")
        row = self.final["row"]
        col = f"post_{event_type}"
        row[col] = 1.0
        if event_type == "payment_link_opened":
            row["post_hours_to_link_open"] = hours_after_call
        row["post_confirmation_unanswered"] = float(row.get("post_whatsapp_confirmation_sent", pd.Series([0])).iloc[0] == 1 and
                                                    row.get("post_confirmation_reply_yes", pd.Series([0])).iloc[0] != 1 and
                                                    row.get("post_confirmation_reply_no", pd.Series([0])).iloc[0] != 1)
        for c in self.engine.post.features:
            if c not in row or pd.isna(row[c].iloc[0]):
                row[c] = 0.0 if c != "post_hours_to_link_open" else np.nan
        p = float(self.engine.post.predict(np.array([self.final["p_keep"]]), row)[0])
        return {"session_id": self.session_id, "event": event_type, "p_keep_updated": round(p, 4),
                "credibility_band": self.engine.band(p)}


class CredibilityEngine:
    def __init__(self, artifact: Path = ARTIFACT, store: Optional[ContextStore] = None, explain_live: bool = True):
        a = joblib.load(artifact)
        self.keep, self.stream, self.suff, self.type, self.post = a["keep"], a["stream"], a["sufficiency"], a["type"], a["post"]
        self.thresholds, self.versions = a["thresholds"], a["versions"]
        self.store = store or ContextStore()
        self.explain_live = explain_live
        self.sessions: Dict[str, CallSession] = {}

    def band(self, p: float) -> str:
        if p >= self.thresholds["high"]:
            return "high"
        if p < self.thresholds["low"]:
            return "low"
        return "medium"

    def start(self, ptp_id: str, voice_consent: bool = True, captured_ts: Optional[str] = None) -> CallSession:
        ctx = self.store.context(ptp_id)
        ctx["ptp_id"] = ptp_id
        # strip anything the live system cannot know yet (outcomes, post-call, scores, conversation)
        drop = {"kept", "partial", "any_payment", "paid_fraction", "recovered_amount", "outcome_status",
                "first_payment_ts", "resolution_ts", "annotated_type", "annotator_confidence", "p_keep", "p_raw",
                "p_casemix", "ptp_type"}
        conv_keys = set(ConversationState().features().keys())
        ctx = {k: v for k, v in ctx.items() if k not in drop and k not in conv_keys and not k.startswith(("post_", "type_p_"))}
        if captured_ts:
            ctx["captured_ts"] = pd.Timestamp(captured_ts)
        s = CallSession(self, ctx, voice_consent=voice_consent)
        self.sessions[s.session_id] = s
        return s
