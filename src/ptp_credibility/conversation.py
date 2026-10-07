"""Incremental conversation analyser.

``ConversationState`` consumes one transcript turn at a time and maintains
O(1)-updatable counters, so the live service can re-score after every turn
inside the latency budget. Offline training calls exactly the same code on
every prefix of every historical call, which guarantees train / serve parity.

All features are interpretable by construction: each one corresponds to a
signal named in the problem statement (hedging, conditions, vague dates,
who proposed the date, negotiation, pressure-then-agree, promise-before-
hangup, hesitation proxies, borrower-stated amount, and so on).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import lexicon as lx

AGENT_SPEAKERS = {"agent", "bot"}


@dataclass
class Turn:
    speaker: str
    text: str
    pause_before_ms: float = 0.0
    start_ms: float = 0.0
    end_ms: float = 0.0
    asr_confidence: float = 1.0


@dataclass
class ConversationState:
    """Streaming feature state for a single call."""

    n_turns: int = 0
    n_customer: int = 0
    n_agent: int = 0
    cust: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in lx.CUSTOMER})
    agent: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in lx.AGENT})
    first_cust_date_turn: Optional[int] = None
    first_agent_date_proposal_turn: Optional[int] = None
    first_confirm_turn: Optional[int] = None
    cust_amounts: List[float] = field(default_factory=list)
    cust_days: List[int] = field(default_factory=list)
    agent_amounts: List[float] = field(default_factory=list)
    confirm_amount: Optional[float] = None
    confirm_day: Optional[int] = None
    confirm_month: Optional[int] = None
    stated_salary_day: Optional[int] = None
    last_speaker: str = ""
    last_agent_flags: Dict[str, bool] = field(default_factory=dict)
    recent_pressure_turn: int = -99
    pressure_then_agree: int = 0
    agent_push_after_deferral: int = 0
    deferral_seen: bool = False
    agent_consecutive_max: int = 0
    _agent_run: int = 0
    agent_turns_without_reply_after_confirm: int = 0
    cust_turns_after_confirm: int = 0
    closing_after_date: int = 0
    agree_and_close_same_turn: int = 0
    negotiation_turns: int = 0
    response_pauses_to_asks: List[float] = field(default_factory=list)
    cust_pauses: List[float] = field(default_factory=list)
    cust_asr: List[float] = field(default_factory=list)
    cust_chars: int = 0
    cust_speech_ms: float = 0.0
    agent_chars: int = 0
    fillers: int = 0
    inaudible: int = 0
    call_end_ms: float = 0.0
    cust_text: List[str] = field(default_factory=list)
    agent_text: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ update
    def update(self, turn: Turn) -> None:
        text = turn.text or ""
        low = text.lower()
        idx = self.n_turns
        self.n_turns += 1
        self.call_end_ms = max(self.call_end_ms, float(turn.end_ms or 0))
        self.fillers += len(lx.FILLER.findall(low))
        self.inaudible += len(lx.INAUDIBLE.findall(low))

        if turn.speaker in AGENT_SPEAKERS:
            self._update_agent(idx, low, turn)
        else:
            self._update_customer(idx, low, turn)
        self.last_speaker = "agent" if turn.speaker in AGENT_SPEAKERS else "customer"

    def _update_agent(self, idx: int, low: str, turn: Turn) -> None:
        self.n_agent += 1
        self.agent_chars += len(low)
        self.agent_text.append(low)
        flags = {k: bool(p.search(low)) for k, p in lx.AGENT.items()}
        for k, v in flags.items():
            self.agent[k] += int(v)
        if flags["proposes_date"] and self.first_agent_date_proposal_turn is None:
            self.first_agent_date_proposal_turn = idx
        if flags["pressure"]:
            self.recent_pressure_turn = idx
        if self.deferral_seen and (flags["today_ask"] or flags["proposes_date"] or flags["pressure"]
                                   or flags["part_payment_ask"] or flags["open_date_question"]):
            self.agent_push_after_deferral += 1
        amts = lx.extract_amounts(low)
        self.agent_amounts.extend(amts)
        if flags["confirm_ptp"]:
            if self.first_confirm_turn is None:
                self.first_confirm_turn = idx
            m = lx.DATE_WITH_MONTH.search(low)
            if m:
                self.confirm_day, self.confirm_month = int(m.group(1)), lx.MONTHS[m.group(2).lower()]
            if amts:
                self.confirm_amount = amts[-1]
        if self.first_confirm_turn is not None and idx > self.first_confirm_turn and self.last_speaker == "agent":
            self.agent_turns_without_reply_after_confirm += 1
        self._agent_run = self._agent_run + 1 if self.last_speaker == "agent" else 1
        self.agent_consecutive_max = max(self.agent_consecutive_max, self._agent_run)
        self.last_agent_flags = flags

    def _update_customer(self, idx: int, low: str, turn: Turn) -> None:
        self.n_customer += 1
        self._agent_run = 0
        self.cust_text.append(low)
        flags = {k: bool(p.search(low)) for k, p in lx.CUSTOMER.items()}
        for k, v in flags.items():
            self.cust[k] += int(v)
        pause = float(turn.pause_before_ms or 0)
        self.cust_pauses.append(pause)
        self.cust_asr.append(float(turn.asr_confidence if turn.asr_confidence is not None else 1.0))
        self.cust_chars += len(low)
        self.cust_speech_ms += max(1.0, float((turn.end_ms or 0) - (turn.start_ms or 0)))

        has_date = flags["specific_date"] or flags["vague_date"]
        amts = lx.extract_amounts(low)
        self.cust_amounts.extend(amts)
        days = lx.extract_day_numbers(low)
        if flags["salary_day_info"]:
            # "har mahine 28 ko aa jaati hai" is information about cash flow, not a promise date
            has_date = False
            bare = [int(x) for x in re.findall(r"\b(\d{1,2})\b", low) if 1 <= int(x) <= 31]
            if days or bare:
                self.stated_salary_day = (days or bare)[0]
            for k in ("specific_date", "vague_date"):
                self.cust[k] -= int(flags[k])
        else:
            self.cust_days.extend(days)
        if has_date and self.first_cust_date_turn is None:
            self.first_cust_date_turn = idx
        la = self.last_agent_flags
        asked = la.get("open_date_question") or la.get("proposes_date") or la.get("part_payment_ask") \
            or la.get("today_ask") or la.get("confirm_ptp")
        if asked:
            self.response_pauses_to_asks.append(pause)
        if (la.get("part_payment_ask") or la.get("proposes_date")) and (amts or has_date or flags["partial_offer"]):
            self.negotiation_turns += 1
        agrees = flags["passive_agree"] or flags["firm_commitment"] or flags["verbal_commit"] or has_date
        if agrees and idx - self.recent_pressure_turn <= 2:
            self.pressure_then_agree += 1
        if flags["closing_escape"] or flags["refusal"]:
            self.deferral_seen = True
        if flags["closing_escape"] and self.first_cust_date_turn is not None:
            self.closing_after_date += 1
        if agrees and flags["closing_escape"]:
            self.agree_and_close_same_turn += 1
        if self.first_confirm_turn is not None and idx > self.first_confirm_turn:
            self.cust_turns_after_confirm += 1

    # ------------------------------------------------------------------ features
    def features(self) -> Dict[str, float]:
        f: Dict[str, float] = {}
        f["n_turns"] = self.n_turns
        f["n_customer_turns"] = self.n_customer
        f["n_agent_turns"] = self.n_agent
        f["call_duration_s"] = self.call_end_ms / 1000.0
        for k, v in self.cust.items():
            f[f"cust_{k}"] = v
        for k, v in self.agent.items():
            f[f"agent_{k}"] = v
        hardship_keys = [k for k in self.cust if k.startswith("hardship_")]
        f["cust_hardship_any"] = float(sum(self.cust[k] for k in hardship_keys) > 0)
        f["cust_hardship_kinds"] = float(sum(self.cust[k] > 0 for k in hardship_keys))
        f["borrower_stated_date"] = float(self.first_cust_date_turn is not None)
        f["borrower_stated_specific_date"] = float(self.cust["specific_date"] > 0)
        f["borrower_stated_amount"] = float(len(self.cust_amounts) > 0)
        f["borrower_only_vague_date"] = float(self.cust["vague_date"] > 0 and self.cust["specific_date"] == 0)
        f["agent_proposed_date_first"] = float(
            self.first_agent_date_proposal_turn is not None and (
                self.first_cust_date_turn is None or self.first_agent_date_proposal_turn < self.first_cust_date_turn))
        f["borrower_silent_on_terms"] = float(self.first_cust_date_turn is None and not self.cust_amounts)
        f["confirm_seen"] = float(self.first_confirm_turn is not None)
        f["stated_salary_day"] = float(self.stated_salary_day) if self.stated_salary_day else np.nan
        f["confirm_day_matches_borrower"] = float(
            self.confirm_day is not None and self.confirm_day in self.cust_days) if self.confirm_day else np.nan
        f["confirm_amount_matches_borrower"] = float(
            self.confirm_amount is not None and any(abs(a - self.confirm_amount) < 1 for a in self.cust_amounts)
        ) if self.confirm_amount else np.nan
        f["borrower_offer_below_confirm"] = float(
            bool(self.cust_amounts) and self.confirm_amount is not None and max(self.cust_amounts) < self.confirm_amount - 1
        ) if self.confirm_amount else np.nan
        f["negotiation_turns"] = self.negotiation_turns
        f["pressure_then_agree"] = self.pressure_then_agree
        f["agent_push_after_deferral"] = self.agent_push_after_deferral
        f["agent_consecutive_max"] = self.agent_consecutive_max
        f["agent_turns_no_reply_after_confirm"] = self.agent_turns_without_reply_after_confirm
        f["cust_turns_after_confirm"] = self.cust_turns_after_confirm
        f["closing_after_date"] = self.closing_after_date
        f["agree_and_close_same_turn"] = self.agree_and_close_same_turn
        f["turns_before_first_cust_date"] = float(self.first_cust_date_turn) if self.first_cust_date_turn is not None else np.nan
        f["confirm_turn_position"] = (self.first_confirm_turn / max(1, self.n_turns)) if self.first_confirm_turn is not None else np.nan
        f["agent_talk_share"] = self.agent_chars / max(1, self.agent_chars + self.cust_chars)
        f["cust_hedge_rate"] = self.cust["hedge"] / max(1, self.n_customer)
        f["cust_passive_rate"] = self.cust["passive_agree"] / max(1, self.n_customer)
        # "How it was said" proxies (sensitive under DPDP; gated by the voice consent flag)
        cp = np.array(self.cust_pauses) if self.cust_pauses else np.array([np.nan])
        f["voice_mean_pause_ms"] = float(np.nanmean(cp)) if self.cust_pauses else np.nan
        f["voice_max_pause_ms"] = float(np.nanmax(cp)) if self.cust_pauses else np.nan
        f["voice_long_pauses"] = float(np.sum(cp > 1500)) if self.cust_pauses else 0.0
        f["voice_response_pause_to_ask_ms"] = float(np.mean(self.response_pauses_to_asks)) if self.response_pauses_to_asks else np.nan
        f["voice_max_response_pause_to_ask_ms"] = float(np.max(self.response_pauses_to_asks)) if self.response_pauses_to_asks else np.nan
        f["voice_mean_asr_conf"] = float(np.mean(self.cust_asr)) if self.cust_asr else np.nan
        f["voice_min_asr_conf"] = float(np.min(self.cust_asr)) if self.cust_asr else np.nan
        f["voice_speech_rate_cps"] = self.cust_chars / (self.cust_speech_ms / 1000.0) if self.cust_speech_ms else np.nan
        f["voice_filler_rate"] = self.fillers / max(1, self.n_turns)
        f["voice_inaudible"] = float(self.inaudible)
        return f

    @property
    def customer_text(self) -> str:
        return " | ".join(self.cust_text)

    @property
    def agent_text_joined(self) -> str:
        return " | ".join(self.agent_text)


VOICE_PREFIX = "voice_"
