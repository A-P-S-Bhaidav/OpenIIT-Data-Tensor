"""Decision policy: in-call prompts and post-call disposition.

The policy is a transparent, version-controlled decision table on top of the
calibrated keep probability, the likely PTP type and the hardship flag.
Hard guardrails are enforced *after* the table, so no score can override
them:

1. **Hardship guardrail** - if any hardship signal is present, the account can
   never receive a harsher treatment (shorter parking, escalation, today-
   payment asks). It is routed to restructuring options or to a human.
2. **Fair Practices Code** - prompts come only from a vetted template library;
   every prompt is checked against a banned-phrase list (threats, credit-score
   intimidation, escalation language). Follow-ups are scheduled only inside
   permitted contact hours (08:00 - 19:00) and within a daily contact cap.
3. **Explainability** - every decision carries the rule id that produced it.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .config import CONFIG, PolicyConfig

POLICY_VERSION = "2026.10.1"
CONTACT_START_HOUR, CONTACT_END_HOUR = 8, 19
MAX_CONTACTS_PER_DAY = 1

BANNED = re.compile(
    r"cibil|credit score|escalat|legal|police|court|warrant|last reminder|case aage|case munde|"
    r"karna padega|maadle beku|otherwise|warna|seize|recovery agent|ghar aayenge|mane ge bartivi",
    re.IGNORECASE)

HARDSHIP_FEATURES = ["cust_hardship_medical", "cust_hardship_job_loss", "cust_hardship_business",
                     "cust_hardship_no_funds", "cust_hardship_salary", "dial_rpc_hardship_30d"]

# --------------------------------------------------------------------------- prompt library
PROMPTS: Dict[str, Dict[str, str]] = {
    "P_HARDSHIP_SUPPORT": {
        "agent": "Acknowledge the difficulty. Offer a date after income arrives or a smaller instalment, and mention restructuring support. Do not ask for payment today.",
        "english": "I understand this is a difficult time. Would a date after your income comes in, or a smaller amount, work better for you? We also have options to restructure the payments.",
        "hinglish": "Main samajh sakta hoon abhi mushkil samay hai. Kya income aane ke baad ki date ya thodi kam amount aapke liye theek rahegi? Hamare paas EMI restructure karne ke options bhi hain.",
        "kanglish": "Ivaga kashta ide antha arthavaagutte. Income banda mele date athva swalpa kammi amount nimge sari aagutta? EMI restructure maadoke options kooda ive.",
    },
    "P_MOVE_AFTER_SALARY": {
        "agent": "The promised date falls before the borrower's salary credit (day {salary_day}). Offer {suggested_date} instead.",
        "english": "Since your salary comes on the {salary_day}, would {suggested_date} be more comfortable for you?",
        "hinglish": "Aapki salary {salary_day} tareekh ko aati hai, toh kya {suggested_date} aapke liye zyada aasaan rahega?",
        "kanglish": "Nimma salary {salary_day} ne tareekhu barutte, {suggested_date} nimge sulabha aagutta?",
    },
    "P_ASK_TERMS_OWN_WORDS": {
        "agent": "The borrower has not stated a date or amount. Ask them to say the date and amount in their own words before recording a PTP.",
        "english": "Just to be sure I record this correctly, could you tell me the date and the amount you are planning to pay?",
        "hinglish": "Sahi se note karne ke liye, kya aap bata sakte hain kis tareekh ko aur kitna amount aap pay karenge?",
        "kanglish": "Sariyaagi note maadoke, yaava tareekhu mattu eshtu amount kattuttira antha heltira?",
    },
    "P_PART_PAYMENT_LINK": {
        "agent": "Promise looks weak. Offer a payment link for Rs {part_amount} now, with the balance on the promised date. Accept a no politely.",
        "english": "If it helps, I can send a link for Rs {part_amount} now, and the rest can be paid on {promised_date}. Would that be possible?",
        "hinglish": "Agar aasaan ho toh main abhi Rs {part_amount} ka link bhej sakta hoon, baaki {promised_date} ko. Kya yeh ho payega?",
        "kanglish": "Sulabha aadre, eega Rs {part_amount} ge link kalistini, baaki {promised_date} ge. Aagutta?",
    },
    "P_REACH_BORROWER": {
        "agent": "The promise is from a third party. Do not record it as a borrower PTP. Ask when the borrower can be reached directly.",
        "english": "Thank you. When would be a good time to speak with {name_placeholder} directly?",
        "hinglish": "Dhanyavaad. {name_placeholder} ji se seedhe baat karne ka sahi samay kya rahega?",
        "kanglish": "Dhanyavaada. {name_placeholder} avra jothe neravaagi maataadoke yaava samaya sari?",
    },
    "P_REPEAT_SMALLER_NEARER": {
        "agent": "Repeat promiser. Respectfully reference the earlier promise and agree a smaller, nearer instalment the borrower is confident about.",
        "english": "Last time the payment could not go through. What amount are you fully confident you can pay this week?",
        "hinglish": "Pichli baar payment nahi ho paaya tha. Is hafte kitna amount aap pakka kar payenge?",
        "kanglish": "Hinde payment aagilla. Ee vaara eshtu amount khachitavaagi kattbahudu?",
    },
    "P_CONFIRM_LINK": {
        "agent": "Credible promise. Confirm the date and amount once and offer the payment link for convenience.",
        "english": "Thank you. I am noting Rs {promised_amount} on {promised_date}. I will send a link so you can pay easily.",
        "hinglish": "Dhanyavaad. Main {promised_date} ko Rs {promised_amount} note kar raha hoon. Aasaani ke liye link bhej deta hoon.",
        "kanglish": "Dhanyavaada. {promised_date} ge Rs {promised_amount} note maadtini. Sulabhavaagi kattoke link kalistini.",
    },
}

LANG_KEY = {"hinglish": "hinglish", "kanglish": "kanglish", "english": "english"}


def fpc_check(text: str) -> bool:
    return BANNED.search(text or "") is None


assert all(fpc_check(v) for p in PROMPTS.values() for v in p.values()), "Prompt library violates FPC filter"


@dataclass
class Prompt:
    prompt_id: str
    agent_hint: str
    script: str
    rule: str


@dataclass
class Disposition:
    treatment: str
    parking_days: int
    reminders: List[str]
    third_party_confirmation_call: bool
    integrity_review: bool
    route: str
    rule: str
    guardrails_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- helpers
def hardship_flag(f: Dict[str, float], type_proba: Optional[Dict[str, float]] = None) -> bool:
    lex = any((f.get(k) or 0) > 0 for k in HARDSHIP_FEATURES if not (isinstance(f.get(k), float) and math.isnan(f.get(k))))
    model = bool(type_proba) and type_proba.get("genuine_infeasible", 0.0) >= 0.35
    return bool(lex or model)


def suggested_part_amount(promised_amount: float, ability: Optional[float]) -> int:
    frac = 0.25 if ability is None or (isinstance(ability, float) and math.isnan(ability)) else 0.15 + 0.25 * float(ability)
    amt = max(500.0, min(promised_amount, promised_amount * frac))
    return int(round(amt / 500.0) * 500) or 500


def _date_after_salary(captured: datetime, salary_day: Optional[float]) -> Optional[datetime]:
    if salary_day is None or (isinstance(salary_day, float) and math.isnan(salary_day)):
        return None
    sd = int(salary_day)
    y, m = captured.year, captured.month
    for _ in range(3):
        try:
            cand = datetime(y, m, min(sd, 28)) + timedelta(days=2)
        except ValueError:
            cand = None
        if cand and cand.date() > captured.date():
            return cand
        m = 1 if m == 12 else m + 1
        y = y + 1 if m == 1 else y
    return None


def _in_contact_hours(ts: datetime) -> datetime:
    if ts.hour < CONTACT_START_HOUR:
        return ts.replace(hour=CONTACT_START_HOUR + 2, minute=0, second=0)
    if ts.hour >= CONTACT_END_HOUR:
        nxt = ts + timedelta(days=1)
        return nxt.replace(hour=CONTACT_START_HOUR + 2, minute=0, second=0)
    return ts


# --------------------------------------------------------------------------- in-call prompts
def in_call_prompts(p_keep: float, ptype: str, f: Dict[str, float], ctx: Dict, hardship: bool,
                    cfg: PolicyConfig = CONFIG.policy, max_prompts: int = 2) -> List[Prompt]:
    lang = LANG_KEY.get(str(ctx.get("language", "english")), "english")
    captured = ctx.get("captured_ts") or datetime.now()
    salary_day = ctx.get("salary_day")
    after_sal = _date_after_salary(captured, salary_day)
    fill = {
        "salary_day": int(salary_day) if after_sal else "",
        "suggested_date": after_sal.strftime("%d %B") if after_sal else "",
        "part_amount": suggested_part_amount(float(ctx.get("promised_amount") or 2000), ctx.get("ability_to_pay")),
        "promised_date": ctx.get("promised_date_str", "the promised date"),
        "promised_amount": int(ctx.get("promised_amount") or 0),
        "name_placeholder": "[NAME]",
    }
    out: List[Prompt] = []

    def add(pid: str, rule: str):
        if any(p.prompt_id == pid for p in out) or len(out) >= max_prompts:
            return
        t = PROMPTS[pid]
        out.append(Prompt(pid, t["agent"].format(**fill), t[lang].format(**fill), rule))

    misaligned = after_sal is not None and (f.get("terms_salary_aligned") == 0.0)
    if ptype == "third_party":
        add("P_REACH_BORROWER", "R1_third_party")
    if hardship:
        if misaligned:
            add("P_MOVE_AFTER_SALARY", "R2_hardship_salary_misaligned")
        add("P_HARDSHIP_SUPPORT", "R2_hardship")
        return out  # guardrail: no part-payment asks under hardship
    if (f.get("borrower_silent_on_terms") or 0) >= 1 or ptype == "agent_recorded_or_pushed":
        add("P_ASK_TERMS_OWN_WORDS", "R3_terms_not_from_borrower")
    if misaligned and p_keep < cfg.high_credibility:
        add("P_MOVE_AFTER_SALARY", "R4_salary_misaligned")
    if ptype == "repeat_promiser" and p_keep < cfg.high_credibility:
        add("P_REPEAT_SMALLER_NEARER", "R5_repeat")
    if p_keep < cfg.low_credibility or ptype == "escape":
        add("P_PART_PAYMENT_LINK", "R6_weak_promise")
    if p_keep >= cfg.high_credibility and not out:
        add("P_CONFIRM_LINK", "R7_credible")
    for p in out:
        assert fpc_check(p.script) and fpc_check(p.agent_hint)
    return out


# --------------------------------------------------------------------------- disposition
def disposition(p_keep: float, ptype: str, hardship: bool, days_to_promise: float,
                captured_ts: Optional[datetime] = None, cfg: PolicyConfig = CONFIG.policy) -> Disposition:
    captured_ts = captured_ts or datetime.now()
    dtp = int(max(1, min(days_to_promise if not math.isnan(days_to_promise) else 7, cfg.default_parking_days_cap)))
    guard: List[str] = []

    if ptype == "agent_recorded_or_pushed":
        d = Disposition("integrity_review_and_confirm", min(dtp, 1), ["confirmation_call_by_different_agent_or_bot:+1d"],
                        False, True, "integrity_team", "D6_agent_recorded")
    elif ptype == "third_party":
        d = Disposition("confirm_with_borrower", min(dtp, 1), ["borrower_confirmation_call:+1d"], True, False,
                        "standard", "D4_third_party")
    elif p_keep >= cfg.high_credibility and ptype in ("genuine_feasible", "genuine_infeasible"):
        d = Disposition("park_until_due", dtp, ["whatsapp_reminder:due-1d", "payment_link:due"], False, False,
                        "standard", "D1_credible")
    elif ptype == "genuine_infeasible":
        d = Disposition("renegotiate_or_restructure", min(dtp, 3), ["supportive_call:+2d"], False, False,
                        cfg.hardship_route, "D2_infeasible")
    elif ptype == "repeat_promiser":
        d = Disposition("stronger_treatment_short_parking", min(dtp, cfg.repeat_parking_cap_days),
                        ["human_agent_call:+%dd" % min(dtp, cfg.repeat_parking_cap_days)], False, False,
                        "senior_tele_agent", "D5_repeat")
    elif p_keep < cfg.low_credibility or ptype == "escape":
        d = Disposition("part_payment_follow_up", min(dtp, cfg.weak_promise_parking_cap_days),
                        ["payment_link_whatsapp:+%dh" % cfg.escape_follow_up_hours, "voice_bot_check_in:+1d"],
                        False, False, "standard", "D3_weak_or_escape")
    else:
        d = Disposition("park_with_mid_check", dtp, ["voice_bot_check_in:mid", "whatsapp_reminder:due-1d"], False,
                        False, "standard", "D7_moderate")

    # -------- hardship guardrail: never harsher than the default fixed rule
    if hardship and d.treatment in ("stronger_treatment_short_parking", "part_payment_follow_up",
                                    "park_with_mid_check", "integrity_review_and_confirm"):
        guard.append("G1_hardship_no_harsher_treatment")
        integrity = d.integrity_review
        d = Disposition("supportive_restructure_or_human", dtp, ["supportive_call_by_human:+2d"],
                        d.third_party_confirmation_call, integrity, cfg.hardship_route, d.rule)
    if hardship and d.route == "standard":
        d.route = cfg.hardship_route
        guard.append("G1_hardship_route")
    # -------- contact-hour guardrail is applied when reminders are materialised
    guard.append("G2_contact_hours_08_19")
    d.guardrails_applied = guard
    return d


def materialise_schedule(d: Disposition, captured_ts: datetime, promised_date: datetime) -> List[dict]:
    """Turn reminder specs into concrete, contact-hour compliant timestamps (max one per day)."""
    out, used_days = [], set()
    for spec in d.reminders:
        action, _, when = spec.partition(":")
        if when.startswith("due-"):
            ts = promised_date - timedelta(days=int(when[4:-1]))
        elif when == "due":
            ts = promised_date
        elif when == "mid":
            ts = captured_ts + (promised_date - captured_ts) / 2
        elif when.endswith("h"):
            ts = captured_ts + timedelta(hours=int(when[1:-1]))
        else:
            ts = captured_ts + timedelta(days=int(when[1:-1]))
        ts = _in_contact_hours(ts.replace(hour=max(ts.hour, 10)) if ts.hour == 0 else ts)
        while ts.date() in used_days:
            ts = _in_contact_hours(ts + timedelta(days=1))
        used_days.add(ts.date())
        out.append({"action": action, "scheduled_ts": ts.isoformat(timespec="minutes")})
    return out
