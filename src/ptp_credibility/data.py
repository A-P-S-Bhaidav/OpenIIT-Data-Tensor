"""Data access layer with schema validation.

Every table is loaded with explicit dtypes and timestamp parsing, and basic
referential-integrity checks are run so that silent join failures surface
immediately rather than as degraded model quality.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .config import CONFIG, Paths


@dataclass
class Tables:
    ptps: pd.DataFrame
    transcripts: pd.DataFrame
    post_call_events: pd.DataFrame
    annotations: pd.DataFrame
    accounts: pd.DataFrame
    agents: pd.DataFrame
    lenders: pd.DataFrame
    payments: pd.DataFrame
    splits: pd.DataFrame
    dial_attempts: pd.DataFrame
    field_visits: pd.DataFrame
    addresses: pd.DataFrame


REQUIRED_COLUMNS = {
    "ptps": ["ptp_id", "account_id", "source", "attempt_id", "visit_id", "channel", "agent_id",
             "captured_ts", "promised_date", "promised_amount", "overdue_at_capture",
             "answerer_recorded", "payment_link_sent"],
    "transcripts": ["call_id", "turn_idx", "speaker", "start_ms", "end_ms", "pause_before_ms",
                    "text", "asr_confidence"],
    "payments": ["payment_id", "account_id", "payment_ts", "amount", "channel"],
    "accounts": ["account_id", "portfolio", "salary_credit_day", "prev_ptp_count", "prev_ptp_broken"],
}


def _check(name: str, df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in df.columns]
    if missing:
        raise ValueError(f"Table '{name}' is missing required columns: {missing}")


@lru_cache(maxsize=1)
def load_tables(paths: Paths = CONFIG.paths) -> Tables:
    p1, sh = paths.ps1_dir, paths.shared_dir
    ptps = pd.read_csv(p1 / "ptps.csv", parse_dates=["captured_ts", "promised_date"])
    transcripts = pd.read_csv(p1 / "call_transcripts.csv")
    transcripts["text"] = transcripts["text"].fillna("").astype(str)
    events = pd.read_csv(p1 / "post_call_events.csv", parse_dates=["event_ts"])
    ann = pd.read_csv(p1 / "annotated_ptp_sample.csv")
    accounts = pd.read_csv(sh / "accounts.csv")
    agents = pd.read_csv(sh / "agents.csv")
    lenders = pd.read_csv(sh / "lenders.csv")
    payments = pd.read_csv(sh / "payments.csv", parse_dates=["payment_ts"])
    splits = pd.read_csv(sh / "splits.csv")
    dial = pd.read_csv(sh / "dial_attempts.csv", parse_dates=["attempt_ts"])
    visits = pd.read_csv(sh / "field_visits.csv", parse_dates=["visit_date", "start_ts", "checkin_ts"])
    addresses = pd.read_csv(sh / "addresses.csv")

    for name, df in [("ptps", ptps), ("transcripts", transcripts), ("payments", payments),
                     ("accounts", accounts)]:
        _check(name, df)

    ptps["payment_link_sent"] = ptps["payment_link_sent"].astype(str).str.lower().eq("true")
    transcripts = transcripts.sort_values(["call_id", "turn_idx"]).reset_index(drop=True)

    # Referential integrity
    unknown_acc = set(ptps.account_id) - set(accounts.account_id)
    if unknown_acc:
        raise ValueError(f"{len(unknown_acc)} PTP accounts not found in accounts table")
    if ptps.ptp_id.duplicated().any():
        raise ValueError("Duplicate ptp_id values found")

    return Tables(ptps, transcripts, events, ann, accounts, agents, lenders, payments, splits,
                  dial, visits, addresses)
