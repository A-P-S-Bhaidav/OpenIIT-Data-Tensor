"""Dataset construction shared by training, evaluation and the service."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG
from .data import Tables, load_tables
from .features import build_context, post_call_features, transcript_features
from .labels import build_outcomes
from .models import add_derived, mask_for_prefix


@dataclass
class Datasets:
    tables: Tables
    outcomes: pd.DataFrame
    ptp: pd.DataFrame          # one row per PTP, end-of-call features
    prefix: pd.DataFrame       # one row per (call PTP, transcript prefix)
    call_feats: pd.DataFrame


def build_datasets(with_prefix: bool = True) -> Datasets:
    t = load_tables()
    outcomes = build_outcomes(t.ptps, t.payments, CONFIG.label)
    cf = transcript_features(t.transcripts)
    ptp = build_context(t, outcomes, cf)
    ptp = ptp.merge(post_call_features(t.ptps, t.post_call_events, outcomes), on="ptp_id", how="left")
    ptp = add_derived(ptp)
    ptp = ptp.merge(cf[["call_id", "customer_text", "agent_text"]].rename(columns={"call_id": "attempt_id"}),
                    on="attempt_id", how="left")
    prefix = pd.DataFrame()
    if with_prefix:
        pf = transcript_features(t.transcripts, prefixes=True)
        conv_cols = [c for c in pf.columns if c not in ("call_id",)]
        ctx = ptp.drop(columns=[c for c in conv_cols if c in ptp.columns and c not in ("customer_text", "agent_text")])
        ctx = ctx.drop(columns=["customer_text", "agent_text"], errors="ignore")
        prefix = pf.merge(ctx, left_on="call_id", right_on="attempt_id", how="inner")
        prefix = mask_for_prefix(prefix)
        # salary alignment depends on terms -> already masked by prefix (terms_*)
    return Datasets(t, outcomes, ptp, prefix, cf)
