"""Agent integrity analytics: quality-weighted PTPs and agent-pushed PTP detection.

The keep model must stay *independent* of agents (they will adapt to any
scorer they can see). Integrity is therefore assessed with a separate,
case-mix-adjusted view:

* **Quality-weighted PTPs** = sum of calibrated keep probabilities, reported
  next to raw PTP counts for every agent, bot and team.
* **Observed / expected keeps** - expected keeps come from a *case-mix* model
  that uses only account, history and promise-term features (nothing the
  agent controls), so a low O/E is not explained by a harder book.
* **Conversation evidence** - share of the agent's PTPs where the borrower
  never stated a date or amount, PTPs right after pressure language,
  end-of-shift bunching, copied remarks and (field) reused visit photos.

Flags use empirical-Bayes shrinkage and z-scores against peers on the same
channel, and every flag lists the evidence behind it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZFLAG = 2.0


def _z(rate: pd.Series, n: pd.Series, peer_rate: float) -> pd.Series:
    se = np.sqrt(peer_rate * (1 - peer_rate) / n.clip(lower=1))
    return (rate - peer_rate) / se.replace(0, np.nan)


def agent_integrity_table(df: pd.DataFrame, p_col: str = "p_keep", casemix_col: str = "p_casemix") -> pd.DataFrame:
    d = df.copy()
    d["resolved"] = d["kept"].notna()
    d["end_of_shift"] = (d["terms_hours_to_shift_end"] <= 1.0).astype(float)
    d["silent"] = d["borrower_silent_on_terms"]
    d["pressured"] = (d["pressure_then_agree"].fillna(0) > 0).astype(float)
    d["copied_remark"] = (d["remark_copies_before"].fillna(0) >= 3).astype(float)
    g = d.groupby("agent_id")
    t = pd.DataFrame({
        "channel": g["channel"].first(),
        "raw_ptps": g.size(),
        "quality_weighted_ptps": g[p_col].sum(),
        "observed_keeps": g["kept"].sum(),
        "expected_keeps_casemix": g[casemix_col].sum(),
        "silent_terms_share": g["silent"].mean(),
        "pressure_then_agree_share": g["pressured"].mean(),
        "end_of_shift_share": g["end_of_shift"].mean(),
        "copied_remark_share": g["copied_remark"].mean(),
        "photo_reuse_share": g["field_photo_reused"].mean(),
    })
    t["ptp_quality_ratio"] = t.quality_weighted_ptps / t.raw_ptps
    t["obs_to_exp"] = (t.observed_keeps + 2) / (t.expected_keeps_casemix + 2)
    out = []
    for ch, grp in t.groupby("channel"):
        grp = grp.copy()
        peer_keep = grp.observed_keeps.sum() / grp.raw_ptps.sum()
        exp_rate = grp.expected_keeps_casemix / grp.raw_ptps
        obs_rate = grp.observed_keeps / grp.raw_ptps
        se = np.sqrt(exp_rate * (1 - exp_rate) / grp.raw_ptps)
        grp["keep_gap_z"] = (obs_rate - exp_rate) / se.replace(0, np.nan)
        for col in ["silent_terms_share", "pressure_then_agree_share", "end_of_shift_share", "copied_remark_share"]:
            peer = grp[col].mean()
            grp[col + "_z"] = _z(grp[col], grp.raw_ptps, min(max(peer, 1e-3), 0.999))
        evid = []
        for _, r in grp.iterrows():
            e = []
            if r.keep_gap_z <= -ZFLAG:
                e.append(f"keeps {r.observed_keeps:.0f} vs {r.expected_keeps_casemix:.1f} expected for this book (z={r.keep_gap_z:.1f})")
            if r.silent_terms_share_z >= ZFLAG:
                e.append(f"{r.silent_terms_share:.0%} of PTPs without borrower-stated terms (z={r.silent_terms_share_z:.1f})")
            if r.pressure_then_agree_share_z >= ZFLAG:
                e.append(f"{r.pressure_then_agree_share:.0%} of PTPs agreed right after pressure (z={r.pressure_then_agree_share_z:.1f})")
            if r.end_of_shift_share_z >= ZFLAG:
                e.append(f"{r.end_of_shift_share:.0%} of PTPs in the last shift hour (z={r.end_of_shift_share_z:.1f})")
            if r.copied_remark_share_z >= ZFLAG:
                e.append(f"{r.copied_remark_share:.0%} of PTP remarks copy-pasted (z={r.copied_remark_share_z:.1f})")
            if (r.photo_reuse_share or 0) >= 0.05:
                e.append(f"{r.photo_reuse_share:.0%} of field PTPs with a reused visit photo")
            evid.append("; ".join(e))
        grp["integrity_evidence"] = evid
        grp["integrity_flag"] = grp["integrity_evidence"].str.len() > 0
        out.append(grp)
    res = pd.concat(out).sort_values(["integrity_flag", "keep_gap_z"], ascending=[False, True])
    return res.reset_index()
