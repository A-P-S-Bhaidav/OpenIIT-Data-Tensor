"""Replay a historical call through the live engine and print what the agent desktop would show.

Usage:  python scripts/replay_call.py --ptp PTP000162 [--no-voice-consent]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from ptp_credibility.data import load_tables  # noqa: E402
from ptp_credibility.service.engine import CredibilityEngine  # noqa: E402


def bar(p: float, width: int = 20) -> str:
    n = int(round(p * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def main(ptp_id: str, voice_consent: bool) -> None:
    t = load_tables()
    eng = CredibilityEngine()
    ctx = eng.store.df.loc[ptp_id]
    if not isinstance(ctx.attempt_id, str):
        raise SystemExit(f"{ptp_id} is a field-visit PTP without a transcript")
    s = eng.start(ptp_id, voice_consent=voice_consent)
    turns = t.transcripts[t.transcripts.call_id == ctx.attempt_id]
    print(f"PTP {ptp_id} | account {ctx.account_id} | portfolio {ctx.acct_portfolio} | agent {ctx.agent_id} | "
          f"language {ctx.acct_language}\n" + "-" * 110)
    shown = set()
    for _, x in turns.iterrows():
        r = s.add_turn(x.speaker, x.text, x.pause_before_ms, x.start_ms, x.end_ms, x.asr_confidence)
        print(f"{x.speaker:>8}: {x.text[:70]:<70} {bar(r['p_keep'])} {r['p_keep']:.2f} {r['likely_type'][:18]:<18}"
              f"{' ACT' if r['act_now'] else ''} ({r['latency_ms']:.0f} ms)")
        for p in r["prompts"]:
            if p["prompt_id"] not in shown:
                shown.add(p["prompt_id"])
                print(f"{'':>10}>> PROMPT {p['prompt_id']} [{p['rule']}]: {p['agent_hint']}")
                print(f"{'':>10}   Suggested line: {p['script']}")
    end = s.end_call()
    print("-" * 110)
    print(f"Final keep probability {end['p_keep']:.2f} ({end['credibility_band']}), likely type {end['likely_type']}, "
          f"hardship flag {end['hardship_flag']}")
    d = end["disposition"]
    print(f"Disposition: {d['treatment']} | parking {d['parking_days']} day(s) | route {d['route']} | rule {d['rule']} | "
          f"guardrails {', '.join(d['guardrails_applied'])}")
    for x in end["schedule"]:
        print(f"  scheduled: {x['action']} at {x['scheduled_ts']}")
    for direction in ("lowers", "raises"):
        for rc in end["reasons"][direction]:
            print(f"  {direction:>6}: [{rc['code']}] {rc['text']} (log-odds {rc['log_odds']:+.2f})")
    print(f"Actual outcome: {ctx.outcome_status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptp", default="PTP000162")
    ap.add_argument("--no-voice-consent", action="store_true")
    a = ap.parse_args()
    main(a.ptp, not a.no_voice_consent)
