"""Latency benchmark for live in-call scoring.

Replays held-out test calls turn by turn through the real engine (feature
update, streaming score, type model, hardship flag, stopping rule, prompts
and reason codes) and reports latency percentiles per update.

Usage:  python scripts/benchmark_latency.py [--calls 150]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from ptp_credibility.config import CONFIG  # noqa: E402
from ptp_credibility.data import load_tables  # noqa: E402
from ptp_credibility.service.engine import CredibilityEngine  # noqa: E402


def main(n_calls: int) -> None:
    t = load_tables()
    eng = CredibilityEngine()
    scored = eng.store.df
    test_ptps = scored[(scored.split == "test") & scored.attempt_id.notna()].sample(
        n=min(n_calls, int(((scored.split == "test") & scored.attempt_id.notna()).sum())), random_state=0)
    tr = t.transcripts.set_index("call_id")
    lat, lat_noexp, end_lat = [], [], []
    for ptp_id, r in test_ptps.iterrows():
        turns = tr.loc[[r.attempt_id]]
        for explain, sink in ((True, lat), (False, lat_noexp)):
            eng.explain_live = explain
            s = eng.start(ptp_id)
            for _, x in turns.iterrows():
                t0 = time.perf_counter()
                s.add_turn(x.speaker, x.text, x.pause_before_ms, x.start_ms, x.end_ms, x.asr_confidence)
                sink.append((time.perf_counter() - t0) * 1000)
            if explain:
                t0 = time.perf_counter()
                s.end_call()
                end_lat.append((time.perf_counter() - t0) * 1000)
    q = lambda a: {f"p{p}": round(float(np.percentile(a, p)), 2) for p in (50, 90, 95, 99)} | {"max": round(float(np.max(a)), 2), "n_updates": len(a)}
    out = {"per_turn_update_ms_with_reason_codes": q(lat), "per_turn_update_ms_score_only": q(lat_noexp),
           "end_of_call_ms": q(end_lat), "calls": len(test_ptps), "budget_ms": 300,
           "hardware": "single CPU process, no batching, Python 3.11"}
    CONFIG.paths.reports.mkdir(parents=True, exist_ok=True)
    (CONFIG.paths.reports / "latency.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=150)
    main(ap.parse_args().calls)
