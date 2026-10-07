"""Unit tests for the core logic. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ptp_credibility.calibration import HierarchicalPlatt, expected_calibration_error  # noqa: E402
from ptp_credibility.config import LabelConfig  # noqa: E402
from ptp_credibility.conversation import ConversationState, Turn  # noqa: E402
from ptp_credibility.labels import build_outcomes  # noqa: E402
from ptp_credibility import lexicon as lx  # noqa: E402
from ptp_credibility.policy import (PROMPTS, disposition, fpc_check, in_call_prompts,  # noqa: E402
                                    materialise_schedule)


def run(turns):
    s = ConversationState()
    for sp, tx in turns:
        s.update(Turn(sp, tx, 400, 0, 1000, 0.9))
    return s


class TestLexicon(unittest.TestCase):
    def test_hedging_code_mixed(self):
        for t in ["dekhta hoon", "try karunga", "nodtini", "I will try my best", "shayad ho jayega"]:
            self.assertTrue(lx.CUSTOMER["hedge"].search(t), t)

    def test_conditional(self):
        for t in ["salary aayi toh 5 tareekh ko kar dunga", "party payment kottre maadtini", "if my client pays me"]:
            self.assertTrue(lx.CUSTOMER["conditional"].search(t), t)

    def test_third_party(self):
        for t in ["main unki wife bol rahi hoon", "naanu avra amma", "this is her mother, she is at work"]:
            self.assertTrue(lx.CUSTOMER["third_party"].search(t.lower()), t)

    def test_amount_extraction(self):
        self.assertEqual(lx.extract_amounts("4,200 rupees poora kodtini"), [4200.0])
        self.assertEqual(lx.extract_amounts("1,77,600 cash"), [177600.0])
        self.assertEqual(lx.extract_amounts("5 tareekh"), [])


class TestConversation(unittest.TestCase):
    def test_agent_recorded_pattern(self):
        s = run([("agent", "Aapki EMI 800 bounce ho gayi thi"), ("customer", "Baad mein call karo"),
                 ("agent", "Sir aaj hi karna padega, warna account escalate ho jayega"),
                 ("agent", "4 April tak ho jayega na?"), ("customer", "theek hai, kar dunga, rakhta hoon"),
                 ("agent", "Confirm kar raha hoon 4 April tak 800")])
        f = s.features()
        self.assertEqual(f["borrower_silent_on_terms"], 1.0)
        self.assertEqual(f["agent_proposed_date_first"], 1.0)
        self.assertGreaterEqual(f["agent_push_after_deferral"], 1)
        self.assertGreaterEqual(f["agree_and_close_same_turn"], 1)

    def test_genuine_pattern(self):
        s = run([("agent", "Aap payment kab tak kar payenge?"),
                 ("customer", "8 tareekh ko pakka kar dunga. 6,000 rupaye poora kar dunga"),
                 ("agent", "Confirm kar raha hoon 8 April tak 6000 rupaye")])
        f = s.features()
        self.assertEqual(f["borrower_stated_amount"], 1.0)
        self.assertEqual(f["borrower_stated_specific_date"], 1.0)
        self.assertEqual(f["confirm_amount_matches_borrower"], 1.0)
        self.assertEqual(f["confirm_day_matches_borrower"], 1.0)
        self.assertEqual(f["agent_proposed_date_first"], 0.0)

    def test_salary_info_is_not_a_promise_date(self):
        s = run([("agent", "Salary kis tareekh ko aati hai?"), ("customer", "har mahine 28 ko aa jaati hai")])
        f = s.features()
        self.assertEqual(f["stated_salary_day"], 28.0)
        self.assertEqual(f["borrower_stated_date"], 0.0)

    def test_incremental_equals_batch(self):
        turns = [("agent", "kab tak clear kar denge?"), ("customer", "next week dekhta hoon"),
                 ("agent", "kam se kam 2000 aaj kar dijiye"), ("customer", "abhi nahi ho payega")]
        a = run(turns).features()
        s = ConversationState()
        for sp, tx in turns:
            s.update(Turn(sp, tx, 400, 0, 1000, 0.9))
            _ = s.features()  # intermediate reads must not mutate state
        b = s.features()
        for k in a:
            if isinstance(a[k], float) and np.isnan(a[k]):
                self.assertTrue(np.isnan(b[k]))
            else:
                self.assertEqual(a[k], b[k], k)


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.ptps = pd.DataFrame({
            "ptp_id": ["P1", "P2", "P3", "P4"], "account_id": ["A", "B", "C", "C"],
            "captured_ts": pd.to_datetime(["2026-04-01 10:00", "2026-04-01 10:00", "2026-04-01 10:00", "2026-04-03 10:00"]),
            "promised_date": pd.to_datetime(["2026-04-05", "2026-04-05", "2026-04-10", "2026-04-08"]),
            "promised_amount": [1000.0, 1000.0, 1000.0, 1000.0]})
        self.pay = pd.DataFrame({"account_id": ["A", "B", "C"],
                                 "payment_ts": pd.to_datetime(["2026-04-07 12:00", "2026-04-04 12:00", "2026-04-05 12:00"]),
                                 "amount": [1000.0, 400.0, 1000.0], "channel": ["upi_link"] * 3})

    def test_grace_partial_and_supersession(self):
        o = build_outcomes(self.ptps, self.pay, LabelConfig(grace_days=3, min_fraction=0.9)).set_index("ptp_id")
        self.assertEqual(o.loc["P1", "kept"], 1)               # late but within grace
        self.assertEqual(o.loc["P1", "kept_on_time"], 0)
        self.assertEqual(o.loc["P2", "kept"], 0)               # 40 percent -> partial
        self.assertEqual(o.loc["P2", "partial"], 1)
        self.assertEqual(o.loc["P3", "kept"], 0)               # superseded by P4 before payment
        self.assertEqual(o.loc["P4", "kept"], 1)

    def test_strict_definition(self):
        o = build_outcomes(self.ptps, self.pay, LabelConfig(grace_days=0, min_fraction=0.9)).set_index("ptp_id")
        self.assertEqual(o.loc["P1", "kept"], 0)


class TestCalibration(unittest.TestCase):
    def test_hierarchical_platt_improves_ece(self):
        rng = np.random.default_rng(0)
        p_true = rng.uniform(0.05, 0.6, 4000)
        y = rng.binomial(1, p_true)
        seg = np.where(rng.uniform(size=4000) < 0.5, "a", "b")
        p_bad = np.clip(p_true * np.where(seg == "a", 1.5, 0.6), 0.01, 0.99)
        cal = HierarchicalPlatt().fit(p_bad, y, seg)
        p_fix = cal.transform(p_bad, seg)
        for s in ("a", "b"):
            m = seg == s
            self.assertLess(expected_calibration_error(y[m], p_fix[m]), expected_calibration_error(y[m], p_bad[m]))

    def test_monotonic_within_segment(self):
        cal = HierarchicalPlatt().fit(np.linspace(0.05, 0.95, 200), np.r_[np.zeros(100), np.ones(100)], np.array(["x"] * 200))
        out = cal.transform(np.linspace(0.05, 0.95, 50), "x")
        self.assertTrue(np.all(np.diff(out) >= 0))


class TestPolicy(unittest.TestCase):
    CTX = {"language": "hinglish", "captured_ts": datetime(2026, 4, 2, 11), "salary_day": 7.0,
           "promised_amount": 8000, "promised_date_str": "04 April", "ability_to_pay": 0.4}

    def test_prompt_library_is_fpc_compliant(self):
        for p in PROMPTS.values():
            for v in p.values():
                self.assertTrue(fpc_check(v))
        self.assertFalse(fpc_check("Aapka CIBIL kharab ho raha hai"))

    def test_hardship_never_harsher(self):
        for ptype in ["escape", "repeat_promiser", "genuine_infeasible", "genuine_feasible", "agent_recorded_or_pushed"]:
            for p in (0.02, 0.2, 0.5):
                d = disposition(p, ptype, hardship=True, days_to_promise=7)
                self.assertNotIn(d.treatment, {"stronger_treatment_short_parking", "part_payment_follow_up"})

    def test_hardship_prompts_do_not_ask_for_payment_today(self):
        ps = in_call_prompts(0.1, "escape", {"terms_salary_aligned": 0.0}, self.CTX, hardship=True)
        ids = [p.prompt_id for p in ps]
        self.assertNotIn("P_PART_PAYMENT_LINK", ids)
        self.assertIn("P_HARDSHIP_SUPPORT", ids)

    def test_escape_gets_part_payment_prompt(self):
        ps = in_call_prompts(0.1, "escape", {"terms_salary_aligned": 1.0}, self.CTX, hardship=False)
        self.assertIn("P_PART_PAYMENT_LINK", [p.prompt_id for p in ps])

    def test_schedule_inside_contact_hours(self):
        d = disposition(0.1, "escape", False, 5)
        sched = materialise_schedule(d, datetime(2026, 4, 2, 18, 50), datetime(2026, 4, 7))
        for s in sched:
            h = int(s["scheduled_ts"][11:13])
            self.assertTrue(8 <= h < 19, s)
        self.assertEqual(len({s["scheduled_ts"][:10] for s in sched}), len(sched))  # max one contact per day


if __name__ == "__main__":
    unittest.main()
