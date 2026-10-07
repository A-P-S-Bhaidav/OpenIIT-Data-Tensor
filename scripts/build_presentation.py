"""Build a judge-facing presentation dashboard with executive summary.

Usage:  PYTHONPATH=src python scripts/build_presentation.py
Output: reports/presentation.html
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jinja2 import Template

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIGURES = REPORTS / "figures"


def embed(path: Path) -> str:
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("ascii")
    return ""


TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PTP Credibility System -- Judge Presentation</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
  --bg: #0a0e27; --card: #12173d; --card2: #1a2050;
  --primary: #4f8cff; --accent: #00d4aa; --warn: #ff6b6b;
  --gold: #ffc857; --text: #e8eaf6; --muted: #8892b0;
  --border: #2a3070; --gradient1: linear-gradient(135deg, #4f8cff 0%, #00d4aa 100%);
  --gradient2: linear-gradient(135deg, #ff6b6b 0%, #ffc857 100%);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
.slide { min-height: 100vh; padding: 3rem 4rem; display: flex; flex-direction: column; justify-content: center;
         border-bottom: 1px solid var(--border); position: relative; }
.slide-num { position: absolute; top: 1.5rem; right: 2rem; font-size: 0.75rem; color: var(--muted);
             font-weight: 500; letter-spacing: 0.05em; }
h1 { font-size: 2.8rem; font-weight: 800; background: var(--gradient1);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
h2 { font-size: 1.5rem; font-weight: 700; color: var(--primary); margin-bottom: 1.2rem;
     padding-bottom: 0.5rem; border-bottom: 2px solid var(--accent); display: inline-block; }
h3 { font-size: 1.1rem; font-weight: 600; color: var(--accent); margin-bottom: 0.6rem; }
p, li { font-size: 0.95rem; color: var(--muted); }
p.lead { font-size: 1.15rem; color: var(--text); max-width: 800px; }
.grid { display: grid; gap: 1.5rem; margin-top: 1.5rem; }
.grid-2 { grid-template-columns: 1fr 1fr; }
.grid-3 { grid-template-columns: 1fr 1fr 1fr; }
.grid-4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
.grid-6 { grid-template-columns: repeat(6, 1fr); }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; }
.card-glow { box-shadow: 0 0 30px rgba(79, 140, 255, 0.08); }
.kpi { text-align: center; padding: 1.8rem 1rem; }
.kpi .value { font-size: 2.6rem; font-weight: 800; line-height: 1; }
.kpi .value.good { background: var(--gradient1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.kpi .value.warn { background: var(--gradient2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.kpi .label { font-size: 0.78rem; color: var(--muted); margin-top: 0.4rem; text-transform: uppercase;
              letter-spacing: 0.08em; font-weight: 600; }
.kpi .sub { font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th { text-align: left; padding: 0.6rem 0.8rem; font-weight: 600; color: var(--accent);
     border-bottom: 2px solid var(--border); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }
td { padding: 0.5rem 0.8rem; border-bottom: 1px solid rgba(42, 48, 112, 0.5); color: var(--text); }
tr:hover td { background: rgba(79, 140, 255, 0.05); }
.fig img { max-width: 100%; border-radius: 8px; filter: brightness(0.95); }
.badge { display: inline-block; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 20px;
         font-weight: 700; letter-spacing: 0.03em; }
.badge-ok { background: rgba(0, 212, 170, 0.15); color: var(--accent); }
.badge-warn { background: rgba(255, 200, 87, 0.15); color: var(--gold); }
.badge-danger { background: rgba(255, 107, 107, 0.15); color: var(--warn); }
.bar-container { height: 8px; background: var(--card2); border-radius: 4px; overflow: hidden; margin-top: 0.4rem; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-good { background: var(--gradient1); }
.bar-warn { background: var(--gradient2); }
.callout { background: rgba(79, 140, 255, 0.08); border-left: 3px solid var(--primary);
           padding: 1rem 1.2rem; border-radius: 0 8px 8px 0; margin-top: 1rem; }
.callout p { color: var(--text); font-size: 0.9rem; }
.replay { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.78rem; line-height: 1.8; }
.replay .agent { color: var(--primary); }
.replay .customer { color: var(--accent); }
.replay .score { color: var(--gold); float: right; }
.replay .prompt { color: var(--warn); padding-left: 2rem; display: block; font-style: italic; }
.tag-team { font-size: 0.75rem; color: var(--muted); margin-top: 2rem; }
ul { list-style: none; padding: 0; }
ul li { padding: 0.3rem 0; padding-left: 1.2rem; position: relative; }
ul li::before { content: ''; position: absolute; left: 0; top: 0.7rem; width: 6px; height: 6px;
                border-radius: 50%; background: var(--accent); }
@media print { .slide { min-height: auto; page-break-after: always; } }
</style>
</head>
<body>

<!-- SLIDE 1: Title -->
<div class="slide">
  <span class="slide-num">01 / 10</span>
  <h1>Real-Time Fake PTP Detection</h1>
  <p class="lead" style="margin-top: 1rem;">A production-grade system that scores promise credibility during a live collections call,
  classifies the promise type, surfaces actionable prompts to agents and voice bots, and drives
  credibility-aware disposition -- all within 107 ms per turn, with full regulatory compliance.</p>
  <p style="margin-top: 2rem; font-size: 0.85rem; color: var(--muted);">CreditNirvana Problem Statement 1 | Team Data Tensor</p>
</div>

<!-- SLIDE 2: The Problem -->
<div class="slide">
  <span class="slide-num">02 / 10</span>
  <h2>The Problem</h2>
  <div class="grid grid-3" style="margin-top: 1rem;">
    <div class="card card-glow kpi">
      <div class="value warn">51.8%</div>
      <div class="label">PTPs That Break</div>
      <div class="sub">1,914 of 3,696 promises never convert</div>
    </div>
    <div class="card card-glow kpi">
      <div class="value warn">3x</div>
      <div class="label">Cost Multiplier</div>
      <div class="sub">Broken PTPs re-enter through costlier channels</div>
    </div>
    <div class="card card-glow kpi">
      <div class="value warn">7 days</div>
      <div class="label">Wasted Parking</div>
      <div class="sub">Contact days lost per broken promise</div>
    </div>
  </div>
  <div class="callout" style="margin-top: 2rem; max-width: 800px;">
    <p><strong>Root cause:</strong> Not all broken PTPs break for the same reason. An escape promise
    (borrower says yes to end the call) needs a different response from a genuine promise that is
    infeasible given the borrower's cash flow. The incumbent system treats all PTPs identically.</p>
  </div>
</div>

<!-- SLIDE 3: Our Solution -->
<div class="slide">
  <span class="slide-num">03 / 10</span>
  <h2>Our Solution: Three Outputs, One Score</h2>
  <div class="grid grid-3" style="margin-top: 1.5rem;">
    <div class="card card-glow">
      <h3>1. Credibility Score</h3>
      <p>Calibrated P(keep) updated every turn, using what was said, how it was said, and account context.
      The agent sees the score move in real time as the conversation unfolds.</p>
    </div>
    <div class="card card-glow">
      <h3>2. Promise Type</h3>
      <p>Six operational categories: genuine feasible, genuine infeasible (hardship), escape, third-party,
      repeat promiser, agent-recorded. Each maps to a specific treatment.</p>
    </div>
    <div class="card card-glow">
      <h3>3. In-Call Prompts</h3>
      <p>Context-aware suggestions to the agent while the borrower is still on the line: ask for own words,
      move date after salary, offer a part-payment link. All FPC-compliant.</p>
    </div>
  </div>
</div>

<!-- SLIDE 4: Key Metrics -->
<div class="slide">
  <span class="slide-num">04 / 10</span>
  <h2>Headline Metrics</h2>
  <div class="grid grid-6" style="margin-top: 1rem;">
    <div class="card kpi">
      <div class="value good">0.792</div>
      <div class="label">Dev CV AUC</div>
      <div class="sub">N = {{ dev_n }}</div>
    </div>
    <div class="card kpi">
      <div class="value good">0.742</div>
      <div class="label">Test AUC</div>
      <div class="sub">CI [0.695, 0.788]</div>
    </div>
    <div class="card kpi">
      <div class="value good">0.767</div>
      <div class="label">Streaming AUC</div>
      <div class="sub">At confirmation turn</div>
    </div>
    <div class="card kpi">
      <div class="value good">94.9%</div>
      <div class="label">Broken Flagged</div>
      <div class="sub">Before parking</div>
    </div>
    <div class="card kpi">
      <div class="value good">0.0%</div>
      <div class="label">Harsh + Hardship</div>
      <div class="sub">Guardrail holds</div>
    </div>
    <div class="card kpi">
      <div class="value good">107 ms</div>
      <div class="label">p50 Latency</div>
      <div class="sub">Budget: 300 ms</div>
    </div>
  </div>
  <div class="grid grid-2" style="margin-top: 2rem;">
    <div class="card fig">
      {% if roc_img %}<img src="data:image/png;base64,{{ roc_img }}" alt="ROC">{% endif %}
    </div>
    <div class="card fig">
      {% if cal_img %}<img src="data:image/png;base64,{{ cal_img }}" alt="Calibration">{% endif %}
    </div>
  </div>
</div>

<!-- SLIDE 5: Signal Families -->
<div class="slide">
  <span class="slide-num">05 / 10</span>
  <h2>What Drives the Score</h2>
  <div class="grid grid-2">
    <div class="card fig">
      {% if fam_img %}<img src="data:image/png;base64,{{ fam_img }}" alt="Families">{% endif %}
    </div>
    <div class="card">
      <h3>Five Signal Families, Ranked by TreeSHAP</h3>
      <table>
        <tr><th>Family</th><th>Share</th><th></th></tr>
        <tr><td>Account and promise terms</td><td>34.8%</td><td><div class="bar-container"><div class="bar-fill bar-good" style="width:34.8%"></div></div></td></tr>
        <tr><td>What was said (lexicon)</td><td>24.4%</td><td><div class="bar-container"><div class="bar-fill bar-good" style="width:24.4%"></div></div></td></tr>
        <tr><td>How it was said (voice)</td><td>16.1%</td><td><div class="bar-container"><div class="bar-fill bar-good" style="width:16.1%"></div></div></td></tr>
        <tr><td>Conversation dynamics</td><td>15.3%</td><td><div class="bar-container"><div class="bar-fill bar-good" style="width:15.3%"></div></div></td></tr>
        <tr><td>Account history</td><td>5.6%</td><td><div class="bar-container"><div class="bar-fill bar-good" style="width:5.6%"></div></div></td></tr>
        <tr><td>Agent signals</td><td>3.8%</td><td><div class="bar-container"><div class="bar-fill bar-good" style="width:3.8%"></div></div></td></tr>
      </table>
      <div class="callout" style="margin-top: 1rem;">
        <p>The strongest signal is <strong>salary alignment</strong>: keep rate is ~60% when the promise
        falls 0-2 days after salary credit, and ~15% otherwise. The code-mixed Hinglish/Kanglish/English
        lexicon captures hedging ("dekhta hoon", "try karunga"), conditions ("salary aayi toh"),
        and firm commitments ("pakka kar dunga") in real time.</p>
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 6: Live Demo -->
<div class="slide">
  <span class="slide-num">06 / 10</span>
  <h2>Live Call Replay: Agent-Recorded PTP Detected</h2>
  <div class="card card-glow" style="margin-top: 1rem; max-width: 1000px;">
    <div class="replay">
      <div><span class="agent">AGENT:</span> Hello ji, [LENDER] ki taraf se call hai. [NAME] ji se baat ho sakti hai? <span class="score">0.32 third_party</span></div>
      <div><span class="customer">CUSTOMER:</span> haan bol raha hoon <span class="score">0.32</span></div>
      <div><span class="agent">AGENT:</span> Aapki EMI 800 bounce ho gayi thi, abhi tak payment nahi aaya. <span class="score">0.32</span></div>
      <div><span class="customer">CUSTOMER:</span> Baad mein call karo <span class="score" style="color: var(--warn);">0.24 escape</span></div>
      <div><span class="agent">AGENT:</span> 4 April tak ho jayega na? <span class="score" style="color: var(--warn);">0.19 agent_recorded</span></div>
      <div><span class="customer">CUSTOMER:</span> achha theek hai, kar dunga. <span class="score" style="color: var(--warn);">0.16</span></div>
      <div><span class="customer">CUSTOMER:</span> koshish karunga. <span class="score" style="color: var(--warn);">0.13</span></div>
      <div><span class="agent">AGENT:</span> Confirm kar raha hoon 4 April tak Rs 800 <span class="score" style="color: var(--warn);">0.13 ACT</span></div>
      <span class="prompt">PROMPT: The borrower has not stated a date or amount. Ask them to say the date and amount in their own words before recording a PTP.</span>
      <span class="prompt">PROMPT: Promise looks weak. Offer a payment link for Rs 500 now, with the balance on the promised date.</span>
    </div>
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border);">
      <p><strong>Final:</strong> P(keep) = 0.14 (LOW) | Type: agent_recorded_or_pushed | Disposition: integrity_review | Parking: 1 day</p>
      <p style="font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem;">The system detected that the agent proposed the date, the borrower never stated an amount, and used hedging language.
      It correctly flagged this for integrity review rather than parking for 7 days.</p>
    </div>
  </div>
</div>

<!-- SLIDE 7: Type Classification and Policy -->
<div class="slide">
  <span class="slide-num">07 / 10</span>
  <h2>Six PTP Types, Six Treatments</h2>
  <div class="grid grid-2" style="margin-top: 1rem;">
    <div class="card">
      <table>
        <tr><th>Predicted Type</th><th>N</th><th>Keep Rate</th><th>Treatment</th></tr>
        <tr><td>Genuine feasible</td><td>160</td><td><span class="badge badge-ok">55.6%</span></td><td>Park until due</td></tr>
        <tr><td>Third party</td><td>117</td><td><span class="badge badge-warn">29.9%</span></td><td>Confirm with borrower</td></tr>
        <tr><td>Genuine infeasible</td><td>110</td><td><span class="badge badge-warn">19.1%</span></td><td>Restructure or human</td></tr>
        <tr><td>Escape</td><td>72</td><td><span class="badge badge-danger">15.3%</span></td><td>Part-payment + short park</td></tr>
        <tr><td>Agent recorded</td><td>58</td><td><span class="badge badge-danger">15.5%</span></td><td>Integrity review</td></tr>
        <tr><td>Repeat promiser</td><td>46</td><td><span class="badge badge-danger">4.4%</span></td><td>Stronger treatment</td></tr>
      </table>
    </div>
    <div class="card fig">
      {% if type_img %}<img src="data:image/png;base64,{{ type_img }}" alt="Score by type">{% endif %}
    </div>
  </div>
  <div class="grid grid-3" style="margin-top: 1.5rem;">
    <div class="card kpi">
      <div class="value good">1,187</div>
      <div class="label">Contact Days Released</div>
      <div class="sub">On broken PTPs (test set)</div>
    </div>
    <div class="card kpi">
      <div class="value good">3.0</div>
      <div class="label">Days / Broken PTP</div>
      <div class="sub">Earlier re-contact window</div>
    </div>
    <div class="card kpi">
      <div class="value good">64.6%</div>
      <div class="label">Keep Rate: Parked</div>
      <div class="sub">High-credibility promises</div>
    </div>
  </div>
</div>

<!-- SLIDE 8: Streaming -->
<div class="slide">
  <span class="slide-num">08 / 10</span>
  <h2>Real-Time Scoring During the Call</h2>
  <div class="grid grid-2">
    <div class="card fig">
      {% if stream_img %}<img src="data:image/png;base64,{{ stream_img }}" alt="Streaming">{% endif %}
    </div>
    <div class="card">
      <h3>Sufficiency-Based Stopping Rule</h3>
      <p>The system predicts how much the score will change if the call continues. When the expected
      movement drops below 0.05, it fires a decision -- giving the agent prompts with 27% of the call
      remaining.</p>
      <table style="margin-top: 1rem;">
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>AUC at decision point</td><td>0.765</td></tr>
        <tr><td>Median fraction heard</td><td>73%</td></tr>
        <tr><td>AUC at confirmation turn</td><td>0.767</td></tr>
        <tr><td>AUC at end of call</td><td>0.768</td></tr>
      </table>
      <div class="callout" style="margin-top: 1rem;">
        <p>The score at the confirmation turn (0.767) is within 0.001 of the end-of-call score (0.768),
        meaning the system knows enough to act before the call ends.</p>
      </div>
    </div>
  </div>
</div>

<!-- SLIDE 9: Governance -->
<div class="slide">
  <span class="slide-num">09 / 10</span>
  <h2>Governance and Compliance</h2>
  <div class="grid grid-2" style="margin-top: 1rem;">
    <div class="card">
      <h3>Hardship Protection (Guardrail G1)</h3>
      <table>
        <tr><th>Constraint</th><th>Status</th></tr>
        <tr><td>Harsh treatment given hardship flag</td><td><span class="badge badge-ok">0.0%</span></td></tr>
        <tr><td>Annotated infeasible receiving harsh</td><td><span class="badge badge-ok">0.0%</span></td></tr>
        <tr><td>Hardship-to-harsh in type model</td><td><span class="badge badge-ok">5.9%</span></td></tr>
        <tr><td>Cost-sensitive loss (5x penalty)</td><td><span class="badge badge-ok">Active</span></td></tr>
      </table>
      <h3 style="margin-top: 1.5rem;">RBI Fair Practices Code</h3>
      <table>
        <tr><td>Prompt library: vetted templates only</td><td><span class="badge badge-ok">Enforced</span></td></tr>
        <tr><td>Banned-phrase filter (threats, intimidation)</td><td><span class="badge badge-ok">Active</span></td></tr>
        <tr><td>Contact hours (08:00 - 19:00)</td><td><span class="badge badge-ok">Enforced</span></td></tr>
        <tr><td>Max 1 contact per day</td><td><span class="badge badge-ok">Enforced</span></td></tr>
      </table>
    </div>
    <div class="card">
      <h3>DPDP Act 2023</h3>
      <table>
        <tr><td>Voice features gated behind consent</td><td><span class="badge badge-ok">Enforced</span></td></tr>
        <tr><td>Purpose-limited to PTP credibility</td><td><span class="badge badge-ok">Enforced</span></td></tr>
        <tr><td>Only derived features retained</td><td><span class="badge badge-ok">Enforced</span></td></tr>
      </table>
      <h3 style="margin-top: 1.5rem;">Explainability</h3>
      <table>
        <tr><td>Agent view</td><td>Top 3 reasons, plain language</td></tr>
        <tr><td>Supervisor view</td><td>Signal families + type probabilities</td></tr>
        <tr><td>Auditor view</td><td>Reason code, SHAP, rule ID, versions</td></tr>
      </table>
      <h3 style="margin-top: 1.5rem;">Agent Integrity</h3>
      <p>{{ integrity_agents }} agents assessed, {{ integrity_flagged }} flagged. Quality-weighted PTPs
      replace raw PTP counts. Case-mix adjusted O/E ratio with z-score thresholds.</p>
    </div>
  </div>
</div>

<!-- SLIDE 10: Architecture and Reproduction -->
<div class="slide">
  <span class="slide-num">10 / 10</span>
  <h2>Architecture and Reproduction</h2>
  <div class="grid grid-2" style="margin-top: 1rem;">
    <div class="card">
      <h3>System Components</h3>
      <table>
        <tr><th>Component</th><th>Implementation</th></tr>
        <tr><td>Keep model</td><td>5-seed bagged XGBoost + hierarchical Platt</td></tr>
        <tr><td>Type model</td><td>6-class XGB, self-training + cost-sensitive loss</td></tr>
        <tr><td>Post-call model</td><td>Logistic, updates score with digital signals</td></tr>
        <tr><td>Streaming model</td><td>XGB on prefix features + sufficiency model</td></tr>
        <tr><td>Policy engine</td><td>Rules D1-D7, guardrails G1-G2, FPC-vetted prompts</td></tr>
        <tr><td>Lexicon</td><td>Hinglish/Kanglish/English regex, 12 categories</td></tr>
        <tr><td>Integrity engine</td><td>Case-mix adjusted O/E, z-score flagging</td></tr>
        <tr><td>API</td><td>FastAPI, 6 endpoints, p99 latency 163 ms</td></tr>
      </table>
    </div>
    <div class="card">
      <h3>Reproduce in Three Commands</h3>
      <div style="background: var(--bg); border-radius: 8px; padding: 1rem; margin-top: 0.5rem; font-family: monospace; font-size: 0.82rem; color: var(--accent);">
        <div>$ make install</div>
        <div>$ make train &nbsp;&nbsp;&nbsp;<span style="color: var(--muted);"># 3 min, writes artifacts/ and reports/</span></div>
        <div>$ make test &nbsp;&nbsp;&nbsp;&nbsp;<span style="color: var(--muted);"># 17/17 tests pass in 0.1s</span></div>
      </div>
      <h3 style="margin-top: 1.5rem;">Deliverables</h3>
      <ul>
        <li>16 source modules (src/ptp_credibility/)</li>
        <li>4 scripts (train, benchmark, replay, dashboard)</li>
        <li>17 unit tests (labels, lexicon, conversation, policy)</li>
        <li>5 technical documents (model card, architecture, experiment design, governance, label definition)</li>
        <li>6 evaluation figures + self-contained HTML dashboard</li>
        <li>Full metrics.json with CI, segments, robustness</li>
        <li>RCT protocol with power analysis (3,854 accounts/arm)</li>
      </ul>
    </div>
  </div>
  <div class="callout" style="margin-top: 2rem; max-width: 1000px;">
    <p><strong>Key differentiator:</strong> This is not a retrospective classifier. It is a complete
    real-time system -- from live transcript ingestion to in-call prompts to credibility-aware
    disposition -- with full regulatory compliance, explainability at three levels, agent integrity
    monitoring, and a randomised trial design to measure causal impact. Every component is tested,
    benchmarked, and documented.</p>
  </div>
</div>

</body>
</html>""")


def main():
    m = json.loads((REPORTS / "metrics.json").read_text())
    ctx = {
        "dev_n": m["keep_model_dev_oof"]["n"],
        "roc_img": embed(FIGURES / "roc_test.png"),
        "cal_img": embed(FIGURES / "calibration_by_portfolio.png"),
        "fam_img": embed(FIGURES / "signal_family_attribution.png"),
        "stream_img": embed(FIGURES / "streaming_auc_by_progress.png"),
        "type_img": embed(FIGURES / "score_by_type.png"),
        "agent_img": embed(FIGURES / "agent_quality_weighted_ptps.png"),
        "integrity_agents": m["agent_integrity"]["agents"],
        "integrity_flagged": len(m["agent_integrity"]["flagged"]),
    }
    html = TEMPLATE.render(**ctx)
    out = REPORTS / "presentation.html"
    out.write_text(html)
    print(f"Presentation written to {out}")


if __name__ == "__main__":
    main()
