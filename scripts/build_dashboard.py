"""Build a self-contained HTML dashboard from reports/metrics.json and figures.

Usage:  python scripts/build_dashboard.py
Output: reports/dashboard.html
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

TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PTP Credibility -- Evaluation Dashboard</title>
<style>
:root { --bg: #f8f9fa; --card: #ffffff; --primary: #1f3b73; --accent: #2e86ab;
        --text: #212529; --muted: #6c757d; --border: #dee2e6; --success: #3b8f4f;
        --warn: #f18f01; --danger: #a23b72; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
       background: var(--bg); color: var(--text); line-height: 1.6; }
header { background: var(--primary); color: white; padding: 1.5rem 2rem; }
header h1 { font-size: 1.4rem; font-weight: 600; }
header p { font-size: 0.85rem; opacity: 0.85; margin-top: 0.25rem; }
.container { max-width: 1280px; margin: 0 auto; padding: 1.5rem; }
.grid { display: grid; gap: 1.25rem; }
.grid-2 { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
.grid-3 { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; }
.card h2 { font-size: 1rem; color: var(--primary); margin-bottom: 0.75rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.4rem; }
.kpi { text-align: center; }
.kpi .value { font-size: 2rem; font-weight: 700; color: var(--primary); }
.kpi .label { font-size: 0.78rem; color: var(--muted); }
.kpi .sub { font-size: 0.72rem; color: var(--muted); }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin-top: 0.5rem; }
th { background: var(--bg); text-align: left; padding: 0.4rem 0.6rem; font-weight: 600; border-bottom: 2px solid var(--border); }
td { padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--border); }
tr:hover td { background: #f0f4f8; }
.fig { text-align: center; }
.fig img { max-width: 100%; border-radius: 6px; }
.tag { display: inline-block; font-size: 0.7rem; padding: 0.15rem 0.45rem; border-radius: 3px; font-weight: 600; }
.tag-ok { background: #d4edda; color: #155724; }
.tag-warn { background: #fff3cd; color: #856404; }
.tag-err { background: #f8d7da; color: #721c24; }
footer { text-align: center; padding: 1.5rem; color: var(--muted); font-size: 0.75rem; }
</style>
</head>
<body>
<header>
  <h1>PTP Credibility System -- Evaluation Dashboard</h1>
  <p>Package v{{ v.package }} | Lexicon v{{ v.lexicon }} | Policy v{{ v.policy }} | Label: grace {{ ld.grace_days }}d, min fraction {{ ld.min_fraction }}</p>
</header>

<div class="container">

<!-- KPI row -->
<div class="grid grid-3" style="margin-bottom: 1.25rem;">
  <div class="card kpi">
    <div class="value">{{ "%.3f"|format(test.auc) }}</div>
    <div class="label">Test AUC</div>
    <div class="sub">95% CI [{{ "%.3f"|format(test.auc_ci95[0]) }}, {{ "%.3f"|format(test.auc_ci95[1]) }}]</div>
  </div>
  <div class="card kpi">
    <div class="value">{{ "%.3f"|format(dev.auc) }}</div>
    <div class="label">Dev OOF AUC</div>
    <div class="sub">N = {{ dev.n }}</div>
  </div>
  <div class="card kpi">
    <div class="value">{{ "%.3f"|format(test.ece) }}</div>
    <div class="label">Expected Calibration Error</div>
    <div class="sub">Brier {{ "%.3f"|format(test.brier) }}</div>
  </div>
  <div class="card kpi">
    <div class="value">{{ "%.1f%%"|format(data.keep_rate * 100) }}</div>
    <div class="label">Overall Keep Rate</div>
    <div class="sub">{{ data.ptps }} PTPs</div>
  </div>
  <div class="card kpi">
    <div class="value">{{ "%.1f%%"|format(stream_conf_auc * 100 / 100) }}</div>
    <div class="label">Streaming AUC at Confirmation</div>
    <div class="sub">Median {{ "%.0f%%"|format(stop.median_fraction_heard * 100) }} of call heard</div>
  </div>
  <div class="card kpi">
    <div class="value">{{ "%.1f%%"|format(policy.hardship_flagged_share * 100) }}</div>
    <div class="label">Hardship Flag Rate</div>
    <div class="sub">Harsh given hardship: {{ "%.1f%%"|format(policy.harsh_treatment_given_hardship * 100) }}</div>
  </div>
</div>

<!-- ROC and Calibration -->
<div class="grid grid-2">
  <div class="card fig">
    <h2>ROC Curve (Test)</h2>
    {% if roc_img %}<img src="data:image/png;base64,{{ roc_img }}" alt="ROC curve">{% endif %}
  </div>
  <div class="card fig">
    <h2>Calibration by Portfolio</h2>
    {% if cal_img %}<img src="data:image/png;base64,{{ cal_img }}" alt="Calibration plot">{% endif %}
  </div>
</div>

<!-- Baselines -->
<div class="grid grid-2" style="margin-top: 1.25rem;">
  <div class="card">
    <h2>Baselines and Ablations (Test AUC)</h2>
    <table>
      <tr><th>Model Variant</th><th>AUC</th></tr>
      {% for k, v in baselines|dictsort(by='value', reverse=true) %}
      <tr><td>{{ k|replace("_", " ")|title }}</td><td>{{ "%.4f"|format(v) }}</td></tr>
      {% endfor %}
    </table>
  </div>
  <div class="card">
    <h2>Broken PTP Flagging (Test)</h2>
    <table>
      <tr><th>Threshold</th><th>Broken Flagged</th><th>False Alarm</th><th>Precision</th></tr>
      {% for k, v in flagging.items() %}
      <tr>
        <td>{{ k|replace("_", " ")|title }} ({{ "%.2f"|format(v.threshold) }})</td>
        <td>{{ "%.1f%%"|format(v.broken_flagged_share * 100) }}</td>
        <td>{{ "%.1f%%"|format(v.kept_false_alarm_share * 100) }}</td>
        <td>{{ "%.1f%%"|format(v.precision_broken * 100) }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
</div>

<!-- Type model and streaming -->
<div class="grid grid-2" style="margin-top: 1.25rem;">
  <div class="card">
    <h2>Type Model (5x5 CV)</h2>
    <table>
      <tr><th>Variant</th><th>Macro-F1</th><th>Accuracy</th><th>Hardship-to-Harsh</th></tr>
      {% for r in type_cv %}
      <tr>
        <td>{{ r.variant|replace("_", " ")|title }}</td>
        <td>{{ "%.3f"|format(r.macro_f1) }}</td>
        <td>{{ "%.3f"|format(r.accuracy) }}</td>
        <td>{{ "%.1f%%"|format(r.hardship_to_harsh_rate * 100) }}
            {% if r.hardship_to_harsh_rate < 0.06 %}<span class="tag tag-ok">OK</span>{% elif r.hardship_to_harsh_rate < 0.10 %}<span class="tag tag-warn">WARN</span>{% else %}<span class="tag tag-err">HIGH</span>{% endif %}
        </td>
      </tr>
      {% endfor %}
    </table>
  </div>
  <div class="card fig">
    <h2>Streaming AUC by Call Progress</h2>
    {% if stream_img %}<img src="data:image/png;base64,{{ stream_img }}" alt="Streaming AUC">{% endif %}
  </div>
</div>

<!-- Families and Type scores -->
<div class="grid grid-2" style="margin-top: 1.25rem;">
  <div class="card fig">
    <h2>Signal Family Attribution</h2>
    {% if fam_img %}<img src="data:image/png;base64,{{ fam_img }}" alt="Signal families">{% endif %}
  </div>
  <div class="card fig">
    <h2>Score Distribution by PTP Type</h2>
    {% if type_img %}<img src="data:image/png;base64,{{ type_img }}" alt="Score by type">{% endif %}
  </div>
</div>

<!-- Policy simulation -->
<div class="grid grid-2" style="margin-top: 1.25rem;">
  <div class="card">
    <h2>Policy Simulation (Test)</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Broken PTPs flagged before parking</td><td>{{ "%.1f%%"|format(policy.broken_ptps_flagged_before_parking * 100) }}</td></tr>
      <tr><td>Broken with shorter parking</td><td>{{ "%.1f%%"|format(policy.broken_ptps_with_shorter_parking * 100) }}</td></tr>
      <tr><td>Contact days released on broken</td><td>{{ policy.contact_days_released_on_broken }}</td></tr>
      <tr><td>Per broken PTP</td><td>{{ "%.1f"|format(policy.contact_days_released_per_broken_ptp) }} days</td></tr>
      <tr><td>Keep rate: park until due</td><td>{{ "%.1f%%"|format(policy.keep_rate_parked_until_due * 100) }}</td></tr>
      {% if policy.keep_rate_part_payment_follow_up is not none %}
      <tr><td>Keep rate: part-payment follow-up</td><td>{{ "%.1f%%"|format(policy.keep_rate_part_payment_follow_up * 100) }}</td></tr>
      {% endif %}
    </table>
    <h2 style="margin-top: 1rem;">Treatment Mix</h2>
    <table>
      <tr><th>Treatment</th><th>Count</th></tr>
      {% for t, n in policy.treatment_mix|dictsort(by='value', reverse=true) %}
      <tr><td>{{ t|replace("_", " ")|title }}</td><td>{{ n }}</td></tr>
      {% endfor %}
    </table>
  </div>
  <div class="card">
    <h2>Agent Integrity</h2>
    <p style="font-size: 0.85rem; margin-bottom: 0.5rem;">{{ integrity.agents }} agents assessed, {{ integrity.flagged|length }} flagged for review.</p>
    {% if integrity.flagged %}
    <table>
      <tr><th>Flagged Agent IDs</th></tr>
      {% for a in integrity.flagged %}
      <tr><td>{{ a }}</td></tr>
      {% endfor %}
    </table>
    {% endif %}
    {% if agent_img %}
    <div class="fig" style="margin-top: 1rem;">
      <img src="data:image/png;base64,{{ agent_img }}" alt="Agent quality">
    </div>
    {% endif %}
  </div>
</div>

<!-- Top features -->
<div class="card" style="margin-top: 1.25rem;">
  <h2>Top 25 Features by Mean |TreeSHAP| Attribution</h2>
  <table>
    <tr><th>Rank</th><th>Feature</th><th>Mean |SHAP|</th></tr>
    {% for f, v in top_features.items() %}
    <tr><td>{{ loop.index }}</td><td><code>{{ f }}</code></td><td>{{ "%.4f"|format(v) }}</td></tr>
    {% endfor %}
  </table>
</div>

<!-- Trial design -->
<div class="card" style="margin-top: 1.25rem;">
  <h2>Randomised Trial Design</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Baseline keep rate</td><td>{{ "%.1f%%"|format(trial.baseline_keep_rate * 100) }}</td></tr>
    <tr><td>Per-arm N for +3pp lift</td><td>{{ trial["per_arm_n_for_+3pp"] }}</td></tr>
    <tr><td>Per-arm N for +5pp lift</td><td>{{ trial["per_arm_n_for_+5pp"] }}</td></tr>
    <tr><td>Per-arm N clustered (ICC 0.05, m=2, +3pp)</td><td>{{ trial["per_arm_n_for_+3pp_clustered_icc0.05_m2"] }}</td></tr>
  </table>
</div>

</div>

<footer>
  PTP Credibility System v{{ v.package }} | Generated from reports/metrics.json
</footer>
</body>
</html>""")


def embed_image(path: Path) -> str:
    """Base64-encode a PNG file for inline embedding."""
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode("ascii")
    return ""


def main() -> None:
    m = json.loads((REPORTS / "metrics.json").read_text())
    ctx = {
        "v": m["versions"],
        "ld": m["label_definition"],
        "data": m["data"],
        "test": m["keep_model_test"],
        "dev": m["keep_model_dev_oof"],
        "baselines": m["baselines_test_auc"],
        "flagging": m["broken_flagging_test"],
        "type_cv": m["type_model_cv_5x5"],
        "policy": m["policy_simulation_test"],
        "integrity": m["agent_integrity"],
        "top_features": m["top_features_mean_abs_shap"],
        "trial": m["trial_design"],
        "stream_conf_auc": m["streaming_test"]["auc_at_first_confirmation_turn"],
        "stop": m["streaming_test"]["stopping_rule"],
        "roc_img": embed_image(FIGURES / "roc_test.png"),
        "cal_img": embed_image(FIGURES / "calibration_by_portfolio.png"),
        "stream_img": embed_image(FIGURES / "streaming_auc_by_progress.png"),
        "fam_img": embed_image(FIGURES / "signal_family_attribution.png"),
        "type_img": embed_image(FIGURES / "score_by_type.png"),
        "agent_img": embed_image(FIGURES / "agent_quality_weighted_ptps.png"),
    }
    html = TEMPLATE.render(**ctx)
    out = REPORTS / "dashboard.html"
    out.write_text(html)
    print(f"Dashboard written to {out}")


if __name__ == "__main__":
    main()
