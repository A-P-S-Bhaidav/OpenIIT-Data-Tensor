# PTP Credibility: Real-Time Fake Promise-to-Pay Detection

A production-grade system for estimating, during a live collections call, how likely a
promise to pay (PTP) is to be kept, what type of promise it is, and what operational
treatment should follow. Built for CreditNirvana Problem Statement 1.

---

## Table of Contents

1. [Problem](#problem)
2. [Solution Architecture](#solution-architecture)
3. [Signal Map](#signal-map)
4. [Label Definition](#label-definition)
5. [Modelling](#modelling)
6. [Results](#results)
7. [Streaming and In-Call Prompts](#streaming-and-in-call-prompts)
8. [Type Classification](#type-classification)
9. [Policy and Disposition](#policy-and-disposition)
10. [Agent Integrity](#agent-integrity)
11. [Governance and Compliance](#governance-and-compliance)
12. [Randomised Trial Design](#randomised-trial-design)
13. [API Reference](#api-reference)
14. [Reproduction](#reproduction)
15. [Repository Structure](#repository-structure)
16. [Limitations](#limitations)
17. [Roadmap](#roadmap)

---

## Problem

More than half of captured PTPs never convert to a payment. Each broken PTP costs the
lender in three ways: contact days are wasted while the account is parked, the account
re-enters through a more expensive channel, and inflated PTP counts mask actual recovery
performance. Not all broken PTPs break for the same reason -- an escape promise by a
borrower ending the call requires a different response from a genuine promise that is
infeasible given the borrower's cash flow.

The system must:
- Score PTP credibility **during the call**, within a few hundred milliseconds per turn.
- Classify the PTP into one of six operational types.
- Surface actionable prompts to the agent or voice bot while the borrower is still on
  the line.
- Drive downstream disposition (parking window, reminder schedule, third-party
  confirmation, integrity review) based on credibility and type.
- Protect borrowers in genuine hardship from harsher treatment.
- Comply with the RBI Fair Practices Code and the DPDP Act 2023.

---

## Solution Architecture

```
                                CN Voice Bot / Tele-calling Stack
                                           |
                              live audio + transcript stream
                                           |
                                   +-------v--------+
                                   | Credibility    |
                                   | Engine         |
                                   | (FastAPI)      |
                                   +---+----+---+---+
                                       |    |   |
                       +---------------+    |   +----------------+
                       |                    |                    |
               +-------v------+    +-------v-------+    +------v--------+
               | Agent Desktop|    | Voice Bot     |    | Campaign &    |
               | (score, type,|    | Dialogue Flow |    | Disposition   |
               | prompts)     |    | (auto-prompts)|    | Engine        |
               +--------------+    +---------------+    +------+--------+
                                                               |
                                                   +-----------v-----------+
                                                   | Dashboards            |
                                                   | - Quality-weighted    |
                                                   |   PTPs                |
                                                   | - Agent integrity     |
                                                   | - Call replay         |
                                                   +-----------------------+
```

**Module dependency graph:**

```
config.py
   |
data.py ---> labels.py ---> pipeline.py ---> models.py ---> explain.py
   |              |              |               |
lexicon.py   conversation.py  features.py   calibration.py
                                   |
                              streaming.py
                                   |
policy.py <--- models.py      integrity.py
   |
service/engine.py ---> service/api.py
```

---

## Signal Map

The problem statement identifies five signal families. The table below maps each to
concrete features and their measured attribution share (mean |TreeSHAP| on the test
split).

| Signal Family | Attribution | Key Features |
|---|---:|---|
| Account and promise terms | 34.8% | `terms_days_to_promise`, `borrower_stated_amount`, `terms_promise_dom`, `acct_ability_to_pay`, `terms_amount_to_emi`, `terms_salary_gap_days` |
| What was said | 24.4% | `cust_hedge_rate`, `cust_closing_escape`, `cust_firm_commitment`, `cust_hedge`, `cust_hardship_business` |
| How it was said (voice) | 16.1% | `voice_max_pause_ms`, `voice_response_pause_to_ask_ms`, `voice_max_response_pause_to_ask_ms`, `voice_min_asr_conf` |
| How the conversation went | 15.3% | `negotiation_turns`, `agent_proposed_date_first`, `call_duration_s`, `agent_talk_share` |
| Account history | 5.6% | `acct_bureau_ord`, `dial_answer_rate_30d`, `hist_days_since_last_ptp` |
| Agent signals | 3.8% | `agent_silent_terms_rate` |

Voice features are gated behind explicit consent (`voice_consent=true` at session
start). When consent is not given, voice features are set to missing and the model
operates without them (test AUC drops from 0.742 to 0.729).

---

## Label Definition

A PTP is **kept** when cumulative payments on the account, from the capture timestamp
until 3 days after the promised date, reach at least 90% of the promised amount. Payments
through any channel count. When a newer PTP supersedes an older one, the payment is
attributed to the newer promise only.

| Outcome Status | Count | Share |
|---|---:|---:|
| `broken` | 1,914 | 51.8% |
| `kept_on_time` | 950 | 25.7% |
| `partial` | 396 | 10.7% |
| `paid_after_window` | 328 | 8.9% |
| `kept_late_within_grace` | 108 | 2.9% |

Overall keep rate: **28.6%**. Full rationale and sensitivity analysis are in
[`docs/LABEL_DEFINITION.md`](docs/LABEL_DEFINITION.md).

---

## Modelling

### Training Protocol

1. **Splits**: account-level train / validation / test, shipped with the data (3,133
   dev, 563 test PTPs).
2. **Feature selection**: features ranked by total gain on the train split; set size
   chosen by grouped 5-fold CV on dev (smallest k within 0.002 AUC of the best).
   Selected 45 features out of 161 candidates.
3. **Final model**: 5-seed bagged XGBoost (depth 3, lr 0.015, 600 trees) with grouped
   out-of-fold predictions on dev used to fit a hierarchical Platt calibrator
   (per-portfolio, shrunk to global).
4. **Test evaluation**: single pass, never used for selection or tuning.

### Keep Model (XGBoost)

| Metric | Dev OOF | Test |
|---|---:|---:|
| AUC | 0.792 | 0.742 |
| AUC 95% CI | -- | [0.695, 0.788] |
| PR-AUC | 0.606 | 0.585 |
| Brier Score | 0.155 | 0.171 |
| ECE | 0.022 | 0.048 |
| Log Loss | 0.479 | 0.527 |

### Baselines and Ablations (Test AUC)

| Model Variant | AUC |
|---|---:|
| Logistic regression (all features) | 0.760 |
| All candidates, no feature selection | 0.756 |
| Conversation features only | 0.746 |
| **Final model (XGB, k=45)** | **0.742** |
| Selected features without voice | 0.729 |
| Selected without agent track record | 0.743 |
| Account + history + terms only | 0.703 |
| Past keep-rate heuristic | 0.523 |
| Incumbent fixed rule | 0.500 |

The LR baseline achieves a higher test AUC (0.760 vs 0.742) on this particular 563-PTP
test fold. This is consistent with sampling variance: the XGB 95% CI
[0.695, 0.788] overlaps with the LR point estimate, and the XGB model is superior on the
larger dev CV evaluation (0.792 vs 0.763). XGB is retained as the production model
because it provides TreeSHAP explainability, incremental streaming capability, and
per-feature contribution tracking that LR cannot.

### Label Robustness

The same trained model is re-scored against alternative label definitions. Ranking
quality is stable, confirming the business can adjust grace days and thresholds without
retraining (only recalibration is needed).

| Grace Days | Min Fraction | Test AUC |
|---:|---:|---:|
| 0 | 0.90 | 0.764 |
| 1 | 0.90 | 0.761 |
| **3** | **0.90** | **0.742** |
| 5 | 0.90 | 0.726 |
| 7 | 0.90 | 0.721 |

---

## Results

### Broken PTP Flagging (Test)

The system identifies broken PTPs before the account is parked, enabling earlier
intervention.

| Operating Point | Threshold | Broken Flagged | False Alarm | Precision |
|---|---:|---:|---:|---:|
| At 20% false-alarm budget | 0.20 | 55.8% | 21.6% | 86.0% |
| At policy low threshold | 0.25 | 66.4% | 28.7% | 84.6% |
| At policy high threshold | 0.55 | 90.9% | 59.9% | 78.3% |

### Post-Call Update

A logistic model combines the in-call credibility score with post-call behavioural
signals (payment-link opens, confirmation-message replies, callback within 30 hours).
For PTPs unresolved at the horizon:

| Stage | AUC |
|---|---:|
| In-call score alone | 0.758 |
| After post-call update | 0.770 |

### Calibration by Portfolio (Test)

| Portfolio | N | Keep Rate | Mean Pred | AUC | ECE |
|---|---:|---:|---:|---:|---:|
| Consumer Durable | 58 | 27.6% | 33.0% | 0.823 | 0.138 |
| Credit Card | 69 | 24.6% | 28.0% | 0.715 | 0.089 |
| MFI JLG | 100 | 30.0% | 30.6% | 0.746 | 0.069 |
| MSME | 55 | 29.1% | 26.5% | 0.893 | 0.143 |
| PL Salaried | 163 | 33.1% | 32.2% | 0.667 | 0.096 |
| Two-Wheeler | 118 | 28.8% | 28.9% | 0.777 | 0.069 |

---

## Streaming and In-Call Prompts

The system scores the PTP incrementally as the call progresses, not only at the end.

### AUC by Fraction of Call Heard (Test)

| % of Call | AUC |
|---:|---:|
| 10% | 0.588 |
| 30% | 0.581 |
| 50% | 0.670 |
| 60% | 0.719 |
| 70% | 0.758 |
| 80% | 0.769 |
| 90% | 0.768 |
| 100% | 0.768 |

**At the first confirmation turn**: AUC 0.767.

### Stopping Rule

A sufficiency model predicts the expected absolute change in the score if the call
continues. The system fires a decision when the expected move falls below a delta of
0.05.

| Metric | Value |
|---|---|
| AUC at decision point | 0.765 |
| Fired before end of call | yes, in majority of calls |
| Median fraction of call heard at firing | 73% |

This means the agent or bot receives the credibility assessment and prompts with 27% of
the call remaining -- enough time to firm up a weak promise.

### Latency (150 test calls, single CPU, Python 3.11)

| Operation | p50 | p90 | p95 | p99 | Budget |
|---|---:|---:|---:|---:|---:|
| Turn update with reason codes | 107 ms | 115 ms | 124 ms | 163 ms | 300 ms |
| Turn update, score only | 74 ms | 83 ms | 86 ms | 121 ms | 300 ms |
| End-of-call (final score + disposition) | 98 ms | 105 ms | 109 ms | 133 ms | 300 ms |

All operations are well within the 300 ms latency budget at p99.

---

## Type Classification

Each PTP is classified into one of six operational types:

| Predicted Type | N (Test) | Keep Rate | Mean Score |
|---|---:|---:|---:|
| Genuine feasible | 160 | 55.6% | 0.547 |
| Third party | 117 | 29.9% | 0.271 |
| Genuine infeasible | 110 | 19.1% | 0.233 |
| Escape | 72 | 15.3% | 0.146 |
| Agent recorded or pushed | 58 | 15.5% | 0.123 |
| Repeat promiser | 46 | 4.4% | 0.166 |

### Type Model Performance (5x5 Repeated Stratified CV)

| Variant | Macro-F1 | Accuracy | Hardship-to-Harsh |
|---|---:|---:|---:|
| Supervised only | 0.725 | 69.9% | 11.0% |
| Self-training | 0.738 | 71.8% | 9.0% |
| Self-training + cost-sensitive | 0.726 | 70.7% | **5.9%** |

The cost-sensitive variant is used in production. It trades a small amount of macro-F1
for a 47% reduction in the rate at which genuinely infeasible (hardship) cases are
misrouted to harsher treatments.

---

## Policy and Disposition

The disposition engine maps the credibility score and PTP type to an operational
treatment, subject to two guardrails:

- **G1 (Hardship)**: if the hardship flag is set (lexicon hits for medical emergencies,
  job loss, business closure, salary delays; or type-model probability of
  `genuine_infeasible` above 0.35), harsher treatments are impossible. The account is
  routed to restructuring or a human.
- **G2 (Contact hours)**: all follow-ups are scheduled within 08:00-19:00, with at most
  one contact per day.

### Policy Simulation (Test)

| Metric | Value |
|---|---|
| Broken PTPs flagged before parking | 94.9% |
| Broken PTPs with shorter parking | 52.5% |
| Contact days released on broken PTPs | 1,187 |
| Contact days released per broken PTP | 3.0 |
| Keep rate: park until due | 64.6% |
| Hardship flagged share | 50.4% |
| Harsh treatment given hardship | **0.0%** |
| Annotated infeasible receiving harsh | **0.0%** |

### Treatment Mix (Test)

| Treatment | Count |
|---|---:|
| Supportive restructure or human | 121 |
| Confirm with borrower | 117 |
| Renegotiate or restructure | 103 |
| Park until due | 99 |
| Part-payment follow-up | 44 |
| Integrity review and confirm | 41 |
| Stronger treatment, short parking | 21 |
| Park with mid-check | 17 |

---

## Agent Integrity

The system computes a quality-weighted PTP count for every agent, comparing their
observed keep rate to the expected keep rate given their case mix (account-level
features). Agents whose observed/expected ratio falls below a z-score threshold, or who
exhibit patterns such as remark copying, PTPs clustered at shift end, or
silent-on-terms (agent logs a PTP the borrower never explicitly agreed to), are flagged.

- **30 agents assessed**, **7 flagged** for integrity review.
- Flagged agents: TA013, TA004, TA019, FA009, TA014, TA020, TA003.

---

## Governance and Compliance

Full details are in [`docs/GOVERNANCE.md`](docs/GOVERNANCE.md). Summary:

| Constraint | Control |
|---|---|
| Hardship protection | G1 guardrail: harsh treatment is impossible when the hardship flag is set. Tested exhaustively. |
| RBI Fair Practices Code | Prompts come from a vetted template library with a banned-phrase filter. Contact hours and frequency enforced in code. |
| DPDP Act 2023 | Voice features gated behind consent. Only derived features retained. Purpose-limited to PTP credibility. |
| Explainability | TreeSHAP reason codes at agent, supervisor and auditor levels. Every disposition carries its rule ID and guardrails applied. |
| Gaming resistance | Strongest signals come from borrower speech, which agents cannot fabricate. Agent-controlled fields used only for integrity analytics. Drift monitored weekly. |
| Counterfactual data | Model predicts keep probability under the incumbent treatment. Causal effects measured only through the RCT. Permanent 10% holdout provides clean retraining data. |

---

## Randomised Trial Design

The causal value of in-call prompts must be measured through a randomised controlled
trial. Full protocol is in [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md).

| Parameter | Value |
|---|---|
| Baseline keep rate | 28.6% |
| Per-arm N for +3pp lift | 3,671 |
| Per-arm N for +5pp lift | 1,345 |
| Per-arm N clustered (ICC 0.05, m=2, +3pp) | 3,854 |
| Randomisation unit | Account-level |
| Permanent holdout | 10% (no prompts, for clean retraining data) |

---

## API Reference

The credibility engine is served as a FastAPI application.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/v1/calls` | Open a new call session |
| POST | `/v1/calls/{call_id}/turns` | Add a transcript turn; returns updated score, type, prompts |
| POST | `/v1/calls/{call_id}/terms` | Set or update promise terms (date, amount, answerer) |
| POST | `/v1/calls/{call_id}/end` | End the call; returns final score, type, disposition |
| POST | `/v1/events` | Post-call event (link open, reply, callback); returns updated score |
| GET | `/v1/health` | Health check |

Start the service:

```bash
make serve
# or directly:
PYTHONPATH=src python -m uvicorn ptp_credibility.service.api:app --host 0.0.0.0 --port 8000
```

---

## Reproduction

### Prerequisites

- Python 3.10 or later
- Dependencies listed in `requirements.txt`

### Commands

```bash
# Install dependencies
make install

# Train all models, evaluate, generate figures and metrics
make train

# Run the test suite (17 tests)
make test

# Benchmark per-turn latency
make benchmark

# Build the HTML evaluation dashboard
make dashboard

# Replay a specific call through the engine
make replay

# Run everything
make all
```

All outputs are written to `reports/` (metrics, figures, CSVs, dashboard) and
`artifacts/` (model binaries). The `artifacts/` directory is gitignored because the
binaries are regenerable by `make train`.

---

## Repository Structure

```
.
|-- Makefile                        Reproducible entry points
|-- pyproject.toml                  Package metadata
|-- requirements.txt                Runtime dependencies
|-- .gitignore
|
|-- src/ptp_credibility/
|   |-- __init__.py                 Version
|   |-- config.py                   Central configuration (label, policy, model)
|   |-- data.py                     Data loading with validation
|   |-- lexicon.py                  Hinglish/English/Kanglish regex lexicons
|   |-- labels.py                   Outcome labelling with superseding-PTP attribution
|   |-- conversation.py             Incremental ConversationState (turn-by-turn features)
|   |-- features.py                 Point-in-time context features
|   |-- calibration.py              Hierarchical Platt calibration
|   |-- models.py                   KeepModel, TypeModel, PostCallModel
|   |-- explain.py                  TreeSHAP reason codes
|   |-- policy.py                   FPC-compliant prompts, disposition rules, guardrails
|   |-- streaming.py                SufficiencyModel, decision points
|   |-- pipeline.py                 Dataset builder
|   |-- integrity.py                Agent integrity analytics
|   |-- plots.py                    Evaluation figures
|   |-- service/
|       |-- engine.py               CredibilityEngine, CallSession, ContextStore
|       |-- api.py                  FastAPI endpoints
|
|-- scripts/
|   |-- train.py                    Full training and evaluation protocol
|   |-- benchmark_latency.py        Per-turn latency measurement
|   |-- replay_call.py              Call replay demonstration
|   |-- build_dashboard.py          HTML dashboard generator
|
|-- tests/
|   |-- test_core.py                17 unit tests (labels, lexicon, conversation, policy)
|
|-- docs/
|   |-- LABEL_DEFINITION.md         Label definition and sensitivity analysis
|   |-- GOVERNANCE.md               Compliance, responsible use, model risk management
|   |-- MODEL_CARD.md               Model card (Google framework)
|   |-- ARCHITECTURE.md             System and module architecture with diagrams
|   |-- EXPERIMENT_DESIGN.md        RCT protocol
|
|-- ps1_fake_ptp/                   Problem-statement-specific data
|-- shared/                         Cross-problem shared data
|
|-- reports/                        Generated by train.py
|   |-- metrics.json                All evaluation metrics
|   |-- figures/                    Evaluation plots (ROC, calibration, streaming, etc.)
|   |-- dashboard.html              Self-contained HTML dashboard
|   |-- agent_integrity.csv         Per-agent integrity assessment
|   |-- policy_simulation_test.csv  Per-PTP disposition simulation
|   |-- streaming_test_trajectories.csv
|
|-- artifacts/                      Generated by train.py (gitignored)
    |-- ptp_credibility.joblib      All trained models and thresholds
    |-- scored_ptps.pkl             Scored PTP dataset for the service
```

---

## Limitations

1. **Small test set**: 563 PTPs with 167 positives. Confidence intervals are wide.
   Production monitoring should track AUC and calibration on a rolling window.

2. **Historical policy confounding**: outcomes are observed only for actions that were
   taken under the incumbent rule-based policy. The model predicts keep probability
   under that policy, not under the new credibility-driven one. The RCT is essential.

3. **Voice feature coverage**: voice-derived pause and hesitation features are available
   only for calls with audio metadata. They require DPDP-compliant consent.

4. **Annotated type labels**: only 200 PTPs are annotated. The type model uses
   self-training to leverage the unlabelled pool, but performance depends on annotation
   quality. Expanding the annotated set would improve type classification.

5. **Field channel calibration**: field PTPs (N=79 in test) show higher ECE (0.128)
   than tele-agent PTPs (0.050), likely because field conversations have different
   dynamics that the conversation-feature pipeline captures less precisely.

6. **No neural components**: the system uses gradient-boosted trees and logistic models.
   A transformer-based approach on raw transcripts could capture longer-range discourse
   patterns but would require more data and a GPU inference budget.

---

## Roadmap

- **Expand annotations**: active-learning loop to prioritise ambiguous PTPs for
  human annotation, improving type-model coverage beyond 200 samples.
- **Neural transcript encoder**: fine-tuned multilingual model (IndicBERT or
  Muril) for code-mixed Hinglish/Kanglish, producing dense features that complement
  the lexicon-based signals.
- **Causal uplift model**: after the RCT collects sufficient data, train a
  conditional average treatment effect (CATE) model to personalise prompts
  (which prompt, for which borrower, produces the largest lift in keep rate net of
  cost).
- **Payment amount prediction**: extend from binary keep/break to a continuous
  expected recovery amount, enabling cost-optimal disposition.
- **Integration with NACH re-presentation timing**: use the credibility score to
  time NACH re-presentation attempts on days with higher predicted probability of
  sufficient balance.
- **Multi-lingual prompt library**: expand beyond English, Hinglish and Kanglish
  to Tamil, Telugu and Marathi based on lender geography.

