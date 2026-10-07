# Governance, Compliance and Responsible Use

This document explains how the PTP credibility service meets the three constraints in the problem statement (regulatory, counterfactual data and cost asymmetry) and the PS1-specific constraints. Each control is enforced in code and covered by tests, not only described here.

## 1. Hardship protection (PS1 constraint 1)

> A low score must never lead to harsher treatment where there are signs of hardship; those cases go to restructuring or to a human.

| Control | Where | Test |
|---|---|---|
| Hardship flag = any hardship lexicon hit (medical, job loss, business down, no funds, salary delayed or held), a recent `rpc_hardship` disposition, or type-model probability of *genuine but not feasible* of at least 0.35 | `policy.hardship_flag` | `TestPolicy.test_hardship_never_harsher` |
| Guardrail G1: with the hardship flag set, treatments `stronger_treatment_short_parking` and `part_payment_follow_up` are impossible. The account is routed to `restructuring_desk_or_human` with the full parking window. | `policy.disposition` | Same test, swept across every type and score |
| In-call prompts under hardship are limited to supportive prompts (date after income, smaller instalment, restructuring). No pay-today or part-payment ask is shown. | `policy.in_call_prompts` | `test_hardship_prompts_do_not_ask_for_payment_today` |
| Type-model training cost matrix: classifying *genuine but not feasible* as escape, repeat or agent-pushed costs 5 times any other error | `models.type_cost_matrix` | Measured as `hardship_to_harsh_rate` in `reports/metrics.json` |

The guardrail is applied after the decision table, so no future change to scores or thresholds can bypass it.

## 2. RBI Fair Practices Code

| Requirement | Control |
|---|---|
| Tone and conduct: no threats, intimidation or humiliation | Prompts come only from a vetted template library (`policy.PROMPTS`). A banned-phrase filter (credit-score intimidation, escalation, legal or police threats, "last reminder", and Hinglish and Kanglish equivalents) is asserted at import time and checked again on every prompt at runtime. |
| Firming up a promise must not turn into pressure | Every firming prompt is an offer the borrower can decline ("If it helps...", "Would that be possible?"). The agent hint says "Accept a no politely". |
| Contact hours | `materialise_schedule` moves every follow-up into 08:00 to 19:00 local time. |
| Contact frequency | At most one scheduled contact per day per account (`MAX_CONTACTS_PER_DAY`). The disposition never stacks reminders on the same day. |
| Agent conduct monitoring | Pressure language, pushing after a deferral and agree-right-after-pressure are measured per agent (`integrity.py`) and surfaced to the integrity team. They are never shown to the borrower and never used to penalise the borrower. |

## 3. DPDP Act 2023

| Principle | Implementation |
|---|---|
| Purpose limitation | Voice-derived signals (pauses, response latency, ASR confidence, fillers) are used only to estimate promise credibility for the current collection. They are not used for credit decisions, marketing or any other purpose. |
| Consent and sensitive processing | Voice features are isolated under the `voice_` prefix. The service only uses them when the session opens with `voice_consent=true`; otherwise they are set to missing and the response reports `voice_features_used: false`. The model is trained to handle missing voice features, and the measured cost of running without them is published (`baselines_test_auc.selected_without_voice`). |
| Data minimisation | Only derived features are retained for scoring. Raw audio is not needed by the model: prosodic proxies come from turn timing and ASR metadata the telephony stack already produces. |
| Retention | Recommended: derived voice features are retained only until the PTP resolves (promised date + grace), then deleted. Reason codes and scores are retained for the audit period mandated by the lender's policy. |
| Rights of the data principal | Every score is reproducible from the logged feature vector, model version and lexicon version, which supports access and correction requests. |

## 4. Explainability and auditability

* **Agent view**: the top three reasons in each direction, in plain language ("Borrower never stated an amount; only the agent did").
* **Supervisor view**: the same reasons with signal family and the PTP type probabilities.
* **Auditor view**: reason code, feature name, raw value, TreeSHAP log-odds contribution, model version, lexicon version and policy version. Every disposition carries the rule id (`D1` to `D7`) and the guardrails applied (`G1`, `G2`).
* **Global explanation**: signal-family attribution is published with every model release (`reports/figures/signal_family_attribution.png`).

## 5. Counterfactual data and bias

* Historical outcomes reflect the incumbent policy. The model predicts the keep probability **under the incumbent treatment**. It is used to rank and route promises, not to claim causal effects. The causal value of in-call prompts is measured only through the randomised trial in `docs/EXPERIMENT_DESIGN.md`.
* Prompts change the outcome the model predicts (performative prediction). The trial keeps a permanent 10% holdout with no prompts, which provides clean data for retraining.
* Score quality is reported separately for PTPs from the randomised dialling arm (`random_arm_check`) as a selection-bias check.
* Fairness: AUC and calibration error are reported per portfolio, language, income type, lender type and channel.

## 6. Gaming resistance and independence from agents

* The scorer is a separate service owned by the risk team. Agents see the score and reasons for the current call only. They never see model internals, thresholds or feature weights.
* The strongest signals come from **borrower** speech (stated amount and date, hedging, conditions, hesitation), which agents cannot fabricate without leaving a trace. Agent-controlled fields (free-text remarks, the answerer field) are used only for integrity analytics or as end-of-call inputs.
* A bot trained on outcomes must not be trained on the credibility score itself. The score is excluded from the bot's reward signal.
* Drift monitoring: weekly population stability index on the score and on each signal family, per agent and per channel. A sudden rise in an agent's borrower-stated-terms share combined with a falling keep rate is itself an integrity flag.

## 7. Model risk management

| Item | Practice |
|---|---|
| Validation | Account-level splits with a single-pass test evaluation, cluster-bootstrap confidence intervals and label-definition sensitivity analysis |
| Monitoring | AUC, calibration error per portfolio, hardship-to-harsh rate, prompt acceptance and complaint rate, all reported weekly |
| Recalibration | Monthly hierarchical Platt refit on resolved PTPs. Full retrain quarterly or on drift alarm. |
| Change control | Lexicon, policy and model are versioned independently, and every score logs all three versions. |
| Human oversight | Hardship and agent-recorded cases always go to humans. The model never closes, writes off or escalates an account on its own. |
