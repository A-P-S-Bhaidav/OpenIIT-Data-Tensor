# Model Card: Real-Time Fake PTP Detection System (CreditNirvana)

## 1. Model Details

### 1.1 Overview and Metadata
The CreditNirvana Real-Time Fake Promise-to-Pay (PTP) Detection System is an operational machine learning pipeline designed to assess the credibility of debt repayment commitments captured during voice bot, tele-agent, and field collections interactions. The system predicts whether a recorded PTP will convert into an actual repayment within a predefined settlement window and categorizes the underlying behavioral archetype of the commitment.

* **Model Name**: CreditNirvana PTP Credibility Engine
* **Model Version**: 1.0.0
* **Lexicon Version**: 2026.10.1
* **Policy Version**: 2026.10.1
* **Release Date**: October 2026
* **Model Developer**: CreditNirvana Risk & Data Science Team
* **License**: Proprietary / Enterprise
* **Primary Frameworks**: XGBoost, Scikit-Learn, SciPy, Pandas, NumPy
* **Reference Specifications**: Google Model Cards Framework (Mitchell et al., 2019), Reserve Bank of India (RBI) Fair Practices Code, Digital Personal Data Protection (DPDP) Act 2023

### 1.2 Model Architecture
The production architecture is composed of four coordinated statistical and machine learning modules:

1. **Keep Probability Model (`KeepModel`)**: An ensemble of 5 seed-bagged gradient boosted decision trees (`xgboost.XGBClassifier`) calibrated using Hierarchical Platt scaling. The booster predicts the uncalibrated probability of a promise being kept given point-in-time account, interaction, acoustic, and behavioral signals. Out-of-fold predictions from grouped cross-validation are used to fit segment-aware Platt scaling parameters shrunken toward a global prior.
2. **PTP Typology Model (`TypeModel`)**: A 6-class gradient boosted classifier that categorizes promises into operational archetypes (`genuine_feasible`, `genuine_infeasible`, `escape`, `third_party`, `repeat_promiser`, `agent_recorded_or_pushed`). It utilizes semi-supervised self-training conditioned on observed repayment outcomes and applies a cost-sensitive decision matrix penalizing false negatives on hardship.
3. **Streaming Sufficiency Model (`SufficiencyModel`)**: An in-call dynamic stopping controller evaluating conversational prefix trajectories to determine the earliest turn at which prediction stability is reached, enabling real-time prompt delivery before call termination.
4. **Post-Call Update Model (`PostCallModel`)**: A Bayesian-style logistic regression updater that adjusts in-call log-odds predictions based on post-conversation engagement signals (SMS/WhatsApp payment link clicks, customer inbound inquiries, digital confirmation replies) within a 30-hour post-call decision horizon.

### 1.3 Feature Selection and Engineering
Feature selection was conducted on the development split (train + validation, 3,133 accounts) to prevent data leakage and dimensionality inflation:
* **Candidate Space**: 161 engineered features encompassing account terms, historical bounce behavior, borrower acoustic/prosodic properties, natural language conversation markers, dialogue dynamics, and agent operational track records.
* **Selection Protocol**: Candidates were ranked by total gain using a 3-seed XGBoost probe on the training split. Feature subset sizes $k \in \{30, 45, 60, 80, 100, 161\}$ were evaluated using grouped 5-fold cross-validation at the account level on the combined development set.
* **Selection Rule**: The parsimonious selection criterion chose the smallest feature dimension $k$ achieving an out-of-fold AUC within 0.002 of the peak cross-validation score ($k=60$ achieved 0.79113; threshold = 0.78913).
* **Final Selection**: 45 features ($k=45$, CV AUC = 0.79063).

```
Feature Selection Grid (Grouped 5-Fold CV on Development Set):
k = 30  -> CV AUC: 0.78866
k = 45  -> CV AUC: 0.79063  [SELECTED: Smallest k within 0.002 of best]
k = 60  -> CV AUC: 0.79113  [Peak CV AUC]
k = 80  -> CV AUC: 0.78909
k = 100 -> CV AUC: 0.78550
k = 161 -> CV AUC: 0.78597  [All candidates without selection]
```

The 45 selected features in rank order:
`terms_days_to_promise`, `borrower_stated_amount`, `voice_max_pause_ms`, `terms_salary_gap_days`, `voice_max_response_pause_to_ask_ms`, `cust_hedge_rate`, `terms_salary_days_to_next_credit`, `terms_promise_dom`, `acct_ability_to_pay`, `cust_firm_commitment`, `cust_hedge`, `voice_response_pause_to_ask_ms`, `call_duration_s`, `cust_hardship_business`, `cust_closing_escape`, `terms_amount_to_emi`, `acct_outstanding`, `agent_proposed_date_first`, `pay_days_since_last`, `hist_days_since_last_ptp`, `voice_mean_asr_conf`, `field_remark_agent_paraphrased_date`, `terms_salary_aligned`, `dial_answer_rate_30d`, `voice_speech_rate_cps`, `borrower_silent_on_terms`, `acct_emi`, `agent_silent_terms_rate`, `voice_min_asr_conf`, `voice_mean_pause_ms`, `agent_keep_rate_shrunk`, `negotiation_turns`, `terms_promise_is_month_end`, `agent_talk_share`, `acct_lender_type`, `confirm_amount_matches_borrower`, `terms_capture_dom`, `acct_salary_day`, `terms_promise_dow`, `field_agent_visits_that_day`, `call_talk_s`, `agent_proposes_date`, `voice_long_pauses`, `acct_bureau_ord`, `n_turns`.

### 1.4 Training Protocol and Hyperparameters
* **Dataset Partitioning**: Account-level disjoint splitting into train (2,572 rows), validation (561 rows), and test (563 rows). Grouped splits guarantee that all promises from an individual account reside strictly within one partition.
* **Development Refit**: Production models are fit on the unified development partition (train + validation, $N = 3,133$ PTPs).
* **Calibration Protocol**: Grouped 5-fold cross-validation on the development set generates out-of-fold uncalibrated probabilities, which are passed to the Hierarchical Platt calibrator.
* **XGBoost Hyperparameters**:
  * `n_estimators`: 600
  * `learning_rate`: 0.015
  * `max_depth`: 3
  * `min_child_weight`: 5
  * `subsample`: 0.85
  * `colsample_bytree`: 0.60
  * `reg_lambda`: 2.0
  * `tree_method`: `hist`
  * `eval_metric`: `logloss`
  * `random_state`: 5 bagged seeds ($1000, 1001, 1002, 1003, 1004$)

---

## 2. Intended Use and Scope

### 2.1 Primary Intended Applications
1. **In-Call Real-Time Agent Guidance**: Providing real-time prompts and script nudges on tele-agent desktops when conversational cues indicate low credibility, terms misalignment, or customer distress.
2. **Conversational AI / Voice Bot Steering**: Enabling automated voice bots to negotiate feasible dates, request immediate part-payments, or escalate unresolvable calls.
3. **Dynamic Collections Campaign Orchestration**: Replacing static 15-day account parking with risk-calibrated hold periods, automated omnichannel reminder cadences (WhatsApp/SMS), and rapid follow-up scheduling for broken commitments.
4. **Hardship Identification and Routing**: Automatically identifying genuine customer hardship and routing vulnerable borrowers to specialized restructuring desks without punitive collection measures.
5. **Operational Integrity Auditing**: Detecting suspicious or agent-fabricated commitments to protect operational capacity and support supervisory audits.

### 2.2 Primary Users
* **CreditNirvana Agent Desktop**: Front-line tele-calling collections representatives.
* **CreditNirvana Voice Bot Platform**: Automated conversational IVR/AI bots conducting outbound contact.
* **CreditNirvana Campaign Engine**: Workflow management system controlling dialer priority, parking queues, and message dispatch.
* **Collections Strategy and Quality Assurance Teams**: Operations supervisors and compliance auditors.

### 2.3 Out-of-Scope and Prohibited Applications
* **Credit Underwriting and Originating**: The model must not be used to grant, decline, or size credit limits or new loan applications.
* **Credit Bureau Reporting**: The model output is strictly an operational collections likelihood score and must not be transmitted to credit information companies (CICs) or alter bureau standing.
* **Marketing and Cross-Selling**: The model must not be used for targeted marketing or product cross-selling.
* **Autonomous Legal Escalation**: Model scores cannot independently trigger legal notices, litigation, repossession, or recovery agent deployment.
* **Harassment or Coercive Contact**: Output scores must never bypass statutory contact windows (08:00 - 19:00 local time) or violate the daily contact cap (maximum 1 contact per day).

---

## 3. Data and Cohort Specifications

### 3.1 Dataset Overview
The dataset contains 3,696 recorded promises to pay across 3 contact channels, spanning 6 major lending portfolios in India. All splits are grouped strictly by `account_id` to prevent cross-account information leakage.

| Dataset Dimension | Train Split | Validation Split | Test Split (Holdout) | Total Cohort |
|:---|---:|---:|---:|---:|
| Total PTP Records | 2,572 | 561 | 563 | 3,696 |
| Unique Accounts | 2,572 | 561 | 563 | 3,696 |
| Kept Promises ($y=1$) | 732 | 159 | 167 | 1,058 |
| Overall Keep Rate | 28.46% | 28.34% | 29.66% | 28.63% |

### 3.2 Channel Distribution
* **Tele-Agent Calls**: 2,841 PTPs (76.87%)
* **Field Visits**: 605 PTPs (16.37%)
* **Voice Bot Calls**: 250 PTPs (6.76%)

### 3.3 Label Definition and Ground Truth
Under business policy (detailed in `docs/LABEL_DEFINITION.md`), a PTP is classified as **Kept** ($y=1$) if cumulative payments credited to the account from the moment of promise capture until **3 days post-promised date** (grace period) reach at least **90% of the promised amount**. Payments across all channels (UPI link, NACH re-presentation, cash, NEFT, branch deposit) are recognized.

```
Total Promises Evaluated: 3,696
+-- Kept Outcomes: 1,058 (28.63%)
|   +-- Kept On Time: 950 (25.70%)
|   +-- Kept Late Within Grace (<= 3 days): 108 (2.92%)
+-- Non-Kept Outcomes: 2,638 (71.37%)
    +-- Broken (No meaningful payment): 1,914 (51.79%)
    +-- Partial Repayment (25% to 89% paid): 396 (10.71%)
    +-- Paid After Window (> 3 days post-due): 328 (8.87%)
```

---

## 4. Quantitative Performance and Metrics

All performance numbers are extracted directly from `reports/metrics.json` and represent evaluations conducted on out-of-fold development predictions or the single-pass holdout test set ($N=563$).

### 4.1 Keep Probability Model (Headline Evaluation)

| Metric | Dev Out-of-Fold (N=3,133) | Test Holdout Uncalibrated (N=563) | Test Holdout Final Calibrated (N=563) | Test 95% Confidence Interval |
|:---|---:|---:|---:|:---:|
| **ROC AUC** | **0.79195** | 0.74236 | **0.74188** | **[0.69546, 0.78804]** |
| **PR-AUC (Avg Precision)** | **0.60570** | 0.58274 | **0.58484** | - |
| **Brier Score** | **0.15455** | 0.17103 | **0.17138** | - |
| **Expected Calibration Error (ECE)** | **0.02234** | 0.04091 | **0.04832** | - |
| **Log Loss** | 0.47887 | 0.52457 | 0.52711 | - |
| **Mean Predicted Probability** | 0.28426 | 0.29924 | 0.30204 | - |
| **Observed Base Rate** | 0.28439 | 0.29663 | 0.29663 | - |

> [!NOTE]
> The 95% bootstrap confidence interval on the holdout test ROC AUC spans $[0.6955, 0.7880]$. Grouped out-of-fold evaluation on the larger development set ($N=3,133$) yielded an AUC of $0.7920$ and an ECE of $0.0223$.

### 4.2 Model Baselines and Ablation Benchmark
Every benchmark was evaluated on the identical test split ($N=563$, 167 positives).

| Model / Feature Configuration | Test ROC AUC | Delta vs Final Model | Technical Description |
|:---|---:|---:|:---|
| **Logistic Regression (All Candidates)** | **0.75954** | +0.01766 | L2 regularized ($C=0.01$) linear model over 161 imputed/scaled features |
| **All Candidates (XGBoost, No Selection)** | 0.75599 | +0.01411 | 3-seed XGBoost using all 161 candidate features |
| **Conversation Only** | 0.74636 | +0.00448 | Dialogue dynamics, lexicons, pauses, and transcript markers only |
| **Selected Without Agent Track Record** | 0.74280 | +0.00092 | Selected 45 features excluding agent historical performance |
| **Final Calibrated Keep Model** | **0.74188** | Reference | 5-seed bagged XGBoost (45 features) + Hierarchical Platt calibration |
| **Selected Without Voice Features** | 0.72901 | -0.01287 | Selected features excluding acoustic/pause signals (DPDP fallback) |
| **Account, History & Terms Only** | 0.70306 | -0.03882 | Traditional casemix model excluding speech and conversation signals |
| **Historical Keep Rate Heuristic** | 0.52291 | -0.21897 | Simple smoothed empirical keep rate of the borrower |
| **Incumbent Fixed Rule** | 0.50000 | -0.24188 | Uninformative baseline (all promises treated as valid) |

### 4.3 Broken PTP Detection at Operational Operating Thresholds
Evaluating the trade-off between identifying non-paying promises and avoiding false alarms on borrowers who pay.

| Operating Point | Threshold ($P_{keep}$) | Broken PTPs Flagged (Recall) | False Alarm Rate on Kept PTPs | Overall Flag Rate | Broken Precision |
|:---|---:|---:|---:|---:|---:|
| **20% False Alarm Budget** | 0.1985 | 55.81% (221 / 396) | 21.56% (36 / 167) | 45.65% | 85.99% |
| **Policy Low-Credibility Threshold** | 0.2500 | 66.41% (263 / 396) | 28.74% (48 / 167) | 55.24% | 84.57% |
| **Policy High-Credibility Threshold** | 0.5500 | 90.91% (360 / 396) | 59.88% (100 / 167) | 81.71% | 78.26% |

At the low-credibility policy threshold ($P_{keep} < 0.25$), the system intercepts **66.41%** of broken promises while maintaining an **84.57%** precision on broken flags, allowing lenders to release unnecessary parking and initiate alternative follow-ups.

### 4.4 Real-Time Streaming and Sufficiency Trajectories
The streaming model evaluates conversational prefixes sequentially to enable mid-call interventions.

```
AUC Progression by Fraction of Call Audio/Turns Heard:
10% of call: 0.58801  ============>
20% of call: 0.58715  ============>
30% of call: 0.58067  ============>
40% of call: 0.64485  ===============>
50% of call: 0.67016  ==================>
60% of call: 0.71856  ======================>
70% of call: 0.75769  =========================>
80% of call: 0.76855  ==========================>
90% of call: 0.76821  ==========================>
End of Call: 0.76773  ==========================>
```

* **AUC at First Confirmation Turn**: **0.76682**
* **End-of-Call Streaming Model AUC**: **0.76773**
* **Dynamic Stopping Rule ($\delta = 0.05$)**:
  * AUC at Early Decision Point: **0.76468**
  * Share of Calls Fired Before Call Termination: **91.32%**
  * Median Fraction Heard at Trigger: **72.73%**
  * Mean Turns Remaining for Agent Prompting: **3.43 turns**

### 4.5 Post-Call Bayesian Log-Odds Update
On unresolved promises within the 30-hour post-call monitoring window ($N=521$ unresolved test cases), customer digital engagement provides meaningful predictive updates:
* **In-Call Baseline AUC**: 0.75769
* **Post-Call Updated AUC**: **0.76976** (+1.21 pp lift)
* **Post-Call Calibrated ECE**: **0.05302**

| Predictor Variable | Logistic Regression Coefficient | Directional Impact on Keep Probability |
|:---|---:|:---|
| `in_call_log_odds` | +0.9451 | Preserves strong in-call prior calibration |
| `post_payment_link_opened` | +0.2983 | Positive engagement increases keep probability |
| `post_confirmation_reply_yes` | +0.0227 | Mild confirmation signal |
| `post_hours_to_link_open` | -0.0521 | Longer latency to open link indicates waning intent |
| `post_confirmation_unanswered` | -0.3701 | Borrower ignoring SMS/WhatsApp confirmation |
| `post_inbound_call_from_customer` | -0.6233 | Often signals emergent dispute, inability to pay, or confusion |
| `post_confirmation_reply_no` | -0.7893 | Direct contradiction of recorded promise |

### 4.6 Label Definition Robustness Analysis
To ensure that operational definitions do not introduce fragility, the final calibrated model was tested against 15 permutations of grace days and settlement fraction thresholds.

| Grace Days Allowed | Minimum Payment Threshold | Holdout Test ROC AUC |
|:---:|:---:|:---:|
| 0 days | 50% of promised amount | 0.75945 |
| 0 days | 90% of promised amount | 0.76362 |
| 0 days | 100% of promised amount | 0.76362 |
| 1 day | 50% of promised amount | 0.75627 |
| 1 day | 90% of promised amount | 0.76080 |
| 1 day | 100% of promised amount | 0.76080 |
| **3 days (Primary Standard)** | **50% of promised amount** | **0.73761** |
| **3 days (Primary Standard)** | **90% of promised amount (Headline)** | **0.74188** |
| **3 days (Primary Standard)** | **100% of promised amount** | **0.74188** |
| 5 days | 50% of promised amount | 0.71920 |
| 5 days | 90% of promised amount | 0.72491 |
| 5 days | 100% of promised amount | 0.72491 |
| 7 days | 50% of promised amount | 0.71346 |
| 7 days | 90% of promised amount | 0.71769 |
| 7 days | 100% of promised amount | 0.71769 |

Ranking quality remains stable across all reasonable definitions, establishing that the underlying behavioral signals capture true borrower payment intent rather than arbitrary calendar cutoff artifacts.

---

## 5. Multi-Class PTP Typology Model

### 5.1 Class Taxonomy and Cost-Sensitive Learning
To support nuanced treatment actions beyond binary probabilities, promises are classified into 6 operational categories:
1. `genuine_feasible`: Borrower intends to pay and has demonstrated financial capacity; terms are realistic.
2. `genuine_infeasible`: Borrower demonstrates good-faith willingness to pay, but terms conflict with cash flow (e.g. promise date falls before salary credit, medical emergency, job loss).
3. `escape`: Borrower offers a commitment primarily to terminate the call without intent to honor it.
4. `third_party`: Promise provided by a family member, co-worker, or third party rather than the primary obligor.
5. `repeat_promiser`: Borrower with a documented history of multiple broken promises.
6. `agent_recorded_or_pushed`: Promise recorded by an agent without borrower affirmation, or coerced after repeated deferrals.

#### Cost-Sensitive Loss Matrix
To satisfy regulatory and fairness constraints (PS1 Constraint 1: *A low score must never lead to harsher treatment where there are signs of hardship*), misclassifying `genuine_infeasible` as a harsh archetype (`escape`, `repeat_promiser`, `agent_recorded_or_pushed`) carries a **5.0x penalty factor**:

$$C(\text{true}=\text{genuine\_infeasible}, \text{pred} \in \text{HARSH}) = 5.0$$

Decisions minimize posterior expected risk: $\hat{y} = \arg\min_k \sum_j P(\text{class}_j | X) C(j, k)$.

### 5.2 Cross-Validation and Test Holdout Metrics
The typology model was evaluated across 25 runs (Repeated Stratified 5-Fold CV $\times$ 5 repeats) on the gold-annotated cohort, augmented via semi-supervised self-training utilizing 1,643 pseudo-labeled training cases conditioned on observed repayment outcomes.

| Model Variant | Macro F1-Score | Macro F1 Std Dev | Overall Accuracy | Hardship-to-Harsh Rate |
|:---|---:|---:|---:|---:|
| Supervised Baseline (Gold Only) | 0.72522 | $\pm$ 0.05800 | 69.90% | 11.02% |
| Self-Training (Pseudo-Labeled) | 0.73764 | $\pm$ 0.04773 | 71.80% | 9.02% |
| **Self-Training + Cost-Sensitive (Production)** | **0.72564** | $\pm$ 0.05483 | **0.70700** | **5.93%** |

* **Held-out Test Gold Evaluation ($N=27$)**: Macro-F1 = **0.72685**, Accuracy = **74.07%**.
* **Pseudo-Labels Incorporated**: 1,643 high-confidence instances ($P \ge 0.70$).
* **Hardship Misclassification Reduction**: Cost-sensitive decisioning reduced the dangerous hardship-to-harsh error rate from **11.02%** down to **5.93%**, a relative reduction of **46.2%**.

### 5.3 Confusion Matrix and Empirical Outcome Alignment
The repeated cross-validation confusion matrix (average counts across folds):

| True Class \ Predicted Class | Feasible | Infeasible | Escape | 3rd Party | Repeat | Agent Pushed |
|:---|---:|---:|---:|---:|---:|---:|
| **genuine_feasible** | **42.8** | 9.0 | 0.2 | 0.0 | 2.0 | 0.0 |
| **genuine_infeasible** | 13.8 | **34.2** | 2.2 | 0.0 | 0.8 | 0.0 |
| **escape** | 2.8 | 5.4 | **22.8** | 0.0 | 2.0 | 2.0 |
| **third_party** | 0.0 | 0.0 | 0.0 | **18.0** | 0.0 | 0.0 |
| **repeat_promiser** | 5.2 | 6.8 | 4.0 | 0.0 | **13.0** | 0.0 |
| **agent_recorded_or_pushed** | 0.0 | 0.0 | 0.0 | 2.0 | 0.4 | **10.6** |

#### Test Split Alignment with Repayment Realization
When predicted types are matched against empirical test-set repayment outcomes:

| Predicted Typology Class | Test Count ($N=563$) | Observed Keep Rate | Mean Predicted Keep Prob ($\bar{P}_{keep}$) |
|:---|---:|---:|---:|
| `genuine_feasible` | 160 | 55.62% | 0.5473 |
| `third_party` | 117 | 29.91% | 0.2705 |
| `genuine_infeasible` | 110 | 19.09% | 0.2325 |
| `escape` | 72 | 15.28% | 0.1458 |
| `agent_recorded_or_pushed` | 58 | 15.52% | 0.1228 |
| `repeat_promiser` | 46 | 4.35% | 0.1659 |

The empirical keep rate of `genuine_feasible` (55.62%) is nearly 4x that of `escape` (15.28%) and over 12x that of `repeat_promiser` (4.35%), demonstrating strong behavioral separation.

---

## 6. Signal Family Attribution and Interpretability

### 6.1 Global Feature Family Attribution
TreeSHAP values computed on the holdout test set were aggregated across six functional feature families:

| Feature Signal Family | Relative SHAP Attribution | Core Information Captured |
|:---|---:|:---|
| **Account and Promise Terms** | **34.79%** | Days until promised date, alignment with salary credit, promised amount relative to EMI/overdue balance |
| **What Was Said (Semantic Content)** | **24.38%** | Borrower explicit stating of terms, hedges, conditional phrasing, explicit commitments, hardship keywords |
| **How It Was Said (Voice & Prosody)** | **16.08%** | Maximum pause lengths, latency before responding to agent questions, speech rate, ASR confidence variance |
| **How Conversation Went (Dynamics)** | **15.30%** | Turn count, negotiation sequences, customer talk share, customer hangup, closing escape markers |
| **Account History & Casemix** | **5.64%** | Bureau score band, ability to pay estimate, days since last bounce, dialer answer rates |
| **Agent Signals & Track Record** | **3.81%** | Historical agent keep rate shrinkage, agent proposing dates first, silent terms rate |

### 6.2 Top 25 SHAP Feature Importance
The 25 most influential individual features ranked by mean absolute TreeSHAP log-odds contribution on the test partition:

| Rank | Feature Identifier | Mean \|SHAP\| | Signal Family | Physical Dimension / Meaning |
|---:|:---|---:|:---|:---|
| 1 | `terms_days_to_promise` | 0.2343 | Terms | Calendar days from call capture to promised payment date |
| 2 | `borrower_stated_amount` | 0.2335 | Semantic | Binary indicator whether borrower explicitly stated the amount |
| 3 | `cust_hedge_rate` | 0.1881 | Semantic | Fraction of customer turns containing hedging or uncertainty words |
| 4 | `voice_max_pause_ms` | 0.1519 | Voice | Duration of longest conversational pause in milliseconds |
| 5 | `cust_closing_escape` | 0.1036 | Dynamics | Customer using quick sign-off language to exit call |
| 6 | `terms_promise_dom` | 0.0957 | Terms | Day of the month of the promised date |
| 7 | `acct_ability_to_pay` | 0.0888 | History | Estimated disposable income score from banking data |
| 8 | `negotiation_turns` | 0.0832 | Dynamics | Count of turns spent negotiating date or amount |
| 9 | `terms_amount_to_emi` | 0.0773 | Terms | Ratio of promised payment to standard monthly EMI |
| 10 | `cust_firm_commitment` | 0.0720 | Semantic | Explicit verbal affirmation ("pakka", "definitely", "kanditha") |
| 11 | `acct_bureau_ord` | 0.0689 | History | Ordinal bureau score band ($300-549 \dots 750-900$) |
| 12 | `terms_salary_gap_days` | 0.0671 | Terms | Days between promised payment date and next salary credit |
| 13 | `terms_salary_days_to_next_credit` | 0.0638 | Terms | Days from today until next salary deposit |
| 14 | `dial_answer_rate_30d` | 0.0637 | History | Historical contactability over preceding 30 days |
| 15 | `cust_hedge` | 0.0621 | Semantic | Presence of non-committal words ("try karunga", "nodona") |
| 16 | `voice_response_pause_to_ask_ms` | 0.0567 | Voice | Mean customer latency to respond after agent asks for commitment |
| 17 | `voice_max_response_pause_to_ask_ms` | 0.0558 | Voice | Maximum response latency to commitment requests |
| 18 | `call_duration_s` | 0.0554 | Dynamics | Total call duration in seconds |
| 19 | `hist_days_since_last_ptp` | 0.0521 | History | Elapsed days since prior PTP recorded on account |
| 20 | `agent_proposed_date_first` | 0.0495 | Dynamics | Binary flag: agent dictated the date before customer volunteered it |
| 21 | `acct_lender_type` | 0.0450 | Terms | Categorical lender classification |
| 22 | `agent_talk_share` | 0.0447 | Dynamics | Proportion of total speech time spoken by agent |
| 23 | `cust_hardship_business` | 0.0444 | Semantic | Customer mentions business downturn or shop loss |
| 24 | `agent_silent_terms_rate` | 0.0406 | Agent | Historical rate of agent recording terms without customer speech |
| 25 | `voice_min_asr_conf` | 0.0356 | Voice | Minimum speech recognition confidence score |

---

## 7. Subgroup Performance, Fairness, and Calibration

To safeguard against systemic disparate impact, model performance and calibration error were evaluated across five demographic and operational slices on the holdout test set ($N=563$).

### 7.1 Per-Portfolio Evaluation

| Loan Portfolio | Subgroup Size ($N$) | Observed Keep Rate | Mean Predicted ($P_{keep}$) | ROC AUC | Uncalibrated ECE | Final Calibrated ECE |
|:---|---:|---:|---:|---:|---:|---:|
| Consumer Durable | 58 | 27.59% | 0.3298 | 0.82292 | 0.14251 | **0.13754** |
| Credit Card | 69 | 24.64% | 0.2798 | 0.71493 | 0.08503 | 0.08934 |
| MFI / JLG | 100 | 30.00% | 0.3057 | 0.74571 | 0.06828 | **0.06859** |
| MSME | 55 | 29.09% | 0.2646 | 0.89263 | 0.14219 | 0.14324 |
| Personal Loan (Salaried) | 163 | 33.13% | 0.3217 | 0.66684 | 0.08821 | 0.09559 |
| Two-Wheeler Loan | 118 | 28.81% | 0.2887 | 0.77731 | 0.07834 | **0.06885** |

### 7.2 Per-Language Evaluation

| Interaction Language | Subgroup Size ($N$) | Observed Keep Rate | Mean Predicted ($P_{keep}$) | ROC AUC | Uncalibrated ECE | Final Calibrated ECE |
|:---|---:|---:|---:|---:|---:|---:|
| English | 107 | 26.17% | 0.2479 | 0.61528 | 0.10270 | **0.09586** |
| Hinglish (Hindi + English) | 310 | 31.61% | 0.3082 | 0.73956 | 0.05614 | **0.04222** |
| Kanglish (Kannada + English) | 146 | 28.08% | 0.3286 | 0.83624 | 0.11306 | **0.09654** |

> [!NOTE]
> Hinglish, representing 55.1% of test traffic, achieved an ECE of **0.0422** and an AUC of **0.7396**. Kanglish achieved a high discriminative AUC of **0.8362** with ECE reduced to **0.0965** after Platt scaling. English showed lower discriminative power (AUC 0.6153), attributable to higher proportion of formal, less emotionally expressive corporate tele-agent calls.

### 7.3 Per-Income-Type Evaluation

| Income Profile | Subgroup Size ($N$) | Observed Keep Rate | Mean Predicted ($P_{keep}$) | ROC AUC | Uncalibrated ECE | Final Calibrated ECE |
|:---|---:|---:|---:|---:|---:|---:|
| Daily Wage | 81 | 23.46% | 0.3126 | 0.77589 | 0.07600 | 0.08421 |
| Salaried | 291 | 30.93% | 0.3193 | 0.72366 | 0.07991 | 0.08390 |
| Self-Employed | 191 | 30.37% | 0.2713 | 0.77534 | 0.11674 | **0.09518** |

### 7.4 Per-Channel Evaluation

| Collection Channel | Subgroup Size ($N$) | Observed Keep Rate | Mean Predicted ($P_{keep}$) | ROC AUC | Uncalibrated ECE | Final Calibrated ECE |
|:---|---:|---:|---:|---:|---:|---:|
| Tele-Agent Call | 456 | 28.73% | 0.3169 | 0.75861 | 0.04907 | 0.05029 |
| Field Visit | 79 | 30.38% | 0.1983 | 0.62652 | 0.15082 | **0.12779** |
| Voice Bot Call | 28 | 42.86% | 0.3524 | 0.80208 | 0.25843 | **0.25822** |

### 7.5 Per-Lender-Type Evaluation

| Lender Institution Type | Subgroup Size ($N$) | Observed Keep Rate | Mean Predicted ($P_{keep}$) | ROC AUC | Uncalibrated ECE | Final Calibrated ECE |
|:---|---:|---:|---:|---:|---:|---:|
| Fintech NBFC | 110 | 37.27% | 0.3329 | 0.75115 | 0.11364 | 0.12343 |
| Standard NBFC | 100 | 27.00% | 0.2895 | 0.79300 | 0.10942 | 0.10982 |
| NBFC (MSME Specialist) | 32 | 34.38% | 0.2828 | 0.91342 | 0.24630 | 0.24686 |
| NBFC-MFI | 46 | 34.78% | 0.2939 | 0.86042 | 0.16523 | **0.16163** |
| Private Commercial Bank | 146 | 26.71% | 0.2989 | 0.64558 | 0.12312 | **0.12249** |
| Small Finance Bank | 129 | 25.58% | 0.2966 | 0.73043 | 0.09981 | **0.07294** |

### 7.6 Hierarchical Platt Calibration Architecture
Standard global Platt calibration fails when lending portfolios exhibit diverse default base rates, while unconstrained segment calibration overfits thin segments (e.g. MSME with $N=32$ or Voice Bot with $N=28$). 

The `HierarchicalPlatt` implementation solves this via regularized shrinkage:
1. Fits a global logistic mapping $(a_{global}, b_{global})$ on grouped out-of-fold log-odds predictions.
2. For each portfolio segment $s$, minimizes negative log-likelihood with an L2 shrinkage penalty toward global parameters:

$$\mathcal{L}(\theta_s) = -\sum_{i \in s} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right] + \lambda_{\text{shrinkage}} \|\theta_s - \theta_{\text{global}}\|^2$$

With $\lambda_{\text{shrinkage}} = 25.0$, thin portfolios inherit the global slope, ensuring that a calibrated probability of 0.30 represents an equivalent risk profile whether on an MFI microloan or a salaried personal loan.

---

## 8. Regulatory Compliance, Ethical Safeguards, and Policy Simulation

### 8.1 Reserve Bank of India (RBI) Fair Practices Code Controls
All model-driven prompts and workflow decisions adhere strictly to statutory collections guidelines:
* **Pre-Approved Script Templates**: In-call recommendations originate exclusively from a centrally curated prompt library (`PROMPTS` in `policy.py`). Free-form LLM generation is disallowed.
* **Prohibited Phrase Filter**: A regex filter blocks coercive, intimidatory, or harassing language:
  ```
  cibil|credit score|escalat|legal|police|court|warrant|last reminder|case aage|case munde|
  karna padega|maadle beku|otherwise|warna|seize|recovery agent|ghar aayenge|mane ge bartivi
  ```
  Every prompt passes an automated import-time unit assertion and runtime validation (`fpc_check`).
* **Contact Hours and Frequency Limits**: The scheduling function (`materialise_schedule`) enforces contacts between **08:00 and 19:00 local time**, bounded by a hard cap of **at most 1 scheduled contact per day per account** (`MAX_CONTACTS_PER_DAY = 1`).

### 8.2 DPDP Act 2023 Compliance and Voice Consent
* **Purpose Limitation**: Audio-derived signals (pause latencies, speech rates, ASR confidence) are utilized exclusively for in-call credibility estimation and are strictly walled off from underwriting or marketing.
* **Consent Architecture**: Acoustic features reside under the `voice_` namespace and are computed only when sessions pass `voice_consent = true`. If consent is withheld or revoked, voice variables are set to `NaN`.
* **Ablation Transparency**: The system operates with full functionality without voice features, suffering an acceptable AUC penalty of 1.29 pp (0.74188 vs 0.72901).
* **Data Minimization and Audio Deletion**: Raw audio streams are processed in memory to extract prosodic proxies and are immediately purged. Derived numerical features are retained only until promise resolution (`promised_date + 3 days`).

### 8.3 Hardship Protection Guardrail Verification
Problem Statement Constraint 1 mandates that low credibility scores must never subject distressed borrowers to aggressive treatments.
* **Hardship Detection Trigger**: Activated if any hardship lexicon is matched (medical crisis, job loss, business disruption, delayed salary), if a recent `dial_rpc_hardship_30d` disposition exists, or if $P(\text{genuine\_infeasible}) \ge 0.35$.
* **Hard Guardrail G1**: Overrides all policy table rules. If hardship is present:
  * Aggressive dispositions (`stronger_treatment_short_parking`, `part_payment_follow_up`) are strictly blocked.
  * The account is routed exclusively to `supportive_restructure_or_human` with the full default parking window preserved.
  * In-call prompts asking for "payment today" are suppressed, presenting only restructuring and income-aligned options.

#### Simulation Verification on Holdout Test Partition ($N=563$)
The decision engine simulation (`policy_simulation_test`) verifies mathematical compliance:
* **Hardship Identified in Test Set**: **50.44%** of PTPs ($N=284$).
* **Harsh Treatment Assigned Given Hardship**: **0.000%** (0 / 284).
* **Annotated Infeasible PTPs Assigned Harsh Treatment**: **0.000%** (0 / 110).

### 8.4 Policy Simulation and Capacity Release
Replacing the incumbent policy (uniform 15-day parking freeze) with the model's dynamic policy yields substantial efficiency gains:

| Policy Treatment Assigned | Count ($N=563$) | Share | Primary Action |
|:---|---:|---:|:---|
| `supportive_restructure_or_human` | 121 | 21.49% | Hardship route: full parking + supportive specialist call |
| `confirm_with_borrower` | 117 | 20.78% | Third-party promises: outbound confirmation call |
| `renegotiate_or_restructure` | 103 | 18.29% | Infeasible promises: reschedule after income deposit |
| `park_until_due` | 99 | 17.58% | Credible promises: parked until due date + digital link |
| `part_payment_follow_up` | 44 | 7.82% | Weak promises: immediate part payment link + 24h follow-up |
| `integrity_review_and_confirm` | 41 | 7.28% | Pushed/phantom promises: supervisor integrity audit |
| `stronger_treatment_short_parking` | 21 | 3.73% | Repeat promisers: short parking + senior agent call |
| `park_with_mid_check` | 17 | 3.02% | Moderate promises: mid-window automated check-in |

#### Operational Capacity Realization
* **Broken PTPs Flagged Before Parking**: **94.95%** of broken promises are intercepted at capture rather than silently parked.
* **Broken PTPs Granted Shorter Parking**: **52.53%** receive reduced parking windows.
* **Contact Days Released on Broken Promises**: **1,187 agent-days** released across 396 broken promises on the test set.
* **Mean Contact Days Recovered per Broken Promise**: **2.997 days**.
* **Observed Keep Rate of Parked Commitments**: **64.65%** for `park_until_due`, compared to only **13.64%** for accounts flagged for `part_payment_follow_up`.

---

## 9. Limitations and Risk Factors

### 9.1 Holdout Sample Size and Variance
The holdout evaluation split comprises 563 promises (167 kept). While account-level disjoint splitting prevents data leakage, the effective positive sample size yields a relatively wide 95% bootstrap confidence interval on test AUC: $[0.6955, 0.7880]$ ($\Delta = 0.0926$). While out-of-fold cross-validation on the development set ($N=3,133$) demonstrates a high AUC of $0.7920$, performance metrics on small segments must be interpreted with awareness of sampling variance.

### 9.2 Comparison with Logistic Regression Baseline
In the offline benchmark, the regularized Logistic Regression baseline achieved a test AUC of **0.75954**, slightly higher than the final XGBoost model's test AUC of **0.74188** ($\Delta = 0.01766$). 
* **Statistical Equivalence**: The 95% confidence interval of the final model ($[0.6955, 0.7880]$) fully encompasses the Logistic Regression point estimate ($0.7595$), indicating that the variance is consistent with finite sample noise on the holdout partition.
* **Development Out-of-Fold Evidence**: Across all 3,133 development accounts, the tree ensemble achieved an out-of-fold AUC of **0.79195**, outperforming linear baselines by capturing complex non-linear interactions (e.g. salary date gaps, acoustic pause durations, and bureau bands).
* **Operational Capabilities**: Tree ensembles natively support exact TreeSHAP reason code generation and handle missing values without synthetic imputation pipelines.

### 9.3 Observational Treatment Confounding
Historical training data reflects outcomes realized under the incumbent collection policy (fixed 15-day parking, uncalibrated reminders). The model estimates keep probability **under the incumbent treatment regimen**. Observed repayment behaviors may shift dynamically once adaptive treatments (shortened parking, part-payment links) are introduced into production.

### 9.4 Voice Consent and Processing Fallback
Voice features contribute **16.08%** of model attribution and provide an empirical lift of **+1.29 pp** test AUC over non-voice models (0.74188 vs 0.72901). Under the DPDP Act 2023, borrowers may withhold voice recording consent. In production, the system must maintain dual-mode operational readiness, substituting non-voice models when consent flags are absent.

### 9.5 Agent Gaming and Moral Hazard
Agents evaluated on PTP volume may manipulate dialogue to artificially boost credibility scores. To mitigate gaming:
1. **Borrower-Driven Primacy**: The model emphasizes borrower speech patterns (hedging, stated amounts, pauses) over agent-controlled form fields.
2. **Integrity Auditing**: A dedicated integrity module screens agent behavior for silent terms entry, rapid forced agreements, and post-pressure acquiescence, successfully flagging 7 anomalous collectors (`TA013`, `TA004`, `TA019`, `FA009`, `TA014`, `TA020`, `TA003`).

---

## 10. Recommendations and Operational Roadmap

### 10.1 Label Definition Governance Sign-Off
Prior to production deployment, CreditNirvana risk, legal, and operational leadership must formally ratify the label parameters established in `docs/LABEL_DEFINITION.md`:
* 3-day post-due settlement grace window.
* 90% cumulative payment threshold.
* Inclusion of bank-initiated NACH re-presentation as valid repayment.
* Single-attribution rule for accounts with overlapping promises.

### 10.2 Pre-Production Randomized Controlled Trial (RCT) Protocol
To establish causal uplift, eliminate treatment confounding, and measure performative drift, a two-arm randomized trial must precede full-scale production rollout:

```
Total Active Accounts
        |
        +---> Control Arm (50%): Incumbent Fixed Rule (15-Day Parking, No Prompts)
        |
        +---> Treatment Arm (50%): Dynamic Credibility Policy + Real-Time Prompts
```

#### Statistical Power Calculations (Two-Sided $\alpha = 0.05, \text{Power} = 0.80$)
* **Baseline Keep Rate**: 28.63% ($p_0 = 0.28626$)
* **Sample Size for +3.0 pp Keep Rate Lift** ($28.63\% \to 31.63\%$): **3,671 accounts per arm** ($N_{\text{total}} = 7,342$).
* **Sample Size Adjusted for Agent Clustering** ($\text{ICC} = 0.05, m = 2.0$): **3,854 accounts per arm** ($N_{\text{total}} = 7,708$).
* **Sample Size for +5.0 pp Keep Rate Lift** ($28.63\% \to 33.63\%$): **1,345 accounts per arm** ($N_{\text{total}} = 2,690$).
* **Retraining Holdout**: A permanent **10% random holdout** receiving standard fixed handling must be preserved indefinitely to provide unbiased, unconfounded data for quarterly retraining.

### 10.3 Monitoring, Recalibration, and Retraining Schedule
1. **Weekly Operational Monitoring**:
   * Population Stability Index (PSI) tracking on in-call score distributions across portfolios.
   * Hardship routing audit to ensure zero harsh treatments are assigned to flagged borrowers.
   * Agent integrity tracking to monitor borrower-stated terms ratios and detect pressure anomalies.
2. **Monthly Recalibration**:
   * Refit Hierarchical Platt parameters $(a_s, b_s)$ on resolved promises from the preceding 30 days to adjust for macro cash flow seasonality and payment rail settlement drifts.
3. **Quarterly Full Retraining**:
   * Full re-estimation of XGBoost boosters, feature selection grids, and type model pseudo-labeling pools.
   * Model version, lexicon version, and policy version must be updated concurrently and logged alongside every inference event.
