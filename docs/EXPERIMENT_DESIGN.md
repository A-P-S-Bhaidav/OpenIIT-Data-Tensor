# Randomised Controlled Trial Protocol: In-Call PTP Credibility Prompts and Adaptive Disposition

**Document Version:** 1.0.0  
**Effective Date:** 2026-10-07  
**Status:** Approved for Implementation  
**Methodology:** Account-Level Multi-Arm Randomised Controlled Trial (RCT)  
**Regulatory Framework:** Reserve Bank of India (RBI) Fair Practices Code, DPDP Act 2023  
**Associated Documents:** `docs/LABEL_DEFINITION.md`, `docs/GOVERNANCE.md`, `reports/metrics.json`  

---

## 1. Executive Summary and Experimental Objective

### 1.1 Context and Problem Statement
In retail credit collections, more than half of recorded Promises to Pay (PTPs) are broken, failing to convert into realized cash recoveries. Historical operational policies rely on rigid heuristics: once a PTP is recorded, the account is placed in a fixed "parking" window during which outbound contact ceases until the promised date lapses. This practice creates two operational vulnerabilities:
1. **Unrealistic Promises:** Borrowers frequently agree to unfeasible dates or inflated amounts merely to terminate the collection call ("escape" promises), leaving genuine financial distress unaddressed while recovery effort is wasted.
2. **Delayed Intervention:** Broken promises are discovered only after the full parking window expires, sacrificing recovery momentum and increasing the probability of roll-forward into severe delinquency buckets.

While predictive models can estimate the observational keep probability $P(Y=1 \mid X)$, deploying real-time scores, conversational prompt recommendations, and dynamic disposition windows introduces active interventions that alter debtor behavior, agent conduct, and downstream contact cadences. Estimating the true business and causal value of these interventions requires a rigorous experimental design that avoids observational selection bias and isolates intervention effects from confounding operational shifts.

### 1.2 Primary Objective
The primary objective of this randomised controlled trial is to establish the causal effect of real-time PTP credibility scoring, in-call dynamic prompts, and credibility-driven adaptive disposition parking windows on:
* PTP conversion success (**Keep Rate**), and
* Financial efficiency (**Net Cash Recovery** per contacted account).

### 1.3 Hypotheses
* **Primary Hypothesis ($H_1$):** Providing real-time credibility scores, PTP subtype diagnoses, and dynamic conversational prompts to agents and bots, combined with credibility-driven disposition parking, generates a statistically significant absolute lift of at least $+3.0$ percentage points ($+3.0\text{pp}$) in the PTP keep rate compared to status-quo fixed-rule operations.
* **Secondary Hypothesis ($H_2$):** Treatment increases net cash recovery and shortens time to resolution without increasing operational costs, customer grievances, or regulatory compliance defects.
* **Null Hypothesis ($H_0$):** In-call credibility guidance and adaptive parking windows have zero causal effect ($\Delta = 0$) on PTP keep rates or net financial recovery relative to standard operational baselines.

---

## 2. Unit of Randomisation and Contamination Control

### 2.1 Account-Level Randomisation
The unit of randomisation is the **borrower account** (`account_id`), rather than individual phone calls or single PTP instances.

```
+-----------------------------------------------------------------------------------+
|                            Account Arrival / Intake                               |
|                Hashed Assignment: Salted SHA-256(account_id + salt)               |
+-----------------------------------------------------------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         | (10%)                         | (45%)                         | (45%)
         v                               v                               v
+------------------+            +------------------+            +------------------+
|    Permanent     |            |     Control      |            |    Treatment     |
|     Holdout      |            |       Arm        |            |       Arm        |
+------------------+            +------------------+            +------------------+
| No scores        |            | No scores        |            | Live score       |
| No prompts       |            | No prompts       |            | PTP type         |
| Fixed parking    |            | Fixed parking    |            | Dynamic prompts  |
| Retraining data  |            | Active benchmark |            | Adaptive parking |
+------------------+            +------------------+            +------------------+
```

### 2.2 Rationale Against Call-Level Randomisation
Randomising at the call or interaction level introduces severe methodological violations:
1. **Intra-Account Interference (SUTVA Violation):** A single delinquent borrower frequently undergoes multiple contact attempts across a billing or delinquency cycle. If call 1 were assigned to Treatment (where an agent restructures an infeasible promise to a post-salary date) and call 2 were assigned to Control (where an agent applies standard fixed scripts), the treatment effect from call 1 contaminates call 2. Stable Unit Treatment Value Assumption (SUTVA) would be fundamentally violated.
2. **Operational Disposition Inconsistency:** Disposition rules govern account-level queue suppression ("parking"). Parking an account for 2 days under Treatment rules cannot coexist with a 15-day parking window generated under Control rules for the same underlying loan account.
3. **Agent Spillover and Cross-Arm Learning:** While tele-agents handle calls across portfolios, account-level persistent assignment ensures that the customer experience, terms agreed, and follow-up schedules remain coherent and uncontaminated.

### 2.3 Deterministic and Immutable Assignment
Assignment occurs at the exact timestamp of first qualification or first contact attempt within the trial lifecycle. To ensure immutability and prevent operational leaks across disparate microservices (dialler, CRM, voice-bot gateway):
* The assignment is computed deterministically using a cryptographic hash:
  $$\text{Bucket}(\text{account\_id}) = \text{SHA256}(\text{account\_id} \mathbin{\Vert} \text{TRIAL\_SALT}) \pmod{100}$$
* The assignment is written to the persistent central operational store (`trial_assignment` table) and cached across dialler and telephony orchestrators.
* Once assigned, an account remains strictly within its designated arm for the entire duration of the trial and any subsequent outcome maturation windows.

---

## 3. Stratification and Allocation Architecture

### 3.1 Stratification Covariates
To maximize statistical power, minimize baseline variance, and guarantee group balance across critical segments, randomisation is stratified along two orthogonal dimensions:

1. **Pre-Contact PTP Credibility Decile (10 Strata):**
   Computed prior to dialling using historical account attributes, bureau records, past payment adherence, and pre-call engagement metrics ($P(\text{keep})_{\text{pre}}$). Stratifying by credibility decile prevents accidental imbalance in debtor creditworthiness across arms.
2. **Product Portfolio (6 Strata):**
   Collections dynamics, recovery yields, and ticket sizes diverge substantially by loan type. Stratification enforces exact balance across:
   * Personal Loans - Salaried (`pl_salaried`)
   * Two-Wheeler Loans (`two_wheeler`)
   * Credit Cards (`credit_card`)
   * Microfinance Joint Liability Groups (`mfi_jlg`)
   * Consumer Durable Loans (`consumer_durable`)
   * Micro, Small and Medium Enterprises (`msme`)

This yields a total of $10 \times 6 = 60$ distinct stratification blocks.

### 3.2 Allocation Ratios
Within each stratum, accounts are assigned via permuted block randomisation to prevent allocation order predictability:

| Arm Name | Target Allocation | Primary Purpose |
|---|---:|---|
| **Treatment Arm** | 45.0% | Measures efficacy of live scores, dynamic prompts, and adaptive parking |
| **Control Arm** | 45.0% | Active operational benchmark under status-quo fixed heuristics |
| **Permanent Holdout** | 10.0% | Uncontaminated baseline preserving clean observational distributions |

### 3.3 Theoretical Role of the Permanent Holdout (Performative Prediction)
A critical vulnerability in deploying machine learning within interactive loops is **performative prediction** (Perdomo et al., 2020). When predictive models dictate operational interventions (e.g., prompting agents to alter terms or shortening parking windows), the underlying data-generating distribution shifts:
$$P_{\text{treatment}}(Y \mid X) \neq P_{\text{control}}(Y \mid X)$$
If future model iterations are trained exclusively on data generated under Treatment, the training distribution becomes confounded by model-induced actions. Highly credible borrowers may appear less credible because aggressive follow-ups altered their trajectory, or weak promises may appear artificially resolved due to prompt-driven renegotiations.

The **Permanent Holdout (10%)** provides an essential operational safeguard:
* It never receives score-driven prompts, subtype classifications, or adaptive parking windows.
* It supplies an untainted, stationary data stream reflecting unprompted borrower behavior.
* It enables periodic drift audits, unconfounded calibration updates, and serves as the ground-truth benchmark for long-term retraining pipelines without performative degradation.

---

## 4. Experimental Arms and Intervention Protocol

### 4.1 Treatment Arm: Full Credibility Suite
In the Treatment arm, the collection workflow incorporates real-time streaming intelligence and adaptive post-call disposition:

#### Real-Time In-Call Guidance (Tele-Agents and Voice-Bots)
1. **Live Calibrated Credibility Score:** Displays $P(\text{keep}) \in [0, 1]$, updated dynamically as conversational turns progress (streaming inference with early decision firing at median $72.7\%$ of call duration).
2. **PTP Subtype Classification:** Surfaces the classified intent profile:
   * `genuine_feasible`: Realistic commitment; confirm terms and provide instant digital link.
   * `genuine_infeasible`: Debtor intends to pay but terms clash with cash flows; prompt for realistic restructuring.
   * `escape`: Debtor agreeing purely to end call; prompt for immediate token commitment or closer check-in.
   * `repeat_promiser`: Serial defaulter; prompt for smaller, nearer installment.
   * `third_party`: Family member/colleague answering; prompt for direct borrower contact time without recording false PTP.
   * `agent_recorded_or_pushed`: Agent coercing an unverified promise; flags need for independent confirmation.
3. **Dynamic Prompt Injection:** Injects compliant, vetted prompts from the template library (`policy.py`), including:
   * `P_MOVE_AFTER_SALARY`: Reschedules promises falling prior to verified salary credit dates.
   * `P_ASK_TERMS_OWN_WORDS`: Prompts agent to elicit explicit dates/amounts when the borrower remains passive.
   * `P_PART_PAYMENT_LINK`: Offers immediate low-barrier payment links for weak promises.
   * `P_REPEAT_SMALLER_NEARER`: Proposes manageable weekly milestones for repeat promisers.
   * `P_CONFIRM_LINK`: Seamless link dispatch for high-credibility promises.

#### Credibility-Driven Adaptive Disposition
Post-call queue management executes rules `D1` through `D7` configured in `src/ptp_credibility/policy.py`:
* **High Credibility ($P(\text{keep}) \ge 0.55$):** Parked until due date (`dtp` days up to 15 days), with single reminder on `due - 1d`. Prevents intrusive over-contacting.
* **Weak Credibility ($P(\text{keep}) < 0.25$) or Escape:** Shortened parking window capped at 2 days; automated instant payment link sent via WhatsApp within 24 hours, followed by voice-bot follow-up at $+1\text{d}$.
* **Infeasible Promise:** Short parking capped at 3 days; routed to supportive restructuring desk (`restructuring_desk_or_human`) with human follow-up at $+2\text{d}$.
* **Repeat Promiser:** Short parking capped at 3 days; routed to senior recovery specialists.
* **Third-Party Promise:** 1-day parking; triggers borrower confirmation call at $+1\text{d}$.
* **Agent-Recorded / Pushed:** 1-day parking; routed to integrity audit queue for secondary verification.

### 4.2 Control Arm: Incumbent Status Quo
The Control arm reflects standard industry operations:
1. **No Real-Time Intelligence:** Agents and bots operate standard call interfaces without visibility into credibility scores, predicted subtypes, or dynamic prompt cards.
2. **Fixed Disposition Parking Rules:** All recorded PTPs receive standard fixed parking (default 7 to 15 days until promised due date), regardless of whether the promise was an evasive escape or a highly credible commitment.
3. **Uniform Reminders:** Scheduled reminders follow generic fixed templates dispatched on standard calendar triggers.

### 4.3 Permanent Holdout Arm: Pure Benchmark
Identical in execution to the Control arm, but permanently excluded from trial rotation, model rollouts, and pilot interventions across multi-quarter horizons.

### 4.4 Summary Comparison of Experimental Arms

| Feature / Dimension | Permanent Holdout (10%) | Control Arm (45%) | Treatment Arm (45%) |
|---|---|---|---|
| **Live Credibility Score** | Hidden | Hidden | Displayed ($P(\text{keep})$ in real-time) |
| **PTP Subtype Display** | Hidden | Hidden | Displayed (6 behavioral classes) |
| **Dynamic In-Call Prompts** | Disabled | Disabled | Enabled (Context-aware scripts) |
| **Salary Alignment Guidance** | Disabled | Disabled | Enabled (`P_MOVE_AFTER_SALARY`) |
| **Disposition Parking** | Fixed rules (7–15 days) | Fixed rules (7–15 days) | Adaptive (1–15 days by credibility) |
| **Early Broken PTP Contact** | Disabled (waits for full window) | Disabled (waits for full window) | Enabled (releases contact in 24–48h) |
| **Hardship Guardrail (G1)** | Mandatory Active | Mandatory Active | Mandatory Active |
| **FPC / Contact Hour Caps** | Mandatory Active | Mandatory Active | Mandatory Active |
| **Data Usage Role** | Unconfounded retraining | Trial comparative benchmark | Trial treatment evaluation |

---

## 5. Sample Size and Statistical Power Calculations

### 5.1 Power Calculation Methodology
Sample size determination follows the two-sided two-proportion comparison framework with adjustment for intra-account clustering (Design Effect):
$$n_{\text{unclustered}} = \frac{\left( z_{1-\alpha/2} \sqrt{2\bar{p}(1-\bar{p})} + z_{1-\beta} \sqrt{p_0(1-p_0) + p_1(1-p_1)} \right)^2}{\Delta^2}$$
$$\text{DEFF} = 1 + (m - 1)\rho$$
$$n_{\text{clustered}} = \lceil n_{\text{unclustered}} \times \text{DEFF} \rceil$$

Where:
* $p_0$: Baseline keep rate under incumbent operations ($0.28626 \approx 28.6\%$, derived from `reports/metrics.json` across 3,696 observed PTPs).
* $\Delta$: Minimum Detectable Effect size ($\Delta = p_1 - p_0$).
* $\bar{p}$: Pooled proportion $\frac{p_0 + p_1}{2}$.
* $\alpha$: Type-I error rate ($0.05$, two-sided, $z_{0.975} = 1.95996$).
* $1 - \beta$: Statistical power ($0.80$, $z_{0.80} = 0.84162$).
* $m$: Average cluster size (mean number of PTP attempts per account, empirical $m = 2.0$).
* $\rho$: Intracluster Correlation Coefficient ($\text{ICC} = 0.05$).

### 5.2 Power Scenarios and Sample Size Targets
The trial power targets are calibrated directly to empirical parameters in `reports/metrics.json`:

```
+-----------------------------------------------------------------------------------------+
|                                Sample Size Requirements                                 |
+-----------------------------------------------------------------------------------------+
|  Metric Scenario                          | Lift (Delta) | Per-Arm Size | Total Trial*  |
+-------------------------------------------+--------------+--------------+---------------+
|  Unclustered (+3pp Lift)                  |   +3.00 pp   |    3,671     |     8,158     |
|  Unclustered (+5pp Lift)                  |   +5.00 pp   |    1,345     |     2,990     |
|  Clustered (ICC=0.05, m=2, +3pp Lift)     |   +3.00 pp   |    3,854     |     8,564     |
+-----------------------------------------------------------------------------------------+
* Total Trial includes Treatment (45%), Control (45%), and Permanent Holdout (10%).
```

#### Detailed Parameter Derivation
1. **Unclustered $+3\text{pp}$ Lift ($p_0 = 0.28626 \to p_1 = 0.31626$):**
   * $\bar{p} = 0.30126$
   * Required per-arm sample size: **3,671 accounts**
2. **Unclustered $+5\text{pp}$ Lift ($p_0 = 0.28626 \to p_1 = 0.33626$):**
   * $\bar{p} = 0.31126$
   * Required per-arm sample size: **1,345 accounts**
3. **Clustered Sizing ($+3\text{pp}$ Lift, $\text{ICC} = 0.05, m = 2.0$):**
   * $\text{DEFF} = 1 + (2.0 - 1) \times 0.05 = 1.05$
   * Required per-arm sample size: $3,671 \times 1.05 =$ **3,854 accounts**

### 5.3 Full Trial Cohort Sizing
Accounting for the tri-arm split (45% Treatment, 45% Control, 10% Permanent Holdout):
* **Treatment Arm ($n_T$):** 3,854 accounts
* **Control Arm ($n_C$):** 3,854 accounts
* **Permanent Holdout ($n_H$):** $\lceil \frac{3854}{0.45} \times 0.10 \rceil = 856$ accounts
* **Total Enrolled Sample Size ($N$):** **8,564 unique accounts**
* **Expected Interaction Volume:** At $m = 2.0$ promises per account, this corresponds to approximately **17,128 PTP observations**.

---

## 6. Outcome Metrics and Measurement Framework

### 6.1 Primary Outcome
The primary endpoint is the **PTP Keep Rate** ($Y_{ij} \in \{0, 1\}$), evaluated in strict alignment with the signed-off business specification in `docs/LABEL_DEFINITION.md`:

$$\text{Keep Rate} = \frac{\sum \text{Kept PTPs}}{\sum \text{Captured PTPs}}$$

#### Exact Label Definition
A PTP is formally defined as **Kept ($Y = 1$)** if and only if:
1. **Measurement Window:** Cumulative payments are received from the promise capture timestamp (`captured_ts`) through **3 days following the promised date** (`promised_date + 3 days`, inclusive).
2. **Value Threshold:** Total payments within this window equal or exceed **90% of the promised amount** (`min_fraction = 0.90`).
3. **Omni-Channel Attribution:** Credit is given for all payment rails: UPI instant links, debtor UPI apps, NACH/eNACH re-presentation, direct branch cash, NEFT, RTGS, BBPS, and field collections.
4. **Superseded Promise Rule:** If a borrower generates a new PTP prior to the expiration of an earlier date, payments are attributed exclusively to the most recent preceding PTP.

### 6.2 Secondary Outcomes
To evaluate commercial efficiency and operational velocity, five secondary endpoints are measured:

1. **Net Recovered Amount ($\text{INR}$):**
   Continuous financial realization per account:
   $$\text{Net Recovery} = \sum \text{Recovered Amount} - \text{Direct Contact Cost}$$
   Where direct contact costs incorporate telephony dial minutes, SMS/WhatsApp messaging API fees, and agent labor allocations.
2. **Cost per Rupee Recovered ($\text{Cost Ratio}$):**
   Total variable operational expenses incurred divided by total gross recoveries realized:
   $$\text{Cost Ratio} = \frac{\text{Total Collections Expenditure (INR)}}{\text{Total Gross Recovery (INR)}}$$
3. **Customer Complaint Rate:**
   Total formal grievances registered per 1,000 contacted accounts across Central Banking Ombudsman, lender grievance desks, and in-call escalations.
4. **Hardship Escalation Rate:**
   Percentage of contacted accounts identified as experiencing genuine insolvency and successfully transitioned to formal debt restructuring or moratorium review desks.
5. **Time to Resolution (Cash Acceleration):**
   Mean elapsed calendar days from initial delinquency qualification to account regularization or full payment clearance.

### 6.3 Granular Behavioral Taxonomy
In addition to the binary keep label, all outcomes are mapped into the standardized 5-class taxonomy:
* `kept_on_time`: $\ge 90\%$ paid by promised date.
* `kept_late_within_grace`: $\ge 90\%$ paid within 3-day grace window.
* `partial`: $25\%$ to $90\%$ paid within window.
* `paid_after_window`: Payment landed after grace window but within 30 days.
* `broken`: $< 25\%$ paid within window.

---

## 7. Guardrail Metrics and Safety Stopping Criteria

To protect consumers and maintain strict regulatory compliance, the trial implements non-negotiable guardrail metrics. A breach of any guardrail triggers an automated freeze of treatment allocation.

```
+-----------------------------------------------------------------------------------------+
|                                Guardrail Monitoring Grid                                |
+-----------------------------------------------------------------------------------------+
| Guardrail Parameter              | Threshold / Ceiling | Audit Frequency | Enforcement  |
+----------------------------------+---------------------+-----------------+--------------+
| Hardship-to-Harsh Treatment Rate | <= 5.0%             | Weekly Real-Time| Automated G1 |
| RBI FPC Banned Phrase Violations | Exactly 0.0%        | Daily / Stream  | Instant Stop |
| Permitted Contact Hours Adherence| 100.0% (08:00-19:00)| Automated Engine| Blocked in DB|
| Max Contacts Per Day Ceiling     | Exactly <= 1 / day  | Dialler Gate    | Hard DB Rule |
| Excess Customer Complaint Rate   | <= Control + 0.15pp | Weekly DSMB     | Trial Pause  |
+-----------------------------------------------------------------------------------------+
```

### 7.1 Hardship Protection Guardrail (Mandatory Ceiling: $\le 5.0\%$)
* **Definition:** A low credibility score must never cause an account experiencing genuine hardship to receive harsher collection treatment (such as accelerated follow-ups, short parking, or coercive payment demands).
* **Hardship Detection:** An account is flagged for hardship if it exhibits:
  * Any hardship lexicon hit (medical catastrophe, job loss, business collapse, no food/funds, delayed salary),
  * Historical `dial_rpc_hardship_30d > 0`, or
  * Type-model probability $P(\text{genuine\_infeasible}) \ge 0.35$.
* **Enforcement Guardrail (G1):** Accounts with the hardship flag are strictly prohibited from receiving short parking or part-payment push prompts. They are routed exclusively to `restructuring_desk_or_human` with full parking windows.
* **Safety Standard:** In production audits, the rate of harsh treatments administered to identified hardship cases must **not exceed 5.0%** (the validated policy simulation achieved $0.0\%$).

### 7.2 RBI Fair Practices Code (FPC) Compliance
* **Zero-Tolerance Banned Phrase Filter:** Real-time speech and text filtering asserts that no prompt or agent script contains intimidation, legal/police threats, credit bureau humiliation ("CIBIL खराब कर देंगे"), or vulgarity. Verified in `fpc_check()`.
* **Contact Window Constraints:** Follow-up contacts and reminder dispatches are strictly restricted to local hours between **08:00 and 19:00**. All automated reminders outside this window are automatically shifted by `_in_contact_hours()`.
* **Daily Contact Ceiling:** Strictly enforces a maximum of **1 contact attempt per day per account** (`MAX_CONTACTS_PER_DAY = 1`). Stacking reminders is blocked at the telephony gateway.

### 7.3 Customer Grievance Non-Inferiority
* **Safety Bound:** The customer complaint rate in the Treatment arm must not exceed that of the Control arm by more than **$0.15$ percentage points** ($p_{\text{complaint}, T} - p_{\text{complaint}, C} \le 0.0015$).

---

## 8. Trial Duration, Cadence, and Interim Monitoring

### 8.1 Trial Duration: 8-Week Active Enrollment
The trial is structured for a minimum active enrollment horizon of **8 weeks**, followed by a **4-week outcome maturation tail** (12 weeks total trial lifecycle).

#### Operational Justification for 8-Week Enrollment
1. **Full PTP Lifecycle Capture:** The maximum parking window under policy rules is 15 days. Adding the 3-day grace period, a 2-day bank clearance buffer, and handling superseded promises requires up to 20 calendar days for a single PTP cycle to resolve.
2. **Salary Credit Alignment:** Indian consumer credit repayment cycles are intimately linked to month-end and early-month payroll credits (typically 28th to 7th). An 8-week enrollment window guarantees observation across **two full consecutive payroll cycles**, ensuring that prompts such as `P_MOVE_AFTER_SALARY` can be evaluated across month boundaries.
3. **Multi-Touch Resolution:** Approximately 35% of accounts require repeat contacts and renegotiated terms before resolving. A shorter trial window would artificially truncate multi-touch accounts and bias estimates toward early single-call payers.

### 8.2 Interim Analysis and O'Brien-Fleming Boundaries
To balance early efficacy detection against the risk of premature termination, the trial incorporates formal interim evaluations governed by group sequential methodology using **O'Brien-Fleming alpha-spending functions**.

```
Information Fraction (t)
  0.0                                0.5                               1.0
---|----------------------------------|---------------------------------|--->
  Start                           Month 1                           Month 2
                               (Interim Look)                   (Final Analysis)
                              Z_crit = 2.963                    Z_crit = 1.969
                              p < 0.00305                       p < 0.0489
```

#### Stopping Boundaries Specification
The overall two-sided family-wise error rate is constrained to $\alpha = 0.05$.

| Analysis Look | Calendar Horizon | Information Fraction ($t$) | Cum. Account Sample ($N$) | O'Brien-Fleming $Z_{\text{crit}}$ | Nominal Significance ($\alpha_t$) |
|---|---|---:|---:|---:|---:|
| **Interim Look 1** | Week 4 (Month 1) | $50.0\%$ | 4,282 | $\pm 2.963$ | $p < 0.00305$ |
| **Final Analysis** | Week 8 (Month 2) | $100.0\%$ | 8,564 | $\pm 1.969$ | $p < 0.04890$ |

#### Stopping Criteria Protocol
1. **Early Efficacy Stopping:** If at Interim Look 1 the test statistic for the primary endpoint exceeds $|Z| > 2.963$, the Data and Safety Monitoring Board (DSMB) may recommend early full-scale deployment to the Treatment policy.
2. **Futility Stopping (Non-Binding):** If at Interim Look 1 the observed conditional power of achieving statistical significance at final readout is less than $10\%$, the DSMB may terminate the trial early for futility.
3. **Safety / Guardrail Stopping:** If at any weekly audit the Hardship-to-Harsh rate exceeds $5.0\%$ or the complaint non-inferiority bound is breached with $p < 0.01$, treatment assignment is paused immediately.

---

## 9. Statistical Analysis Plan (SAP)

### 9.1 Intention-to-Treat (ITT) Framework
The primary analysis is conducted strictly under the **Intention-to-Treat (ITT)** principle. Every randomised account is analyzed within its assigned arm, regardless of:
* Tele-agent adherence to displayed prompt recommendations,
* Premature call termination by the debtor, or
* Subsequent operational routing deviations.

Analyzing strictly by ITT protects the trial against compliance-induced selection bias and measures the true real-world policy impact of providing the system to collections operations.

### 9.2 Account-Clustered Econometric Specification
Because accounts may record multiple PTPs during the observation window ($m \approx 2$), observations within the same account are correlated. Standard ordinary least squares or unclustered standard errors would underestimate parameter variance and inflate false-positive rates.

#### Primary Regression Specification
The primary treatment effect is estimated using a Linear Probability Model (LPM) with **Cluster-Robust Variance Estimators (CRVE / Huber-White Sandwich SE)** clustered at the account level:

$$Y_{ij} = \beta_0 + \beta_1 \text{Treatment}_i + \sum_{s=1}^{S-1} \gamma_s \text{Stratum}_{s, i} + \mathbf{X}_{ij}' \boldsymbol{\delta} + \varepsilon_{ij}$$

Where:
* $Y_{ij} \in \{0, 1\}$ is the keep outcome for PTP $j$ on account $i$.
* $\text{Treatment}_i$ is the binary treatment indicator ($1$ if Treatment, $0$ if Control).
* $\beta_1$ represents the primary causal estimand: the **Average Treatment Effect (ATE)** on PTP keep rate.
* $\text{Stratum}_{s, i}$ are dummy variables representing the 60 stratification blocks (decile $\times$ portfolio).
* $\mathbf{X}_{ij}$ is a vector of pre-treatment baseline covariates (delinquency bucket, bureau score, loan amount, historical keep rate) included for precision enhancement.
* Standard errors are clustered on `account_id` $i$:
  $$\mathbb{V}_{\text{cluster}}(\hat{\boldsymbol{\beta}}) = (X'X)^{-1} \left( \sum_{i=1}^{K} X_i' \hat{\varepsilon}_i \hat{\varepsilon}_i' X_i \right) (X'X)^{-1}$$

#### Secondary Non-Linear Model
As a robustness check, a Generalized Estimating Equations (GEE) logit model with an exchangeable working correlation matrix is estimated:
$$\text{logit}(P(Y_{ij} = 1)) = \alpha_0 + \alpha_1 \text{Treatment}_i + \mathbf{Z}_{ij}' \boldsymbol{\theta}$$
Reporting both the odds ratio ($\exp(\alpha_1)$) and marginal effects ($\frac{\partial P}{\partial \text{Treatment}}$).

### 9.3 Secondary Outcome Estimation
1. **Net Recovered Amount:** Estimated via Tobit regression or two-part hurdle models to handle zero-inflation (unpaid promises) and positive continuous recovery amounts.
2. **Cost per Recovery:** Evaluated via non-parametric cluster bootstrap (1,000 resamples at the account level) to construct 95% confidence intervals for the ratio metric.
3. **Time to Resolution:** Evaluated using Cox Proportional Hazards regression clustered by account, treating unrecovered accounts as right-censored at trial completion.

### 9.4 Pre-Specified Subgroup and Heterogeneity Analyses
To uncover heterogeneous treatment effects, the analysis plan pre-specifies interaction models across four operational dimensions:
1. **Loan Portfolio:** Interaction terms $\text{Treatment}_i \times \text{Portfolio}_i$ across the 6 loan categories.
2. **Credibility Score Bands:** Interaction across 3 operating score tiers:
   * Low Credibility: $P(\text{keep}) < 0.25$
   * Moderate Credibility: $0.25 \le P(\text{keep}) < 0.55$
   * High Credibility: $P(\text{keep}) \ge 0.55$
3. **Interaction Channel:** Performance comparison across `tele_agent`, `voice_bot`, and `field`.
4. **Primary Language:** Subgroup readouts across `hinglish`, `kanglish`, and `english`.

Wald tests of joint significance across interaction coefficients will determine whether treatment efficacy varies meaningfully across operational cohorts.

### 9.5 Instrumental Variables (IV) / As-Treated Analysis
To evaluate the causal effect of *agent compliance with prompts* (Treatment-on-the-Treated, or TOT), an instrumental variables regression is specified:
* **Instrument:** Random assignment $\text{Treatment}_i$.
* **Endogenous Variable:** Adherence indicator $D_{ij}$ ($1$ if agent executed the displayed prompt script).
* **Two-Stage Least Squares (2SLS):**
  $$\text{Stage 1: } D_{ij} = \pi_0 + \pi_1 \text{Treatment}_i + \mathbf{X}_{ij}' \boldsymbol{\lambda} + \nu_{ij}$$
  $$\text{Stage 2: } Y_{ij} = \delta_0 + \tau_{\text{LATE}} \hat{D}_{ij} + \mathbf{X}_{ij}' \boldsymbol{\mu} + \omega_{ij}$$
Where $\tau_{\text{LATE}}$ captures the Local Average Treatment Effect of executed prompt interventions among compliant interactions.

---

## 10. Ethical Considerations, Fairness, and Consumer Protection

### 10.1 Universal Hardship Protection
Under no circumstances is financial hardship exploited or penalized. The experimental design guarantees that:
* The **Hardship Guardrail (G1)** applies identically across **both Treatment and Control arms**.
* Regardless of assignment, any debtor exhibiting signs of severe illness, job loss, business insolvency, or family bereavement is immediately shielded from aggressive collections.
* Distressed accounts are systematically routed to compassionate restructuring desks or specialized human counsellors.

### 10.2 Equal Access to Restructuring and Debt Relief
Randomisation into the Control arm does not withhold debt restructuring or forbearance opportunities. While the Treatment arm uses model triggers to prompt restructuring earlier during the call, borrowers in all arms retain unhindered access to statutory repayment restructuring, tenure modifications, and settlement options upon request.

### 10.3 Regulatory and Statutory Compliance
All experimental operations operate in strict conformity with:
1. **Reserve Bank of India (RBI) Guidelines on Recovery Agents:**
   * Prohibition of uncivil language, intimidation, and harassment.
   * Universal enforcement of approved calling windows (08:00 to 19:00).
   * Absolute respect for customer privacy and contact frequency caps.
2. **Digital Personal Data Protection (DPDP) Act 2023:**
   * Voice-derived prosodic features (hesitation, pauses, latency) are processed strictly for in-call credibility assessment and deleted upon PTP resolution.
   * Voice analytics require documented borrower consent (`voice_consent=true`); accounts lacking consent automatically default to textual and tabular features without operational penalty.
   * All predictive inputs and decision rule IDs are logged to maintain an auditable decision trail.

---

## 11. Operational Execution and Systems Architecture

### 11.1 Technical Systems Integration
The RCT is supported by low-latency integration across the telephony, CRM, and risk evaluation engines:

```
[ Inbound/Outbound Telephony Gateway ]
                 |
                 v
   [ Randomisation & Hashing Layer ]  ---> Salted SHA-256 Lookup
                 |
                 v
      [ Real-Time Scoring Engine ]     ---> PTP Credibility & Subtype
                 |
                 v
+-------------------------------------------------------------+
|                     Execution Routing                       |
+-------------------------------------------------------------+
| Treatment Arm:                 | Control Arm:               |
| * Streaming latency < 200ms    | * Standard CRM View        |
| * In-call prompt card injection| * No scores or prompts     |
| * Dynamic disposition schedule | * Fixed disposition parking|
+-------------------------------------------------------------+
                 |
                 v
    [ Central Audit & Telemetry Log ] ---> Daily DSMB Pipeline
```

1. **Randomisation Service:** Operates as a stateless microservice returning assigned arm within $<10\text{ms}$ based on `account_id` hash lookup.
2. **Inference Latency Budget:** Streaming speech-to-text and model feature evaluation run on dedicated inferencing hardware, delivering prompt injection within $<200\text{ms}$ of turn completion.
3. **Telemetry and Audit Logging:** Every interaction records an immutable audit payload containing:
   * `account_id`, `call_id`, and `ptp_id`
   * Assigned experimental arm (`treatment`, `control`, `holdout`)
   * Displayed credibility score, predicted subtype, and rule ID
   * Prompt delivered and agent adherence outcome
   * Materialized disposition timestamps and parking window length

### 11.2 Trial Governance and DSMB Oversight
A dedicated **Data and Safety Monitoring Board (DSMB)** comprising representatives from Credit Risk, Collections Operations, Compliance, and Data Science will meet weekly throughout the 8-week enrollment window. The board holds sole authority to review interim unblinded readouts, inspect guardrail dashboards, and execute stopping rules.

---

## 12. Document Approval and Sign-Off

This experimental design protocol represents the binding operational standard for evaluating in-call PTP credibility systems. Any modification to sample sizes, allocation ratios, label definitions, or stopping boundaries requires unanimous sign-off by all authorizing stakeholders.

| Role | Name / Function | Signature | Date |
|---|---|---|---|
| **Head of Credit Risk** | Risk Governance Committee | `[APPROVED]` | 2026-10-07 |
| **Head of Collections Operations** | Collections Strategy Group | `[APPROVED]` | 2026-10-07 |
| **Chief Compliance Officer** | Regulatory Compliance & Legal | `[APPROVED]` | 2026-10-07 |
| **Lead Machine Learning Scientist** | Risk Analytics & AI Engineering | `[APPROVED]` | 2026-10-07 |
