# Label Definition: When Is a Promise to Pay "Kept"?

The problem statement notes that late, partial and other-channel payments blur the outcome, and that the definition must be agreed with the business before training. This document is the proposed definition for sign-off. Every parameter is in `src/ptp_credibility/config.py` (`LabelConfig`) and can be changed without code changes.

## 1. Primary definition (used for training and all headline metrics)

A PTP is **kept** if cumulative payments on the account, from the moment the promise is captured until **3 days after the promised date**, reach **at least 90% of the promised amount**.

| Element | Decision | Rationale |
|---|---|---|
| Window start | `captured_ts` of the PTP | Payments before the promise cannot be caused by it. |
| Window end | `promised_date + 3 days` (inclusive) | Covers settlement and posting lag for NACH re-presentation, branch cash, NEFT and BBPS. A borrower who pays one or two days late has kept the substance of the promise, and the operational treatment (a reminder on due date + 1) would be the same. |
| Amount threshold | at least 90% of `promised_amount` | Tolerates rounding and small fee differences without counting genuine part payments as kept. |
| Channels | All (UPI link, UPI app, NACH re-presentation, branch cash, NEFT, BBPS, field cash) | The question is whether the borrower paid, not which rail was used. Other-channel payments are explicitly part of the definition. |
| Overlapping promises | A payment is attributed only to the most recent PTP captured before it | When a borrower re-promises before the old date, the old promise is superseded; without this rule one payment would be double counted. |

## 2. Outcome taxonomy (reported alongside the binary label)

| Status | Definition | Count | Share |
|---|---|---:|---:|
| `kept_on_time` | 90% or more paid by the promised date | 950 | 25.7% |
| `kept_late_within_grace` | 90% or more paid within the 3-day grace period | 108 | 2.9% |
| `partial` | 25% to 90% paid within the window | 396 | 10.7% |
| `paid_after_window` | Some payment within 30 days, after the window | 328 | 8.9% |
| `broken` | None of the above | 1,914 | 51.8% |

Overall keep rate: **28.6%**. This is consistent with the problem statement, which says more than half of captured PTPs never turn into a payment.

`recovered_amount` (payment within the window, capped at the promised amount) is stored as a continuous outcome for value metrics and trial readouts.

## 3. Sensitivity of the definition

Keep rate under alternative definitions (whole dataset):

| Grace (days) | 50% of amount | 75% | 90% | 100% |
|---:|---:|---:|---:|---:|
| 0 | 28.4% | - | 25.7% | - |
| 1 | 30.0% | - | 26.6% | - |
| 3 | 33.4% | - | **28.6%** | - |
| 5 | 35.9% | - | 30.5% | - |
| 7 | 37.5% | - | 32.1% | - |

The trained model is re-scored against every alternative definition in `reports/metrics.json` (`label_robustness_test`). Ranking quality is stable across definitions, so the business can choose the definition that matches its operations without retraining. A retrain is recommended for calibration only.

## 4. Point-in-time use of outcomes

An outcome becomes known at `resolution_ts = promised_date + grace + 1 day`. Features built from earlier PTPs (account history, agent track record) only use outcomes whose `resolution_ts` is earlier than the capture time of the PTP being scored. Agent track-record features additionally use only training-split accounts, so no validation or test label ever enters a feature.

## 5. Items for business sign-off

1. Grace period of 3 days (alternatives: 0, 1, 5, 7).
2. Amount threshold of 90% (alternatives: 50%, 75%, 100%).
3. Whether NACH re-presentation should count as a kept promise or be tracked separately, since it is lender-initiated. It is counted by default.
4. Whether a promise superseded by a new promise should be scored as broken (current behaviour) or excluded from training.
