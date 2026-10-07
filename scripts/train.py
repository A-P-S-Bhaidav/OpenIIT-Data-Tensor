"""End-to-end training and evaluation.

Protocol
--------
* Splits are the account-level splits shipped with the data (train / validation / test).
* Features are ranked on *train*; the feature-set size is chosen by grouped 5-fold CV on train + validation.
* Final models are refit on train + validation ("dev") with grouped 5-fold
  out-of-fold predictions used to fit the calibrators.
* The test split is scored exactly once, at the end, for every reported number.

Usage:  python scripts/train.py            (writes artifacts/ and reports/)
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from ptp_credibility import __version__
from ptp_credibility.calibration import expected_calibration_error
from ptp_credibility.config import CONFIG, LabelConfig
from ptp_credibility.evaluation import (binary_report, broken_flag_recall, safe_auc, segment_report,
                                        threshold_for_false_alarm, two_proportion_sample_size)
from ptp_credibility.explain import family_attribution
from ptp_credibility.integrity import agent_integrity_table
from ptp_credibility.labels import build_outcomes, label_sensitivity
from ptp_credibility.lexicon import LEXICON_VERSION
from ptp_credibility.models import (HARSH_TYPES, TYPE_CLASSES, KeepModel, PostCallModel, TypeModel,
                                    candidate_features)
from ptp_credibility.pipeline import build_datasets
from ptp_credibility.policy import POLICY_VERSION, disposition, hardship_flag
from ptp_credibility.streaming import SufficiencyModel, decision_points
from ptp_credibility import plots

P = CONFIG.paths.ensure()
RS = CONFIG.model.random_state
XGB = dict(n_estimators=600, learning_rate=0.015, max_depth=3, min_child_weight=5, subsample=0.85,
           colsample_bytree=0.6, reg_lambda=2.0, tree_method="hist", n_jobs=8)
CASEMIX_PREFIXES = ("acct_", "hist_", "pay_", "dial_", "terms_")
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main() -> None:
    metrics: dict = {"versions": {"package": __version__, "lexicon": LEXICON_VERSION, "policy": POLICY_VERSION},
                     "label_definition": {"grace_days": CONFIG.label.grace_days,
                                          "min_fraction": CONFIG.label.min_fraction}}
    ds = build_datasets()
    df, pf = ds.ptp, ds.prefix
    log(f"datasets: {len(df)} PTPs, {len(pf)} prefix rows")
    tr, va, te = (df.split == s for s in ("train", "validation", "test"))
    dev = tr | va
    y = df.kept.values
    groups_all = df.account_id.values
    metrics["data"] = {"ptps": int(len(df)), "keep_rate": float(y.mean()),
                       "outcome_status": df.outcome_status.value_counts().to_dict(),
                       "split_sizes": df.split.value_counts().to_dict(),
                       "channels": df.channel.value_counts().to_dict()}
    metrics["label_sensitivity_keep_rate"] = label_sensitivity(ds.tables.ptps, ds.tables.payments).to_dict("records")

    # ------------------------------------------------------------------ feature selection (grouped CV on dev)
    # Ranking uses the train split only; the size of the feature set is chosen by grouped 5-fold CV on
    # train + validation (a single validation split proved too noisy). Smallest k within 0.002 AUC of the best.
    cand = candidate_features(df)
    cats = {c: sorted(df[c].dropna().astype(str).unique()) for c in cand if isinstance(df[c].dtype, pd.CategoricalDtype)}
    probe = KeepModel(XGB, cand, n_seeds=3)
    probe.categories = cats
    ms = probe._fit_raw(df[tr], y[tr])
    gain = pd.concat([pd.Series(m.get_booster().get_score(importance_type="total_gain")) for m in ms], axis=1).mean(1)
    ranked = list(gain.reindex(cand).fillna(0).sort_values(ascending=False).index)
    D, yd, gd = df[dev].reset_index(drop=True), y[dev], groups_all[dev]
    folds = list(GroupKFold(5).split(D, yd, gd))
    sel_rows = []
    for k in (30, 45, 60, 80, 100, len(ranked)):
        km = KeepModel(XGB, ranked[:k], n_seeds=2)
        km.categories = cats
        oof = np.zeros(len(D))
        for a, b in folds:
            oof[b] = km._predict_raw(km._fit_raw(D.iloc[a], yd[a]), km._frame(D.iloc[b]))
        sel_rows.append({"k": k, "cv_auc": safe_auc(yd, oof)})
        log(f"feature selection k={k}: dev CV AUC {sel_rows[-1]['cv_auc']:.4f}")
    best = max(r["cv_auc"] for r in sel_rows)
    best_k = min(r["k"] for r in sel_rows if r["cv_auc"] >= best - 0.002)
    features = ranked[:best_k]
    metrics["feature_selection"] = {"candidates": len(cand), "grid": sel_rows, "selected_k": best_k,
                                    "rule": "smallest k within 0.002 of best grouped-CV AUC on dev"}

    # ------------------------------------------------------------------ final keep model (dev) and test
    groups = df.account_id.values
    keep = KeepModel(XGB, features, n_seeds=5).fit(df[dev], y[dev], groups[dev])
    df.loc[dev, "p_keep"] = keep.calibrator.transform(keep.oof_raw_, df.loc[dev, "acct_portfolio"].astype(str).values)
    df.loc[te, "p_keep"] = keep.predict(df[te])
    df.loc[dev, "p_raw"] = keep.oof_raw_
    df.loc[te, "p_raw"] = keep.predict_raw(df[te])
    log("final keep model trained")
    T = df[te].copy()
    metrics["keep_model_test"] = binary_report(T.kept, T.p_keep, groups=T.account_id.values)
    metrics["keep_model_test_uncalibrated"] = binary_report(T.kept, T.p_raw, ci=False)
    metrics["keep_model_dev_oof"] = binary_report(df.kept[dev], df.p_keep[dev], ci=False)
    log(f"TEST AUC {metrics['keep_model_test']['auc']:.4f} CI {metrics['keep_model_test']['auc_ci95']}")

    # per-segment calibration and fairness
    seg = {}
    for col in ("acct_portfolio", "acct_language", "acct_income_type", "channel", "acct_lender_type"):
        r_cal = segment_report(T, "kept", "p_keep", col)
        r_raw = segment_report(T, "kept", "p_raw", col)
        r_cal["ece_uncalibrated"] = r_raw["ece"].values
        seg[col] = r_cal.to_dict("records")
    metrics["segments_test"] = seg

    # label-definition robustness (same predictions, alternative labels)
    alt = []
    for g in CONFIG.label.sensitivity_grace_days:
        for f in (0.5, 0.9, 1.0):
            o = build_outcomes(ds.tables.ptps, ds.tables.payments, LabelConfig(grace_days=g, min_fraction=f)).set_index("ptp_id")
            alt.append({"grace_days": g, "min_fraction": f, "test_auc": safe_auc(T.ptp_id.map(o.kept), T.p_keep)})
    metrics["label_robustness_test"] = alt

    # ------------------------------------------------------------------ baselines and ablations (all on test)
    base = {"incumbent_fixed_rule": 0.5,
            "history_keep_rate_heuristic": safe_auc(T.kept, T.hist_keep_rate_smoothed)}
    casemix_feats = [c for c in cand if c.startswith(CASEMIX_PREFIXES)]
    casemix = KeepModel(XGB, casemix_feats, n_seeds=3).fit(df[dev], y[dev], groups[dev])
    df.loc[dev, "p_casemix"] = casemix.calibrator.transform(casemix.oof_raw_, df.loc[dev, "acct_portfolio"].astype(str).values)
    df.loc[te, "p_casemix"] = casemix.predict(df[te])
    base["account_history_terms_only"] = safe_auc(df.kept[te], df.p_casemix[te])

    def quick(feats, n_seeds=3):
        km = KeepModel(XGB, feats, n_seeds=n_seeds)
        km.categories = {c: sorted(df[c].dropna().astype(str).unique()) for c in feats
                         if isinstance(df[c].dtype, pd.CategoricalDtype)}
        mm = km._fit_raw(df[dev], y[dev])
        return safe_auc(y[te], km._predict_raw(mm, km._frame(df[te])))

    base["conversation_only"] = quick([c for c in cand if c.startswith(("cust_", "agent_", "borrower_", "confirm",
                                                                        "negotiation", "pressure", "voice_", "n_",
                                                                        "closing", "agree", "call_duration"))
                                       and not c.startswith(("agent_keep", "agent_resolved", "agent_ptps", "agent_silent"))])
    base["all_candidates_no_selection"] = quick(cand)
    base["selected_without_voice"] = quick([c for c in features if not c.startswith("voice_")])
    base["selected_without_agent_track_record"] = quick([c for c in features if not c.startswith(
        ("agent_keep", "agent_resolved", "agent_ptps", "agent_silent"))])
    Xn = pd.get_dummies(df[cand], dummy_na=False).astype(float)
    lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                       LogisticRegression(C=0.01, max_iter=5000)).fit(Xn[dev], y[dev])
    base["logistic_regression_all"] = safe_auc(y[te], lr.predict_proba(Xn[te])[:, 1])
    base["final_model"] = metrics["keep_model_test"]["auc"]
    metrics["baselines_test_auc"] = base
    log(f"baselines: {json.dumps({k: round(v, 4) for k, v in base.items()})}")

    # selection-bias check: PTPs from randomised dialling arm
    da = ds.tables.dial_attempts.set_index("attempt_id")
    T["dialling_arm"] = T.attempt_id.map(da.dialling_arm)
    rnd = T[T.dialling_arm == "random_contact_point"]
    metrics["random_arm_check"] = {"n": int(len(rnd)), "auc": safe_auc(rnd.kept, rnd.p_keep) if len(rnd) > 10 else None}

    # ------------------------------------------------------------------ explainability
    contrib = keep.contributions(T)
    fam = family_attribution(contrib)
    metrics["signal_family_attribution_test"] = (fam / fam.sum()).round(4).to_dict()
    top = contrib.drop(columns="__bias__").abs().mean().sort_values(ascending=False).head(25)
    metrics["top_features_mean_abs_shap"] = top.round(4).to_dict()

    # ------------------------------------------------------------------ operating point: broken PTPs flagged before parking
    thr = threshold_for_false_alarm(df.kept[dev].values, df.p_keep[dev].values, 0.20)
    metrics["broken_flagging_test"] = {
        "at_20pct_false_alarm_budget": broken_flag_recall(T.kept, T.p_keep, thr),
        "at_policy_low_threshold": broken_flag_recall(T.kept, T.p_keep, CONFIG.policy.low_credibility),
        "at_policy_high_threshold": broken_flag_recall(T.kept, T.p_keep, CONFIG.policy.high_credibility),
    }

    # ------------------------------------------------------------------ type model
    gold = df[df.annotated_type.notna()].copy()
    yg = gold.annotated_type.values
    wg = gold.annotator_confidence.map({"high": 1.0, "medium": 0.7, "low": 0.4}).values
    pool_mask = dev & df.annotated_type.isna() & (df.has_transcript == 1)
    pool = df[pool_mask]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RS)
    rows, cms = [], np.zeros((6, 6))
    for variant in ("supervised_only", "self_training", "self_training_cost_sensitive"):
        f1s, hard, accs = [], [], []
        for a, b in cv.split(gold, yg):
            tm = TypeModel(cost_sensitive=variant.endswith("cost_sensitive"))
            if variant == "supervised_only":
                tm.fit(gold.iloc[a], yg[a], wg[a])
            else:
                tm.fit(gold.iloc[a], yg[a], wg[a], pool=pool, kept_pool=pool.kept.values, kept_gold=gold.kept.values[a])
            pr = tm.predict(gold.iloc[b])
            f1s.append(f1_score(yg[b], pr, average="macro"))
            accs.append(accuracy_score(yg[b], pr))
            gi = yg[b] == "genuine_infeasible"
            hard.append(np.isin(pr[gi], list(HARSH_TYPES)).mean())
            if variant == "self_training_cost_sensitive":
                cms += confusion_matrix(yg[b], pr, labels=TYPE_CLASSES)
        rows.append({"variant": variant, "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                     "accuracy": float(np.mean(accs)), "hardship_to_harsh_rate": float(np.mean(hard))})
        log(f"type model {variant}: macroF1 {np.mean(f1s):.3f} hardship->harsh {np.mean(hard):.3f}")
    metrics["type_model_cv_5x5"] = rows
    metrics["type_model_confusion_cv"] = {"labels": TYPE_CLASSES, "matrix": (cms / 5).round(1).tolist()}
    # clean holdout: train on train/validation gold, evaluate on test gold
    gtr, gte = gold.split.ne("test").values, gold.split.eq("test").values
    tm_h = TypeModel().fit(gold[gtr], yg[gtr], wg[gtr], pool=pool, kept_pool=pool.kept.values, kept_gold=gold.kept.values[gtr])
    ph = tm_h.predict(gold[gte])
    metrics["type_model_test_holdout"] = {"n": int(gte.sum()), "macro_f1": float(f1_score(yg[gte], ph, average="macro")),
                                          "accuracy": float(accuracy_score(yg[gte], ph))}
    tmodel = TypeModel().fit(gold, yg, wg, pool=pool, kept_pool=pool.kept.values, kept_gold=gold.kept.values)
    proba = tmodel.predict_proba(df)
    df["ptp_type"] = tmodel.decide(proba)
    for i, c in enumerate(tmodel.classes):
        df[f"type_p_{c}"] = proba[:, i]
    metrics["type_model_pseudo_labels_used"] = tmodel.n_pseudo_
    T = df[te].copy()
    metrics["keep_rate_by_predicted_type_test"] = T.groupby("ptp_type").agg(
        n=("kept", "size"), keep_rate=("kept", "mean"), mean_p_keep=("p_keep", "mean")).round(4).to_dict("index")

    # ------------------------------------------------------------------ post-call update
    H = CONFIG.post_call.horizon_hours
    unres = df.post_resolved_before_horizon == 0
    post = PostCallModel().fit(df.p_keep[dev & unres].values, df[dev & unres], y[dev & unres])
    m_te = te & unres
    p_post = post.predict(df.p_keep[m_te].values, df[m_te])
    metrics["post_call_update_test"] = {
        "horizon_hours": H, "n_unresolved": int(m_te.sum()),
        "auc_in_call": safe_auc(y[m_te], df.p_keep[m_te]), "auc_after_post_call": safe_auc(y[m_te], p_post),
        "ece_after_post_call": expected_calibration_error(y[m_te], p_post),
        "coefficients": post.coefficients()}
    log(f"post-call: AUC {metrics['post_call_update_test']['auc_in_call']:.4f} -> {metrics['post_call_update_test']['auc_after_post_call']:.4f}")

    # ------------------------------------------------------------------ policy simulation (test)
    disp_rows = []
    for _, r in T.iterrows():
        f = r.to_dict()
        tp = {c: r[f"type_p_{c}"] for c in tmodel.classes}
        hs = hardship_flag(f, tp)
        d = disposition(float(r.p_keep), r.ptp_type, hs, float(r.terms_days_to_promise), r.captured_ts.to_pydatetime())
        disp_rows.append({"ptp_id": r.ptp_id, "hardship": hs, "treatment": d.treatment, "parking_days": d.parking_days,
                          "route": d.route, "rule": d.rule, "guardrails": ",".join(d.guardrails_applied),
                          "kept": r.kept, "fixed_parking_days": int(max(1, r.terms_days_to_promise)),
                          "annotated_type": r.annotated_type})
    D = pd.DataFrame(disp_rows)
    harsh = {"stronger_treatment_short_parking", "part_payment_follow_up"}
    D["harsh"] = D.treatment.isin(harsh)
    broken = D.kept == 0
    shorter = D.parking_days < D.fixed_parking_days
    metrics["policy_simulation_test"] = {
        "treatment_mix": D.treatment.value_counts().to_dict(),
        "hardship_flagged_share": float(D.hardship.mean()),
        "harsh_treatment_given_hardship": float(D.loc[D.hardship, "harsh"].mean()) if D.hardship.any() else 0.0,
        "annotated_infeasible_receiving_harsh": float(D.loc[D.annotated_type == "genuine_infeasible", "harsh"].mean())
        if (D.annotated_type == "genuine_infeasible").any() else None,
        "broken_ptps_flagged_before_parking": float((shorter | D.treatment.ne("park_until_due"))[broken].mean()),
        "broken_ptps_with_shorter_parking": float(shorter[broken].mean()),
        "kept_ptps_with_shorter_parking": float(shorter[~broken].mean()),
        "contact_days_released_on_broken": int((D.fixed_parking_days - D.parking_days)[broken].clip(lower=0).sum()),
        "contact_days_released_per_broken_ptp": float((D.fixed_parking_days - D.parking_days)[broken].clip(lower=0).mean()),
        "keep_rate_parked_until_due": float(D.loc[D.treatment == "park_until_due", "kept"].mean()),
        "keep_rate_part_payment_follow_up": float(D.loc[D.treatment == "part_payment_follow_up", "kept"].mean())
        if (D.treatment == "part_payment_follow_up").any() else None,
    }
    D.to_csv(P.reports / "policy_simulation_test.csv", index=False)

    # ------------------------------------------------------------------ streaming model
    pf = pf.merge(df[["ptp_id", "split"]].rename(columns={"split": "_s"}), on="ptp_id", how="left")
    pdev, pte = pf[pf._s != "test"].reset_index(drop=True), pf[pf._s == "test"].reset_index(drop=True)
    w = 1.0 / pdev.groupby("call_id").call_id.transform("size").values
    sfeat = [c for c in features if c in pf.columns]
    stream = KeepModel(XGB, sfeat, n_seeds=2).fit(pdev, pdev.kept.values, pdev.account_id.values, w=w)
    p_dev = stream.calibrator.transform(stream.oof_raw_, pdev.acct_portfolio.astype(str).values)
    p_te = stream.predict(pte)
    log("streaming model trained")
    curve = []
    for q in np.linspace(0.1, 1.0, 10):
        sel = pte.groupby("call_id").apply(lambda g: g.index[np.searchsorted(g.prefix_frac.values, q - 1e-9)]
                                           if (g.prefix_frac.values >= q - 1e-9).any() else g.index[-1])
        idx = sel.values
        curve.append({"fraction_of_call_heard": round(float(q), 2), "auc": safe_auc(pte.kept.values[idx], p_te[idx])})
    at_conf = pte[pte.confirm_reached == 1].groupby("call_id").head(1).index.values
    last = pte.groupby("call_id").tail(1).index.values
    pdev_final = pd.Series(p_dev).groupby(pdev.call_id).transform("last").values
    suff = SufficiencyModel(features=sfeat, categories=stream.categories).fit(pdev, p_dev, pdev_final, w)
    em_te = suff.predict(pte, p_te)
    dp = decision_points(pte, p_te, em_te, suff.delta)
    dp = dp.merge(pte.groupby("call_id").kept.first().rename("kept"), left_on="call_id", right_index=True)
    metrics["streaming_test"] = {
        "auc_by_fraction_heard": curve,
        "auc_at_first_confirmation_turn": safe_auc(pte.kept.values[at_conf], p_te[at_conf]),
        "auc_end_of_call_streaming_model": safe_auc(pte.kept.values[last], p_te[last]),
        "stopping_rule": {"delta": suff.delta, "auc_at_decision": safe_auc(dp.kept, dp.p_decision),
                          "fired_before_end_share": float(dp.fired_before_end.mean()),
                          "median_fraction_heard": float(dp.frac_heard.median()),
                          "mean_turns_remaining_for_prompt": float(dp.turns_remaining.mean())},
    }
    log(f"streaming: AUC@confirm {metrics['streaming_test']['auc_at_first_confirmation_turn']:.4f}, "
        f"stop rule AUC {metrics['streaming_test']['stopping_rule']['auc_at_decision']:.4f} at median "
        f"{metrics['streaming_test']['stopping_rule']['median_fraction_heard']:.2f} of call")

    # ------------------------------------------------------------------ agent integrity (all PTPs, OOF on dev)
    integ = agent_integrity_table(df)
    integ.to_csv(P.reports / "agent_integrity.csv", index=False)
    metrics["agent_integrity"] = {"agents": int(len(integ)), "flagged": integ.loc[integ.integrity_flag, "agent_id"].tolist()}
    # agent-recorded detection: AUC of silent-terms signal vs annotated agent_recorded
    g2 = df[df.annotated_type.notna()]
    metrics["agent_recorded_detection_annotated"] = {
        "auc_type_prob": safe_auc((g2.annotated_type == "agent_recorded_or_pushed").astype(int), g2.type_p_agent_recorded_or_pushed),
        "note": "in-sample for annotated rows; see type_model_cv_5x5 for out-of-sample type metrics"}

    # ------------------------------------------------------------------ randomised trial design
    p0 = float(y.mean())
    metrics["trial_design"] = {
        "baseline_keep_rate": p0,
        "per_arm_n_for_+3pp": two_proportion_sample_size(p0, 0.03),
        "per_arm_n_for_+5pp": two_proportion_sample_size(p0, 0.05),
        "per_arm_n_for_+3pp_clustered_icc0.05_m2": two_proportion_sample_size(p0, 0.03, icc=0.05, cluster_size=2.0),
    }

    # ------------------------------------------------------------------ persist
    artifacts = {"keep": keep, "stream": stream, "sufficiency": suff, "type": tmodel, "post": post,
                 "casemix": casemix, "features": features, "stream_features": sfeat,
                 "thresholds": {"false_alarm_20": thr, "low": CONFIG.policy.low_credibility,
                                "high": CONFIG.policy.high_credibility},
                 "versions": metrics["versions"]}
    joblib.dump(artifacts, P.artifacts / "ptp_credibility.joblib", compress=3)
    df.drop(columns=["customer_text", "agent_text"]).to_pickle(P.artifacts / "scored_ptps.pkl")
    pte.assign(p_stream=p_te, expected_move=em_te)[["call_id", "ptp_id", "prefix_turns", "prefix_frac", "confirm_reached",
                                                    "kept", "p_stream", "expected_move"]].to_csv(
        P.reports / "streaming_test_trajectories.csv", index=False)
    (P.reports / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2))
    log("artifacts and metrics written")

    plots.make_all(df, T, metrics, fam, integ, P.figures)
    log("figures written")


if __name__ == "__main__":
    main()
