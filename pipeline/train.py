"""Compare models on the dev rows, fix a threshold, evaluate once on the locked rows,
then refit on everything and export. Run: python -m pipeline.train
"""
import json
import time

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import root
from .data import dev_folds, load, locked_split
from .export import compact_trees, write_json
from .settings import always_down, always_up, cat_params, lgb_params, policy_drop, questions
from creditbot import features as fx

results_dir = root / "report" / "results"
model_dir = root / "server" / "creditbot" / "model"
local_dir = root / "server" / "creditbot" / "model_local"

# cost of approving a bad case vs turning away a good one, see report section 6
cost_ratio = 8.0
threads = 6


# decided from report/results/field_search.json, see report section 4
final_drop_questions = ["residence", "education"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(raw, y):
    model = LogisticRegression(C=1e6).fit(raw.reshape(-1, 1), y)
    return {"a": float(model.coef_[0][0]), "b": float(model.intercept_[0])}


def apply_platt(raw, c):
    return 1 / (1 + np.exp(-(c["a"] * raw + c["b"])))


def choose_threshold(p, y, ratio):
    grid = np.linspace(0.01, 0.6, 591)
    cost = [ratio * np.sum((p < g) & (y == 1)) + np.sum((p >= g) & (y == 0)) for g in grid]
    return float(grid[int(np.argmin(cost))])


def at_threshold(p, y, cut):
    flag = p >= cut
    tp, fp = int(np.sum(flag & (y == 1))), int(np.sum(flag & (y == 0)))
    fn, tn = int(np.sum(~flag & (y == 1))), int(np.sum(~flag & (y == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"threshold": round(cut, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round((tp + tn) / len(y), 4), "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
            "declined_share": round(float(flag.mean()), 4),
            "approval_recall": round(tn / max(tn + fp, 1), 4)}


def bootstrap(p, y, cut, rounds=500, seed=5):
    rng = np.random.default_rng(seed)
    out = {"auc": [], "ap": [], "precision": [], "recall": [], "accuracy": []}
    for _ in range(rounds):
        i = rng.integers(0, len(y), len(y))
        out["auc"].append(roc_auc_score(y[i], p[i]))
        out["ap"].append(average_precision_score(y[i], p[i]))
        m = at_threshold(p[i], y[i], cut)
        for k in ("precision", "recall", "accuracy"):
            out[k].append(m[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in out.items()}


def direction(name):
    return 1 if name in always_up else -1 if name in always_down else 0


def fit_lgb(x, y, trees, seed=7, extra=None, eval_set=None):
    params = {**lgb_params, "random_state": seed, "n_jobs": threads,
              "monotone_constraints": [direction(n) for n in fx.feature_names],
              "monotone_constraints_method": "advanced", **(extra or {})}
    m = lgb.LGBMClassifier(**params, n_estimators=trees)
    if eval_set is None:
        m.fit(x, y)
    else:
        m.fit(x, y, eval_set=[eval_set], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


def cat_frame(t, look):
    nums, cats = fx.catboost_rows(t, look)
    return [list(a) + list(b) for a, b in zip(nums.tolist(), zip(*cats))], nums.shape[1]


def fit_cat(t, y, look, iters, seed=8, eval_part=None):
    frame, width = cat_frame(t, look)
    idx = list(range(width, width + len(fx.catboost_cats)))
    params = {**cat_params, "random_seed": seed, "thread_count": threads}
    m = CatBoostClassifier(iterations=iters, cat_features=idx, **params)
    if eval_part is None:
        params.pop("od_type"), params.pop("od_wait")
        m = CatBoostClassifier(iterations=iters, cat_features=idx, **params)
        m.fit(frame, y)
    else:
        m.fit(frame, y, eval_set=eval_part, use_best_model=True)
    return m


def blanked(t, dropped):
    return fx.blank_out(t, dropped)


def cv_models(t, y, folds, drop_fields):
    """Out of fold predictions per candidate. Everything label based is refit inside each fold."""
    names = ["logistic", "lightgbm", "lightgbm_weighted", "catboost"]
    oof = {n: np.zeros(len(y)) for n in names}
    best = {"lightgbm": [], "catboost": []}
    for k, (tr, va) in enumerate(folds):
        start = time.time()
        tt = blanked(fx.take(t, tr), drop_fields)
        tv = blanked(fx.take(t, va), drop_fields)
        look = fx.fit_lookups(tt)
        xtr = fx.train_matrix(tt, y[tr], look)
        xva = fx.score_matrix(tv, look, fx.fit_rates(tt, y[tr], look))

        base = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.5))
        oof["logistic"][va] = base.fit(xtr, y[tr]).predict_proba(xva)[:, 1]

        m = fit_lgb(xtr, y[tr], 3000, eval_set=(xva, y[va]))
        best["lightgbm"].append(m.best_iteration_)
        oof["lightgbm"][va] = m.predict_proba(xva)[:, 1]

        m = fit_lgb(xtr, y[tr], m.best_iteration_, extra={"scale_pos_weight": 18.0})
        oof["lightgbm_weighted"][va] = m.predict_proba(xva)[:, 1]

        frame_va, _ = cat_frame(tv, look)
        c = fit_cat(tt, y[tr], look, 4000, eval_part=(frame_va, y[va]))
        best["catboost"].append(c.get_best_iteration())
        oof["catboost"][va] = c.predict_proba(frame_va)[:, 1]
        print(f"fold {k} done in {time.time() - start:.0f}s", flush=True)
    return oof, best


def group_table(p, y, cut, groups):
    out = {}
    for name, values in groups.items():
        rows = {}
        for v in sorted(set(values)):
            mask = values == v
            if v and mask.sum() >= 100:
                m = at_threshold(p[mask], y[mask], cut)
                rows[str(v)] = {"n": int(mask.sum()), "actual_decline": round(float(y[mask].mean()), 4),
                                "flagged": m["declined_share"], "recall": m["recall"],
                                "precision": m["precision"]}
        out[name] = rows
    return out


def main():
    df, t, y, groups = load()
    dev, test = locked_split(y, groups)
    assert not set(groups[dev]) & set(groups[test])
    t_dev, y_dev = fx.take(t, dev), y[dev]
    folds = dev_folds(y_dev, groups[dev])

    drop_fields = list(policy_drop)
    for q in final_drop_questions:
        drop_fields += questions[q]

    oof, best = cv_models(t_dev, y_dev, folds, drop_fields)
    oof["blend"] = 1 / (1 + np.exp(-(logit(oof["lightgbm"]) + logit(oof["catboost"])) / 2))
    table = {n: {"auc": round(roc_auc_score(y_dev, p), 4),
                 "ap": round(average_precision_score(y_dev, p), 4)} for n, p in oof.items()}
    print(json.dumps(table, indent=1), flush=True)

    lgb_trees = int(np.median(best["lightgbm"]) * 1.1)
    cat_trees = int(np.median(best["catboost"]) * 1.1)

    # calibration and threshold from out of fold scores only
    hosted_cal = fit_platt(logit(oof["lightgbm"]), y_dev)
    blend_cal = fit_platt(logit(oof["blend"]), y_dev)
    hosted_p = apply_platt(logit(oof["lightgbm"]), hosted_cal)
    blend_p = apply_platt(logit(oof["blend"]), blend_cal)
    cut = choose_threshold(hosted_p, y_dev, cost_ratio)
    blend_cut = choose_threshold(blend_p, y_dev, cost_ratio)

    sensitivity = []
    for r in (2, 3, 5, 8, 10, 15):
        c = choose_threshold(hosted_p, y_dev, r)
        sensitivity.append({"ratio": r, **at_threshold(hosted_p, y_dev, c)})
    grid = np.linspace(0.01, 0.5, 99)
    curve = [{"threshold": round(float(g), 3), **{k: v for k, v in at_threshold(hosted_p, y_dev, g).items()
                                                 if k in ("precision", "recall", "accuracy", "declined_share")},
              "cost": round((cost_ratio * np.sum((hosted_p < g) & (y_dev == 1))
                             + np.sum((hosted_p >= g) & (y_dev == 0))) / len(y_dev), 4)} for g in grid]

    # the single look at the locked rows, settings above are frozen by now
    t_fit = blanked(t_dev, drop_fields)
    t_test = blanked(fx.take(t, test), drop_fields)
    y_test = y[test]
    look = fx.fit_lookups(t_fit)
    rates = fx.fit_rates(t_fit, y_dev, look)
    x_fit = fx.train_matrix(t_fit, y_dev, look)
    x_test = fx.score_matrix(t_test, look, rates)

    hosted = fit_lgb(x_fit, y_dev, lgb_trees)
    raw_test = hosted.predict_proba(x_test)[:, 1]
    p_test = apply_platt(logit(raw_test), hosted_cal)

    cat_model = fit_cat(t_fit, y_dev, look, cat_trees)
    frame_test, _ = cat_frame(t_test, look)
    cat_test = cat_model.predict_proba(frame_test)[:, 1]
    blend_test = apply_platt((logit(raw_test) + logit(cat_test)) / 2, blend_cal)

    naive = at_threshold(np.zeros(len(y_test)), y_test, 0.5)
    other_costs = [{"ratio": r["ratio"], **at_threshold(p_test, y_test, r["threshold"])} for r in sensitivity]
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    step = max(len(prec) // 150, 1)

    report = {
        "rows": {"total": int(len(y)), "dev": int(len(dev)), "test": int(len(test)),
                 "decline_rate": round(float(y.mean()), 4)},
        "dropped_fields": drop_fields,
        "cv": table,
        "trees": {"lightgbm": lgb_trees, "catboost": cat_trees},
        "cost_ratio": cost_ratio,
        "threshold": {"hosted": round(cut, 4), "local": round(blend_cut, 4)},
        "calibration": {"hosted": hosted_cal, "local": blend_cal},
        "threshold_sensitivity": sensitivity,
        "threshold_curve": curve,
        "test": {
            "always_approve": naive,
            "other_costs": other_costs,
            "hosted": {"auc": round(roc_auc_score(y_test, p_test), 4),
                       "ap": round(average_precision_score(y_test, p_test), 4),
                       "brier": round(float(np.mean((p_test - y_test) ** 2)), 5),
                       **at_threshold(p_test, y_test, cut),
                       "ci95": bootstrap(p_test, y_test, cut)},
            "local": {"auc": round(roc_auc_score(y_test, blend_test), 4),
                      "ap": round(average_precision_score(y_test, blend_test), 4),
                      **at_threshold(blend_test, y_test, blend_cut),
                      "ci95": bootstrap(blend_test, y_test, blend_cut)},
        },
        "pr_curve": [{"precision": round(float(a), 4), "recall": round(float(b), 4)}
                     for a, b in zip(prec[::step], rec[::step])],
        "fairness": group_table(p_test, y_test, cut, {
            "gender": fx.take(t, test)["Gender"],
            "tier": fx.repair(fx.take(t, test), look)["Final_Tier"],
            "employment": fx.take(t, test)["Employment_Type"]}),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["test"], indent=1), flush=True)

    # final fit on all rows for the hidden set and the chatbot
    t_all = blanked(t, drop_fields)
    look = fx.fit_lookups(t_all)
    rates = fx.fit_rates(t_all, y, look)
    x_all = fx.train_matrix(t_all, y, look)
    final = fit_lgb(x_all, y, lgb_trees)

    meta = {"version": time.strftime("%Y.%m.%d"), "threshold": round(cut, 4), "calibration": hosted_cal,
            "dropped_fields": drop_fields, "feature_names": fx.feature_names, "kind": "lightgbm"}
    write_json(model_dir / "trees.json", compact_trees(final.booster_))
    write_json(model_dir / "lookups.json", look)
    write_json(model_dir / "rates.json", rates)
    write_json(model_dir / "meta.json", meta)

    final_cat = fit_cat(t_all, y, look, cat_trees)
    local_dir.mkdir(parents=True, exist_ok=True)
    final_cat.save_model(str(local_dir / "catboost.cbm"))
    write_json(local_dir / "meta.json", {**meta, "kind": "lightgbm+catboost",
                                         "threshold": round(blend_cut, 4), "calibration": blend_cal})
    print("exported to", model_dir, "and", local_dir)


if __name__ == "__main__":
    main()
