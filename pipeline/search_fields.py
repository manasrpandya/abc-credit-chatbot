"""Backward search over questions, not columns. Writes report/results/field_search.json."""
import json

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import root
from .data import dev_folds, load, locked_split
from .settings import lgb_params, policy_drop, questions
from creditbot import features as fx

df, t, y, groups = load()
dev, _ = locked_split(y, groups)
t_dev, y_dev, g_dev = fx.take(t, dev), y[dev], groups[dev]
folds = dev_folds(y_dev, g_dev, n=3)
use = list(range(len(fx.feature_names)))


def blank_fields(asked_out):
    out = list(policy_drop)
    for q in asked_out:
        out += questions[q]
    return out


def cv_score(asked_out, drop_gender=True):
    fields = blank_fields(asked_out) if drop_gender else [f for q in asked_out for f in questions[q]]
    oof = np.zeros(len(y_dev))
    for tr, va in folds:
        tt = fx.blank_out(fx.take(t_dev, tr), fields)
        tv = fx.blank_out(fx.take(t_dev, va), fields)
        look = fx.fit_lookups(tt)
        xtr = fx.train_matrix(tt, y_dev[tr], look)[:, use]
        xva = fx.score_matrix(tv, look, fx.fit_rates(tt, y_dev[tr], look))[:, use]
        m = lgb.LGBMClassifier(**lgb_params, n_estimators=3000, n_jobs=6)
        m.fit(xtr, y_dev[tr], eval_set=[(xva, y_dev[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(xva)[:, 1]
    return round(roc_auc_score(y_dev, oof), 4), round(average_precision_score(y_dev, oof), 4)


results = {"with_gender": cv_score([], drop_gender=False), "all_questions": cv_score([])}
print(results, flush=True)

results["drop_one"] = {}
for q in questions:
    results["drop_one"][q] = cv_score([q])
    print(q, results["drop_one"][q], flush=True)

order = sorted(questions, key=lambda q: -results["drop_one"][q][0])
path = [{"asked": len(questions), "removed": None, "auc": results["all_questions"][0],
         "ap": results["all_questions"][1]}]
gone = []
for q in order[:-1]:
    gone.append(q)
    auc, ap = cv_score(gone)
    path.append({"asked": len(questions) - len(gone), "removed": q, "auc": auc, "ap": ap})
    print(path[-1], flush=True)
results["path"] = path

out = root / "report" / "results"
out.mkdir(parents=True, exist_ok=True)
(out / "field_search.json").write_text(json.dumps(results, indent=1))
