"""Is one holdout score luck? Repeats the whole fit on fresh 80/20 splits, and shuffles the labels
as a leakage check: with random labels any honest pipeline has to land on an auc of 0.5.
Settings are already frozen, nothing here feeds back into the model.
Run: python -m pipeline.stability"""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import root
from .data import load, locked_split
from .settings import policy_drop, questions
from .train import final_drop_questions, fit_lgb
from creditbot import features as fx

trees = json.loads((root / "report" / "results" / "metrics.json").read_text())["trees"]["lightgbm"]


def one_run(t, y, groups, seed):
    dev, test = locked_split(y, groups, seed=seed)
    fields = list(policy_drop)
    for q in final_drop_questions:
        fields += questions[q]
    a, b = fx.blank_out(fx.take(t, dev), fields), fx.blank_out(fx.take(t, test), fields)
    look = fx.fit_lookups(a)
    x = fx.train_matrix(a, y[dev], look)
    xt = fx.score_matrix(b, look, fx.fit_rates(a, y[dev], look))
    p = fit_lgb(x, y[dev], trees).predict_proba(xt)[:, 1]
    return round(roc_auc_score(y[test], p), 4), round(average_precision_score(y[test], p), 4)


def main():
    _, t, y, groups = load()
    runs = [one_run(t, y, groups, s) for s in (1, 2, 3, 4, 5, 6, 7, 8)]
    rng = np.random.default_rng(99)
    shuffled = [one_run(t, rng.permutation(y), groups, s) for s in (1, 2)]
    aucs = [r[0] for r in runs]
    out = {"seeds": runs, "mean_auc": round(float(np.mean(aucs)), 4), "sd_auc": round(float(np.std(aucs)), 4),
           "min_auc": min(aucs), "max_auc": max(aucs), "mean_ap": round(float(np.mean([r[1] for r in runs])), 4),
           "shuffled_labels": shuffled}
    (root / "report" / "results" / "stability.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
