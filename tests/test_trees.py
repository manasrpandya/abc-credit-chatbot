"""Our pure python tree walker has to match lightgbm exactly, including shap values."""
import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from creditbot import trees as tr
from pipeline.export import compact_trees


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(3000, 8))
    x[rng.random(x.shape) < 0.1] = np.nan
    x[:, 7] = rng.normal(size=3000)  # no missing values in this one
    signal = np.nan_to_num(x[:, 0]) * 1.5 - np.nan_to_num(x[:, 1]) + x[:, 7] * np.nan_to_num(x[:, 2])
    y = (signal + rng.normal(size=3000) > 1).astype(int)
    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=15, min_child_samples=20, verbose=-1,
                               colsample_bytree=0.7, n_jobs=1)
    model.fit(x, y)
    test = rng.normal(size=(200, 8))
    test[rng.random(test.shape) < 0.2] = np.nan
    return model.booster_, compact_trees(model.booster_), test


def test_scores_match(fitted):
    booster, trees, test = fitted
    theirs = booster.predict(test, raw_score=True)
    ours = [tr.raw_score(trees, row) for row in test.tolist()]
    assert np.allclose(theirs, ours, atol=1e-9)


def test_shap_matches(fitted):
    booster, trees, test = fitted
    theirs = booster.predict(test[:40], pred_contrib=True)[:, :-1]
    ours = np.array([tr.contributions(trees, row, 8) for row in test[:40].tolist()])
    assert np.allclose(theirs, ours, atol=1e-7)
