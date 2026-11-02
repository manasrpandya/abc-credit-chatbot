"""The ways this kind of project usually fools itself."""
import numpy as np
import pytest

from creditbot import features as fx


def small_data(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    pins = rng.choice([f"{p}000" for p in range(110, 140)], n)
    rows = {
        "Pincode": list(pins),
        "Age": rng.integers(18, 60, n).tolist(),
        "Loan_Amount": rng.integers(40000, 140000, n).tolist(),
        "LTV": rng.uniform(40, 95, n).round(2).tolist(),
        "Net_salary": rng.integers(15000, 60000, n).tolist(),
        "Employment_Type": rng.choice(["SAL", "SEP", "AGR"], n).tolist(),
        "Model_Description": rng.choice(["A", "B", "C"], n).tolist(),
        "Make_Code": rng.choice(["M1", "M2"], n).tolist(),
        "Model_Variant": ["V"] * n,
        "Product_Code": ["MC"] * n,
        "Final_Tier": ["07 Rural"] * n,
        "PAST_LOANS_ACTIVE": ["NO_PAST_LOANS"] * n,
        "Qualifications": ["HSC"] * n,
        "Gender": ["Male"] * n,
        "Resident_Type": ["O"] * n,
    }
    y = rng.integers(0, 2, n)  # pure noise
    return fx.tidy(rows), y


def test_rate_features_carry_nothing_on_pure_noise():
    """With random labels, out of fold rates must not predict the label on the training rows.
    Plain target encoding fails this badly, which is exactly the leak we're avoiding."""
    t, y = small_data()
    look = fx.fit_lookups(t)
    x = fx.out_of_fold_rates(t, y, look, folds=5)
    # a row in the fold sees rates fit without it, so the correlation can even go slightly negative
    for j in range(x.shape[1]):
        if np.std(x[:, j]) > 0:
            assert np.corrcoef(x[:, j], y)[0, 1] < 0.05


def test_naive_encoding_would_have_leaked():
    """Keeps the test above honest: the same check does catch the leaky version."""
    t, y = small_data()
    look = fx.fit_lookups(t)
    leaky = fx.rate_matrix(t, fx.fit_rates(t, y, look), look)
    col = fx.rate_names.index("rate_pin3")
    assert np.corrcoef(leaky[:, col], y)[0, 1] > 0.05


def test_lookups_use_no_labels():
    t, y = small_data()
    assert fx.fit_lookups(t) == fx.fit_lookups(t)  # takes no y at all, by signature
    import inspect
    assert list(inspect.signature(fx.fit_lookups).parameters) == ["t"]


def test_nothing_in_the_features_comes_from_id_or_decision():
    assert "ID" not in fx.all_fields
    assert "Decision" not in fx.all_fields


def test_full_pincode_is_not_stored_in_rates():
    t, y = small_data()
    rates = fx.fit_rates(t, y, fx.fit_lookups(t))
    assert "pin6" not in rates
    assert all(len(k) == 3 for k in rates["pin3"])


def test_split_keeps_identical_profiles_together():
    pipeline_data = pytest.importorskip("pipeline.data")
    n = 5000
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 1500, n)
    y = rng.integers(0, 2, n)
    dev, test = pipeline_data.locked_split(y, groups)
    assert not set(groups[dev]) & set(groups[test])
    assert len(set(dev) | set(test)) == n


def test_unseen_categories_do_not_break_scoring():
    t, y = small_data(800)
    look = fx.fit_lookups(t)
    rates = fx.fit_rates(t, y, look)
    fresh = fx.tidy({"Pincode": ["999999"], "Model_Description": ["A BRAND NEW BIKE"], "Age": [" "],
                     "Loan_Amount": ["50000"], "LTV": ["0"], "Net_salary": ["100"],
                     "Employment_Type": ["ASTRONAUT"], "Final_Tier": ["09 Moon"]})
    x = fx.score_matrix(fresh, look, rates)
    assert x.shape == (1, len(fx.feature_names))
    assert not np.isinf(x).any()


def test_cleaning_rules():
    t = fx.tidy({"Model_Description": ["STARCITY PLUS ES BSVI â€“ MONO TONE"], "Model_Variant": ["125CC DISC"],
                 "LTV": ["0"], "Net_salary": ["100"], "Age": [" "], "Pincode": ["12"], "Loan_Amount": ["1000"]})
    assert t["Model_Description"][0] == "STARCITY PLUS ES BSVI - MONO TONE"
    assert t["Model_Variant"][0] == "125 CC DISC"
    assert np.isnan(t["LTV"][0]) and t["ltv_was_zero"][0] == 1
    assert np.isnan(t["Net_salary"][0]) and t["income_placeholder"][0] == 1
    assert np.isnan(t["Age"][0]) and t["Pincode"][0] == ""


def test_zero_income_stays_real():
    t = fx.tidy({"Net_salary": ["0"]})
    assert t["Net_salary"][0] == 0 and t["income_placeholder"][0] == 0
