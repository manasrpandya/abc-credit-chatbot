"""Cleaning and feature building, shared by training and the live service.

Only numpy on purpose: the same code path scores a csv of 100k rows and one
chat session, so training and serving can't drift apart, and the hosted
function stays small.
"""
import math
import re

import numpy as np

text_fields = ["Qualifications", "Employment_Type", "Pincode", "Gender", "Product_Code",
               "Resident_Type", "Final_Tier", "Make_Code", "Model_Description",
               "Model_Variant", "PAST_LOANS_ACTIVE"]
number_fields = ["Age", "Loan_Amount", "LTV", "Net_salary"]
all_fields = text_fields + number_fields

# anything below this is a placeholder (1, 2, 25, 100 show up a lot), not a real monthly income
lowest_real_income = 1000
policy_max_age = 60

simple_rates = ["Employment_Type", "Qualifications", "Resident_Type", "Gender",
                "Final_Tier", "Product_Code"]
coded_fields = simple_rates + ["Make_Code", "Model_Variant", "PAST_LOANS_ACTIVE"]
smoothing = 30.0


def fix_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value)
    # some dashes in the file are utf-8 read as cp1252
    s = s.replace("â€“", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip().upper()


def fix_variant(s):
    return re.sub(r"(\d+)\s*CC", r"\1 CC", s)


def to_number(value):
    if value is None:
        return math.nan
    try:
        return float(str(value).strip().replace(",", ""))
    except ValueError:
        return math.nan


def tidy(rows):
    """rows: dict of field -> list of raw values. Missing fields are treated as blank."""
    n = len(next(iter(rows.values())))
    t = {}
    for f in text_fields:
        t[f] = np.array([fix_text(v) for v in rows.get(f, [""] * n)], dtype=object)
    for f in number_fields:
        t[f] = np.array([to_number(v) for v in rows.get(f, [math.nan] * n)], dtype=float)

    t["Model_Variant"] = np.array([fix_variant(v) for v in t["Model_Variant"]], dtype=object)

    pins = t["Pincode"]
    good = np.array([len(p) == 6 and p.isdigit() for p in pins])
    t["Pincode"] = np.where(good, pins, "")

    # a zero ltv is impossible on a financed vehicle
    t["ltv_was_zero"] = (t["LTV"] <= 0).astype(float)
    t["LTV"] = np.where(t["LTV"] > 0, t["LTV"], np.nan)

    pay = t["Net_salary"]
    fake = (pay > 0) & (pay < lowest_real_income)
    t["income_placeholder"] = fake.astype(float)
    t["Net_salary"] = np.where(fake, np.nan, pay)
    return t


def take(t, idx):
    return {k: v[idx] for k, v in t.items()}


def blank_out(t, fields):
    out = dict(t)
    for f in fields:
        if f in number_fields:
            out[f] = np.full(len(t[f]), np.nan)
        else:
            out[f] = np.full(len(t[f]), "", dtype=object)
    return out


def _medians(keys, values):
    out = {}
    for k in sorted(set(keys)):
        if k:
            v = values[keys == k]
            v = v[~np.isnan(v)]
            if len(v):
                out[k] = round(float(np.median(v)), 3)
    return out


def _mode_table(keys, values):
    counts = {}
    for k, v in zip(keys, values):
        if k and v:
            counts.setdefault(k, {}).setdefault(v, 0)
            counts[k][v] += 1
    return {k: max(c, key=c.get) for k, c in counts.items()}


def prefix(pins, size):
    return np.array([p[:size] for p in pins], dtype=object)


def fit_lookups(t):
    """Everything here is computed without labels, on training rows only."""
    models = t["Model_Description"]
    value = t["Loan_Amount"] / (t["LTV"] / 100)
    pin2, pin3 = prefix(t["Pincode"], 2), prefix(t["Pincode"], 3)

    model_price = _medians(models, value)
    price_share = value / np.array([model_price.get(m, np.nan) for m in models])

    tiers = sorted({x for x in t["Final_Tier"] if x})
    codes = {}
    for f in coded_fields:
        codes[f] = {v: i for i, v in enumerate(sorted({x for x in t[f] if x}))}

    return {
        "pin_tier": _mode_table(t["Pincode"], t["Final_Tier"]),
        "pin3_tier": _mode_table(pin3, t["Final_Tier"]),
        "model_make": _mode_table(models, t["Make_Code"]),
        "model_variant": _mode_table(models, t["Model_Variant"]),
        "model_product": _mode_table(models, t["Product_Code"]),
        "model_price": model_price,
        "model_size": {k: int(v) for k, v in zip(*np.unique(models[models != ""], return_counts=True))},
        "model_ltv": _medians(models, t["LTV"]),
        "model_loan": _medians(models, t["Loan_Amount"]),
        "pin3_ltv": _medians(pin3, t["LTV"]),
        "pin3_income": _medians(pin3, t["Net_salary"]),
        "pin3_size": {k: int(v) for k, v in zip(*np.unique(pin3[pin3 != ""], return_counts=True))},
        "pin2_price_share": _medians(pin2, price_share),
        "price_all": round(float(np.nanmedian(value)), 2),
        "ltv_all": round(float(np.nanmedian(t["LTV"])), 2),
        "income_all": round(float(np.nanmedian(t["Net_salary"])), 2),
        "tier_rank": {x: int(x[:2]) for x in tiers if x[:2].isdigit()},
        "codes": codes,
    }


def repair(t, look):
    """Fill fields that are fully determined by another field."""
    out = dict(t)
    n = len(t["Pincode"])

    def fill(field, key_field, table):
        cur = out[field]
        keys = t[key_field]
        out[field] = np.array([cur[i] or table.get(keys[i], "") for i in range(n)], dtype=object)

    fill("Make_Code", "Model_Description", look["model_make"])
    fill("Model_Variant", "Model_Description", look["model_variant"])
    fill("Product_Code", "Model_Description", look["model_product"])

    tier = out["Final_Tier"]
    pins = t["Pincode"]
    out["Final_Tier"] = np.array(
        [tier[i] or look["pin_tier"].get(pins[i]) or look["pin3_tier"].get(pins[i][:3], "")
         for i in range(n)], dtype=object)

    # ltv of zero or blank: estimate it from what this model usually costs
    price = np.array([look["model_price"].get(m, look["price_all"]) for m in t["Model_Description"]])
    guess = np.round(out["Loan_Amount"] / price * 100, 2)
    guess = np.minimum(guess, 100)
    out["LTV"] = np.where(np.isnan(out["LTV"]), guess, out["LTV"])
    return out


base_names = ["age", "loan_amount", "ltv", "monthly_income", "vehicle_value", "down_payment",
              "loan_to_income", "price_vs_model", "ltv_vs_model", "loan_vs_model",
              "ltv_vs_area", "income_vs_area", "price_share_area", "area_size", "tier_rank",
              "zero_income", "income_placeholder", "ltv_was_zero", "age_over_policy",
              "income_round_5k"] + ["code_" + f.lower() for f in coded_fields]


def base_matrix(t, look):
    t = repair(t, look)
    models = t["Model_Description"]
    pin2, pin3 = prefix(t["Pincode"], 2), prefix(t["Pincode"], 3)

    def lookup(table, keys, default=np.nan):
        return np.array([table.get(k, default) for k in keys], dtype=float)

    age, loan, ltv, pay = t["Age"], t["Loan_Amount"], t["LTV"], t["Net_salary"]
    value = loan / (ltv / 100)
    usable_pay = np.where(pay > 0, pay, np.nan)
    tier_rank = lookup(look["tier_rank"], t["Final_Tier"])

    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [
            age, loan, ltv, pay, value, value - loan,
            loan / usable_pay,
            value / lookup(look["model_price"], models),
            ltv - lookup(look["model_ltv"], models),
            loan / lookup(look["model_loan"], models),
            ltv - lookup(look["pin3_ltv"], pin3),
            pay / lookup(look["pin3_income"], pin3),
            lookup(look["pin2_price_share"], pin2),
            lookup(look["pin3_size"], pin3, 0.0),
            tier_rank,
            (pay == 0).astype(float),
            t["income_placeholder"],
            t["ltv_was_zero"],
            (age > policy_max_age).astype(float),
            np.where(np.isnan(pay), np.nan, (pay % 5000 == 0).astype(float)),
        ]
    for f in coded_fields:
        cols.append(lookup(look["codes"][f], t[f]))
    return np.column_stack(cols), t


rate_names = ["rate_" + f.lower() for f in simple_rates] + \
             ["rate_make", "rate_variant", "rate_model", "rate_pin2", "rate_pin3"]


def _rate_table(keys, y, prior_for):
    """Smoothed decline rate per key. prior_for(key) supplies the parent level's rate."""
    out = {}
    uniq, inv = np.unique(keys, return_inverse=True)
    n = np.bincount(inv)
    hits = np.bincount(inv, weights=y)
    for k, ni, hi in zip(uniq, n, hits):
        if k:
            p = prior_for(k)
            out[k] = round(float((hi + smoothing * p) / (ni + smoothing)), 5)
    return out


def fit_rates(t, y, look):
    """Label based, so only ever fit on rows that are not being encoded."""
    t = repair(t, look)
    base = float(np.mean(y))
    rates = {"all": round(base, 5)}
    for f in simple_rates:
        rates[f] = _rate_table(t[f], y, lambda k: base)

    rates["make"] = _rate_table(t["Make_Code"], y, lambda k: base)
    make_of = {}
    for v, m in zip(t["Model_Variant"], t["Make_Code"]):
        make_of[v] = m
    rates["variant"] = _rate_table(t["Model_Variant"], y,
                                   lambda k: rates["make"].get(make_of.get(k), base))
    rates["model"] = _rate_table(t["Model_Description"], y,
                                 lambda k: rates["variant"].get(look["model_variant"].get(k), base))

    pin2, pin3 = prefix(t["Pincode"], 2), prefix(t["Pincode"], 3)
    rates["pin2"] = _rate_table(pin2, y, lambda k: base)
    rates["pin3"] = _rate_table(pin3, y, lambda k: rates["pin2"].get(k[:2], base))
    return rates


def rate_matrix(t, rates, look):
    t = repair(t, look)
    base = rates["all"]
    pins = t["Pincode"]

    def simple(table, keys):
        return np.array([table.get(k, base) for k in keys], dtype=float)

    cols = [simple(rates[f], t[f]) for f in simple_rates]
    make = simple(rates["make"], t["Make_Code"])
    variant = np.array([rates["variant"].get(v, m) for v, m in zip(t["Model_Variant"], make)])
    model = np.array([rates["model"].get(k, v) for k, v in zip(t["Model_Description"], variant)])
    pin2 = np.array([rates["pin2"].get(p[:2], base) for p in pins])
    pin3 = np.array([rates["pin3"].get(p[:3], b) for p, b in zip(pins, pin2)])
    cols += [make, variant, model, pin2, pin3]
    return np.column_stack(cols)


def out_of_fold_rates(t, y, look, folds, seed=11):
    """Rate features for training rows, each row encoded by the other folds only."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(y))
    out = np.zeros((len(y), len(rate_names)))
    for k in range(folds):
        hold = np.where(fold == k)[0]
        keep = np.where(fold != k)[0]
        rates = fit_rates(take(t, keep), y[keep], look)
        out[hold] = rate_matrix(take(t, hold), rates, look)
    return out


feature_names = base_names + rate_names


def train_matrix(t, y, look, folds=5):
    base, _ = base_matrix(t, look)
    return np.hstack([base, out_of_fold_rates(t, y, look, folds)])


def score_matrix(t, look, rates):
    base, _ = base_matrix(t, look)
    return np.hstack([base, rate_matrix(t, rates, look)])


catboost_cats = ["Employment_Type", "Qualifications", "Resident_Type", "Gender", "Final_Tier",
                 "Product_Code", "Make_Code", "Model_Variant", "Model_Description",
                 "PAST_LOANS_ACTIVE", "pin3", "pin2"]


def catboost_rows(t, look):
    """Numbers from base_matrix plus raw categories, since catboost encodes those itself."""
    base, fixed = base_matrix(t, look)
    keep = [i for i, name in enumerate(base_names) if not name.startswith("code_")]
    cats = []
    for f in catboost_cats:
        if f == "pin3":
            cats.append(prefix(fixed["Pincode"], 3))
        elif f == "pin2":
            cats.append(prefix(fixed["Pincode"], 2))
        else:
            cats.append(fixed[f])
    return base[:, keep], [np.array([x or "missing" for x in c], dtype=object) for c in cats]
