"""Turns a filled application into a probability, a decision and its reasons."""
import json
import math
import time
from pathlib import Path

import numpy as np

from . import features as fx
from . import trees as tr

here = Path(__file__).resolve().parent

# which answer each model feature really comes from, used to group shap values
feature_group = {
    "age": "age", "age_over_policy": "age",
    "ltv": "financing", "ltv_vs_model": "financing", "ltv_vs_area": "financing",
    "down_payment": "financing", "loan_amount": "financing", "loan_vs_model": "financing",
    "vehicle_value": "price", "price_vs_model": "price", "price_share_area": "location",
    "monthly_income": "income", "loan_to_income": "income", "income_vs_area": "income",
    "zero_income": "income", "income_placeholder": "income", "income_round_5k": "income",
    "area_size": "location", "tier_rank": "location", "rate_final_tier": "location",
    "rate_pin2": "location", "rate_pin3": "location", "code_final_tier": "location",
    "rate_model": "vehicle", "rate_make": "vehicle", "rate_variant": "vehicle",
    "rate_product_code": "vehicle", "code_product_code": "vehicle", "code_make_code": "vehicle",
    "code_model_variant": "vehicle",
    "code_employment_type": "employment", "rate_employment_type": "employment",
    "code_past_loans_active": "active_loans",
    "code_qualifications": "education", "rate_qualifications": "education",
    "code_resident_type": "residence", "rate_resident_type": "residence",
    "code_gender": "gender", "rate_gender": "gender", "ltv_was_zero": "financing",
}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Scorer:
    def __init__(self, folder=None, better=False):
        folder = folder or here / "model"
        self.trees = tr.load_trees(folder / "trees.json")
        self.look = _read(folder / "lookups.json")
        self.rates = _read(folder / "rates.json")
        self.meta = _read(folder / "meta.json")
        self.better = None
        if better:
            from catboost import CatBoostClassifier
            local = here / "model_local"
            self.meta = _read(local / "meta.json")
            self.better = CatBoostClassifier()
            self.better.load_model(str(local / "catboost.cbm"))
        self.threshold = self.meta["threshold"]
        self.cal = self.meta["calibration"]
        self.version = self.meta["version"]
        self.kind = self.meta["kind"]
        self.width = len(self.meta["feature_names"])

    def prepare(self, rows):
        t = fx.blank_out(fx.tidy(rows), self.meta["dropped_fields"])
        return t, fx.score_matrix(t, self.look, self.rates)

    def _calibrate(self, raw):
        return 1 / (1 + np.exp(-(self.cal["a"] * raw + self.cal["b"])))

    def _raw(self, t, x):
        lgb_raw = np.array([tr.raw_score(self.trees, row) for row in x.tolist()])
        if self.better is None:
            return lgb_raw
        nums, cats = fx.catboost_rows(t, self.look)
        frame = [list(a) + list(b) for a, b in zip(nums.tolist(), zip(*cats))]
        p = np.clip(self.better.predict_proba(frame)[:, 1], 1e-6, 1 - 1e-6)
        return (lgb_raw + np.log(p / (1 - p))) / 2

    def batch_probability(self, rows):
        t, x = self.prepare(rows)
        return self._calibrate(self._raw(t, x))

    def probability(self, application):
        rows = {k: [v] for k, v in application.items()}
        return float(self.batch_probability(rows)[0])

    def explain(self, application):
        """Shap values grouped by the answer they came from, largest pull first."""
        rows = {k: [v] for k, v in application.items()}
        _, x = self.prepare(rows)
        phi = tr.contributions(self.trees, x[0].tolist(), self.width)
        groups = {}
        for name, value in zip(self.meta["feature_names"], phi):
            g = feature_group.get(name)
            if g:
                groups[g] = groups.get(g, 0.0) + value
        ranked = sorted(groups.items(), key=lambda kv: -abs(kv[1]))
        return [{"factor": g, "effect": round(v, 4)} for g, v in ranked if abs(v) >= 0.005]

    def what_if_down_payment(self, application, price):
        """Smallest extra down payment that would flip a decline, by re-scoring the model."""
        loan = float(application["Loan_Amount"])
        if price <= 0:
            return None
        lowest_ltv = 20.0
        for ltv_step in np.arange(math.floor(float(application["LTV"])) - 1, lowest_ltv, -1.0):
            new_loan = round(price * ltv_step / 100)
            changed = {**application, "Loan_Amount": new_loan, "LTV": round(ltv_step, 2)}
            if self.probability(changed) < self.threshold:
                return {"extra_down_payment": int(round((loan - new_loan) / 500.0) * 500),
                        "new_loan": int(new_loan), "new_ltv": round(float(ltv_step), 1)}
        return None

    def decide(self, application):
        start = time.perf_counter()
        p = self.probability(application)
        decision = "declined" if p >= self.threshold else "approved"
        ms = (time.perf_counter() - start) * 1000
        return {"probability": round(p, 5), "threshold": self.threshold, "decision": decision,
                "scoring_ms": round(ms, 2), "model_version": self.version, "model_kind": self.kind}


_cache = {}


def load_scorer(kind="hosted"):
    if kind not in _cache:
        _cache[kind] = Scorer(better=kind == "local")
    return _cache[kind]
