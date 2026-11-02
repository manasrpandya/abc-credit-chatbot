"""Copies the numbers the site shows into web/lib. Run: python -m pipeline.site_data"""
import json

from . import root

results = root / "report" / "results"
report = json.loads((results / "metrics.json").read_text())
site = {k: report[k] for k in ("rows", "dropped_fields", "cv", "cost_ratio", "threshold",
                               "threshold_sensitivity", "threshold_curve", "test", "fairness")}
site["search"] = json.loads((results / "field_search.json").read_text())
site["stability"] = json.loads((results / "stability.json").read_text())
(root / "web" / "lib" / "site_metrics.json").write_text(json.dumps(site))
