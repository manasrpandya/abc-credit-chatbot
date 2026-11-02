"""Score a csv in the training layout. Run: python -m pipeline.predict path/to/test.csv out.csv

Works with the hosted model by default, add --better for the lightgbm + catboost blend.
Unseen categories and blank cells are fine, see creditbot.features.
"""
import argparse

import pandas as pd

from creditbot.features import all_fields
from creditbot.scorer import load_scorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--better", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    df = df[df.drop(columns=["ID"], errors="ignore").apply(lambda c: c.str.strip()).ne("").any(axis=1)]
    rows = {f: df[f].tolist() if f in df else [""] * len(df) for f in all_fields}

    scorer = load_scorer("local" if args.better else "hosted")
    probs = scorer.batch_probability(rows)
    out = pd.DataFrame({"ID": df["ID"] if "ID" in df else range(1, len(df) + 1),
                        "declined_probability": probs.round(6),
                        "Decision": ["DECLINED" if p >= scorer.threshold else "APPROVED" for p in probs]})
    out.to_csv(args.out, index=False)
    print(out["Decision"].value_counts().to_dict(), "written to", args.out)


if __name__ == "__main__":
    main()
