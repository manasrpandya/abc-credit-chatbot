import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import root
from creditbot.features import all_fields, tidy

raw_csv = root / "training_data.csv"


def read_raw(path=raw_csv):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    df = df.apply(lambda c: c.str.strip())
    # the file ends with one fully empty row
    return df[df["ID"] != ""].reset_index(drop=True)


def profile_groups(df):
    # identical applications must land on the same side of any split
    joined = df[all_fields].astype(str).agg("|".join, axis=1)
    return np.array([int(hashlib.md5(s.encode()).hexdigest()[:12], 16) for s in joined])


def load():
    df = read_raw()
    y = (df["Decision"].str.upper() == "DECLINED").astype(int).to_numpy()
    rows = {f: df[f].tolist() for f in all_fields}
    return df, tidy(rows), y, profile_groups(df)


def locked_split(y, groups, seed=2026):
    """80/20 split. The 20 is only touched by evaluate.py, once."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    dev, test = next(splitter.split(np.zeros(len(y)), y, groups))
    return dev, test


def dev_folds(y, groups, n=5, seed=7):
    return list(StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), y, groups))
