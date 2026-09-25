"""LightGBM ranker that turns per-race driver features into win probabilities."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS

LGBM_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)


def _make_groups(df: pd.DataFrame) -> list[int]:
    return df.groupby(["Year", "Round"], sort=False).size().tolist()


def train_ranker(train_df: pd.DataFrame) -> lgb.LGBMRanker:
    """Trains an LGBMRanker on races sorted by (Year, Round) so each race is
    one ranking group; label = relevance (higher for a better finish)."""
    train_df = train_df.sort_values(["Year", "Round"]).reset_index(drop=True)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["relevance"]
    groups = _make_groups(train_df)

    ranker = lgb.LGBMRanker(**LGBM_PARAMS)
    ranker.fit(X, y, group=groups)
    return ranker


def predict_win_probabilities(model: lgb.LGBMRanker, race_df: pd.DataFrame) -> pd.Series:
    """Raw ranker scores for a single race's grid, softmaxed into probabilities
    that sum to 1 across the entrants in `race_df`."""
    X = race_df[FEATURE_COLUMNS]
    scores = model.predict(X)
    scores = scores - scores.max()  # numerical stability
    exp_scores = np.exp(scores)
    return pd.Series(exp_scores / exp_scores.sum(), index=race_df.index)
