from pathlib import Path
from typing import Tuple

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedKFold


def load_train_dataset(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.train}.csv")
    kfold = StratifiedKFold(n_splits=cfg.data.n_splits, shuffle=True, random_state=cfg.data.seed)
    train_idx, valid_idx = next(kfold.split(train[cfg.store.selected_features], train[cfg.data.target]))

    X_train = train[cfg.store.selected_features].iloc[train_idx]
    y_train = train[cfg.data.target].iloc[train_idx]
    X_valid = train[cfg.store.selected_features].iloc[valid_idx]
    y_valid = train[cfg.data.target].iloc[valid_idx]

    return X_train, y_train, X_valid, y_valid


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    test = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.test}.csv")
    X_test = test[cfg.store.selected_features]
    return X_test
