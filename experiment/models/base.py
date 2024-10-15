from pathlib import Path
from typing import Tuple

import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import GroupKFold

from .sampling import negative_sampling_train_dataset


def load_train_dataset(cfg: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = negative_sampling_train_dataset(cfg, cfg.data.sampling)
    kfold = GroupKFold(n_splits=2)
    train_idx, valid_idx = next(kfold.split(train, groups=train["User"]))

    X_train = train[cfg.store.selected_features].iloc[train_idx]
    y_train = train[cfg.data.target].iloc[train_idx]
    X_valid = train[cfg.store.selected_features].iloc[valid_idx]
    y_valid = train[cfg.data.target].iloc[valid_idx]

    return X_train, y_train, X_valid, y_valid


def load_test_dataset(cfg: DictConfig) -> pd.DataFrame:
    test = pd.read_parquet(Path(cfg.data.path) / cfg.data.test)
    return test[cfg.store.selected_features], test[cfg.data.target]