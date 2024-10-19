from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class BaseModel(ABC):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @abstractmethod
    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        raise NotImplementedError

    def save_model(self, model: Any, save_dir: Path) -> None:
        if isinstance(model, xgb.Booster):
            model.save_model(save_dir / f"{self.cfg.models.results}")

        elif isinstance(model, CatBoostClassifier):
            model.save_model(save_dir / f"{self.cfg.models.results}")

        elif isinstance(model, lgb.Booster):
            model.save_model(save_dir / f"{self.cfg.models.results}")

        else:
            joblib.dump(self.result, save_dir / f"{self.cfg.models.results}")

    def load_model(self, model_path: Path) -> Any:
        if self.cfg.models.name == "xgboost":
            model = xgb.Booster()
            model.load_model(model_path)

        elif self.cfg.models.name == "catboost":
            model = CatBoostClassifier()
            model.load_model(model_path)

        elif self.cfg.models.name == "lightgbm":
            model = lgb.Booster(model_file=model_path)

        else:
            model = joblib.load(model_path)

        return model

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> Any:
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    def _predict(self, model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(model, xgb.Booster):
            return model.predict(xgb.DMatrix(X))

        elif isinstance(model, CatBoostClassifier):
            return model.predict_proba(X)[:, 1]

        elif isinstance(model, lgb.Booster):
            return model.predict(X)

        else:
            raise ValueError("Model not supported")

    def run_cv_training(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        oof_preds = np.zeros(X.shape[0])
        models = {}
        kfold = StratifiedKFold(n_splits=self.cfg.data.n_splits, shuffle=True, random_state=self.cfg.data.seed)

        for fold, (train_idx, valid_idx) in enumerate(iterable=kfold.split(X, group=X["User"]), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = self.fit(X_train, y_train, X_valid, y_valid)
            oof_preds[valid_idx] = self._predict(model, X_valid)

            models[f"fold_{fold}"] = model

        del model, X_train, X_valid, y_train, y_valid
        gc.collect()

        print(f"CV Score: {roc_auc_score(y, oof_preds):.6f}")

        return oof_preds
