from pathlib import Path

import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].mean())
    df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].mean())
    return df


def encode_train_categorical(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in cfg.store.cat_features:
        df[col] = le.fit_transform(df[col])

    joblib.dump(le, Path(cfg.data.encoder) / "label_encoder.pkl")

    return df


def encode_test_categorical(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    le = joblib.load(Path(cfg.data.encoder) / "label_encoder.pkl")
    for col in cfg.store.cat_features:
        df[col] = le.transform(df[col])

    return df
