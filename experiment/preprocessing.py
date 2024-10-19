from __future__ import annotations

from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig
from store import add_features, encode_train_categorical


@hydra.main(config_path="../config/", config_name="preprocessing", version_base="1.3.1")
def _main(cfg: DictConfig):
    train = pd.read_csv(Path(cfg.data.path) / "train.csv")
    test = pd.read_csv(Path(cfg.data.path) / "test.csv")

    train = encode_train_categorical(cfg, train)
    test = encode_train_categorical(cfg, test)

    train = add_features(train)
    test = add_features(test)

    train.to_csv(Path(cfg.data.path) / "train_features.csv", index=False)
    test.to_csv(Path(cfg.data.path) / "test_features.csv", index=False)


if __name__ == "__main__":
    _main()
