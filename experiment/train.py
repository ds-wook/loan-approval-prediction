from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from data import load_train_dataset
from model import build_model
from omegaconf import DictConfig


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        # load dataset
        X_train, y_train, X_valid, y_valid = load_train_dataset(cfg)

        # choose trainer
        trainer = build_model(cfg)

        # train model
        model = trainer.fit(X_train, y_train, X_valid, y_valid)

        # save model
        trainer.save_model(model, Path(cfg.models.path))


if __name__ == "__main__":
    _main()
