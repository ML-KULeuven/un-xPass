"""Model architectures."""
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xgboost as xgb
from rich.progress import track
from sklearn.model_selection import cross_val_score, train_test_split
from torch.utils.data import DataLoader, Subset, random_split

from unxpass.config import logger as log
from unxpass.datasets import PassesDataset
from unxpass.features import simulate_features


class UnxpassComponent(ABC):
    """Base class for all components."""

    component_name = "default"

    def __init__(
        self, features: Union[List, Dict], label: List, transform: Optional[Callable] = None
    ):
        self.features = features
        self.label = label
        self.transform = transform

    def initialize_dataset(self, dataset: Union[PassesDataset, Callable]) -> PassesDataset:
        if callable(dataset):
            return dataset(xfns=self.features, yfns=self.label, transform=self.transform)
        return dataset

    @abstractmethod
    def train(self, dataset: Callable, optimized_metric=None) -> Optional[float]:
        pass

    @abstractmethod
    def test(self, dataset: Callable) -> Dict[str, float]:
        pass

    def _get_metrics(self, y_true, y_hat):
        return {}

    @abstractmethod
    def predict(self, dataset: Callable) -> pd.Series:
        pass

    def save(self, path: Path):
        pickle.dump(self, path.open(mode="wb"))

    @classmethod
    def load(cls, path: Path):
        return pickle.load(path.open(mode="rb"))


class UnxPassSkLearnComponent(UnxpassComponent):
    """Base class for an SkLearn-based component."""

    def __init__(self, model, features, label):
        super().__init__(features, label)
        self.model = model

    def train(self, dataset, optimized_metric=None) -> Optional[float]:
        mlflow.sklearn.autolog()

        # Load data
        data = self.initialize_dataset(dataset)
        X_train, y_train = data.features, data.labels

        # Train the model
        log.info("Fitting model on train set")
        self.model.fit(X_train, y_train)

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            cv_score = cross_val_score(
                self.model, X_train, y_train, cv=5, scoring=optimized_metric
            )
            return np.mean(cv_score, dtype=float)

        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        y_hat = self.model.predict_proba(X_test)[:, 1]
        return self._get_metrics(y_test, y_hat)

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        y_hat = self.model.predict_proba(data.features)[:, 1]
        return pd.Series(y_hat, index=data.features.index)


class UnxPassXGBoostComponent(UnxpassComponent):
    """Base class for an XGBoost-based component."""

    def __init__(self, model, features, label):
        super().__init__(features, label)
        self.model = model

    def train(self, dataset, optimized_metric=None, **train_cfg) -> Optional[float]:
        mlflow.xgboost.autolog()

        # Load data
        data = self.initialize_dataset(dataset)
        X_train, X_val, y_train, y_val = train_test_split(
            data.features, data.labels, test_size=0.2
        )

        # Train the model
        log.info("Fitting model on train set")
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **train_cfg)

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            idx = self.model.best_iteration
            return self.model.evals_result()["validation_0"][optimized_metric][idx]

        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        if isinstance(self.model, xgb.XGBClassifier):
            y_hat = self.model.predict_proba(X_test)[:, 1]
        elif isinstance(self.model, xgb.XGBRegressor):
            y_hat = self.model.predict(X_test)
        else:
            raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return self._get_metrics(y_test, y_hat)

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        if isinstance(self.model, xgb.XGBClassifier):
            y_hat = self.model.predict_proba(data.features)[:, 1]
        elif isinstance(self.model, xgb.XGBRegressor):
            y_hat = self.model.predict(data.features)
        else:
            raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return pd.Series(y_hat, index=data.features.index)

    def predict_locations(self, dataset, game_id, db, xy_coo, result=None) -> pd.Series:
        data = self.initialize_dataset(dataset)
        games = data.features.index.unique(level=0)
        assert game_id in games, "Game ID not found in dataset!"
        sim_features = simulate_features(
            db,
            game_id,
            xfns=list(data.xfns.keys()),
            actionfilter=data.actionfilter,
            xy=xy_coo,
            result=result,
        )
        cols = [item for sublist in data.xfns.values() for item in sublist]
        if isinstance(self.model, xgb.XGBClassifier):
            y_hat = self.model.predict_proba(sim_features[cols])[:, 1]
        elif isinstance(self.model, xgb.XGBRegressor):
            y_hat = self.model.predict(sim_features[cols])
        else:
            raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return pd.Series(y_hat, index=data.features.index)

    def predict_surface(
        self, dataset, game_id, config=None, db=None, x_bins=104, y_bins=68, result=None
    ) -> Dict:
        data = self.initialize_dataset(dataset)
        games = data.features.index.unique(level=0)
        assert game_id in games, "Game ID not found in dataset!"
        sim_features = simulate_features(
            db,
            game_id,
            xfns=list(data.xfns.keys()),
            actionfilter=data.actionfilter,
            x_bins=x_bins,
            y_bins=y_bins,
            result=result,
        )

        out = {}
        cols = [item for sublist in data.xfns.values() for item in sublist]
        for action_id in sim_features.index.unique(level=1):
            if isinstance(self.model, xgb.XGBClassifier):
                out[f"action_{action_id}"] = (
                    self.model.predict_proba(sim_features.loc[(game_id, action_id), cols])[:, 1]
                    .reshape(x_bins, y_bins)
                    .T
                )
            elif isinstance(self.model, xgb.XGBRegressor):
                out[f"action_{action_id}"] = (
                    self.model.predict(sim_features.loc[(game_id, action_id), cols])
                    .reshape(x_bins, y_bins)
                    .T
                )
            else:
                raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return out


class UnxPassPytorchComponent(UnxpassComponent):
    """Base class for a PyTorch-based component."""

    def __init__(self, model, features, label, transform):
        super().__init__(features, label, transform)
        self.model = model

    def train(
        self,
        dataset,
        optimized_metric=None,
        callbacks=None,
        logger=None,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        **train_cfg,
    ) -> Optional[float]:
        mlflow.pytorch.autolog()

        # Init lightning trainer
        trainer = pl.Trainer(callbacks=callbacks, logger=logger, **train_cfg["trainer"])

        # Load data
        data = self.initialize_dataset(dataset)
        nb_train = int(len(data) * 0.8)
        lengths = [nb_train, len(data) - nb_train]
        _data_train, _data_val = random_split(data, lengths)
        train_dataloader = DataLoader(
            _data_train,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            _data_val,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )

        # Train the model
        log.info("Fitting model on train set")
        trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        # Print path to best checkpoint
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            return trainer.callback_metrics[optimized_metric]

        return None

    def test(
        self, dataset, batch_size=1, num_workers=0, pin_memory=False, **test_cfg
    ) -> Dict[str, float]:
        # Load dataset
        data = self.initialize_dataset(dataset)
        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model.eval()

        # Apply model on test set
        all_preds, all_targets = [], []
        for batch in track(dataloader):
            loss, y_hat, y = self.model.step(batch)
            all_preds.append(y_hat)
            all_targets.append(y)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]
        all_targets = torch.cat(all_targets, dim=0).detach().numpy()[:, 0]

        # Compute metrics
        return self._get_metrics(all_targets, all_preds)

    def predict(self, dataset, batch_size=1, num_workers=0, pin_memory=False) -> pd.Series:
        # Load dataset
        data = self.initialize_dataset(dataset)
        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model.eval()

        # Apply model on test set
        all_preds = []
        for batch in track(dataloader):
            loss, y_hat, y = self.model.step(batch)
            all_preds.append(y_hat)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]
        return pd.Series(all_preds, index=data.features.index)

    def predict_surface(
        self, dataset, game_id=None, batch_size=1, num_workers=0, pin_memory=False, **predict_cfg
    ) -> Dict:
        # Load dataset
        data = self.initialize_dataset(dataset)
        actions = data.features.reset_index()
        if game_id is not None:
            actions = actions[actions.game_id == game_id]
            data = Subset(data, actions.index.values)
        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        predictor = pl.Trainer(**predict_cfg.get("trainer", {}))
        predictions = torch.cat(predictor.predict(self.model, dataloaders=dataloader))

        output = defaultdict(dict)
        for i, action in actions.iterrows():
            output[action.game_id][action.action_id] = predictions[i][0].detach().numpy()
        return dict(output)

    @classmethod
    def load(cls, path: Path):
        return pickle.load(path.open(mode="rb"))
