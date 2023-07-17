"""A PyTorch dataset containing all passes."""
import itertools
import os
from copy import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd
from rich.progress import track
from torch.utils.data import Dataset

from unxpass import features, labels
from unxpass.config import logger as log


class PassesDataset(Dataset):
    """A dataset containing passes.

    Parameters
    ----------
    xfns : dict(str or callable -> list(str))
        The feature generators and columns to use.
    yfns : list(str or callable)
        The label generators.
    transform : Callable, optional
        A function/transform that takes a sample and returns a transformed
        version of it.
    path : Path
        The path to the directory where pre-computed features are stored. By
        default all features and labels are computed on the fly, but
        pre-computing them will speed up training significantly.
    load_cached : bool, default: True
        Whether to attempt to load the dataset from disk.
    """

    def __init__(
        self,
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        yfns: List[Union[str, Callable]],
        transform: Optional[Callable] = None,
        path: Optional[os.PathLike[str]] = None,
        load_cached: bool = True,
    ):
        # Check requested features and labels
        self.xfns = self._parse_xfns(xfns)
        self.yfns = self._parse_yfns(yfns)
        self.transform = transform

        # Try to load the dataset
        self.store = Path(path) if path is not None else None
        self._features = None
        self._labels = None
        if load_cached:
            if self.store is None:
                raise ValueError("No path to cached dataset provided.")
            try:
                log.info("Loading dataset from %s", self.store)
                if len(self.xfns):
                    self._features = pd.concat(
                        [
                            pd.read_parquet(self.store / f"x_{xfn.__name__}.parquet")[cols]
                            for xfn, cols in self.xfns.items()
                        ],
                        axis=1,
                    )
                if len(self.yfns):
                    self._labels = pd.concat(
                        [
                            pd.read_parquet(self.store / f"y_{yfn.__name__}.parquet")
                            for yfn in self.yfns
                        ],
                        axis=1,
                    )
            except FileNotFoundError:
                log.error(
                    "No complete dataset found at %s. Run 'create' to create it.", self.store
                )

    @staticmethod
    def _parse_xfns(
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]]
    ) -> Dict[Callable, Optional[List]]:
        parsed_xfns = {}
        if isinstance(xfns, list):
            xfns = {xfn: None for xfn in xfns}
        for xfn, cols in xfns.items():
            parsed_xfn = xfn
            parsed_cols = cols
            if isinstance(parsed_xfn, str):
                try:
                    parsed_xfn = getattr(features, parsed_xfn)
                except AttributeError:
                    raise ValueError(f"No feature function found that matches '{parsed_xfn}'.")
            if parsed_cols is None:
                parsed_cols = features.feature_column_names([parsed_xfn])
            parsed_xfns[parsed_xfn] = parsed_cols
        return parsed_xfns

    @staticmethod
    def _parse_yfns(yfns: List[Union[str, Callable]]) -> List[Callable]:
        parsed_yfns = []
        for yfn in yfns:
            if isinstance(yfn, str):
                try:
                    parsed_yfns.append(getattr(labels, yfn))
                except AttributeError:
                    raise ValueError(f"No labeling function found that matches '{yfn}'.")
            else:
                parsed_yfns.append(yfn)
        return parsed_yfns

    @staticmethod
    def actionfilter(actions: pd.DataFrame) -> pd.Series:
        is_pass = actions.type_name.isin(["pass", "cross"])
        by_foot = actions.bodypart_name.str.contains("foot")
        in_freeze_frame = actions.in_visible_area_360
        return is_pass & by_foot & in_freeze_frame

    def create(self, db, filters: Optional[List[Dict]] = None) -> None:
        """Create the dataset.

        Parameters
        ----------
        db : Database
            The database with raw data.
        filters : list(dict), optional
            A list of filters used to select a sample of the dataset. Each filter
            is a dictionary with (a subset of) the following keys:
                - "competition_id": int
                - "season_id": int
                - "game_id": int
            For example, [{"competition_id": 55, "season_id": 43}] will select
            all passes from EURO2020. If no filters are provided, all available
            passes are returned.
        """
        # Create directory
        if self.store is not None:
            self.store.mkdir(parents=True, exist_ok=True)

        # Select games to include
        self.games_idx = list(
            itertools.chain.from_iterable(
                [x["game_id"]] if "game_id" in x else db.games(**x).index.tolist() for x in filters
            )
            if filters is not None
            else db.games().index
        )

        # Compute features for each pass
        if len(self.xfns):
            df_features = []
            for xfn, _ in self.xfns.items():
                df_features_xfn = []
                for game_id in track(
                    self.games_idx, description=f"Computing {xfn.__name__} feature"
                ):
                    df_features_xfn.append(
                        features.get_features(
                            db,
                            game_id=game_id,
                            xfns=[xfn],
                            nb_prev_actions=3,
                            actionfilter=PassesDataset.actionfilter,
                        )
                    )
                df_features_xfn = pd.concat(df_features_xfn)
                if self.store is not None:
                    assert self.store is not None
                    df_features_xfn.to_parquet(self.store / f"x_{xfn.__name__}.parquet")
                df_features.append(df_features_xfn)
            self._features = pd.concat(df_features, axis=1)

        # Compute labels for each pass
        if len(self.yfns):
            df_labels = []
            for yfn in self.yfns:
                df_labels_yfn = []
                for game_id in track(
                    self.games_idx, description=f"Computing {yfn.__name__} label"
                ):
                    df_labels_yfn.append(
                        labels.get_labels(
                            db,
                            game_id=game_id,
                            yfns=[yfn],
                            actionfilter=PassesDataset.actionfilter,
                        )
                    )
                df_labels_yfn = pd.concat(df_labels_yfn)
                if self.store is not None:
                    df_labels_yfn.to_parquet(self.store / f"y_{yfn.__name__}.parquet")
                df_labels.append(df_labels_yfn)
            self._labels = pd.concat(df_labels, axis=1)

    @property
    def features(self):
        if self._features is None:
            assert self._labels is not None, "First, create the dataset."
            return pd.DataFrame(index=self._labels.index)
        return self._features

    @property
    def labels(self):
        if self._labels is None:
            assert self._features is not None, "First, create the dataset."
            return pd.DataFrame(index=self._features.index)
        return self._labels

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.features is not None:
            return len(self.features)
        if self.labels is not None:
            return len(self.labels)
        return 0

    def __getitem__(self, idx: int) -> Dict:
        """Return a sample from the dataset at the given index.

        Parameters
        ----------
        idx : int
            The index of the sample to return.

        Returns
        -------
        sample : (dict, dict)
            A dictionary containing the sample and target.
        """
        game_id, action_id = None, None
        sample_features = {}
        if self.features is not None:
            sample_features = self.features.iloc[idx].to_dict()
            game_id, action_id = self.features.iloc[idx].name
        sample_target = {}
        if self.labels is not None:
            sample_target = self.labels.iloc[idx].to_dict()
            game_id, action_id = self.labels.iloc[idx].name
        # freezedf = self.db.freeze_frame(game_id=sample_idx[0], action_id=sample_idx[1], ltr=True)
        sample = {
            "game_id": game_id,
            "action_id": action_id,
            **sample_features,
            **sample_target,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def apply_overrides(
        self,
        db,
        overrides: pd.DataFrame,
    ):
        """Override a set of action attributes before applying the feature generators.

        Parameters
        ----------
        db : Database
            The database with raw data.
        overrides : pd.DataFrame
            A dataframe with action attributes that override the values in the
            database. The dataframe should be indexed by game_id and action_id.

        Returns
        -------
        pd.DataFrame
            A dataframe with the features computed from the modified actions.
        """
        # select features that are not affected by overrides
        xfns_fixed = [
            fn
            for fn in self.xfns
            if not any(field in fn.required_fields for field in overrides.columns)
        ]
        xfns_fixed_cols = list(itertools.chain(*[self.xfns[fn] for fn in xfns_fixed]))
        df_fixed_features = self.features.loc[:, xfns_fixed_cols]

        games_idx = self.features.index.get_level_values("game_id").unique()
        df_simulated_features = []
        for xfn, cols in self.xfns.items():
            # skip fixed features
            if xfn in xfns_fixed:
                continue

            # simulate other features
            df_features_xfn = []
            for game_id in track(games_idx, description=f"Computing {xfn.__name__} feature"):
                df_features_xfn.append(
                    features.get_features(
                        db,
                        game_id=game_id,
                        xfns=[xfn],
                        nb_prev_actions=3,
                        actionfilter=PassesDataset.actionfilter,
                        overrides=overrides,
                    )[cols]
                )
            df_simulated_features.append(pd.concat(df_features_xfn))

        modified_dataset = copy(self)
        modified_dataset._features = pd.concat(
            [df_fixed_features] + df_simulated_features, axis=1
        )[self.features.columns]
        return modified_dataset


class CompletedPassesDataset(PassesDataset):
    """A dataset containing only completed passes."""

    def __init__(
        self,
        path: os.PathLike[str],
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        yfns: List[Union[str, Callable]],
        transform: Optional[Callable] = None,
    ):
        super().__init__(path, xfns, yfns + ["success"], transform)

    @property
    def features(self):
        if self._features is None:
            return pd.DataFrame(index=self.labels.index)
        return self._features.loc[self.labels.index]

    @property
    def labels(self):
        assert self._labels is not None
        df_labels = self._labels[self._labels.success]
        df_labels.drop("success", axis=1, inplace=True)
        return df_labels


class FailedPassesDataset(PassesDataset):
    """A dataset containing only failed passes."""

    def __init__(
        self,
        path: os.PathLike[str],
        xfns: Union[List, Dict[Union[str, Callable], Optional[List]]],
        yfns: List[Union[str, Callable]],
        transform: Optional[Callable] = None,
    ):
        super().__init__(path, xfns, yfns + ["success"], transform)

    @property
    def features(self):
        if self._features is None:
            return pd.DataFrame(index=self.labels.index)
        return self._features.loc[self.labels.index]

    @property
    def labels(self):
        assert self._labels is not None
        df_labels = self._labels[~self._labels.success]
        df_labels.drop("success", axis=1, inplace=True)
        return df_labels
