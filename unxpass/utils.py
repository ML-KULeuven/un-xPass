"""Provides some utilities widely used by other modules."""
import warnings
from typing import Dict, List, Sequence, Union

import hydra
import pandas as pd
import pytorch_lightning as pl
import rich.syntax
import rich.tree
import socceraction.spadl.config as spadlconfig
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from unxpass.config import logger as log


def play_left_to_right(gamestates: List[pd.DataFrame], home_team_id: int) -> List[pd.DataFrame]:
    """Perform all action in the same playing direction.

    This changes the start and end location of each action and the freeze
    frame, such that all actions are performed as if the team plays from left
    to right.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    GameStates
        The game states with all actions performed left to right.
    """
    a0 = gamestates[0]
    away_idx = a0.team_id != home_team_id
    for actions in gamestates:
        for col in ["start_x", "end_x"]:
            actions.loc[away_idx, col] = spadlconfig.field_length - actions[away_idx][col].values
        for col in ["start_y", "end_y"]:
            actions.loc[away_idx, col] = spadlconfig.field_width - actions[away_idx][col].values
        for idx, action in actions.loc[away_idx].iterrows():
            freeze_frame = action["freeze_frame_360"]
            if freeze_frame is not None:
                freezedf = pd.DataFrame(freeze_frame).fillna(
                    {"teammate": False, "actor": False, "keeper": False}
                )
                freezedf["x"] = spadlconfig.field_length - freezedf["x"].values
                freezedf["y"] = spadlconfig.field_width - freezedf["y"].values
                actions.at[idx, "freeze_frame_360"] = freezedf.to_dict(orient="records")
    return gamestates


def set_seeds(seed: int = 42) -> None:
    """Set seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Number to be used as the seed. Defaults to 42.
    """
    seed_everything(seed, workers=True)


def extras(config: DictConfig) -> None:
    """Enable a couple of optional utilities, controlled by main config file.

    Does the following:
        - disabling warnings
        - easier access to debug mode
        - forcing debug friendly configuration

    Modifies DictConfig in place.

    Parameters
    ----------
    config : DictConfig
        Configuration composed by Hydra.
    """
    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        if config.get("model_cfg"):
            model = hydra.utils.get_class(config.model_cfg._target_)
            if issubclass(model, xgb.XGBModel):
                config.model_cfg.n_estimators = 1
                config.model_cfg.verbosity = 2
            elif issubclass(model, pl.LightningModule):
                config.train_cfg.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.get("train_cfg", {}).get("trainer", {}).get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.train_cfg.trainer.get("gpus"):
            config.train_cfg.trainer.gpus = 0
        if config.train_cfg.get("pin_memory"):
            config.train_cfg.pin_memory = False
        if config.train_cfg.get("num_workers"):
            config.train_cfg.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = ("component", "train_cfg"),
    resolve: bool = True,
) -> None:
    """Print content of DictConfig using Rich library and its tree structure.

    Parameters
    ----------
    config : DictConfig
        Configuration composed by Hydra.
    fields : Sequence[str], optional
        Determines which main fields from config will be printed and in what order.
    resolve : bool, optional
        Whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def instantiate_callbacks(
    train_cfg: Dict,
) -> Union[List[pl.Callback], List[xgb.callback.TrainingCallback]]:
    """Instantiates callbacks from config."""
    callbacks: Union[List[pl.Callback], List[xgb.callback.TrainingCallback]] = []
    if "callbacks" in train_cfg:
        for _, cb_conf in train_cfg["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf['_target_']}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
        train_cfg["callbacks"] = callbacks


def instantiate_loggers(train_cfg: Dict) -> List[pl.loggers.Logger]:
    """Instantiates loggers from config."""
    logger: List[pl.loggers.Logger] = []
    if "logger" in train_cfg:
        for _, lg_conf in train_cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf['_target_']}>")
                logger.append(hydra.utils.instantiate(lg_conf))
        train_cfg["logger"] = logger


def nested_to_record(dictionary, parent_key="", separator="/"):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, dict):
            items.extend(nested_to_record(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)
