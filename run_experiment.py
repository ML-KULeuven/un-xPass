from functools import partial
from pathlib import Path
from typing import Callable

# import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf


# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# dotenv.load_dotenv(override=True)


@hydra.main(version_base="1.2", config_path="config/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from unxpass import utils
    from unxpass.components.base import UnxpassComponent
    from unxpass.config import logger
    from unxpass.datasets import PassesDataset

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Set seed
    if config.get("seed"):
        utils.set_seeds(config.seed)

    # Train model
    logger.info("Instantiating training dataset")
    dataset_train: Callable = partial(
        PassesDataset, path=Path("stores") / "datasets" / "default" / "train"
    )
    logger.info(f"Instantiating model component <{config.component._target_}>")
    component: UnxpassComponent = hydra.utils.instantiate(config.component, _convert_="partial")
    # Setup callbacks
    train_cfg = OmegaConf.to_object(config.get("train_cfg", DictConfig({})))
    utils.instantiate_callbacks(train_cfg)
    utils.instantiate_loggers(train_cfg)
    logger.info("⌛ Starting training!")
    result = component.train(
        dataset_train, optimized_metric=config.get("optimized_metric"), **train_cfg
    )
    logger.info("✅ Finished training.")
    return result


if __name__ == "__main__":
    main()
