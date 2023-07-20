<div align="center">
<img src="docs/logo.png" height="250">
<br/>

[![Python Version: 3.9+](https://img.shields.io/badge/Python-3.7.1+-blue.svg)](https://pypi.org/project/socceraction)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/license/apache-2-0/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<br/>

</div>

## What is it?

un-xPass is a framework to **evaluate the creative abilities of soccer players** using [StatsBomb 360 data](https://statsbomb.com/what-we-do/soccer-data/360-2/). The methodology is based on the intuition that creativity involves not only accomplishing something useful, but doing it in a unique or atypical way. Therefore, it assesses whether a player's passes (1) differ from the typical pass that most other players would have selected in a similar situation and (2) have more promising results than the typical pass. A combination of **three machine learning models** are used to estimate the originality and expected value of a pass:

1.  **Pass selection:** the likelihood of each possible pass destination
2.  **Pass value**: the long-term reward of each passing option
3.  **Pass success:** the success probability of each passing option

For each model, the framework provides a deep learning model based on the [SoccerMap](https://arxiv.org/abs/2010.10202) architecture, a feature-based [XGBoost](https://xgboost.readthedocs.io/en/stable/) model and a set of baseline models.

## Installation

You can install a development version directly from GitHub. This requires [Poetry](https://python-poetry.org/).

```sh
# Clone the repository
$ git clone git://github.com/ML-KULeuven/un-xPass.git
$ cd un-xPass
# Create a virtual environment
$ python3 -m venv .venv
$ source venv/bin/activate
# Install the package and its dependencies
$ poetry install
```

You should now be able to run the command-line interface (CLI).

```
$ unxpass --help
```

## Getting started

This section gives a quick introduction on how to get started using the CLI. For examples on how to use the library in your own code, please refer to the [notebooks](./notebooks) directory.

<details>
<summary><b>STEP 1: Obtain StatsBomb 360 data.</b></summary>

The models are built on [StatsBomb 360 event stream data](https://statsbomb.com/what-we-do/soccer-data/360-2/). StatsBomb has made data of certain leagues freely available for public non-commercial use at <https://github.com/statsbomb/open-data>. This open data can be accessed without the need of authentication, but its use is subject to a [user agreement](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf). The code below shows how to fetch the public data of EURO 2020 from the repository and store it in an SQLite database.

```bash
unxpass load-data \
  sqlite://$(pwd)/stores/database.sql \
  --getter="remote" \
  --competition-id="55" \
  --season-id="43"
```

Apart from the SQLite interface, the unxpass library also supports storing data in a HDF file. To use this data storage interface, replace `sqlite://` with `hdf://` in the above command. Additional interfaces can be supported by subclassing `unxpass.databases.Database`.

</details>

<details>
<summary><b>STEP 2: Create train and test data.</b></summary>

Now we will extract all passes from the data, create a feature representation and assign a label to each pass. The code below shows how to create a train and test set in `./stores/datasets/euro2020` with all features and labels required to train and evaluate the models. The [`./config/dataset/euro2020/train.yaml`](./config/dataset/euro2020/train.yaml) file defines which leagues, seasons and games should be used to create the training dataset. Similarly, the [`./config/dataset/euro2020/test.yaml`](./config/dataset/euro2020/test.yaml) file defines which leagues, seasons and games should be used to create the evaluation set.

```bash
unxpass create-dataset \
  sqlite://$(pwd)/stores/database.sql \
  $(pwd)/stores/datasets/euro2020/train \
  $(pwd)/config/dataset/euro2020/train.yaml
```

```bash
unxpass create-dataset \
  sqlite://$(pwd)/stores/database.sql \
  $(pwd)/stores/datasets/euro2020/test \
  $(pwd)/config/dataset/euro2020/test.yaml
```

_(this will take ~2 hours to run)_

It is also possible to generate a specific set of features and labels. For example, to generate only the "relative start location" features and "success" label, you can add `--xfn="relative_startlocation" --yfn="success"` to the above command.

</details>

<details>
<summary><b>STEP 3: Train model components.</b></summary>

All models are dynamically instantiated from a hierarchical configuration file managed by the [Hydra](https://github.com/facebookresearch/hydra) framework. The main config is available in [config/config.yaml](./config/config.yaml) and a set of example configurations for training specific models is available in [config/experiment](./config/experiment). The experiment configs allow you to overwrite parameters from the main config and allow you to easily iterate over new model configurations! You can run a chosen experiment config with:

```bash
unxpass train \
  $(pwd)/config \
  $(pwd)/stores/datasets/euro2020/train \
  experiment="pass_success/threesixty"
```

Experiments are tracked using [MLFlow](https://mlflow.org/). You can view the results of your experiments by running `mlflow ui --backend-store-uri stores/model` in the root directory of the project and browsing to <http://localhost:5000>.

To optimize the model's hyperparameters, you can use the `run_experiment.py` script. This script uses [Optuna](https://optuna.org/) to automate the search and (optionally) [Ray](https://www.ray.io/) to run the search in parallel on a computing cluster. The script can be run with:

```bash
python run_experiment.py \
  experiment="pass_success/threesixty" \
  hparams_search="xgboost_optuna" \
  hydra/launcher="ray" \
  hydra.launcher.ray.init.address="ray://123.45.67.89:10001"
```

</details>

<details>
<summary><b>STEP 4: Compute creativity ratings.</b></summary>

Once you have trained all required models, they can be used to compute creativity ratings. Therefore, specify a dataset to compute ratings for and the run ID of a Soccermap-based pass selection model, an XGBoost-based pass selection model and a VAEP model. The run IDs are printed after training a component or can be found in the MLFlow UI.

```bash
unxpass rate \
  sqlite://$(pwd)/stores/database.sql \
  $(pwd)/stores/datasets/euro2020/test \
  runs:/788ec5a232af46e59ac984d50ecfc1d5 \
  runs:/f0d0458824324fbbb257550bf09d924a \
  runs:/f4f4efb5f0534f03a1d513141e06c962
```

</details>

## Contributing

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome. However, be aware that the code is not actively developed. Its primary use is to enable reproducibility of our research. If you believe there is a feature missing, feel free to raise a feature request, but please do be aware that the overwhelming likelihood is that your feature request will not be accepted.
To learn more on how to contribute, see the [Contributor Guide](./CONTRIBUTING.rst).

## Research

If you make use of this package in your research, please consider citing the following paper:

- Pieter Robberechts, Maaike Van Roy and Jesse Davis. **un-xPass: Measuring Soccer Player’s Creativity.** Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2023. <br/>[ [pdf](https://people.cs.kuleuven.be/~pieter.robberechts/repo/robberechts-kdd23-unxpass.pdf) | [bibtex](./docs/unxpass.bibtex) ]

## License

Distributed under the terms of the [Apache License, Version 2.0](https://opensource.org/license/apache-2-0/), un-xPass is free and open source software. Although not strictly required, we appreciate it if you include a link to this repo or cite our research in your work if you make use of it.
