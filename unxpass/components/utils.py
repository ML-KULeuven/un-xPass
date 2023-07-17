"""Model saving and loading utilities for unxpass components.

Implements a custom MLFlow flavor for the unxpass model components such such
that they can be saved to MLFlow.
"""
from pathlib import Path

from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import _get_flavor_configuration

import unxpass.components
from unxpass.components.base import UnxpassComponent


def fullname(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + klass.__qualname__


# options commented out are not necessary
def save_model(
    component: UnxpassComponent,
    path,
    # conda_env=None,
    mlflow_model=None,
    # code_paths=None,
    # signature: ModelSignature = None,
    # input_example: ModelInputExample = None,
    # requirements_file=None,
    # extra_files=None,
    # pip_requirements=None,
    # extra_pip_requirements=None,
):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    mlflow_mlmodel_file_path = path / MLMODEL_FILE_NAME
    model_subpath = path / "component.pkl"
    if mlflow_model is None:
        mlflow_model = Model()
    mlflow_model.add_flavor(
        "unxpass_component", component_name=component.component_name, loader=fullname(component)
    )
    mlflow_model.save(mlflow_mlmodel_file_path)
    component.save(model_subpath)


def load_model(model_uri, dst_path=None):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    model_subpath = Path(local_model_path) / "component.pkl"
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name="unxpass_component"
    )
    return eval(flavor_conf["loader"]).load(model_subpath)


def log_model(
    model: UnxpassComponent,
    artifact_path,
    # conda_env=None,
    # code_paths=None,
    # registered_model_name=None,
    # signature: ModelSignature = None,
    # input_example: ModelInputExample = None,
    # pip_requirements=None,
    # extra_pip_requirements=None,
    **kwargs,
):
    return Model.log(
        artifact_path=str(artifact_path),  # must be string, numbers etc
        flavor=unxpass.components.utils,  # points to this module itself
        # registered_model_name=registered_model_name,
        component=model,
        # conda_env=conda_env,
        # code_paths=code_paths,
        # signature=signature,
        # input_example=input_example,
        # pip_requirements=pip_requirements,
        # extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )
