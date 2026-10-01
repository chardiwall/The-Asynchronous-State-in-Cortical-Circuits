"""Single loader for config.yaml -- the only source of numeric constants and paths."""
import os

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"{path} is empty or contains no top-level mapping")
    return config


def output_path(config: dict, key: str) -> str:
    """Resolve a named output to a path under the configured artifacts root.

    Every output location lives in config.yaml (`paths.artifacts` + `outputs`), so no
    path is hardcoded and redirecting the root redirects every output. Filenames the
    code generates per item (task_<i>.json, panel_<name>.csv) are composed by the caller
    from a directory resolved here.
    """
    return os.path.join(config["paths"]["artifacts"], config["outputs"][key])
