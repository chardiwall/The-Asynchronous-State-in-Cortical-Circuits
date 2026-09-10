"""Single loader for config.yaml -- the only source of numeric constants (CLAUDE.md)."""
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"{path} is empty or contains no top-level mapping")
    return config
