"""Single loader for config.yaml -- the only source of numeric constants (CLAUDE.md)."""
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
