from __future__ import annotations

"""Central registry for shared experiment settings."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

import tomllib

CONFIG_PATH = Path(__file__).resolve().parent / "experiments_config.toml"


class ExperimentConfigError(RuntimeError):
    """Raised when the shared experiment configuration cannot be loaded."""


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    num_rounds: int
    clients_qtd: int
    epochs: int


@lru_cache
def _load_configs() -> Dict[str, ExperimentConfig]:
    if not CONFIG_PATH.exists():
        raise ExperimentConfigError(f"Config file not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("rb") as stream:
        raw_data = tomllib.load(stream)

    configs: Dict[str, ExperimentConfig] = {}
    for name, values in raw_data.items():
        try:
            num_rounds = int(values["num_rounds"])
            clients_qtd = int(values["clients_qtd"])
            epochs = int(values["epochs"])
        except KeyError as exc:
            missing = exc.args[0]
            raise ExperimentConfigError(
                f"Missing '{missing}' entry for experiment '{name}' in {CONFIG_PATH}"
            ) from exc

        configs[name] = ExperimentConfig(
            name=name,
            num_rounds=num_rounds,
            clients_qtd=clients_qtd,
            epochs=epochs,
        )
    return configs


def get_experiment_config(experiment_name: str) -> ExperimentConfig:
    try:
        return _load_configs()[experiment_name]
    except KeyError as exc:
        raise ExperimentConfigError(
            f"Experiment '{experiment_name}' not defined in {CONFIG_PATH}"
        ) from exc


def reload_configs() -> None:
    """Clear the in-memory cache so that future calls reload from disk."""

    _load_configs.cache_clear()
