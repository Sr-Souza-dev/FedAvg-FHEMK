from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import os
import re
import shutil
from threading import Lock

from models.registry import ModelSpec, get_model_spec


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
PLOTS_ROOT = ROOT_DIR / "plots"
LOGS_DIR = ROOT_DIR / "logs"
EXPERIMENT_ENV_VAR = "AQUIPLACA_EXPERIMENT_NAME"
LOGGING_ENV_VAR = "AQUIPLACA_ENABLE_LOGS"
_RUN_ID_LOCK = Lock()


def current_model_spec(model_name: str | None = None) -> ModelSpec:
    return get_model_spec(model_name)


def current_model_name(model_name: str | None = None) -> str:
    return current_model_spec(model_name).name


def current_output_root(model_name: str | None = None) -> Path:
    return _ensure_dir(OUTPUT_DIR / current_model_name(model_name))


def current_plots_root(model_name: str | None = None) -> Path:
    return _ensure_dir(PLOTS_ROOT / current_model_name(model_name))


def _sanitize_experiment_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return sanitized or "default"


def _resolve_experiment_name(explicit: str | None = None) -> str:
    """Return the sanitized experiment name, favouring the env flag when present."""
    env_name = os.environ.get(EXPERIMENT_ENV_VAR, "").strip()
    candidate = env_name or (explicit or "")
    return _sanitize_experiment_name(candidate or "default")


def current_logs_root(model_name: str | None = None) -> Path:
    return _ensure_dir(LOGS_DIR / current_model_name(model_name))


def current_logs_dir(experiment: str | None = None) -> Path:
    """Return the log directory for the active experiment, namespaced by model."""
    root = current_logs_root()
    exp_name = experiment or os.environ.get(EXPERIMENT_ENV_VAR, "").strip()
    if not exp_name:
        return _ensure_dir(root / "default")
    return _ensure_dir(root / _sanitize_experiment_name(exp_name))


def logging_enabled() -> bool:
    """Return True if log capturing should be performed."""
    value = os.environ.get(LOGGING_ENV_VAR, "1").strip().lower()
    return value not in {"0", "false", "no", "n"}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def next_run_id(experiment: str) -> str:
    """Return the next sequential run identifier (1,2,3,...) for the experiment."""
    sanitized = _resolve_experiment_name(experiment)
    experiment_dir = _ensure_dir(current_output_root() / sanitized)
    with _RUN_ID_LOCK:
        numeric_runs = [
            int(entry.name)
            for entry in experiment_dir.iterdir()
            if entry.is_dir() and entry.name.isdigit()
        ]
        return str(max(numeric_runs, default=0) + 1)


def experiment_output_dir(
    experiment: str,
    encrypted: bool,
    execution_id: str,
    subdir: str = "",
) -> Path:
    experiment_dir = _ensure_dir(current_output_root() / _resolve_experiment_name(experiment))
    run_label = str(execution_id).strip()
    if not run_label:
        raise ValueError("execution_id cannot be empty; provide a run identifier.")
    if not run_label.isdigit():
        raise ValueError(f"execution_id '{run_label}' must be numeric to match the expected run layout.")
    path = experiment_dir / run_label
    if subdir:
        path = path / subdir
    return _ensure_dir(path)


def write_numbers_to_file(
    filename: str,
    values: Iterable[Iterable[float]],
    base_path: str | Path | None = None,
    extension: str = ".dat",
    open_mode: str = "a",
    **kwargs,
) -> None:
    if "basePath" in kwargs:
        base_path = kwargs.pop("basePath")
    if "type" in kwargs:
        extension = kwargs.pop("type")
    target_dir = _ensure_dir(Path(base_path)) if base_path else current_output_root()
    if not filename.endswith(extension):
        filename += extension
    full_path = target_dir / filename
    with open(full_path, open_mode) as file:
        for row in values:
            file.write("\t".join(str(v) for v in row) + "\n")


def load_numbers_file(
    filename: str,
    base_path: str | Path | None = None,
    extension: str = ".dat",
    **kwargs,
) -> List[float]:
    if "basePath" in kwargs:
        base_path = kwargs.pop("basePath")
    if "type" in kwargs:
        extension = kwargs.pop("type")
    target_dir = Path(base_path) if base_path else current_output_root()
    if not filename.endswith(extension):
        filename += extension
    full_path = target_dir / filename
    if not full_path.exists():
        return []
    with open(full_path, "r") as file:
        values = []
        for line in file:
            row = []
            for token in line.split():
                try:
                    row.append(int(token))
                except ValueError:
                    row.append(float(token))
            values.append(row)
        return values if len(values) > 1 else (values[0] if values else [])


def register_logs(title: str, value: str, file_name: str = "default") -> None:
    if not logging_enabled():
        return
    target_dir = _ensure_dir(current_logs_dir())
    write_string_to_file(
        filename=file_name,
        value=title + "\n" + value,
        base_path=target_dir,
        extension=".txt",
        open_mode="a",
    )


def write_string_to_file(
    filename: str,
    value: str,
    base_path: str | Path = ROOT_DIR / "ckks" / "keys",
    extension: str = ".txt",
    open_mode: str = "a",
    **kwargs,
) -> None:
    if "basePath" in kwargs:
        base_path = kwargs.pop("basePath")
    if "type" in kwargs:
        extension = kwargs.pop("type")
    target_dir = _ensure_dir(Path(base_path))
    if not filename.endswith(extension):
        filename += extension
    full_path = target_dir / filename
    with open(full_path, open_mode) as file:
        file.write(value + "\n")


def load_string_file(
    filename: str,
    base_path: str | Path = ROOT_DIR / "ckks" / "keys",
    extension: str = ".txt",
    **kwargs,
) -> str:
    if "basePath" in kwargs:
        base_path = kwargs.pop("basePath")
    if "type" in kwargs:
        extension = kwargs.pop("type")
    target_dir = Path(base_path)
    if not filename.endswith(extension):
        filename += extension
    full_path = target_dir / filename
    with open(full_path, "r") as file:
        return file.read()


def delete_directory_files(dir_path: str | Path | None = None, **kwargs) -> None:
    """Remove files/subdirectories within the provided directory (non-recursive)."""
    target = dir_path if dir_path is not None else kwargs.pop("dir", None)
    if target is None:
        raise ValueError("Expected a directory path.")
    directory = Path(target)
    if directory.exists() and directory.is_dir():
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        print(f"O diretorio '{directory}' foi deletado com sucesso.")
