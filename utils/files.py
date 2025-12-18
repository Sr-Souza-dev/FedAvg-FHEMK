from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
import os
import shutil


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
LOGS_DIR = ROOT_DIR / "logs"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiment_output_dir(
    experiment: str,
    encrypted: bool,
    execution_id: str,
    subdir: str = "",
) -> Path:
    mode = "cypher" if encrypted else "plain"
    path = OUTPUT_DIR / experiment / mode / execution_id
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
    target_dir = _ensure_dir(Path(base_path)) if base_path else _ensure_dir(OUTPUT_DIR)
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
    target_dir = Path(base_path) if base_path else OUTPUT_DIR
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
    write_string_to_file(
        filename=file_name,
        value=title + "\n" + value,
        base_path=LOGS_DIR,
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
        print(f"O diretorio '{directory}' nao existe.")
