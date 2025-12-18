#!/usr/bin/env python3
"""Helper script to launch the CKKS Flower experiments."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = BASE_DIR / "experiments"
EXPERIMENTS = {
    "1": ("new_ckks-fl (baseline CKKS)", EXPERIMENTS_ROOT / "new_ckks-fl"),
    "2": ("full_ckks-fl (Pyfhel FHE)", EXPERIMENTS_ROOT / "full_ckks-fl"),
}
MENU = """Escolha qual experimento deseja executar:
  1 - new_ckks-fl (baseline CKKS)
  2 - full_ckks-fl (Pyfhel FHE)
  3 - Ambos (executa em sequencia)
Digite sua opcao (1/2/3): """

ENV_FLAG = "RUN_EXPERIMENTS_IN_VENV"
_RAY_AVAILABLE = importlib.util.find_spec("ray") is not None


def venv_python() -> Path:
    scripts = "Scripts" if os.name == "nt" else "bin"
    return BASE_DIR / ".venv" / scripts / ("python.exe" if os.name == "nt" else "python")


def scripts_dir() -> Path:
    return BASE_DIR / ".venv" / ("Scripts" if os.name == "nt" else "bin")


def extend_path(env: dict[str, str]) -> None:
    scripts = str(scripts_dir())
    env["PATH"] = scripts + os.pathsep + env.get("PATH", "")


def ensure_environment() -> None:
    py_path = venv_python()
    if not py_path.exists():
        print("Ambiente virtual nao encontrado. Execute `python -m venv .venv` na raiz primeiro.")
        sys.exit(1)

    already_inside = Path(sys.executable).resolve() == py_path.resolve()

    if os.environ.get(ENV_FLAG) != "1":
        print("Instalando dependencias no ambiente virtual...")
        install_cmd = [str(py_path), "-m", "pip", "install", "-r", str(BASE_DIR / "requirements.txt")]
        result = subprocess.run(install_cmd)
        if result.returncode != 0:
            print("Falha ao instalar dependencias. Verifique o log acima.")
            sys.exit(result.returncode)

        if not already_inside:
            env = os.environ.copy()
            env[ENV_FLAG] = "1"
            extend_path(env)
            print("Reiniciando o script dentro do ambiente virtual...")
            result = subprocess.run([str(py_path), __file__], env=env)
            sys.exit(result.returncode)

    extend_path(os.environ)


def run_experiment(key: str) -> None:
    name, path = EXPERIMENTS[key]
    if not path.exists():
        print(f"[ERRO] Diretorio do experimento '{name}' nao encontrado em {path}")
        sys.exit(1)
    print(f"\n=== Iniciando {name} ===")
    if _RAY_AVAILABLE:
        cmd = ["flwr", "run", "."]
    else:
        num_supernodes = load_num_supernodes(path)
        print(
            "Ray nao esta disponivel para esta versao do Python, utilizando o backend"
            " inline."
        )
        cmd = [
            "flower-simulation",
            "--app",
            ".",
            "--num-supernodes",
            str(num_supernodes),
            "--backend",
            "inline",
        ]
    result = subprocess.run(cmd, cwd=path)
    if result.returncode != 0:
        print(f"[ERRO] Execucao de '{name}' falhou (codigo {result.returncode}).")
        sys.exit(result.returncode)


def load_num_supernodes(path: Path) -> int:
    pyproject = path / "pyproject.toml"
    with pyproject.open("rb") as stream:
        data = tomllib.load(stream)
    try:
        options = data["tool"]["flwr"]["federations"]["local-simulation"]["options"]
        return int(options["num-supernodes"])
    except Exception as exc:
        raise RuntimeError(
            f"Nao consegui ler 'options.num-supernodes' em {pyproject}"
        ) from exc


def main() -> None:
    ensure_environment()
    choice = input(MENU).strip()
    if choice == "1":
        run_experiment("1")
    elif choice == "2":
        run_experiment("2")
    elif choice == "3":
        for key in ("1", "2"):
            run_experiment(key)
    else:
        print("Opcao invalida. Encerrando.")
        sys.exit(1)


if __name__ == "__main__":
    main()
