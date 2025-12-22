#!/usr/bin/env python3
"""Helper script to launch the CKKS Flower experiments."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from experiment_config import get_experiment_config

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = BASE_DIR / "experiments"
EXPERIMENTS = {
    "1": ("baseline-fl (sem criptografia)", EXPERIMENTS_ROOT / "baseline-fl"),
    "2": ("new_ckks-fl (baseline CKKS)", EXPERIMENTS_ROOT / "new_ckks-fl"),
    "3": ("full_ckks-fl (Pyfhel FHE)", EXPERIMENTS_ROOT / "full_ckks-fl"),
    "4": ("selective_ckks-fl (Mask-guided CKKS)", EXPERIMENTS_ROOT / "selective_ckks-fl"),
}
MENU = """Escolha qual experimento deseja executar:
  1 - baseline-fl (sem criptografia)
  2 - new_ckks-fl (baseline CKKS)
  3 - full_ckks-fl (Pyfhel FHE)
  4 - selective_ckks-fl (mask-guided CKKS)
  5 - Todos (executa em sequencia)
Digite sua opcao (1/2/3/4/5): """

ENV_FLAG = "RUN_EXPERIMENTS_IN_VENV"
LOGGING_ENV_FLAG = "AQUIPLACA_ENABLE_LOGS"
EXPERIMENT_ENV_FLAG = "AQUIPLACA_EXPERIMENT_NAME"
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


def ask_logging_preference() -> bool:
    """Ask the user whether logs should be recorded for this run."""
    prompt = "Deseja registrar logs desta execucao? (s/n) [s]: "
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("", "s", "sim", "y", "yes"):
            return True
        if answer in ("n", "nao", "no"):
            return False
        print("Opcao invalida. Responda com 's' ou 'n'.")


def set_logging_flag(enabled: bool) -> None:
    os.environ[LOGGING_ENV_FLAG] = "1" if enabled else "0"


def run_experiment(key: str) -> None:
    name, path = EXPERIMENTS[key]
    if not path.exists():
        print(f"[ERRO] Diretorio do experimento '{name}' nao encontrado em {path}")
        sys.exit(1)
    shared_cfg = get_experiment_config(path.name)
    print(f"\n=== Iniciando {name} ===")
    child_env = os.environ.copy()
    child_env[EXPERIMENT_ENV_FLAG] = path.name
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    path_entries = [str(path)]
    if existing_pythonpath:
        path_entries.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(path_entries)
    if _RAY_AVAILABLE:
        cmd = ["flwr", "run", "."]
    else:
        print(
            "Ray nao esta disponivel para esta versao do Python, utilizando o backend"
            " inline."
        )
        cmd = [
            "flower-simulation",
            "--app",
            ".",
            "--num-supernodes",
            str(shared_cfg.clients_qtd),
            "--backend",
            "inline",
        ]
    result = subprocess.run(cmd, cwd=path, env=child_env)
    if result.returncode != 0:
        print(f"[ERRO] Execucao de '{name}' falhou (codigo {result.returncode}).")
        sys.exit(result.returncode)


def main() -> None:
    ensure_environment()
    logging_enabled = ask_logging_preference()
    set_logging_flag(logging_enabled)
    choice = input(MENU).strip()
    if choice in EXPERIMENTS:
        run_experiment(choice)
    elif choice == "5":
        for key in EXPERIMENTS:
            run_experiment(key)
    else:
        print("Opcao invalida. Encerrando.")
        sys.exit(1)


if __name__ == "__main__":
    main()
