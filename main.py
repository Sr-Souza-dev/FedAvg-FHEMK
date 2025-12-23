#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import run_experiments as re
from models.registry import MODEL_ENV_VAR, ModelSpec, list_models

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = BASE_DIR / "plots"
BACKUP_DIR = BASE_DIR / "backup"
ANALYSIS_SCRIPT = BASE_DIR / "generate_analysis.py"
REQUIREMENTS = BASE_DIR / "requirements.txt"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dir_has_content(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def existing_run_numbers() -> list[int]:
    if not BACKUP_DIR.exists():
        return []
    numbers: list[int] = []
    for entry in BACKUP_DIR.iterdir():
        if entry.is_dir() and entry.name.startswith("run_"):
            try:
                numbers.append(int(entry.name.split("_", maxsplit=1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(numbers)


def archive_results(run_number: int, *, move: bool) -> bool:
    """Move or copy current output/plots to backup/run_{run_number}."""
    has_output = dir_has_content(OUTPUT_DIR)
    has_plots = dir_has_content(PLOTS_DIR)
    if not has_output and not has_plots:
        return False

    run_dir = BACKUP_DIR / f"run_{run_number}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    for source in (OUTPUT_DIR, PLOTS_DIR):
        destination = run_dir / source.name
        if dir_has_content(source):
            if move:
                shutil.move(str(source), destination)
                ensure_dir(source)
            else:
                shutil.copytree(source, destination)
        else:
            destination.mkdir(parents=True, exist_ok=True)
    return True


def prompt_iterations() -> int:
    while True:
        raw = input("Quantas execucoes completas deseja realizar? ").strip()
        try:
            value = int(raw)
        except ValueError:
            print("Informe um numero inteiro valido.")
            continue
        if value <= 0:
            print("Informe um numero inteiro maior que zero.")
            continue
        return value


def run_analysis(model: ModelSpec) -> None:
    if not ANALYSIS_SCRIPT.exists():
        print("Nao encontrei generate_analysis.py; pulei a etapa de analise.")
        return
    print(f"\n=== Executando analise final para {model.label} ===")
    python_exec = re.venv_python()
    if not python_exec.exists():
        raise RuntimeError("Ambiente virtual nao encontrado; impossivel executar as analises.")
    env = os.environ.copy()
    env[MODEL_ENV_VAR] = model.name
    result = subprocess.run([str(python_exec), str(ANALYSIS_SCRIPT)], env=env)
    if result.returncode != 0:
        raise RuntimeError("Falha ao executar generate_analysis.py")


def prepare_workspace() -> int:
    """Move old outputs to the backup and ensure fresh directories."""
    ensure_dir(BACKUP_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOTS_DIR)

    existing_numbers = existing_run_numbers()
    next_run_number = (existing_numbers[-1] + 1) if existing_numbers else 1

    if dir_has_content(OUTPUT_DIR) or dir_has_content(PLOTS_DIR):
        print("Movendo resultados anteriores para o backup antes de iniciar...")
        if archive_results(next_run_number, move=True):
            next_run_number += 1

    # Guarantee the working directories are empty
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    if PLOTS_DIR.exists():
        shutil.rmtree(PLOTS_DIR)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOTS_DIR)

    return next_run_number


def install_dependencies() -> None:
    """Install project dependencies inside the virtual environment."""
    py_path = re.venv_python()
    if not py_path.exists():
        print("Ambiente virtual nao encontrado. Execute `python -m venv .venv` na raiz primeiro.")
        sys.exit(1)
    print("Instalando dependencias no ambiente virtual...")
    install_cmd = [str(py_path), "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    result = subprocess.run(install_cmd)
    if result.returncode != 0:
        print("Falha ao instalar dependencias. Verifique o log acima.")
        sys.exit(result.returncode)


def main() -> None:
    # 1. Mover resultados anteriores e preparar diretórios vazios
    next_run_number = prepare_workspace()

    # 2. Instalar dependências
    install_dependencies()

    # 3. Garantir execução dentro do ambiente e desativar logs
    os.environ[re.ENV_FLAG] = "1"
    re.ensure_environment()
    os.environ["AQUIPLACA_ENABLE_LOGS"] = "0"
    re.set_logging_flag(False)

    # 4. Perguntar quantas execuções completas serão feitas
    total_runs = prompt_iterations()

    experiment_items = list(re.EXPERIMENTS)
    model_specs = list(list_models())
    if not model_specs:
        raise RuntimeError("Nenhum modelo disponivel para execucao.")

    for model in model_specs:
        os.environ[MODEL_ENV_VAR] = model.name
        print(f"\n=== Modelo: {model.label} ({model.name}) ===")
        for iteration in range(1, total_runs + 1):
            print(f"\n=== Execucao {iteration} de {total_runs} ({model.label}) ===")
            for entry in experiment_items:
                print(f"\n--- Executando {entry.label} ---")
                re.run_experiment(entry, model)
        run_analysis(model)
    if archive_results(next_run_number, move=False):
        print(f"Resultados atuais copiados para backup/run_{next_run_number}.")

    print("\nProcesso concluido. Resultados finais permanecem em ./output e ./plots.")


if __name__ == "__main__":
    main()
