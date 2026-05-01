#!/usr/bin/env python3
"""Helper script to launch the CKKS Flower experiments."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from experiment_config import get_experiment_config
from models.registry import MODEL_ENV_VAR, ModelSpec, list_models

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = BASE_DIR / "experiments"
SELECTIVE_PATH = EXPERIMENTS_ROOT / "selective_ckks-fl"


@dataclass(frozen=True)
class ExperimentEntry:
    key: str
    label: str
    path: Path
    env_name: str
    mask_ratio: float | None = None


EXPERIMENTS: tuple[ExperimentEntry, ...] = (
    ExperimentEntry("1", "baseline-fl (sem criptografia)", EXPERIMENTS_ROOT / "baseline-fl", "baseline-fl"),
    ExperimentEntry("2", "new_ckks-fl (baseline CKKS)", EXPERIMENTS_ROOT / "new_ckks-fl", "new_ckks-fl"),
    ExperimentEntry("3", "full_ckks-fl (Pyfhel FHE)", EXPERIMENTS_ROOT / "full_ckks-fl", "full_ckks-fl"),
    ExperimentEntry("4", "selective_ckks-fl (mask 10%)", SELECTIVE_PATH, "selective_ckks-fl-10", 0.10),
    ExperimentEntry("5", "selective_ckks-fl (mask 20%)", SELECTIVE_PATH, "selective_ckks-fl-20", 0.20),
    ExperimentEntry("6", "selective_ckks-fl (mask 40%)", SELECTIVE_PATH, "selective_ckks-fl-40", 0.40),
    ExperimentEntry("7", "selective_ckks-fl (mask 80%)", SELECTIVE_PATH, "selective_ckks-fl-80", 0.80),
)

# Experimentos de escalabilidade: NEWCKKS e baseline com P = 5, 10, 20 clientes
SCALABILITY_EXPERIMENTS: tuple[ExperimentEntry, ...] = (
    ExperimentEntry("s1", "baseline-fl P=5", EXPERIMENTS_ROOT / "baseline-fl", "baseline-fl-p5"),
    ExperimentEntry("s2", "baseline-fl P=10", EXPERIMENTS_ROOT / "baseline-fl", "baseline-fl"),
    ExperimentEntry("s3", "baseline-fl P=20", EXPERIMENTS_ROOT / "baseline-fl", "baseline-fl-p20"),
    ExperimentEntry("s4", "new_ckks-fl P=5", EXPERIMENTS_ROOT / "new_ckks-fl", "new_ckks-fl-p5"),
    ExperimentEntry("s5", "new_ckks-fl P=10", EXPERIMENTS_ROOT / "new_ckks-fl", "new_ckks-fl-p10"),
    ExperimentEntry("s6", "new_ckks-fl P=20", EXPERIMENTS_ROOT / "new_ckks-fl", "new_ckks-fl-p20"),
)

EXPERIMENT_LOOKUP = {entry.key: entry for entry in EXPERIMENTS}
SCALABILITY_LOOKUP = {entry.key: entry for entry in SCALABILITY_EXPERIMENTS}
MODEL_OPTIONS: tuple[ModelSpec, ...] = tuple(list_models())


def _build_menu(options: Iterable[ExperimentEntry]) -> str:
    lines = ["Escolha qual experimento deseja executar:"]
    for entry in options:
        lines.append(f"  {entry.key} - {entry.label}")
    lines.append("  0 - Todos (executa em sequencia)")
    available = "/".join(entry.key for entry in options)
    lines.append(f"Digite sua opcao (0/{available}): ")
    return "\n".join(lines)


MENU = _build_menu(EXPERIMENTS)
SCALABILITY_MENU = _build_menu(SCALABILITY_EXPERIMENTS)
MODEL_MENU = "\n".join(
    ["Escolha o modelo para os experimentos:"]
    + [
        f"  {idx} - {spec.label} ({spec.name})"
        for idx, spec in enumerate(MODEL_OPTIONS, start=1)
    ]
    + ["Digite sua opcao: "]
)

ENV_FLAG = "RUN_EXPERIMENTS_IN_VENV"
LOGGING_ENV_FLAG = "AQUIPLACA_ENABLE_LOGS"
EXPERIMENT_ENV_FLAG = "AQUIPLACA_EXPERIMENT_NAME"
MASK_RATIO_ENV_FLAG = "AQUIPLACA_MASK_RATIO"
_RAY_AVAILABLE = importlib.util.find_spec("ray") is not None


def warmup(model: ModelSpec) -> None:
    """Run a dummy training step to warm up OS caches, PyTorch, and dataset I/O.

    This eliminates the cold-start penalty that would otherwise bias the first
    experiment to run (typically the baseline) with higher ``client_train_time``.
    """
    import torch
    from models.loader import get_backend

    print("Aquecendo caches (dataset, PyTorch, bibliotecas)...")
    backend = get_backend(model.name)
    net = backend.Net()
    trainloader, _ = backend.load_data(0, 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    backend.train(net, trainloader, 1, device)
    print("Warmup concluido.")


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


def ask_model_choice() -> ModelSpec:
    if not MODEL_OPTIONS:
        raise RuntimeError("Nenhum modelo disponivel para execucao.")
    while True:
        choice = input(MODEL_MENU).strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(MODEL_OPTIONS):
                return MODEL_OPTIONS[idx - 1]
        print("Opcao invalida. Escolha um dos indices listados.")


def run_experiment(entry: ExperimentEntry, model: ModelSpec | str) -> None:
    if not entry.path.exists():
        print(f"[ERRO] Diretorio do experimento '{entry.label}' nao encontrado em {entry.path}")
        sys.exit(1)
    shared_cfg = get_experiment_config(entry.env_name)
    model_name = model.name if isinstance(model, ModelSpec) else str(model)
    print(f"\n=== Iniciando {entry.label} [{model_name}] ===")
    child_env = os.environ.copy()
    child_env[EXPERIMENT_ENV_FLAG] = entry.env_name
    child_env[MODEL_ENV_VAR] = model_name
    if entry.mask_ratio is not None:
        child_env[MASK_RATIO_ENV_FLAG] = str(entry.mask_ratio)
    else:
        child_env.pop(MASK_RATIO_ENV_FLAG, None)
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    path_entries = [str(entry.path)]
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
    result = subprocess.run(cmd, cwd=entry.path, env=child_env)
    if result.returncode != 0:
        print(f"[ERRO] Execucao de '{entry.label}' falhou (codigo {result.returncode}).")
        sys.exit(result.returncode)


def ask_experiment_mode() -> str:
    """Ask whether to run standard experiments or scalability analysis."""
    prompt = (
        "Modo de execucao:\n"
        "  1 - Experimentos padrao\n"
        "  2 - Analise de escalabilidade (P = 5, 10, 20 clientes)\n"
        "Digite sua opcao (1/2): "
    )
    while True:
        answer = input(prompt).strip()
        if answer in ("1", "2"):
            return answer
        print("Opcao invalida.")


def main() -> None:
    ensure_environment()
    logging_enabled = ask_logging_preference()
    set_logging_flag(logging_enabled)
    selected_model = ask_model_choice()
    os.environ[MODEL_ENV_VAR] = selected_model.name
    warmup(selected_model)

    mode = ask_experiment_mode()

    if mode == "2":
        choice = input(SCALABILITY_MENU).strip()
        if choice == "0":
            for entry in SCALABILITY_EXPERIMENTS:
                run_experiment(entry, selected_model)
        elif choice in SCALABILITY_LOOKUP:
            run_experiment(SCALABILITY_LOOKUP[choice], selected_model)
        else:
            print("Opcao invalida. Encerrando.")
            sys.exit(1)
    else:
        choice = input(MENU).strip()
        if choice == "0":
            for entry in EXPERIMENTS:
                run_experiment(entry, selected_model)
        elif choice in EXPERIMENT_LOOKUP:
            run_experiment(EXPERIMENT_LOOKUP[choice], selected_model)
        else:
            print("Opcao invalida. Encerrando.")
            sys.exit(1)


if __name__ == "__main__":
    main()
