#!/usr/bin/env python3
"""
Script completo para reproduzir todos os experimentos do artigo NEWCKKS.

Executa, em sequência:
  1. Todos os experimentos do MNIST (tabelas principais + escalabilidade)
  2. Todos os experimentos do CIFAR-10 IID (tabelas principais)
  3. Análise e geração de gráficos para cada modelo
  4. Geração das tabelas LaTeX finais

Uso:
    python run_article_experiments.py          # sistema (relança no venv automaticamente)
    .venv/Scripts/python run_article_experiments.py   # direto no venv
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Garante execução dentro do venv
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def _venv_python() -> Path:
    scripts = "Scripts" if os.name == "nt" else "bin"
    return BASE_DIR / ".venv" / scripts / ("python.exe" if os.name == "nt" else "python")


def _ensure_venv() -> None:
    py = _venv_python()
    if not py.exists():
        print(
            "[ERRO] Ambiente virtual não encontrado.\n"
            "Execute primeiro:\n"
            "  python -m venv .venv\n"
            "  .venv/Scripts/pip install -r requirements.txt"
        )
        sys.exit(1)

    # Se já estamos rodando dentro do venv, não relançar
    if Path(sys.executable).resolve() == py.resolve():
        return

    # Relançar com o Python do venv
    scripts = "Scripts" if os.name == "nt" else "bin"
    env = os.environ.copy()
    env["PATH"] = str(BASE_DIR / ".venv" / scripts) + os.pathsep + env.get("PATH", "")
    env["RUN_EXPERIMENTS_IN_VENV"] = "1"
    result = subprocess.run([str(py), __file__] + sys.argv[1:], env=env)
    sys.exit(result.returncode)


_ensure_venv()

# ---------------------------------------------------------------------------
# Imports (disponíveis após garantir o venv)
# ---------------------------------------------------------------------------
import run_experiments as runner  # noqa: E402
from models.registry import MODEL_ENV_VAR, get_model_spec  # noqa: E402

EXPERIMENTS_ROOT = BASE_DIR / "experiments"
SELECTIVE_PATH = EXPERIMENTS_ROOT / "selective_ckks-fl"
E = runner.ExperimentEntry  # alias curto

# ---------------------------------------------------------------------------
# Definição dos experimentos por modelo
# ---------------------------------------------------------------------------

#
# MNIST — tabelas principais + escalabilidade
#
MNIST_MAIN = [
    E("1", "baseline-fl                 (sem criptografia)", EXPERIMENTS_ROOT / "baseline-fl",   "baseline-fl"),
    E("2", "full_ckks-fl                (FHE chave única)",  EXPERIMENTS_ROOT / "full_ckks-fl",  "full_ckks-fl"),
    E("3", "selective_ckks-fl 10%",                          SELECTIVE_PATH,                      "selective_ckks-fl-10", 0.10),
    E("4", "selective_ckks-fl 20%",                          SELECTIVE_PATH,                      "selective_ckks-fl-20", 0.20),
    E("5", "selective_ckks-fl 40%",                          SELECTIVE_PATH,                      "selective_ckks-fl-40", 0.40),
    E("6", "selective_ckks-fl 80%",                          SELECTIVE_PATH,                      "selective_ckks-fl-80", 0.80),
    E("7", "new_ckks-fl                 (protocolo proposto)",EXPERIMENTS_ROOT / "new_ckks-fl",   "new_ckks-fl"),
]

MNIST_SCALABILITY = [
    E("s1", "baseline-fl  P=5",  EXPERIMENTS_ROOT / "baseline-fl",  "baseline-fl-p5"),
    E("s2", "baseline-fl  P=20", EXPERIMENTS_ROOT / "baseline-fl",  "baseline-fl-p20"),
    E("s3", "new_ckks-fl  P=5",  EXPERIMENTS_ROOT / "new_ckks-fl",  "new_ckks-fl-p5"),
    E("s4", "new_ckks-fl  P=20", EXPERIMENTS_ROOT / "new_ckks-fl",  "new_ckks-fl-p20"),
]

#
# CIFAR-10 IID — apenas tabelas principais
# (escalabilidade é demonstrada no MNIST para reduzir tempo total)
#
CIFAR_MAIN = [
    E("1", "baseline-fl                 (sem criptografia)", EXPERIMENTS_ROOT / "baseline-fl",   "baseline-fl"),
    E("2", "full_ckks-fl                (FHE chave única)",  EXPERIMENTS_ROOT / "full_ckks-fl",  "full_ckks-fl"),
    E("3", "selective_ckks-fl 10%",                          SELECTIVE_PATH,                      "selective_ckks-fl-10", 0.10),
    E("4", "selective_ckks-fl 20%",                          SELECTIVE_PATH,                      "selective_ckks-fl-20", 0.20),
    E("5", "selective_ckks-fl 40%",                          SELECTIVE_PATH,                      "selective_ckks-fl-40", 0.40),
    E("6", "selective_ckks-fl 80%",                          SELECTIVE_PATH,                      "selective_ckks-fl-80", 0.80),
    E("7", "new_ckks-fl                 (protocolo proposto)",EXPERIMENTS_ROOT / "new_ckks-fl",   "new_ckks-fl"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str = "") -> None:
    line = "=" * 64
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def _hms(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _run_experiment(entry: E, model_spec, idx: int, total: int) -> float:
    print(f"\n  [{idx}/{total}] {entry.label}")
    t0 = time.time()
    runner.run_experiment(entry, model_spec)
    elapsed = time.time() - t0
    print(f"  ✓ concluído em {_hms(elapsed)}")
    return elapsed


def _run_analysis(model_name: str) -> None:
    _sep(f"Gerando análise — {model_name}")
    py = _venv_python()
    env = os.environ.copy()
    env[MODEL_ENV_VAR] = model_name
    env["AQUIPLACA_ENABLE_LOGS"] = "0"
    result = subprocess.run([str(py), str(BASE_DIR / "generate_analysis.py")], env=env)
    if result.returncode != 0:
        print(f"  [AVISO] Análise falhou para {model_name} (código {result.returncode})")
    else:
        print(f"  ✓ Análise de {model_name} concluída.")


def _run_tables() -> None:
    _sep("Gerando tabelas LaTeX")
    py = _venv_python()
    env = os.environ.copy()
    env["AQUIPLACA_ENABLE_LOGS"] = "0"
    result = subprocess.run([str(py), str(BASE_DIR / "generate_latex_tables.py")], env=env)
    if result.returncode != 0:
        print(f"  [AVISO] Geração de tabelas falhou (código {result.returncode})")
    else:
        print(f"  ✓ Tabelas LaTeX salvas em: {BASE_DIR / 'tables'}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    # Adiciona scripts do venv ao PATH (necessário para flower-simulation)
    runner.extend_path(os.environ)
    os.environ["AQUIPLACA_ENABLE_LOGS"] = "0"
    runner.set_logging_flag(False)

    mnist_model  = get_model_spec("mlp-mnist")
    cifar_model  = get_model_spec("resnet20-cifar10-iid")

    total_experiments = len(MNIST_MAIN) + len(MNIST_SCALABILITY) + len(CIFAR_MAIN)

    _sep("EXPERIMENTOS DO ARTIGO NEWCKKS")
    print(f"  Modelos  : mlp-mnist  +  resnet20-cifar10-iid")
    print(f"  Total    : {total_experiments} experimentos")
    print(f"  Logs     : desativados (economiza tempo)")
    print(
        "\n  Estimativa de tempo:\n"
        "    MNIST  (principais)   ~ 30-90 min\n"
        "    MNIST  (escalabili.)  ~ 20-60 min\n"
        "    CIFAR  (principais)   ~ 3-6 h\n"
        "  ─────────────────────────────────\n"
        "    Total                 ~ 4-8 h"
    )
    _sep()

    wall_start = time.time()
    done = 0

    # ── 1. MNIST — Tabelas principais ─────────────────────────────────────
    _sep(f"[1/3]  MNIST — Tabelas Principais  ({len(MNIST_MAIN)} experimentos)")
    print("  Aquecendo caches para mlp-mnist...")
    runner.warmup(mnist_model)

    for entry in MNIST_MAIN:
        done += 1
        _run_experiment(entry, mnist_model, done, total_experiments)

    # ── 2. MNIST — Escalabilidade ──────────────────────────────────────────
    _sep(f"[2/3]  MNIST — Escalabilidade P={{5,10,20}}  ({len(MNIST_SCALABILITY)} experimentos)")
    print("  (note: new_ckks-fl P=10 já foi executado acima)")

    for entry in MNIST_SCALABILITY:
        done += 1
        _run_experiment(entry, mnist_model, done, total_experiments)

    _run_analysis("mlp-mnist")

    # ── 3. CIFAR-10 IID — Tabelas principais ──────────────────────────────
    _sep(f"[3/3]  CIFAR-10 IID — Tabelas Principais  ({len(CIFAR_MAIN)} experimentos)")
    print("  Aquecendo caches para resnet20-cifar10-iid...")
    runner.warmup(cifar_model)

    for entry in CIFAR_MAIN:
        done += 1
        _run_experiment(entry, cifar_model, done, total_experiments)

    _run_analysis("resnet20-cifar10-iid")

    # ── 4. Tabelas LaTeX ───────────────────────────────────────────────────
    _run_tables()

    # ── Resumo final ───────────────────────────────────────────────────────
    total_time = time.time() - wall_start
    _sep("CONCLUÍDO")
    print(f"  Tempo total : {_hms(total_time)}")
    print(f"  Dados brutos: {BASE_DIR / 'output'}")
    print(f"  Gráficos    : {BASE_DIR / 'plots'}")
    print(f"  Tabelas LaTeX (geradas): {BASE_DIR / 'tables'}")
    print(
        "\n  Próximos passos para o artigo:\n"
        "    1. Execute generate_latex_tables.py para gerar tabelas atualizadas\n"
        "    2. Atualize os valores de setup_time nas tabelas do artigo\n"
        "    3. Inclua a tabela de escalabilidade na seção de experimentos"
    )
    _sep()


if __name__ == "__main__":
    main()
