#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import run_experiments as re
from models.registry import MODEL_ENV_VAR, list_models

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = BASE_DIR / "plots"
ANALYSIS_SCRIPT = BASE_DIR / "generate_analysis.py"
REQUIREMENTS = BASE_DIR / "requirements.txt"


def install_dependencies() -> None:
    """Install project dependencies inside the virtual environment."""
    py_path = re.venv_python()
    if not py_path.exists():
        print("Ambiente virtual não encontrado. Execute `python -m venv .venv` na raiz primeiro.")
        sys.exit(1)
    print("Instalando dependências no ambiente virtual...")
    install_cmd = [str(py_path), "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    result = subprocess.run(install_cmd)
    if result.returncode != 0:
        print("Falha ao instalar dependências. Verifique o log acima.")
        sys.exit(result.returncode)


def run_analysis(model) -> None:
    """Execute analysis script for a specific model."""
    if not ANALYSIS_SCRIPT.exists():
        print(f"Não encontrei {ANALYSIS_SCRIPT}; pulei a etapa de análise.")
        return
    
    print(f"\n=== Executando análise para {model.label} ===")
    python_exec = re.venv_python()
    if not python_exec.exists():
        raise RuntimeError("Ambiente virtual não encontrado; impossível executar as análises.")
    
    env = os.environ.copy()
    env[MODEL_ENV_VAR] = model.name
    result = subprocess.run([str(python_exec), str(ANALYSIS_SCRIPT)], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Falha ao executar análise para {model.label}")


def check_output_dir() -> bool:
    """Check if output directory exists and has content."""
    if not OUTPUT_DIR.exists():
        print(f"❌ Diretório {OUTPUT_DIR} não encontrado.")
        return False
    
    if not any(OUTPUT_DIR.iterdir()):
        print(f"❌ Diretório {OUTPUT_DIR} está vazio.")
        return False
    
    return True


def main() -> None:
    print("=" * 60)
    print("Gerando análises a partir dos resultados existentes")
    print("=" * 60)
    
    # 1. Verificar se existe diretório de output com dados
    if not check_output_dir():
        print("\n⚠️  Não há resultados para analisar.")
        print("Execute primeiro o script main.py para gerar os resultados.")
        sys.exit(1)
    
    # 2. Instalar dependências
    install_dependencies()
    
    # 3. Garantir execução dentro do ambiente e desativar logs
    os.environ[re.ENV_FLAG] = "1"
    re.ensure_environment()
    os.environ["AQUIPLACA_ENABLE_LOGS"] = "0"
    re.set_logging_flag(False)
    
    # 4. Obter lista de modelos
    model_specs = list(list_models())
    if not model_specs:
        raise RuntimeError("Nenhum modelo disponível para análise.")
    
    # 5. Executar análise para cada modelo
    print(f"\n📊 Encontrados {len(model_specs)} modelo(s) para análise:")
    for model in model_specs:
        print(f"  - {model.label} ({model.name})")
    
    print("\n" + "=" * 60)
    for model in model_specs:
        # Verificar se existe output para este modelo
        model_output_dir = OUTPUT_DIR / model.name
        if not model_output_dir.exists() or not any(model_output_dir.iterdir()):
            print(f"\n⚠️  Pulando {model.label}: sem resultados em {model_output_dir}")
            continue
        
        os.environ[MODEL_ENV_VAR] = model.name
        run_analysis(model)
    
    print("\n" + "=" * 60)
    print("✅ Análises concluídas!")
    if PLOTS_DIR.exists():
        print(f"📈 Gráficos salvos em: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
