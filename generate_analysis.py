#!/usr/bin/env python3
from __future__ import annotations

from analysis import run_full_analysis
from models.registry import get_model_spec


def main() -> None:
    summary = run_full_analysis()
    model_spec = get_model_spec()
    experiments = summary.get("experiments", [])
    print(f"=== Analise de Resultados ({model_spec.label}) ===")
    if not experiments:
        print("Nao encontrei execucoes em output/. Execute os experimentos primeiro.")
        return
    for experiment in experiments:
        averaged = "sim" if summary["averages"].get(experiment) else "nao"
        charts = summary["experiment_plots"].get(experiment, 0)
        print(f"- {experiment}: average={averaged}, plots={charts}")
    comparisons = summary.get("comparisons", {})
    if comparisons:
        print("\nGraficos comparativos gerados:")
        for metric, path in comparisons.items():
            status = path if path else "dados insuficientes"
            print(f"  * {metric}: {status}")
    stats = summary.get("statistics", {})
    table = stats.get("table") if isinstance(stats, dict) else None
    if table:
        print(f"\nResumo estatistico salvo em: {table}")
    boxplots = summary.get("boxplots", {})
    if boxplots:
        print("\nBoxplots (variacao por rodada):")
        for metric, path in boxplots.items():
            status = path if path else "dados insuficientes"
            print(f"  * {metric}: {status}")
    barplots = summary.get("barplots", {})
    if barplots:
        print("\nBarras comparando metricas finais:")
        for metric, path in barplots.items():
            status = path if path else "dados insuficientes"
            print(f"  * {metric}: {status}")
    tradeoff = summary.get("tradeoff")
    if tradeoff:
        print(f"\nTrade-off tempo x acuracia: {tradeoff}")


if __name__ == "__main__":
    main()
