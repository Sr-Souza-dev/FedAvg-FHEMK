#!/usr/bin/env python3
"""
Script para gerar tabelas LaTeX com os resultados dos experimentos.
"""

import os
from pathlib import Path


def read_final_value(filepath, silent=False):
    """Lê o último valor (última coluna da última linha) de um arquivo .dat"""
    filepath = Path(filepath)
    if not filepath.exists():
        if not silent:
            print(f"Aviso: {filepath} não encontrado")
        return None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if not lines:
                return None
            last_line = lines[-1].strip()
            values = last_line.split()
            if not values:
                return None
            return float(values[-1])
    except Exception as e:
        print(f"Erro ao ler {filepath}: {e}")
        return None


def get_experiment_data(base_path, model_name, experiment_name):
    """Extrai os dados de um experimento específico"""
    avg_path = Path(base_path) / "output" / model_name / experiment_name / "average"
    
    if not avg_path.exists():
        print(f"Aviso: {avg_path} não encontrado")
        return None
    
    data = {
        'accuracy': read_final_value(avg_path / "accuracy.dat"),
        'client_execution_time': read_final_value(avg_path / "client_execution_time.dat"),
        'client_encrypt_time': read_final_value(avg_path / "client_encrypt_time.dat"),
        'client_decrypt_time': read_final_value(avg_path / "client_decrypt_time.dat"),
        'client_train_time': read_final_value(avg_path / "client_train_time.dat"),
        'server_execution_time': read_final_value(avg_path / "server_execution_time.dat"),
        'server_aggregation_time': read_final_value(avg_path / "server_aggregation_time.dat"),
        'server_decrypt_time': read_final_value(avg_path / "server_decrypt_time.dat"),
        'client_size': read_final_value(avg_path / "client_size.dat"),
        # setup_time só existe para new_ckks-fl; silencia o aviso para os demais
        'setup_time': read_final_value(avg_path / "setup_time.dat", silent=True),
    }
    
    # Converter tamanho de bytes para MB
    if data['client_size'] is not None:
        data['client_size_mb'] = data['client_size'] / (1024 * 1024)
    else:
        data['client_size_mb'] = None
    
    return data


def generate_table_simple(model_name, model_label, experiments_config, base_path):
    """Gera tabela LaTeX simples (formato do exemplo fornecido)"""
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Resultados experimentais no dataset {model_label}}}")
    label = model_name.replace("-", "").replace("_", "")
    lines.append(f"\\label{{tab:{label}}}")
    lines.append(r"\begin{tabular}{|l|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Método} & \textbf{Acurácia} & \textbf{Tempo/rodada (s)} & \textbf{Comunicação (MB)} \\")
    lines.append(r"\hline")
    
    for exp_config in experiments_config:
        exp_name = exp_config['folder']
        exp_label = exp_config['label']
        is_new = exp_config.get('highlight', False)
        
        data = get_experiment_data(base_path, model_name, exp_name)
        
        if data is None or data['accuracy'] is None:
            continue
        
        # Calcular tempo total por rodada (cliente + servidor)
        total_time = 0
        if data['client_execution_time'] is not None:
            total_time += data['client_execution_time']
        if data['server_execution_time'] is not None:
            total_time += data['server_execution_time']
        
        acc = f"{data['accuracy']:.4f}"
        time_str = f"{total_time:.2f}"
        comm = f"{data['client_size_mb']:.2f}" if data['client_size_mb'] is not None else "N/A"
        
        if is_new:
            lines.append(f"\\textbf{{{exp_label}}} & \\textbf{{{acc}}} & \\textbf{{{time_str}}} & \\textbf{{{comm}}} \\\\")
        else:
            lines.append(f"{exp_label} & {acc} & {time_str} & {comm} \\\\")
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_table_detailed_client(model_name, model_label, experiments_config, base_path):
    """Gera tabela LaTeX detalhada com tempos do cliente"""
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Tempos de execução no cliente - {model_label}}}")
    label = model_name.replace("-", "").replace("_", "") + "client"
    lines.append(f"\\label{{tab:{label}}}")
    lines.append(r"\begin{tabular}{|l|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Método} & \textbf{Treino (s)} & \textbf{Encrypt (s)} & \textbf{Decrypt (s)} & \textbf{Total (s)} & \textbf{Tamanho (MB)} \\")
    lines.append(r"\hline")
    
    for exp_config in experiments_config:
        exp_name = exp_config['folder']
        exp_label = exp_config['label']
        is_new = exp_config.get('highlight', False)
        
        data = get_experiment_data(base_path, model_name, exp_name)
        
        if data is None:
            continue
        
        train = f"{data['client_train_time']:.2f}" if data['client_train_time'] is not None else "N/A"
        encrypt = f"{data['client_encrypt_time']:.2f}" if data['client_encrypt_time'] is not None else "0.00"
        decrypt = f"{data['client_decrypt_time']:.2f}" if data['client_decrypt_time'] is not None else "0.00"
        total = f"{data['client_execution_time']:.2f}" if data['client_execution_time'] is not None else "N/A"
        size = f"{data['client_size_mb']:.2f}" if data['client_size_mb'] is not None else "N/A"
        
        if is_new:
            lines.append(f"\\textbf{{{exp_label}}} & \\textbf{{{train}}} & \\textbf{{{encrypt}}} & \\textbf{{{decrypt}}} & \\textbf{{{total}}} & \\textbf{{{size}}} \\\\")
        else:
            lines.append(f"{exp_label} & {train} & {encrypt} & {decrypt} & {total} & {size} \\\\")
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_table_detailed_server(model_name, model_label, experiments_config, base_path):
    """Gera tabela LaTeX detalhada com tempos do servidor"""
    
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Tempos de execução no servidor - {model_label}}}")
    label = model_name.replace("-", "").replace("_", "") + "server"
    lines.append(f"\\label{{tab:{label}}}")
    lines.append(r"\begin{tabular}{|l|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Método} & \textbf{Setup/Fase 1 (s)} & \textbf{Decrypt (s)} & \textbf{Agregação (s)} & \textbf{Total (s)} \\")
    lines.append(r"\hline")

    for exp_config in experiments_config:
        exp_name = exp_config['folder']
        exp_label = exp_config['label']
        is_new = exp_config.get('highlight', False)

        data = get_experiment_data(base_path, model_name, exp_name)

        if data is None:
            continue

        setup = f"{data['setup_time']:.2f}" if data.get('setup_time') is not None else "N/A"
        decrypt = f"{data['server_decrypt_time']:.2f}" if data['server_decrypt_time'] is not None else "0.00"
        aggregation = f"{data['server_aggregation_time']:.2f}" if data['server_aggregation_time'] is not None else "N/A"
        total = f"{data['server_execution_time']:.2f}" if data['server_execution_time'] is not None else "N/A"

        if is_new:
            lines.append(f"\\textbf{{{exp_label}}} & \\textbf{{{setup}}} & \\textbf{{{decrypt}}} & \\textbf{{{aggregation}}} & \\textbf{{{total}}} \\\\")
        else:
            lines.append(f"{exp_label} & {setup} & {decrypt} & {aggregation} & {total} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_table_scalability(model_name, model_label, scalability_configs, base_path):
    """Gera tabela LaTeX de escalabilidade: overhead vs número de clientes P."""

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Análise de escalabilidade: overhead do \\gls{{newckks}} vs. número de clientes $P$ -- {model_label}}}")
    label = model_name.replace("-", "").replace("_", "") + "scalability"
    lines.append(f"\\label{{tab:{label}}}")
    lines.append(r"\begin{tabular}{|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{$P$} & \textbf{Acurácia} & \textbf{Tempo/rodada (s)} & \textbf{Comunicação (MB)} & \textbf{Setup (s)} \\")
    lines.append(r"\hline")

    for cfg in scalability_configs:
        exp_name = cfg['folder']
        p_val = cfg['p']
        data = get_experiment_data(base_path, model_name, exp_name)
        if data is None:
            continue
        acc = f"{data['accuracy']:.4f}" if data['accuracy'] is not None else "N/A"
        total_time = 0.0
        if data['client_execution_time'] is not None:
            total_time += data['client_execution_time']
        if data['server_execution_time'] is not None:
            total_time += data['server_execution_time']
        time_str = f"{total_time:.2f}"
        comm = f"{data['client_size_mb']:.2f}" if data['client_size_mb'] is not None else "N/A"
        setup = f"{data['setup_time']:.2f}" if data.get('setup_time') is not None else "N/A"
        lines.append(f"{p_val} & {acc} & {time_str} & {comm} & {setup} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_table_complete(model_name, model_label, experiments_config, base_path):
    """Gera tabela LaTeX completa com todas as informações"""
    
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Resultados completos dos experimentos - {model_label}}}")
    label = model_name.replace("-", "").replace("_", "") + "complete"
    lines.append(f"\\label{{tab:{label}}}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Método} & \textbf{Acurácia} & ")
    lines.append(r"\multicolumn{4}{c|}{\textbf{Tempo Cliente (s)}} & ")
    lines.append(r"\multicolumn{2}{c|}{\textbf{Tempo Servidor (s)}} & ")
    lines.append(r"\textbf{Comunicação} \\")
    lines.append(r"\cline{3-8}")
    lines.append(r" & & \textbf{Treino} & \textbf{Encrypt} & \textbf{Decrypt} & \textbf{Total} & ")
    lines.append(r"\textbf{Agregação} & \textbf{Decrypt} & \textbf{(MB)} \\")
    lines.append(r"\hline")
    
    for exp_config in experiments_config:
        exp_name = exp_config['folder']
        exp_label = exp_config['label']
        is_new = exp_config.get('highlight', False)
        
        data = get_experiment_data(base_path, model_name, exp_name)
        
        if data is None:
            continue
        
        # Valores
        acc = f"{data['accuracy']:.4f}" if data['accuracy'] is not None else "N/A"
        c_train = f"{data['client_train_time']:.2f}" if data['client_train_time'] is not None else "N/A"
        c_encrypt = f"{data['client_encrypt_time']:.2f}" if data['client_encrypt_time'] is not None else "0.00"
        c_decrypt = f"{data['client_decrypt_time']:.2f}" if data['client_decrypt_time'] is not None else "0.00"
        c_total = f"{data['client_execution_time']:.2f}" if data['client_execution_time'] is not None else "N/A"
        s_agg = f"{data['server_aggregation_time']:.2f}" if data['server_aggregation_time'] is not None else "N/A"
        s_decrypt = f"{data['server_decrypt_time']:.2f}" if data['server_decrypt_time'] is not None else "0.00"
        comm = f"{data['client_size_mb']:.2f}" if data['client_size_mb'] is not None else "N/A"
        
        if is_new:
            lines.append(f"\\textbf{{{exp_label}}} & \\textbf{{{acc}}} & \\textbf{{{c_train}}} & \\textbf{{{c_encrypt}}} & \\textbf{{{c_decrypt}}} & \\textbf{{{c_total}}} & \\textbf{{{s_agg}}} & \\textbf{{{s_decrypt}}} & \\textbf{{{comm}}} \\\\")
        else:
            lines.append(f"{exp_label} & {acc} & {c_train} & {c_encrypt} & {c_decrypt} & {c_total} & {s_agg} & {s_decrypt} & {comm} \\\\")
    
    lines.append(r"\hline")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def main():
    base_path = Path(__file__).parent
    tables_dir = base_path / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    # Configuração dos experimentos
    experiments = [
        {'folder': 'baseline-fl', 'label': r'\gls{fedavg} (baseline)', 'highlight': False},
        {'folder': 'full_ckks-fl', 'label': r'\gls{fhe} single-key', 'highlight': False},
        {'folder': 'selective_ckks-fl-10', 'label': r'M-\gls{fhe} 10\%', 'highlight': False},
        {'folder': 'selective_ckks-fl-20', 'label': r'M-\gls{fhe} 20\%', 'highlight': False},
        {'folder': 'selective_ckks-fl-40', 'label': r'M-\gls{fhe} 40\%', 'highlight': False},
        {'folder': 'selective_ckks-fl-80', 'label': r'M-\gls{fhe} 80\%', 'highlight': False},
        {'folder': 'new_ckks-fl', 'label': r'\acrshort{newckks}', 'highlight': True},
    ]
    
    # Configuração dos modelos
    models = [
        {'name': 'mlp-mnist', 'label': r'\gls{mnist}'},
        {'name': 'resnet20-cifar10-iid', 'label': r'\gls{cifar}-10 \gls{iid}'},
    ]
    
    # Gerar tabelas para cada modelo
    for model in models:
        model_name = model['name']
        model_label = model['label']
        
        print(f"Gerando tabelas para {model_name}...")
        
        # Tabela simples (formato do exemplo)
        table_simple = generate_table_simple(model_name, model_label, experiments, base_path)
        simple_file = tables_dir / f"{model_name}_simple.tex"
        with open(simple_file, 'w', encoding='utf-8') as f:
            f.write(table_simple)
        print(f"  - Tabela simples salva em: {simple_file}")
        
        # Tabela detalhada - Cliente
        table_client = generate_table_detailed_client(model_name, model_label, experiments, base_path)
        client_file = tables_dir / f"{model_name}_client.tex"
        with open(client_file, 'w', encoding='utf-8') as f:
            f.write(table_client)
        print(f"  - Tabela de tempos do cliente salva em: {client_file}")
        
        # Tabela detalhada - Servidor
        table_server = generate_table_detailed_server(model_name, model_label, experiments, base_path)
        server_file = tables_dir / f"{model_name}_server.tex"
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write(table_server)
        print(f"  - Tabela de tempos do servidor salva em: {server_file}")
        
        # Tabela completa (NOVA)
        table_complete = generate_table_complete(model_name, model_label, experiments, base_path)
        complete_file = tables_dir / f"{model_name}_complete.tex"
        with open(complete_file, 'w', encoding='utf-8') as f:
            f.write(table_complete)
        print(f"  - Tabela completa salva em: {complete_file}")
    
    # Tabelas de escalabilidade
    # P=10 usa o experimento padrão new_ckks-fl (clients_qtd=10 no config)
    scalability_configs = [
        {'folder': 'new_ckks-fl-p5',  'p': 5},
        {'folder': 'new_ckks-fl',     'p': 10},
        {'folder': 'new_ckks-fl-p20', 'p': 20},
    ]
    for model in models:
        model_name = model['name']
        model_label = model['label']
        table_scal = generate_table_scalability(model_name, model_label, scalability_configs, base_path)
        scal_file = tables_dir / f"{model_name}_scalability.tex"
        with open(scal_file, 'w', encoding='utf-8') as f:
            f.write(table_scal)
        print(f"  - Tabela de escalabilidade salva em: {scal_file}")

    print("\nTabelas LaTeX geradas com sucesso!")
    print(f"Localização: {tables_dir}")


if __name__ == "__main__":
    main()
