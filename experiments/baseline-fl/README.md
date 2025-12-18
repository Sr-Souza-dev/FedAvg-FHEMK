# Baseline FL

Este experimento executa o mesmo fluxo de treino federado dos demais diretórios porém **sem aplicar criptografia** sobre os pesos trocados entre clientes e servidor. Ele reutiliza o mesmo modelo CNN e o particionamento IID do KMNIST, mas transmite os vetores de pesos em texto bruto para servir como linha de base.

## Execução

```bash
flwr run .
```

ou via utilitário da raiz:

```bash
python ..\..\run_experiments.py
```

## Estrutura

- `fl_simulation/client_app.py`: implementação do cliente Flower
- `fl_simulation/server_app.py`: configuração do servidor e agregação
- `tests/`: pequenos testes unitários para validar o modelo base

Os artefatos de log e métricas são gravados nos diretórios globais `logs/` e `output/`, seguindo o mesmo padrão dos demais experimentos.
