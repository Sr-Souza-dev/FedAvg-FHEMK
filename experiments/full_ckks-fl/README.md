# full-ckks-fl

Implementação de um experimento tradicional de FL com Flower utilizando criptografia homomórfica CKKS em todos os pesos do modelo. O projeto foi inspirado em `new_ckks-fl`, mas utiliza a biblioteca [Pyfhel](https://github.com/ibarrond/Pyfhel) para prover um conjunto único de chaves CKKS compartilhadas entre todos os clientes.

## Estrutura

```
full_ckks-fl/
+-- fl_simulation/
¦   +-- client_app.py      # Cliente Flower com PyTorch + CKKS
¦   +-- server_app.py      # Server Flower com FedAvg homomórfico
¦   +-- crypto/            # Gerenciamento do contexto CKKS
¦   +-- model/             # Rede LeNet e carregamento KMNIST
¦   +-- strategies/        # Estratégia FedAvg customizada
+-- utils/                 # Funções de flatten/unflatten e IO simples
+-- keys/                  # Contexto + chaves CKKS (geradas em runtime)
+-- pyproject.toml
+-- requirements.txt
+-- README.md
```

## Como executar

1. Instale as dependências: `pip install -e .`
2. Gere e sincronize o contexto/chaves executando o servidor pelo menos uma vez (`flwr run .`). O servidor cria `keys/context.ckks`, `keys/private.key`, etc., que são compartilhados com todos os clientes.
3. Rode a simulação local dentro de `fl_simulation`: `flwr run .`

### Configurações importantes
- `num-server-rounds`: total de rodadas do FL.
- `fraction-fit/evaluate`: frações de clientes utilizados por rodada.
- `is-encrypted`: quando `1`, o cliente sempre criptografa o vetor de pesos completo antes de devolver ao servidor.

O servidor agrega os modelos criptografados via operações CKKS (soma ponderada + divisão escalar) e somente descriptografa o resultado final para atualizar o modelo global.
