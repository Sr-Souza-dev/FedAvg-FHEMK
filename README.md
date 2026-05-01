# Federated Learning com CKKS

Este repositório reúne experimentos de aprendizado federado com Flower e PyTorch, comparando uma linha de base sem criptografia com diferentes estratégias baseadas em criptografia homomórfica aproximada CKKS. O projeto também inclui scripts para executar simulações, organizar resultados, gerar análises estatísticas, gráficos e tabelas LaTeX.

## Visão geral

O fluxo principal é:

1. Clientes Flower treinam localmente uma partição do dataset.
2. Cada cliente envia pesos ou atualizações para o servidor.
3. O servidor agrega os modelos por FedAvg.
4. Dependendo do experimento, os pesos trafegam em texto claro, totalmente cifrados ou parcialmente cifrados.
5. As métricas de treino, avaliação, tempo e comunicação são salvas em arquivos `.dat`.
6. Os scripts de análise calculam médias, estatísticas, gráficos e tabelas comparativas.

Os experimentos disponíveis são:

- `baseline-fl`: FedAvg sem criptografia.
- `new_ckks-fl`: implementação própria de CKKS usada como baseline cifrado.
- `full_ckks-fl`: CKKS com Pyfhel cifrando o vetor completo de pesos.
- `selective_ckks-fl-10`: CKKS seletivo com 10% dos parâmetros cifrados.
- `selective_ckks-fl-20`: CKKS seletivo com 20% dos parâmetros cifrados.
- `selective_ckks-fl-40`: CKKS seletivo com 40% dos parâmetros cifrados.
- `selective_ckks-fl-80`: CKKS seletivo com 80% dos parâmetros cifrados.

## Estrutura de pastas

```text
.
├── analysis/                  # Rotinas de média, estatística, gráficos e tabelas comparativas
├── articles/                  # Artigos e materiais de apoio relacionados ao projeto
├── backup/                    # Backups de execuções anteriores gerados pelo script principal
├── experiments/               # Aplicações Flower com as variantes experimentais
├── images/                    # Figuras usadas na documentação/artigo
├── logs/                      # Logs opcionais das simulações
├── models/                    # Registro, carregamento, datasets e arquiteturas dos modelos
├── output/                    # Métricas brutas e médias dos experimentos
├── plots/                     # Gráficos gerados pela análise
├── public/                    # Dados públicos auxiliares da implementação CKKS própria
├── tables/                    # Tabelas LaTeX geradas a partir dos resultados
├── utils/                     # Funções utilitárias compartilhadas
├── experiment_config.py       # Leitor da configuração central dos experimentos
├── experiments_config.toml    # Número de rodadas, clientes e épocas por experimento
├── generate_analysis.py       # Gera médias, estatísticas e gráficos para um modelo
├── generate_latex_tables.py   # Gera tabelas LaTeX finais
├── main.py                    # Executa bateria completa de experimentos e análises
├── main_analysis.py           # Executa apenas as análises sobre resultados existentes
├── run_experiments.py         # Menu interativo para rodar um experimento ou todos
├── run_test.py                # Executa as suítes de teste dos experimentos
└── requirements.txt           # Dependências Python do projeto
```

## Módulos principais

### `experiments/`

Contém as aplicações Flower executáveis por `flwr run .` ou pelo script `run_experiments.py`.

- `baseline-fl/`: experimento de referência sem criptografia. O servidor aplica FedAvg diretamente sobre os pesos recebidos.
- `new_ckks-fl/`: implementação própria de CKKS, incluindo codificação, polinômios, NTT, amostragem de ruído, criptogramas e operações homomórficas.
- `full_ckks-fl/`: usa Pyfhel para cifrar todos os pesos do modelo antes da agregação. A estratégia `HomomorphicFedAvg` agrega os ciphertexts e descriptografa o vetor agregado no servidor.
- `selective_ckks-fl/`: cifra apenas uma máscara dos parâmetros. A estratégia `SelectiveHomomorphicFedAvg` combina agregação em claro para o complemento da máscara e agregação homomórfica para as posições protegidas.

Cada experimento possui:

- `fl_simulation/client_app.py`: cliente Flower, treino local, serialização de pesos e coleta de métricas.
- `fl_simulation/server_app.py`: servidor Flower, configuração de rodadas e avaliação.
- `fl_simulation/strategies/`: variações de FedAvg usadas em cada experimento.
- `tests/`: testes unitários específicos.
- `pyproject.toml`: configuração da aplicação Flower.

### `models/`

Define os modelos e datasets disponíveis:

- `models/registry.py`: lista os modelos executáveis e resolve o modelo ativo via variável `AQUIPLACA_MODEL_NAME`.
- `models/mlp_mnist/`: modelo convolucional simples para KMNIST particionado de forma IID.
- `models/resnet20_cifar10_iid/`: ResNet-20 para CIFAR-10 com partições IID.
- `models/resnet20_cifar10_noniid/`: ResNet-20 para CIFAR-10 com partições non-IID por Dirichlet.
- `models/resnet20_cifar10/shared.py`: blocos básicos e arquitetura ResNet-20.
- `models/common/`: funções comuns de treino, avaliação, transformação de imagens e manipulação de pesos.

### `analysis/`

Responsável pelo pós-processamento:

- `averager.py`: calcula médias entre execuções repetidas.
- `data_utils.py`: lê e escreve arquivos `.dat`, localiza experimentos e define estilos/nomes das métricas.
- `plotters.py`: gera gráficos por experimento, comparações, boxplots, barras e trade-off acurácia versus tempo.
- `statistics.py`: calcula média, mediana, desvio padrão, mínimo, máximo, valor final, delta, tendência e AUC.
- `service.py`: orquestra o pipeline completo de análise.

### `utils/`

Funções compartilhadas:

- `files.py`: controle de diretórios, logs, saída por modelo e identificadores de execução.
- `weights.py` e `flatten.py`: flatten/unflatten dos pesos do modelo.
- `flwr_inline_backend.py`: suporte ao backend inline do Flower quando Ray não está disponível.
- `numbers.py`, `uuid.py`, `vandermode.py`: utilidades auxiliares usadas pelas implementações CKKS.

### `output/`, `plots/`, `tables/` e `logs/`

- `output/<modelo>/<experimento>/<run>/`: métricas brutas de cada execução.
- `output/<modelo>/<experimento>/average/`: médias calculadas entre execuções.
- `plots/<modelo>/<experimento>/`: gráficos individuais por experimento.
- `plots/<modelo>/analysis/`: gráficos comparativos e tabelas CSV/LaTeX da análise.
- `tables/`: tabelas LaTeX geradas pelo script `generate_latex_tables.py`.
- `logs/`: logs opcionais da execução, separados por modelo e experimento.

## Conceitos matemáticos utilizados

### Aprendizado federado

O projeto usa aprendizado federado horizontal: cada cliente possui uma partição dos dados, treina localmente e envia seus pesos ao servidor. O servidor não acessa diretamente os dados dos clientes.

### FedAvg

A agregação principal é o Federated Averaging. Se o cliente `k` treinou com `n_k` exemplos e retornou pesos `w_k`, o servidor calcula:

```text
w_global = sum_k (n_k / N) * w_k
N = sum_k n_k
```

Assim, clientes com mais exemplos têm maior peso na média global.

### Flatten e unflatten dos pesos

Para criptografar ou serializar os modelos, os tensores de pesos são convertidos em um único vetor numérico. Depois da agregação, esse vetor é reconstruído no formato original das camadas.

### CKKS

CKKS é um esquema de criptografia homomórfica aproximada para números reais ou complexos. Ele permite realizar operações como soma e multiplicação por escalar diretamente sobre dados cifrados. Isso é útil no FedAvg porque a média ponderada pode ser calculada sem revelar todos os pesos individuais em texto claro.

No projeto, CKKS aparece de duas formas:

- `new_ckks-fl`: implementação própria com polinômios, codificação aproximada, ruído, chaves, criptogramas e NTT.
- `full_ckks-fl` e `selective_ckks-fl`: uso de contexto CKKS com Pyfhel.

### NTT e aritmética polinomial

A implementação própria usa polinômios e Number Theoretic Transform para acelerar operações em anéis polinomiais. Em termos práticos, isso melhora a eficiência de multiplicações e convoluções sobre coeficientes inteiros modulares.

### Máscara seletiva

Nos experimentos seletivos, somente uma fração dos parâmetros é cifrada. A máscara define quais posições do vetor de pesos serão protegidas. O restante é agregado em claro. A seleção é guiada por escores de importância derivados de gradientes locais, buscando proteger parâmetros mais relevantes com menor custo computacional e de comunicação.

### Métricas estatísticas

A análise calcula estatísticas como média, mediana, desvio padrão, valor final, variação entre início e fim, tendência linear e AUC aproximada pela regra do trapézio. Essas métricas ajudam a comparar desempenho, estabilidade, custo de tempo e comunicação.

## Configuração dos experimentos

A configuração compartilhada fica em `experiments_config.toml`. Atualmente, os experimentos usam:

```toml
num_rounds = 40
clients_qtd = 30
epochs = 5
```

Esses valores são carregados por `experiment_config.py` e usados pelos scripts de execução. Para alterar a quantidade de rodadas, clientes ou épocas locais, edite esse arquivo antes de rodar os experimentos.

## Como executar

Os comandos abaixo assumem que você está na raiz do repositório:

### 1. Criar o ambiente virtual

No Linux/macOS:

```bash
python3 -m venv .venv
```

No Windows:

```powershell
python -m venv .venv
```

### 2. Ativar o ambiente virtual

No Linux/macOS:

```bash
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Atualizar `pip`

```bash
python -m pip install --upgrade pip
```

### 4. Instalar as dependências

```bash
python -m pip install -r requirements.txt
```

Observação: o script `run_experiments.py` também tenta instalar as dependências automaticamente dentro de `.venv`, mas instalar manualmente antes facilita identificar erros de ambiente.

### 5. Executar um experimento pelo menu interativo

```bash
python run_experiments.py
```

O script irá perguntar:

1. Se deseja registrar logs da execução.
2. Qual modelo deseja usar:
   - `MLP - MNIST (IID)`
   - `ResNet-20 - CIFAR10 IID`
   - `ResNet-20 - CIFAR10 non-IID`
3. Qual experimento deseja executar:
   - `1`: baseline sem criptografia
   - `2`: baseline CKKS próprio
   - `3`: CKKS completo com Pyfhel
   - `4`: CKKS seletivo 10%
   - `5`: CKKS seletivo 20%
   - `6`: CKKS seletivo 40%
   - `7`: CKKS seletivo 80%
   - `0`: todos em sequência

Exemplo de fluxo:

```text
Deseja registrar logs desta execucao? (s/n) [s]: n
Escolha o modelo para os experimentos:
  1 - MLP - MNIST (IID) (mlp-mnist)
  2 - ResNet-20 - CIFAR10 IID (resnet20-cifar10-iid)
  3 - ResNet-20 - CIFAR10 non-IID (resnet20-cifar10-noniid)
Digite sua opcao: 1
Digite sua opcao (0/1/2/3/4/5/6/7): 0
```

Os resultados serão gravados em:

```text
output/<modelo>/<experimento>/<numero_da_execucao>/
```

### 6. Executar todos os experimentos e modelos automaticamente

Para uma bateria completa, use:

```bash
python main.py
```

Esse script:

1. Move resultados antigos de `output/` e `plots/` para `backup/run_<n>/`.
2. Instala dependências no `.venv`.
3. Desativa logs detalhados para reduzir custo de escrita.
4. Pergunta quantas execuções completas deseja realizar.
5. Executa todos os experimentos para todos os modelos registrados.
6. Gera análises ao final de cada modelo.
7. Move os resultados finais para `backup/run_<n>/`.

Essa é a opção mais adequada quando você deseja gerar dados comparáveis entre todos os métodos.

### 7. Executar um experimento diretamente com Flower

Também é possível entrar na pasta de um experimento e rodar Flower manualmente:

```bash
cd experiments/baseline-fl
flwr run .
```

Para voltar à raiz:

```bash
cd ../..
```

Na prática, prefira `run_experiments.py`, porque ele configura automaticamente variáveis de ambiente, modelo ativo, nome do experimento, máscara seletiva e backend de simulação.

### 8. Gerar análises a partir de resultados existentes

Se os experimentos já foram executados e existem arquivos em `output/`, rode:

```bash
python main_analysis.py
```

Esse script percorre os modelos registrados e executa `generate_analysis.py` para cada modelo com resultados disponíveis.

Para analisar manualmente apenas o modelo ativo, defina `AQUIPLACA_MODEL_NAME` e execute `generate_analysis.py`.

Linux/macOS:

```bash
export AQUIPLACA_MODEL_NAME=mlp-mnist
python generate_analysis.py
```

Windows PowerShell:

```powershell
$env:AQUIPLACA_MODEL_NAME="mlp-mnist"
python generate_analysis.py
```

Modelos aceitos:

```text
mlp-mnist
resnet20-cifar10-iid
resnet20-cifar10-noniid
```

A análise gera:

- médias em `output/<modelo>/<experimento>/average/`;
- gráficos por experimento em `plots/<modelo>/<experimento>/`;
- gráficos comparativos em `plots/<modelo>/analysis/`;
- resumo estatístico em `plots/<modelo>/analysis/metrics_summary.csv`;
- tabela comparativa em `plots/<modelo>/analysis/comparative_table.csv`;
- tabela LaTeX em `plots/<modelo>/analysis/comparative_table.tex`.

### 9. Gerar tabelas LaTeX finais

Após executar experimentos e análises, gere as tabelas finais com:

```bash
python generate_latex_tables.py
```

O script lê os arquivos médios em:

```text
output/<modelo>/<experimento>/average/
```

e grava tabelas em:

```text
tables/
```

Arquivos gerados:

- `<modelo>_simple.tex`: tabela resumida com acurácia, tempo por rodada e comunicação.
- `<modelo>_client.tex`: tempos detalhados do cliente.
- `<modelo>_server.tex`: tempos detalhados do servidor.
- `<modelo>_complete.tex`: tabela completa.

Observação: no estado atual, `generate_latex_tables.py` gera tabelas para `mlp-mnist` e `resnet20-cifar10-iid`.

### 10. Rodar os testes

Para validar as suítes de teste dos experimentos:

```bash
python run_test.py
```

O script executa `pytest` em:

- `experiments/baseline-fl`
- `experiments/new_ckks-fl`
- `experiments/full_ckks-fl`
- `experiments/selective_ckks-fl`

## Variáveis de ambiente úteis

- `AQUIPLACA_MODEL_NAME`: define o modelo ativo.
- `AQUIPLACA_EXPERIMENT_NAME`: define o nome do experimento usado nos diretórios de saída.
- `AQUIPLACA_MASK_RATIO`: define a razão de máscara para experimentos seletivos.
- `AQUIPLACA_ENABLE_LOGS`: controla logs detalhados (`1` habilita, `0` desabilita).

Normalmente você não precisa definir essas variáveis manualmente, pois `run_experiments.py` faz isso.

## Saídas esperadas

Depois de uma execução, espere encontrar arquivos como:

```text
output/mlp-mnist/baseline-fl/1/accuracy.dat
output/mlp-mnist/baseline-fl/1/loss.dat
output/mlp-mnist/baseline-fl/1/client_execution_time.dat
output/mlp-mnist/baseline-fl/average/accuracy.dat
plots/mlp-mnist/baseline-fl/accuracy.png
plots/mlp-mnist/analysis/accuracy_comparison.png
plots/mlp-mnist/analysis/metrics_summary.csv
tables/mlp-mnist_complete.tex
```

Os arquivos `.dat` são matrizes numéricas separadas por tabulação. Em geral, cada linha representa uma rodada e as colunas representam clientes e/ou valores agregados.

## Observações práticas

- A primeira execução pode demorar porque os datasets são baixados via `flwr-datasets`.
- Experimentos com CKKS completo tendem a ser mais lentos e consumir mais memória.
- Se Ray não estiver disponível para a versão de Python usada, `run_experiments.py` tenta usar o backend inline do Flower.
- Para comparar métodos de forma justa, mantenha `experiments_config.toml` igual entre as execuções.
- Para gerar médias estatisticamente mais estáveis, rode múltiplas execuções completas com `main.py`.
