# Análise da Série Histórica de Cotas do Rio Madeira em Humaitá (AM)

Projeto de preparação e análise exploratória da série temporal diária de **cota fluvial** (nível do rio) da estação fluviométrica de **Humaitá (código ANA 15630000)**, no Rio Madeira, com o objetivo de construir um dataset limpo e contínuo para o treinamento de modelos de deep learning de previsão de nível.

## O que o projeto faz

O repositório tem dois notebooks independentes: [`index.ipynb`](index.ipynb) (exploração + experimento SARIMA × TimesFM 3.0) e [`forecasting.ipynb`](forecasting.ipynb) (análise, preenchimento das lacunas e comparação de baselines, ML e deep learning). Em resumo, o projeto:

1. **Carrega os dados brutos** exportados do Hidroweb (ANA) em formato largo — uma linha por mês, com uma coluna por dia (`Cota01` … `Cota31`) e uma coluna de status por dia (`Cota01Status` … `Cota31Status`).
2. **Explora as colunas de metadados** (`NivelConsistencia`, `hora`, `MediaDiaria`, `TipoMedicaoCotas`) para entender a natureza dos registros.
3. **Filtra apenas os dados consistidos** (`NivelConsistencia == 2`), ou seja, que passaram por análise especializada da ANA.
4. **Converte o formato largo para longo**, gerando uma série diária `(date, river_stage)` ordenada cronologicamente e descartando datas inválidas (ex.: 31 de fevereiro) e valores ausentes.
5. **Recorta o período de 01/01/1991 a 31/12/2023**, alinhado ao ciclo hidrológico anual e livre das grandes lacunas históricas.
6. **Identifica os dias faltantes** nesse intervalo e verifica a possibilidade de preenchê-los com dados da estação vizinha **UHE Santo Antônio (código ANA 15629000)**, cuja dinâmica é fortemente correlacionada com a de Humaitá.
7. **Plota a série histórica** com linhas de referência de média, máxima e mínima.

## Fonte dos dados

Os dados são públicos e foram obtidos no Hidroweb da Agência Nacional de Águas e Saneamento Básico (ANA):

- https://www.snirh.gov.br/hidroweb/serieshistoricas
- Estação principal: **Humaitá — 15630000**
- Estação auxiliar: **UHE Santo Antônio — 15629000**

A série de cotas de Humaitá cobre de **março de 1931 a março de 2026**. Apenas os registros até **dezembro de 2023** possuem nível de consistência 2; os anos seguintes são compostos por dados brutos.

## Decisões sobre os dados

- **Nível de consistência:** usa-se somente o nível 2 (dados tratados), por oferecer maior completude e menor ruído. Todos esses registros são médias diárias medidas por escala (régua).
- **Lacunas históricas:** há um gap entre **1949 e 1967** não sinalizado no dataset, e outro de 5 meses entre **junho e novembro de 1990**. Por isso a série usada começa em **01/01/1991**.
- **Dias faltantes em 2014:** entre abril e maio de 2014 existem 21 dias sem registro (15/04–30/04 e 12/05–16/05). A proposta é preenchê-los com os dados da estação Santo Antônio — abordagem preferida à interpolação linear, que descaracterizaria a dinâmica da curva de cota. A correlação entre as estações foi verificada pela sobreposição dos dados de 01/06 a 20/06/2014.
- Os dados de Santo Antônio têm majoritariamente **status 3 (medição duvidosa)**, o que não invalida seu uso para imputação dada a correlação clara entre as estações.

## Estrutura do repositório

```
.
├── index.ipynb                          # Exploração + experimento SARIMA × TimesFM 3.0
├── forecasting.ipynb                    # Análise, preenchimento de lacunas e comparação de modelos
├── results_forecast_h7.csv              # Métricas de todos os modelos, H = 7 (gerado pelo forecasting.ipynb)
├── results_forecast_multi_h.csv         # Métricas por horizonte 7/15/30 (gerado pelo forecasting.ipynb)
├── results_forecast_multi_h_anom_stacking.csv  # Idem, DL sobre a anomalia + stacking
├── results_rigor_h30.csv                # RMSE médio entre sementes e config. ajustada, H = 30
├── README.md
└── data/
    ├── humaita_cota_diaria_1991_2023.csv   # Série diária contínua (gerada pelo forecasting.ipynb)
    ├── humaita/                # Estação 15630000
    │   ├── 15630000_Cotas.csv  # Série de cotas (usada nos notebooks)
    │   ├── 15630000_Vazoes.csv
    │   ├── 15630000_CurvaDescarga.csv
    │   ├── 15630000_ResumoDescarga.csv
    │   ├── 15630000_PerfilTransversal.csv
    │   ├── 15630000_Sedimentos.csv
    │   └── 15630000_QualAgua.csv
    └── santo_antonio/          # Estação 15629000
        ├── 15629000_Cotas.csv  # Série de cotas (usada nos notebooks)
        └── ...                 # Demais séries exportadas do Hidroweb
```

## Notebooks

### `index.ipynb` — exploração e experimento SARIMA × TimesFM 3.0
Leitura dos CSVs do Hidroweb, distribuição das colunas de metadados, conversão wide → long, estatística descritiva, sazonalidade, autocorrelação, decomposição e eventos extremos. Em seguida, o experimento de previsão: seleção da estrutura SARIMA com dados anteriores ao teste, backtesting do SARIMA, TimesFM 3.0 zero-shot (MLX / Apple Silicon), métricas MAE/RMSE/R² por horizonte e registro do ambiente. Salva as saídas em `results/`.

**Protocolo:** teste a partir de 2021-01-01, horizontes 1, 7, 30 e 90 dias, uma origem a cada 7 dias.

### `forecasting.ipynb` — análise e comparação de modelos
**Protocolo:** teste a partir de 2020-01-01, horizontes 7, 15 e 30 dias, uma origem a cada 7 dias, métricas RMSE/MAE/NSE/KGE. Difere do `index.ipynb` no início do teste, nos horizontes e nas métricas — para juntar os resultados dos dois notebooks em uma única tabela do artigo é preciso alinhar esses três pontos.
1. **Análise dos dados:** cobertura e lacunas, distribuição (bimodal cheia/seca), ciclo anual, ACF/PACF, teste ADF, decomposição STL e relação com Santo Antônio (r = 0,9997).
2. **Preenchimento das lacunas por interpolação linear** (21 dias em 2014), com simulação do erro esperado de cada método (linear, PCHIP, Akima, cúbico) e da alternativa de transferir Santo Antônio.
3. **Protocolo:** treino 1991–2015, validação 2016–2019, teste 2020–2023; horizonte de 7 dias; 208 janelas de previsão no teste, iguais para todos os modelos; métricas RMSE, MAE, NSE e KGE.
4. **Modelos comparados:**
   - Baselines: persistência, naive sazonal, climatologia, persistência da anomalia
   - Estatístico: ARIMA sobre a anomalia sazonal
   - ML com lags: Ridge e Gradient Boosting (estratégia direta, um modelo por horizonte)
   - Deep learning ([NeuralForecast](https://github.com/Nixtla/neuralforecast)): LSTM, GRU, DLinear, N-HiTS e **PatchTST**
   - Ensembles: média PatchTST + LSTM, média de todos os modelos de DL e stacking NNLS
   - Variantes de DL sobre a anomalia sazonal, com exógenas de calendário (LSTM, GRU, N-HiTS)
5. **Resultados (H = 7):** tabela geral, RMSE por dia de antecedência, erro por regime hidrológico, previsões em cheia e seca, resíduos.
6. **Múltiplos horizontes (H = 7, 15 e 30 dias):** mesma comparação repetida por horizonte, com *skill* em relação à persistência e erro por dia de antecedência até 30 dias.
7. **DL com sazonalidade + stacking:** os modelos de deep learning re-treinados sobre a anomalia (cota − climatologia), com seno/cosseno do dia do ano como exógenas onde possível, e stacking com pesos NNLS reestimados a cada origem (sem vazamento do teste).
8. **Rigor estatístico (H = 30):** sensibilidade ao método de preenchimento e aos dias marcados como "estimado" pela ANA; 5 sementes por arquitetura com média, desvio e estabilidade do ranking; **teste de Diebold-Mariano** com variância HAC (Newey-West) e correção de Harvey-Leybourne-Newbold; busca de hiperparâmetros com `AutoPatchTST`/`AutoLSTM` (optuna) feita só na validação e reavaliada uma única vez no teste.

## Como executar

Requisitos: Python 3.12 (há um `.venv` no projeto) e Jupyter.

```bash
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy neuralforecast jupyter
jupyter notebook forecasting.ipynb
```

A primeira célula de código de cada notebook também instala as dependências via `%pip install`.

Os arquivos CSV do Hidroweb usam `;` como separador, **codificação ISO-8859-1 (latin-1)** e possuem 11 linhas de cabeçalho antes da tabela. O `forecasting.ipynb` lê com `encoding='latin-1'`; o `index.ipynb` usa `utf-8`, o que funciona no pandas 2.x porque as linhas de cabeçalho puladas não são decodificadas, mas falha no pandas 3.x. A cota está em **centímetros**.

O treino dos modelos de deep learning roda em CPU em poucos minutos (`MAX_STEPS = 1000`, com early stopping).

## Próximos passos

- **Informação externa** — único caminho identificado para melhorar as previsões: estações a montante do Madeira (Porto Velho, Abunã, Guajará-Mirim), chuva acumulada na bacia alta (CHIRPS ou pluviométricas da ANA) e índices ENSO para h = 90. A estação Santo Antônio **não serve** como preditor: a correlação cruzada é máxima em lag 0 (r = 0,9997) e simétrica, ou seja, é um medidor co-locado sem antecedência.
- **Avaliação orientada a uso:** previsão probabilística (`MQLoss`, CRPS, cobertura dos intervalos), skill em cruzar cotas de alerta (POD/FAR/CSI), erro no pico e no *timing* do pico, e avaliação isolada da seca recorde de 2023 e das cheias de 2021/2022.
- **Outros modelos:** TimesFM zero-shot × fine-tuned, Chronos, Moirai, TFT (quando houver covariáveis) e híbrido SARIMA/climatologia + DL nos resíduos.
- **Unificar o protocolo com o `index.ipynb`** (teste desde 2021, horizontes 1/7/30/90, incluir R²) para uma tabela única no artigo.
- A vazão (`*_Vazoes.csv`) já está no repositório e cobre 100% de 1991–2023, mas correlaciona 0,98 com a cota (é derivada da curva de chave) — serve como alvo alternativo, não como preditor novo.
- Opcionalmente, preencher a lacuna de abril/2014 com Santo Antônio em vez de interpolação (a simulação no notebook mostra erro ~4× menor).
