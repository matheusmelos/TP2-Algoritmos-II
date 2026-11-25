# TP2 – Caixeiro Viajante Geométrico (`EUC_2D`)

Trabalho Prático 2 da disciplina **DCC207 – Algoritmos 2** (UFMG), focado na análise empírica e teórica de algoritmos exatos, aproximativos e heurísticos para o Problema do Caixeiro Viajante (TSP) métrico euclidiano 2D (`EUC_2D`).

**Autores:** João Vitor Vieira Neves, Matheus Costa Melo Silva, Matheus Pimentel Kansaon

---

## Visão Geral do Projeto

O projeto implementa, instrumenta e compara:

1. **Algoritmo Exato — Branch-and-Bound (`src/tsp/bnb.py`)**
   - Estratégias de busca: **DFS iterativa** (`dfs`, padrão, memória $O(n^2)$) e **Best-First** (`best_first` com `heapq`).
   - *Lower bounds* plugáveis: `two_min` (Levitin, atualização incremental $O(1)$) e `one_tree` (padrão: caminho + MST dos não visitados memoizada por bitmask + duas conexões com as pontas, avaliada após pré-filtragem por `two_min`).
   - Otimizações e controle: quebra de simetria (`tour[1] < tour[-1]`), representação de visitados por bitmask, matriz de distâncias pré-convertida em listas Python (`inst.D_list`) no *inner loop*, e tratamento de *timeout* preservando contadores parciais e o melhor custo encontrado.

2. **Algoritmos de Aproximação (`src/tsp/approx.py` e `src/tsp/mst.py`)**
   - **Twice-Around-the-Tree (`tatt`)**: MST (Prim denso vetorizado $O(n^2)$ com NumPy) + percurso em pré-ordem iterativo (garantia $2\times \text{OPT}$).
   - **Christofides (`christofides`)**: MST + vértices de grau ímpar + emparelhamento perfeito mínimo (`networkx.min_weight_matching`) + Hierholzer iterativo + atalhos (garantia $1{,}5\times \text{OPT}$).
   - **Ablação (`christofides_greedy`)**: variante com *matching* guloso para mensurar o trade-off tempo/memória × qualidade frente ao gargalo do algoritmo Blossom.

3. **Estratégias de *Warm-Start* e Heurísticas Construtivas (`src/tsp/heuristics.py`)**
   - `trivial`: ordem canônica $0 \to 1 \to \dots \to n-1$.
   - `nn`: Vizinho Mais Próximo ($O(n^2)$).
   - `hull`: Inserção por Fecho Convexo (`scipy.spatial.ConvexHull` + cache incremental do melhor custo de inserção).
   - Além de `tatt` e `christofides`, totalizando **5 estratégias de *warm-start*** para o *upper bound* inicial do Branch-and-Bound.

4. **Instrumentação, Features e Análise (`src/tsp/common.py`, `src/tsp/features.py`, `src/analysis/`)**
   - `Profiler` por subrotina com passadas separadas para **tempo** (`mem=False`) e **pico de memória** via `tracemalloc` (`mem=True`), evitando distorção temporal.
   - Extração de 8 *features* geométricas/estruturais (`aspect_ratio`, `hull_fraction`, `spatial_entropy`, `clark_evans`, `mst_max_degree`, `mst_mean_degree`, `mst_edge_cv`, `nn_dist_cv`).
   - Ajuste de curvas de complexidade (`a·n^k`, `a·n²+b`, `a·n³+b`, `a·c^n` com seleção por AIC, $R^2$ e RMSE), análise de *warm-start* (poda, trade-off e ponto de colapso) e análise de sensibilidade (Spearman, regressão linear padronizada e `DecisionTreeRegressor(max_depth=3)` com validação LOOCV).

---

## Instâncias (`data/`)

O enunciado exige instâncias `EUC_2D` com pelo menos 80% delas na faixa $N \le 40$. Como a TSPLIB não possui instâncias `EUC_2D` nativas com $N \le 40$ (as menores são `eil51`, `berlin52` e `st70`), o conjunto principal é composto por **16 instâncias** listadas em `data/instances/manifest.csv`:

- **13 sub-instâncias (`N` entre 12 e 40 — 81,25% do total)**: recortadas com *seed* fixa (`2026`) a partir de instâncias reais da TSPLIB (`eil51`, `berlin52`, `pr76`, `eil76`, `kroA100`, `a280`, `ch130`), variando o modo de recorte (`random`, `knn`, `window`) para gerar diversidade estrutural.
- **3 instâncias completas da TSPLIB**: `eil51` ($N=51$, ótimo `426`), `berlin52` ($N=52$, ótimo `7542`) e `st70` ($N=70$, ótimo `675`).

---

## Estrutura do Repositório

```text
TP2_ALG2/
├── data/
│   ├── tsplib/               # Instâncias originais da TSPLIB (.tsp e .opt.tour)
│   └── instances/            # 16 instâncias do experimento + manifest.csv
├── src/
│   ├── tsp/
│   │   ├── common.py         # Contrato comum, I/O TSPLIB, tour_cost, Profiler, schema JSONL
│   │   ├── download_tsplib.py# Downloader das instâncias base (apenas stdlib)
│   │   ├── make_instances.py # Gerador determinístico das sub-instâncias EUC_2D
│   │   ├── heuristics.py     # trivial, nearest_neighbor, convex_hull_insertion
│   │   ├── mst.py            # Prim denso O(n²) vetorizado com NumPy
│   │   ├── approx.py         # twice_around_the_tree, christofides, christofides_greedy
│   │   ├── bnb.py            # Branch-and-Bound (dfs/best_first × two_min/one_tree)
│   │   └── features.py       # Extrator de métricas geométricas e da MST
│   ├── analysis/
│   │   ├── _io.py            # Utilitários de leitura dos resultados
│   │   ├── complexity.py     # Ajuste de curvas de tempo/memória (AIC, R², RMSE)
│   │   ├── warmstart.py      # Poda, trade-off e ponto de colapso do B&B
│   │   └── sensitivity.py    # Spearman, regressão linear, árvore de decisão e LOOCV
│   ├── run_experiments.py    # Orquestrador paralelo (ProcessPoolExecutor) do pipeline
│   ├── benchmark_audit.py    # Microbenchmarks de auditoria (D_list, busca/bound, ablação, GC)
│   ├── validate_results.py   # Validador automático de esquema, limites e sanidade
│   └── generate_insights.py  # Consolidador automático de insights (insights.json / INSIGHTS.md)
├── tests/                    # Suíte de testes unitários e de integração (pytest)
├── results/                  # Resultados consolidados, tabelas, figuras e análises
├── results_pilot/            # Rodada piloto preliminar (20 s)
├── PLANO.md                  # Plano de implementação e checklist do projeto
├── pytest.ini
├── requirements.txt
└── tp2.pdf                   # Enunciado oficial do trabalho
```

---

## Configuração e Execução

### 1. Ambiente Virtual e Dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependências listadas em `requirements.txt`: `numpy`, `scipy`, `scikit-learn`, `pandas`, `networkx`, `matplotlib`, `pytest`.

### 2. Testes Automatizados

```bash
pytest -v
```

Os testes em `tests/` cobrem o contrato comum (`test_common.py`), geração de instâncias (`test_make_instances.py`), heurísticas (`test_heuristics.py`), aproximações e garantias teóricas (`test_approx.py`), B&B contra força bruta para $N \le 9$ e tratamento de timeout (`test_bnb.py`), extração de *features* (`test_features.py`), scripts de análise (`test_analysis.py`), orquestrador (`test_run_experiments.py`), auditoria (`test_benchmark_audit.py`) e validação/insights (`test_insights_and_validation.py`).

### 3. Geração de Instâncias e Execução dos Experimentos

```bash
# (Opcional) Regerar as sub-instâncias em data/instances/
python -m src.tsp.make_instances

# Rodada piloto (time-limit 60s, 5 repetições, análise automática)
python -m src.run_experiments --pilot --out results_pilot

# Rodada completa (padrão: time-limit 1800s, 10 repetições) com análise ao final
python -m src.run_experiments --out results --analyze

# Retomar uma rodada interrompida ou estender o time-limit aproveitando jobs já concluídos
python -m src.run_experiments --out results --time-limit 1800 --resume --analyze
```

### 4. Auditoria, Análise, Validação e Consolidação

```bash
# Microbenchmarks de auditoria (D_list vs NumPy, dfs/best_first × two_min/one_tree, ablação, GC/tracemalloc)
python -m src.benchmark_audit --out results/audit

# Reexecutar apenas os módulos de análise sobre results/
python -m src.analysis.complexity --dir results
python -m src.analysis.warmstart --dir results
python -m src.analysis.sensitivity --dir results

# Validar esquema, ótimos, limites teóricos e sanidade
python -m src.validate_results --dir results

# Gerar results/insights.json e results/INSIGHTS.md
python -m src.generate_insights --dir results
```

---

## Resumo dos Resultados Consolidados (`results/`)

A base em `results/` contém **1.040 linhas** em `runs.jsonl`, **2.800 linhas** em `profile.jsonl`, **16 figuras** em `results/figures/` e foi aprovada com **0 erros e 0 avisos** em `results/tables/validation_report.json` (executada com `--time-limit 60` em CPU de 2 núcleos, além de curvas suplementares até $N=500$ em `results/supplementary/`):

- **Refatoração de estrutura de dados (`D_list` vs NumPy):** usar `inst.D_list` no *inner loop* do B&B gerou *speedup* mediano de **1,53×** (até **1,67×**) sobre exatamente a mesma árvore de busca.
- **Busca e *Lower Bound* (`dfs + one_tree`):** em `kroA100_knn22` ($N=22$), `one_tree` reduziu os nós expandidos de **1.048.576** (`TIMEOUT` em 10 s com `two_min`) para **3.960** (**0,203 s**). O `dfs` manteve a fronteira em $\le 96$ nós ($O(n^2)$), enquanto `best_first + two_min` acumulou **1.409.827** nós na fronteira em 10 s (`berlin52_random20`).
- **Gargalo do Christofides e ablação:** o `matching` perfeito mínimo consumiu **62,3%** ($N=12$) a **99,4%** (`a280`) do tempo total. A versão com *matching* guloso acelerou em até **123,8×** e reduziu o pico de memória em **6,8×** (`a280`), ao custo de **+13,46 p.p.** de piora mediana no gap.
- **Memória e instrumentação:** cópias desnecessárias da matriz no Prim aumentaram a memória auxiliar em até **70,7×** (`a280`), e o `tracemalloc` impôs *slowdown* de **2,51× a 4,25×**, confirmando a necessidade de passadas separadas de tempo e memória.
- **Impacto do *warm-start*:** `hull` obteve o menor gap médio (**2,43%** contra **6,77%** do Christofides, **20,84%** do NN e **33,59%** do TATT), reduziu nós expandidos em **62,9%** (mediana) e deslocou o primeiro `TIMEOUT` do B&B de $N=20$ (`trivial`/`tatt`) para $N=28$ (`nn`/`hull`/`christofides`), com `kroA100_random36` ($N=36$) resolvida até o ótimo por todas as estratégias.
- **Sensibilidade às *features*:** `hull` correlacionou-se negativamente com `hull_fraction` ($\rho = -0{,}594$) e positivamente com $N$ ($\rho = +0{,}723$); `christofides` degradou com *hubs* na MST (`mst_max_degree`, $\rho = +0{,}634$).
