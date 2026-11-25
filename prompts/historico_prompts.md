# Anexo – Histórico de Prompts e Registro de Uso de IA Generativa

Este documento registra o uso de ferramentas de Inteligência Artificial (LLM) ao longo do desenvolvimento do Trabalho Prático 2 (DCC207 – Algoritmos 2), conforme exigido em [tp2.pdf](../tp2.pdf).

---

## 1. Planejamento Inicial e Autoria dos Alunos


- A divisão do trabalho em contrato comum (Etapa 0: [common.py](../src/tsp/common.py)), busca exata/heurísticas/instâncias (Etapa 1 – Dev A), aproximações/features/modelagem estatística (Etapa 2 – Dev B) e integração/experimentos/relatório (Etapa 3).
- A decisão metodológica de construir a faixa principal ($N \le 40$, 81% das 16 instâncias) a partir de sub-instâncias EUC\_2D recortadas de instâncias reais da TSPLIB com semente fixa e três modos de amostragem (`random`, `knn`, `window`), somadas às completas `eil51`, `berlin52` e `st70`, garantindo aderência estrita ao tipo `EUC_2D`.
- O contrato imutável de interface (`TourBuilder`, `Profiler` com passadas separadas de tempo e memória, `RUNS_COLUMNS`, `PROFILE_COLUMNS`, `ResultWriter` append-only e convenção `NA` para `TIMEOUT`), além da implementação inicial das Etapas 0 e 1 (`common.py`, `download_tsplib.py`, `make_instances.py`, `heuristics.py`, `bnb.py` e seus testes).

A IA generativa foi empregada sob direção técnica dos alunos como assistente de codificação, instrumentação, auditoria de gargalos, automação de experimentos e validação estatística. Os prompts abaixo apresentam uma síntese parafraseada e organizada das principais interações conduzidas durante o ciclo de desenvolvimento e experimentação.

---

## 2. Histórico dos Principais Prompts Utilizados

### Prompt 1 — Estruturação do repositório e implementação da Etapa 2 (Dev B)
> **Prompt (síntese):**
> *"Importe a base do projeto (`tp2`, contendo o `PLANO.md`, o contrato `common.py` e a Etapa 1) para o repositório e implemente os módulos da Etapa 2 (Dev B) respeitando rigorosamente o contrato comum, sem alterar `common.py`: Prim denso $O(n^2)$ vetorizado em `mst.py`; Twice-around-the-tree, Christofides (com `networkx.min_weight_matching` e Hierholzer iterativo) e ablação com matching guloso em `approx.py`; extração das 9 features geométricas/topológicas em `features.py`; scripts de análise `complexity.py`, `warmstart.py` e `sensitivity.py` (com LOOCV e árvore `max_depth=3`); e testes unitários cobrindo validade, razões de aproximação ($2\times$ e $1{,}5\times$), features e pipeline de análise."*

- **Artefatos gerados:** `src/tsp/mst.py`, `src/tsp/approx.py`, `src/tsp/features.py`, `src/analysis/{_io,complexity,warmstart,sensitivity}.py`, `tests/test_{approx,features,analysis}.py`.
- **Verificação e auditoria:** 238 testes aprovados via `pytest`; profiling em `a280` confirmou `matching` consumindo 99,4% do tempo do Christofides; corrigido aviso de depreciação do Matplotlib (`cmap.with_extremes`) em `warmstart.py`.

---

### Prompt 2 — Integração dos warm-starts e orquestrador experimental (Etapa 3)
> **Prompt (síntese):**
> *"Implemente a Etapa 3 de integração conforme o `PLANO.md`: gere as 16 instâncias oficiais via `make_instances.py`, compare empiricamente `two_min` vs `one_tree` e `dfs` vs `best_first` para definir o padrão do B&B, unifique o dicionário `WARM_STARTS` com as 5 estratégias em `bnb.py` e implemente `src/run_experiments.py` com paralelismo (`núcleos - 1`), isolamento por processo no B&B com limite interno e kill externo, passada separada de memória, suporte a retomada (`--resume`), backfill de `optimal_ref`/`gap` e testes de integração."*

- **Artefatos gerados:** `data/instances/*`, `src/tsp/bnb.py` (integração de `WARM_STARTS`), `src/run_experiments.py`, `tests/test_run_experiments.py`.
- **Auditoria e refatoração aplicada:** na medição de tempos dos testes de integração (`pytest --durations=5`), identificou-se que os processos filhos de B&B iniciados via `spawn` importavam `matplotlib`, `pandas` e `scikit-learn` no topo de `run_experiments.py`, desperdiçando ~3–4 s e memória por worker. O módulo foi refatorado para carregamento preguiçoso (*lazy imports*) das bibliotecas de análise, acelerando o *spawn* em ~3,2× (245 testes aprovados).

---

### Prompt 3 — Revisão de pendências e planejamento estruturado da etapa experimental
> **Prompt (síntese):**
> *"Audite o que ainda falta implementar ou executar segundo o `PLANO.md` e elabore um plano técnico detalhado para a etapa experimental antes de modificar o código. Inclua no plano a persistência completa de todos os dados brutos e derivados (`.jsonl`, espelhos `.csv`, metadados de hardware/software), microbenchmarks de auditoria (`D_list` vs NumPy, `dfs` vs `best_first`, ablação do Christofides), validador automático do checklist e consolidação estruturada de todos os insights coletados (`INSIGHTS.md` e `insights.json`)."*

- **Decisões de engenharia tomadas na revisão do plano:**
  - Separação das séries do B&B por `warm_start` em `complexity.py` (evitando misturar `trivial` e `christofides` na mesma regressão) e filtragem de linhas `TIMEOUT` em `profile_df` para impedir platôs artificiais nos ajustes de subrotinas.
  - Criação de `src/benchmark_audit.py`, `src/validate_results.py` e `src/generate_insights.py`, integrados ao `run_experiments.py`.
  - Dimensionamento do piloto e dos lotes de execução para o ambiente disponível (2 núcleos de CPU), garantindo reprodutibilidade via `experiment_metadata.json` e `--resume`.
- **Verificação:** piloto completo nas 16 instâncias (`results_pilot/`, 20 s) aprovado sem erros ou avisos no validador automático; 247 testes passando.

---

### Prompt 4 — Análise crítica e científica dos resultados em Markdown
> **Prompt (síntese):**
> *"A partir de todas as tabelas, gráficos e métricas coletadas em `results/` e `results_pilot/`, produza uma análise detalhada em Markdown (`results/ANALISE_RESULTADOS.md`) cobrindo auditoria de estruturas de dados e estratégias de busca, casos difíceis para a arguição (como `berlin52_random20` vs instâncias maiores), gargalo e ablação do Christofides, divergências entre complexidade teórica e empírica (incluindo curvas suplementares até $N=500$), impacto do warm-start e ponto de colapso, além da análise de sensibilidade (Spearman, regressão padronizada, árvore `max_depth=3`, LOOCV e instâncias adversariais)."*

---

### Prompt 5 — Revisão contra o `tp2.pdf`, fechamento de lacunas de teste e consolidação dos 80 jobs de B&B
> **Prompt (síntese):**
> *"Verifique se restou algum teste unitário ou experimento sem executar, releia o `tp2.pdf` na íntegra para identificar qualquer requisito não coberto por testes ou medições empíricas, implemente e execute o que estiver faltando e atualize a base definitiva em `results/` com todos os 80 jobs de B&B, validação e análise atualizada."*

- **Lacunas identificadas e resolvidas:**
  1. **Gargalos de memória, cópias de matriz e GC (`tp2.pdf`, p. 2):** adicionado o 4º microbenchmark (`benchmark_memory_and_gc`) em `src/benchmark_audit.py`, medindo impacto de cópias de matriz no Prim (até 70,7× mais memória em `a280`), impacto do GC cíclico no Christofides (-0,67% a +1,27%) e overhead do `tracemalloc` (2,51×–4,25×).
  2. **Cobertura completa de `WARM_STARTS` e auditoria:** expandido `tests/test_bnb.py` para validar todas as 5 estratégias em todas as combinações `(search, bound)` e criado `tests/test_benchmark_audit.py` (**256 testes passando**, 0 pulados).
  3. **Execução definitiva dos 80 jobs de B&B em `results/` (`time_limit=60s`):** executados em lotes com `--resume`, elevando para 13/16 as instâncias com ótimo conhecido (incluindo `berlin52_random20` e `kroA100_random36`) e comprovando o deslocamento do ponto de colapso de $N=20$ para $N=28$ por influência direta do *warm-start*.

---

### Prompt 6 — Versionamento no GitHub e documentação do anexo de prompts
> **Prompt (síntese):**
> *"Consolide os commits com o código-fonte, testes, dados brutos/derivados, figuras e análises no repositório GitHub e estruture o histórico de prompts em `prompts/historico_prompts.md` para compor o anexo obrigatório do relatório SBC."*

---

## 3. Síntese das Refatorações e Auditorias Conduzidas

| Área | Problema / Risco Identificado | Correção Auditada |
|---|---|---|
| Inner loop do B&B | Overhead de *boxing* escalar NumPy em loops Python | Uso de `Instance.D_list` (speedup mediano de 1,53×) |
| Fronteira do B&B | Explosão de memória do `best_first` (>1,4M nós em 10 s) | Adoção de `dfs + one_tree` com memoização por bitmask (fronteira $\le 96$) |
| Workers `spawn` | Importação de `matplotlib`/`sklearn` a cada processo B&B | *Lazy imports* em `run_experiments.py` (~3,2× menos overhead de inicialização) |
| Regressão de complexidade | Mistura dos 5 *warm-starts* e tempos truncados de `TIMEOUT` | Separação por `algorithm:warm_start` e filtragem `STATUS_OK` em `complexity.py` |
| Profiling de memória | `tracemalloc` distorce tempos em 2,5×–4,3× | Passadas separadas de tempo (`mem=False`) e memória (`mem=True`) |
| Cópias de matriz no Prim | Alocação $O(n^2)$ por cópias de `D` e `np.where` | Prim denso vetorizado sem cópia da matriz (redução de até 70,7× no pico de memória) |
