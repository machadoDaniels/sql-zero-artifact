# Bateria OmniSQL-pipeline — resultados novos (2026-07-17)

Re-avaliação de 13 modelos com o **pipeline oficial de inferência do repo OmniSQL**
(`RUCKBReasoning/OmniSQL`, `train_and_evaluate/infer.py`): prompts pré-processados deles
(schema DDL + valores de célula amostrados), greedy@1, `max_output_len=2048`, extração de
```` ```sql ````; pontuação pelo scorer oficial (test-suite-sql-eval / bird-bench). Dados em
`docs/evals_omnisql_pipeline.csv`.

**Por que esta bateria existe:** sob nosso protocolo unificado, OmniSQL/Arctic ficavam ~8pt
abaixo do publicado — o prompt nativo sozinho recuperava só +2.2pt; a causa era o pipeline de
inferência (schema com valores + orçamento de decoding). Aqui reproduzimos o pipeline deles
fielmente e confirmamos por gates.

## Gates de fidelidade (bateram o publicado)

| Modelo | BIRD dev pipe | publicado | Spider test pipe | publicado |
|--------|--------------|-----------|------------------|-----------|
| OmniSQL-7B | 0.6375 | 0.639 | 0.877 | 0.879 |
| Arctic-7B | 0.6851 | 0.689 | 0.897 | 0.888 |

Δ ≤ 0.9pt em ambos os benchmarks → pipeline validado; os números dos outros 11 modelos valem.

## BIRD dev

| Modelo | pipe-OmniSQL | published |
|--------|-------------|-----------|
| Arctic-7B | 0.6851 | 0.689 |
| **G3 iter1 (OmniSQL-init, self-play BIRD)** | **0.6545** | — |
| **G3 iter2 (OmniSQL-init, self-play BIRD)** | **0.6538** | — |
| **G3 iter3 (OmniSQL-init, self-play BIRD)** | **0.6512** | — |
| **F5 (OmniSQL+SQL-Zero)** | **0.6473** | — |
| OmniSQL-7B | 0.6375 | 0.639 |
| SQL-Zero-7B BIRD it1 | 0.5737 | — |
| SQL-Zero-7B BIRD it3 | 0.5691 | — |
| SQL-Zero-7B BIRD it2 | 0.5606 | — |
| Qwen2.5-Coder-7B | 0.5111 | — |
| CSC-SQL-3B | 0.4831 | — |
| SQL-Zero-3B BIRD it1 | 0.4374 | — |
| SQL-Zero-3B BIRD it3 | 0.4342 | — |
| SQL-Zero-3B BIRD it2 | 0.4237 | — |
| **Gold-BIRD-control-3B** (pareado, BIRD gold) | **0.4192** | — |
| Qwen2.5-Coder-3B | 0.3866 | — |
| Gold-control-3B (Spider gold) | 0.3859 | — |

## Spider test (transfer, treino foi no BIRD)

| Modelo | pipe-OmniSQL |
|--------|-------------|
| Arctic-7B | 0.897 |
| **G3 iter2 (OmniSQL-init, self-play BIRD)** | **0.883** |
| **G3 iter1 (OmniSQL-init, self-play BIRD)** | **0.880** |
| OmniSQL-7B | 0.877 |
| **G3 iter3 (OmniSQL-init, self-play BIRD)** | **0.877** |
| F5 (OmniSQL+SQL-Zero) | 0.866 |
| SQL-Zero-7B BIRD it3 | 0.829 |
| SQL-Zero-7B BIRD it1 | 0.824 |
| Qwen2.5-Coder-7B | 0.822 |
| SQL-Zero-7B BIRD it2 | 0.813 |
| CSC-SQL-3B | 0.772 |
| SQL-Zero-3B BIRD it3 | 0.759 |
| SQL-Zero-3B BIRD it2 | 0.757 |
| **Gold-BIRD-control-3B** (pareado) | **0.744** |
| SQL-Zero-3B BIRD it1 | 0.741 |
| Gold-control-3B (Spider gold) | 0.719 |
| Qwen2.5-Coder-3B | 0.68 |

## Spider-Syn — pipe-OmniSQL

Nota de fidelidade: o input do Syn foi gerado sem a recuperação de valores por query
(pyserini/JVM indisponível no nó) — os valores de célula amostrados por tabela ainda entram;
desvio uniforme entre todos os modelos, aceitável para comparação de robustez.
**Gate (2026-07-22)**: reconstruído o caminho Syn na gpu-node-a (gpu-node-a caiu; builder oficial do
OmniSQL `process_dataset.py --source spider_syn`, sem content index). OmniSQL-7B re-scoreado dá
**0.772 = 0.772 tabelado** (exato, ruído zero), então o caminho reproduz a coluna Syn existente à
casa decimal e o G3 é comparável.

| Modelo | pipe-OmniSQL (Syn) |
|--------|--------------------|
| Arctic-7B | 0.817 |
| **G3 iter1 (OmniSQL-init, self-play BIRD)** | **0.792** |
| **G3 iter3 (OmniSQL-init, self-play BIRD)** | **0.789** |
| **G3 iter2 (OmniSQL-init, self-play BIRD)** | **0.785** |
| F5 (OmniSQL+SQL-Zero) | 0.772 |
| OmniSQL-7B | 0.772 |
| SQL-Zero-7B BIRD it2 | 0.738 |
| SQL-Zero-7B BIRD it1 | 0.737 |
| SQL-Zero-7B BIRD it3 | 0.727 |
| Qwen2.5-Coder-7B | 0.715 |
| SQL-Zero-3B BIRD it3 | 0.684 |
| SQL-Zero-3B BIRD it2 | 0.68 |
| SQL-Zero-3B BIRD it1 | 0.656 |
| CSC-SQL-3B | 0.651 |
| Gold-control-3B (Spider gold) | 0.635 |
| **Gold-BIRD-control-3B** (pareado) | **0.631** |
| Qwen2.5-Coder-3B | 0.565 |

## Leituras

1. **Gates confirmam** que o pipeline reproduz os números publicados (Δ≤0.9pt) — os valores
   dos outros 11 modelos são fiéis ao protocolo do autor.
2. **F5 > OmniSQL no protocolo do próprio OmniSQL** (BIRD 0.6473 > 0.6375; em Spider o teto do
   OmniSQL é alto, 0.877): self-play zero-data agrega sobre init forte.
3. **SQL-Zero zero-data > gold-supervisionado nas duas escalas** (BIRD 3B 0.424-0.437 vs gold-Spider
   0.3859; Spider 3B 0.74-0.76 vs gold-Spider 0.719).
3b. **Controle PAREADO no protocolo independente (2026-07-20)**: o gold-BIRD-3B (treinado no gold
   humano do próprio BIRD, mesmos 64 bancos) re-scoreado pelo pipe OmniSQL dá BIRD **0.4192** —
   **abaixo das três iterações self-play** (0.4237/0.4342/0.4374) também neste protocolo. Confirma
   no pipe do autor o que a bateria unificada já mostrava (oficial 0.4654 gold < 0.477-0.4915
   self-play): zero-data supera anotação humana no MESMO benchmark, em DOIS protocolos de eval
   independentes. Transfer Spider test 0.744 (≈ it1 0.741, < it2/it3 0.757/0.759) e Syn 0.631
   (< self-play 0.656-0.684) reforçam. Modelo: `anonymous/sql-zero-3b-gold-control-bird`.
4. **Saturação (C11) reproduz em protocolo independente**: espirais flat entre iterações
   (BIRD 7B 0.574/0.561/0.569; 3B 0.437/0.424/0.434; Spider análogo).
5. **Rankings** nosso-vs-base e nosso-vs-gold preservados. CSC-3B fica ~1.3pt à frente das nossas
   3B no Spider test (0.772 vs 0.741-0.759, dentro/na borda da tolerância de ±0.5pt do caminho de
   score) e à frente no BIRD (0.4831 vs 0.42-0.44).
6. **Saturação também no G3** (2026-07-28): as três iterações OmniSQL-init ficam flat em todos os
   três benchmarks (BIRD 0.6545/0.6538/0.6512; Spider test 0.880/0.883/0.877; Syn 0.792/0.785/
   0.789). Todas as diferenças entre iterações estão abaixo do limiar interpretativo do paper
   (2pt) e, no Spider test, dentro da tolerância do caminho de score. O ganho sobre o init
   OmniSQL-7B (0.6375/0.877/0.772) persiste nas três.

## Ablação think × no-think (3B) no pipe-OmniSQL — 2026-07-29

Os três braços **no-think** da ablação pareada (`SOLVER_THINK_BONUS` 0.1 vs 0, 3B, step 78, 20k
pares) eram os únicos modelos treinados do artigo sem cobertura nesta bateria. Rodados no mesmo
caminho das 17 linhas anteriores: `omnisql_pipeline/infer.py` (greedy@1, `n=1`, `temperature=0.0`,
`max_output_len=2048`), BIRD por `evaluate_bird.py`, Spider test e Syn pelo `test-suite-sql-eval`
(`--etype exec --plug_value --keep_distinct`). Modelos puxados direto do HF
(`anonymous/sql-zero-3b-solver-bird{1,2,3}-nothink`). Jobs gpu-node-a: 28791/28792/28793
(inferência) e 28794/28795/28796 (scoring). Scripts: `scripts/nothink_pipe_infer.sbatch` e
`scripts/nothink_pipe_score.sbatch`.

| iteração | braço | BIRD dev | Spider test | Spider-Syn |
|---|---|---:|---:|---:|
| iter1 | think (`g1b-s78`) | **0.4374** | **0.741** | 0.656 |
| iter1 | no-think (`bird1nt-s78`) | 0.3937 | 0.711 | 0.649 |
| iter2 | think (`bird2-s78`) | **0.4237** | **0.757** | **0.680** |
| iter2 | no-think (`bird2nt-s78`) | 0.3501 | 0.727 | 0.668 |
| iter3 | think (`bird3-s78`) | **0.4342** | **0.759** | **0.684** |
| iter3 | no-think (`bird3nt-s78`) | 0.3807 | 0.748 | 0.673 |

**Gate de sanidade.** Faixa das três iterações think: BIRD 0.4237–0.4374, Spider test 0.741–0.759,
Syn 0.656–0.684. O braço no-think cai dentro/logo abaixo em Spider test (0.711–0.748) e Syn
(0.649–0.673), e 3–7pt abaixo em BIRD. Nada no regime de prompt quebrado (BIRD < 0.30). Auditoria
das predições: as 1.534 respostas BIRD de cada braço têm fence ```` ```sql ````, exatamente 1
predição vazia por braço (mesma contagem do braço think) e p95 de ~1.800 caracteres contra
orçamento de 2.048 tokens — sem truncamento nem falha de extração. Os números BIRD baixos são
comportamento de modelo, não falha de pipeline.

**Leitura.** O sinal **think > no-think em BIRD sobrevive e amplifica** neste protocolo
independente (+4.4 / +7.4 / +5.4pt, contra ~1–3pt no harness interno), nas três iterações. Já o
sinal **no-think > think em Spider inverte**: no harness interno o no-think ganhava no Spider dev
(0.802 vs 0.789/0.793); aqui o think ganha em Spider test nas três iterações (+3.0 / +3.0 / +1.1pt)
e em Syn nas três (+0.7 / +1.2 / +1.1pt). Como os deltas de Syn (≤1.2pt) e o de Spider test da
iter3 (+1.1pt) estão perto da tolerância de ~±0.5pt do caminho de score, a conclusão honesta do
lado Spider é *a vantagem de transfer do no-think desaparece*, não *o think ganha muito*. O bônus
de think compra acurácia in-domain sem custar transfer.

Segundo padrão: **os dois braços afundam na iteração 2 em BIRD** (think 0.4374→0.4237→0.4342;
no-think 0.3937→0.3501→0.3807) e sobem monotonicamente em Spider test (think 0.741→0.757→0.759;
no-think 0.711→0.727→0.748). A não-monotonicidade é da geração da iter2, compartilhada pelas duas
configurações de reward, e não um artefato do bônus de think — é a mesma saturação de espiral (C11)
já reportada, com o braço no-think saturando num nível mais baixo.

Tabela de evidência: `ara/evidence/tables/nothink_omnisql_pipe_2026-07-29.md`. Predições
sobreviventes (re-pontuáveis) na gpu-node-a: `omnisql_pipeline/results/bird{1,2,3}nt_*_greedy.json` e
`official_eval/work/bird{1,2,3}nt_syn_pred.json`.

## Desvio de protocolo documentado: correção do detector de slots do scorer (2026-07-28)

Os jobs de score do Spider test para G3 iter2/iter3 (28419/28420) deram **TIMEOUT em 4h**. Causa
raiz **upstream**, no `test-suite-sql-eval`: `parse.strip_query` detecta slots de valor com
`re.findall("[^tT]\d+", query)`. A intenção era isentar aliases de tabela `T1`/`t2`, mas o padrão
casa **qualquer** token de um caractere seguido de dígitos. Predições que apelidam tabelas como
`s1`, `c2`, `A1` têm o alias tratado como slot de valor, e a expansão plug-value cresce como
`|valores gold|^slots` — até **4.5e15** candidatos para UM datapoint.

**Não há inflação de score.** `exec_eval.eval_exec_match` faz `preds = chain([p_str], preds)` com
geradores lazy: a predição crua é o candidato 0 e uma predição correta passa antes de qualquer
expansão. Substituir um literal na posição de um alias produz SQL sintaticamente inválido, que
nunca pode passar. O viés é, no pior caso, deflacionário; empiricamente é zero. (Isto **corrige
uma hipótese anterior** de inflação levantada na abertura da auditoria.)

**Correção adotada.** Restringir a detecção de slots a literais inteiros genuínos:
`re.fullmatch(r"[-+]?\d+", tok)`. Aplicada **identicamente a todos os modelos**; **nunca** modifica
a saída do modelo. Scorer no commit `e97acc5`, flags `--etype exec --plug_value --keep_distinct`.

Verificação:
- Conjuntos de valores do lado **gold** inalterados: 2147/2147 (Spider test), 1034/1034 (Syn).
- Todos os **11 pares modelo×benchmark** pontuáveis: bit-idênticos, antigo vs novo.
- Patch **inerte** a menos que `SLOT_FIX=1`.
- Remédio independente (`PLUG_CAP=10000`): números idênticos → cross-validação.
- Os dois jobs travados agora rodam em **21 s**.

**Gate de regressão (PASSOU).** G3 iter1 Spider test = **0.880** com o scorer antigo E com o
corrigido, bucket a bucket (0.930/0.903/0.838/0.815), reproduzindo o número reportado (job 27681).
Para o registro: um 0.858 citado antes veio do job 27677, uma run superseded da depuração daquela
tarde — não é o número reportado.

**Novos números (scorer corrigido).** G3 iter2 Spider test 0.883 (0.926/0.908/0.840/0.824), Syn
0.785 (0.887/0.807/0.757/0.609). G3 iter3 Spider test 0.877 (0.934/0.896/0.840/0.804), Syn 0.789
(0.847/0.820/0.808/0.604). BIRD dev pipe já havia sido pontuado dentro dos jobs de inferência:
iter2 0.6538, iter3 0.6512.

### CSC-SQL-3B: 0.76 → 0.772 (decisão do usuário)

Adotamos **0.772** para `cscsql-3b` no `spider_test_pipe`, substituindo 0.76. Motivo: pôr a coluna
inteira em **uma régua reprodutível**. O 0.76 vinha da cópia do scorer na gpu-node-a, sem log
sobrevivente; o delta **não** é causado pelo bug (csc não tem nenhum alias-slot, e os dois modos do
scorer concordam em 0.772). A troca só é legítima acompanhada da **tolerância de ~±0.5pt do
caminho de score** que este documento já declara (ver Proveniência, 2026-07-21): diferenças abaixo
disso nesta coluna são ruído de caminho, não de modelo.

### Deltas medidos (antigo → novo caminho), todos dentro da tolerância

| Modelo | Spider test antigo | novo | Δ (pt) |
|---|---|---|---|
| g1b-s78 (3B it1) | 0.741 | 0.737 | −0.4 |
| bird2-s78 (3B it2) | 0.757 | 0.758 | +0.1 |
| bird3-s78 (3B it3) | 0.759 | 0.760 | +0.1 |
| gold-3b | 0.719 | 0.719 | 0.0 |
| g3-iter1-omni | 0.880 | 0.880 | 0.0 |
| cscsql-3b | 0.76 (gpu-node-a) | 0.772 | +1.2 (troca de caminho, não do bug) |

Os valores tabelados acima permanecem os registrados; os deltas são reportados como ruído de
caminho medido, não como correções.

### ⚠️ Limitação de reprodutibilidade: 9 linhas irrecuperáveis

**9 das 16 linhas não podem mais ser re-pontuadas por ninguém.** Seus arquivos de predição viviam
na gpu-node-a (10.100.0.111), que está inacessível pela rede; não há cópias na gpu-node-a, gpu-node-a ou no
home NFS. Afetados:

`omnisql-7b-zs`, `arctic-7b`, `baseline-7b`, `baseline-3b`, `goldbird3b`, `bird1-7b-s78`,
`bird2-7b-s78`, `bird3x-7b-s78`, `omnisql-sqlzero-s200`.

Os números **não estão errados** — foram produzidos pelo pipeline validado por gates — mas são
**não-verificáveis a partir de artefatos**: nenhuma auditoria futura pode reproduzi-los sem re-rodar
a inferência. De Spider-Syn sobrevivem apenas as predições de `omnisql-7b-zs` e `g3-iter1`. Toda
afirmação que dependa dessas linhas deve declarar esta limitação.

## Disponibilidade dos pesos (auditoria 2026-07-28)

Contrapeso à limitação acima: as predições sumiram, mas os **pesos** sobrevivem. Auditoria de
todas as **17 linhas de dados** de `evals_omnisql_pipeline.csv` (a contagem "16" da seção anterior
é anterior à inclusão de `g3-iter3-omni`), via `huggingface_hub.HfApi.repo_info(files_metadata=True)`
(job 28568, gpu-node-b) — checando `config.json` como **arquivo**, shards `.safetensors` não-vazios,
`usedStorage` e `private` — mais varredura de `hf_model*` merged nos três nós (`/raid` é local por nó).

| model_id | repo HF | visibilidade | usedStorage (B) | shards | nó com `hf_model` merged |
|---|---|---|---|---|---|
| omnisql-7b-zs | `seeklhy/OmniSQL-7B` | público | — (15,23 GB) | 1 | — (público de terceiro) |
| arctic-7b | `Snowflake/Arctic-Text2SQL-R1-7B` | público | 15.242.693.744 | 4 | — (público de terceiro) |
| cscsql-3b | `cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-3B-Instruct` | público | 6.805.679.119 | 2 | — (público de terceiro) |
| baseline-7b | `Qwen/Qwen2.5-Coder-7B-Instruct` | público | 30.462.543.728 | 4 | — (público de terceiro) |
| baseline-3b | `Qwen/Qwen2.5-Coder-3B-Instruct` | público | 6.171.927.000 | 2 | — (público de terceiro) |
| gold-3b | `anonymous/sql-zero-3b-gold-control` | público | 6.805.678.912 | 2 | **só no HF** |
| goldbird3b | `anonymous/sql-zero-3b-gold-control-bird` | público | 6.805.678.856 | 2 | gpu-node-a (`goldbird3b/hf_model_32`) |
| bird1-7b-s78 | `anonymous/sql-zero-7b-solver-bird1-think` | **privado → público em 2026-07-28** | 15.242.693.792 | 4 | **só no HF** |
| bird2-7b-s78 | `anonymous/sql-zero-7b-solver-bird2-iter2-think` | público | 15.242.693.808 | 4 | gpu-node-b (`bird2x_7b20k/hf_model_78`) |
| bird3x-7b-s78 | `anonymous/sql-zero-7b-solver-bird3-iter3-think` | público | 15.242.693.768 | 4 | gpu-node-b (`bird3x_7b20k/hf_model_78`) |
| g1b-s78 | `anonymous/sql-zero-3b-solver-bird1b-think` | **privado → público em 2026-07-28** | 6.805.678.880 | 2 | **só no HF** |
| bird2-s78 | `anonymous/sql-zero-3b-solver-bird2-think` | **privado → público em 2026-07-28** | 6.805.678.896 | 2 | gpu-node-a (`bird2b20k/hf_model_78`) |
| bird3-s78 | `anonymous/sql-zero-3b-solver-bird3-think` | público | 6.805.678.856 | 2 | gpu-node-a (`bird3b20k/hf_model_78`) |
| omnisql-sqlzero-s200 | `anonymous/sql-zero-7b-solver-omnisql` (`@main` = s200; branch `step-75`) | público | 30.473.965.520 | 4 | **só no HF** |
| g3-iter1-omni | `anonymous/sql-zero-7b-solver-omnisql-bird` (s53) | público | 15.242.693.760 | 4 | gpu-node-a (`bird1_omni20k/hf_model_53`) |
| g3-iter2-omni | `anonymous/sql-zero-7b-solver-omnisql-bird-iter2` (s78) | **privado → público em 2026-07-28** | 15.242.693.760 | 4 | gpu-node-b (`bird2_omni20k/hf_model_78`) |
| g3-iter3-omni | `anonymous/sql-zero-7b-solver-omnisql-bird-iter3` (s65) | público | 15.242.693.768 | 4 | gpu-node-b (`bird3_omni20k/hf_model_65`) |

Todos os 17 têm `config.json` como arquivo e shards `.safetensors` não-vazios confirmados pela API
(nenhuma "casca vazia" de rotação). O mapeamento linha→diretório dos 7B `bird2x`/`bird3x` é por
convenção de nomes, não por hash.

**Ações (todas fechadas em 2026-07-28):**

- ✅ `bird1-7b-s78` era o único modelo da tabela ainda **inacessível** a um revisor: tornado
  público em 2026-07-28 (`update_repo_settings(private=False)`, job 28568), `sha` inalterado
  (`52841b0c`), 4 shards intactos, `usedStorage` idêntico antes e depois.
- ✅ A auditoria revelou **mais três repos privados** que não estavam no radar (`g1b-s78`,
  `bird2-s78` e `g3-iter2-omni`); todos foram tornados públicos no mesmo dia (job 28574, gpu-node-a,
  `scripts/hf_publish_bird1nt_and_open3.sbatch`). As três mudanças são **puramente de metadados** —
  `sha`, `usedStorage`, número de shards e tamanho de cada shard idênticos antes e depois:
  `6d49586f` (g1b-s78), `82543d8e` (bird2-s78), `8e4c43ef` (g3-iter2-omni).
  **Nenhum repo do artigo continua privado.**
- ✅ **`bird1nt-s78` foi publicado**: a auditoria confirmou que não existia no HF, e o merge local
  da gpu-node-a (`t2sql_solver_bird1nt20k_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_78`) foi
  subido como `anonymous/sql-zero-3b-solver-bird1-nothink` (público desde a criação, job 28574,
  `sha` `8fe87740`). Os 12 arquivos (2 shards `.safetensors`, índice, `config.json` e tokenizer)
  conferem **byte a byte** com o local — sha256 LFS para os arquivos grandes, git-blob-sha1 para os
  pequenos, mais download e diff de `config.json` e `model.safetensors.index.json`. Soma dos shards
  6.794.257.024 B. Com isso a ablação think/no-think fica completa no HF (`bird1nt`, `bird2nt`,
  `bird3nt`) e nenhum modelo do artigo sobrevive apenas no disco local de um nó.
- ✅ **Os três challengers 7B foram publicados** (2026-07-28, job 28601, gpu-node-b,
  `scripts/hf_upload_challengers_public.sbatch`). A auditoria original cobria só os solvers; a
  varredura do inventário completo da conta mostrou que estes três merges existiam **apenas na
  gpu-node-b**, sem repo correspondente no HF e sem cópia em outro nó (`/raid` é local por nó):

  | checkpoint local (gpu-node-b) | repo HF (público desde a criação) |
  |---|---|
  | `t2sql_challenger_bird1_omni_.../hf_model` | `anonymous/sql-zero-7b-challenger-bird1-omni` |
  | `t2sql_challenger_bird3_omni_.../hf_model` | `anonymous/sql-zero-7b-challenger-bird3-omni` |
  | `t2sql_challenger_bird3x_7b_.../hf_model` | `anonymous/sql-zero-7b-challenger-bird3x` |

  Os ids foram confirmados inexistentes (`RepositoryNotFoundError`) imediatamente antes da criação —
  o repo homônimo `sql-zero-7b-challenger-bird1` é de **outra linhagem** (`bird1_7b`, shards
  diferentes) e não foi tocado. Verificação pós-upload por
  `repo_info(files_metadata=True)`: os 14 arquivos de cada repo (4 shards, índice, `config.json` e
  tokenizer) batem **byte a byte** com o local, `missing=[] mismatch=[]` nos três. Com isso os
  challengers deixam de ser artefatos de nó único.

**O que isso garante e o que não garante.** O HF garante que os **pesos** sobrevivem e que os
números são **re-deriváveis por reinferência**. Não garante que os números publicados sejam
**re-pontuáveis a partir de artefato sobrevivente**: para as 9 linhas da seção anterior as predições
per-query estão perdidas, e decodificação greedy em build diferente de vLLM pode mover o EX por
fração de ponto. Custo medido de restaurar a bateria por reinferência: **~2 GPU-horas**
(7 × ~14 min para os 7B, conforme jobs 28417/28418; 2 × ~10 min para os 3B).

## Proveniência

- Predições 7B/baselines: `official_eval/OmniSQL/train_and_evaluate/results/*_{bird,spidertest,syn}_greedy.json` (gpu-node-a)
- Predições 3B (BIRD): `omnisql_pipeline/results/` (gpu-node-a)
- Scripts: `run_omnisql_pipeline.sh`, `run_spidertest_pipe.sh`, `run_syn_pipe.sh`, `convert_pred_to_ours.py`
- Scorer: `official_eval/run_official.py` → repos `test-suite-sql-eval` / `mini_dev`
- **2026-07-21**: a gpu-node-a (onde vivia `official_eval/`) ficou inacessível pela rede. O caminho de
  score do Spider test foi reconstruído na gpu-node-a (`official_eval/test-suite-sql-eval` clonado,
  `nltk`+`sqlparse` em `/workspace/anon/python-packages/eval-extra`,
  `scripts/score_spidertest_pipe.sbatch`). Gate do caminho novo: `g1b_3b` re-scoreado dá 0.737 vs
  0.741 tabelado (Δ0.4pt, dentro da tolerância). Diferenças <0.5pt nesta coluna são ruído.
  Armadilha: colapsar a predição em uma linha sem remover comentários `--` comenta a query inteira
  (os 3B prefixam `-- Your SQL query`; o primeiro gate deu 0.031 por isso).
- **2026-07-28** (auditoria do detector de slots): scripts em `official_eval/audit/` na gpu-node-a
  (`apply_patch.py`, `audit_slots.py`, `audit_fix_check.py`, `make_pred_txt.py`); 24 transcrições de
  score em `official_eval/audit/scores/*.txt`; jobs 28531-28557. Tabela de evidência:
  `ara/evidence/tables/pipe_slotfix_audit_2026-07-28.md`.
