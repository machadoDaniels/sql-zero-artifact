# Decisões de Arquitetura — Dr. Zero → Text-to-SQL

Registro das escolhas de design para adaptar o **Dr. Zero** (self-play proposer/solver, zero-data, GRPO) do domínio de busca em Wikipedia para **Text-to-SQL** com reward de execução em Spider. Cada linha é uma decisão travada. Fontes citadas em `../references/papers/RESUMOS.md`.

---

## 1. Escopo e ambiente

| Decisão | Escolha | Porquê |
|---|---|---|
| Domínio-alvo | Text-to-SQL (não busca) | Objetivo do projeto |
| Dados | Bancos padrão de pesquisa (**não** ANON-LAB) | Comparabilidade com a literatura |
| Ambiente de treino | **Spider** (166 bancos SQLite, cross-domain) | Padrão de generalização; bancos em arquivo, leves e paralelizáveis por rollout |
| Ambiente futuro | BIRD como stretch (não no 1º ciclo) | Mais realista/pesado (33GB, valores sujos) |
| Avaliação | **Spider dev (1.034 ex., 20 bancos não vistos), execution accuracy (EX)** | Cross-DB mede generalização; métrica canônica; números publicados p/ comparação |
| verl | **Fixar em 0.5.0** (vendorizado no repo) | Atualizar p/ 0.7.1 quebraria o código self-play customizado |

## 2. Self-play (proposer ↔ solver)

| Decisão | Escolha | Porquê |
|---|---|---|
| Sinal de gold | **Self-play puro** (proposer gera o gold SQL) | Zero dados rotulados, fiel ao Dr. Zero |
| Modo do proposer | **Single-shot**, **SQL-first → NL** | Solvabilidade por construção; padrão OmniSQL/SENSE |
| Modo do solver | **Single-shot** (emite `<sql>`) | Simplicidade; Arctic mostra que basta |
| Agrupamento do baseline | **HRPO-lite: agrupar por nível de complexidade** (substitui hops) | Sem agrupamento, o baseline do batch mistura níveis e enviesa o advantage dos "complex"; `grpo_batch` agrupa por `data_source` |
| Estimativa de dificuldade | **N gerações diretas** do solver + `compare_results` | Solver é single-shot → dispensa a máquina multi-turn |
| Fórmula de dificuldade | `(n−k)/(n−1)`, gate `0<k<n` (pico em k=1) | Ablação do próprio Dr. Zero: **bate** o pico-em-50% do R-Zero |
| Round-trip NL↔SQL | **Dispensado** | O gate `k>0` já garante que a pergunta é solúvel-ao-gold |
| Validação do gold | Executa; **rejeita**: SQL inválido, não-SELECT, **resultado vazio**, e **não-determinismo** (`LIMIT` sem `ORDER BY`, `random()`, `datetime('now')`) | Evita reward degenerado e gold semanticamente indefinido (filtro de determinismo do Absolute Zero) |
| Dedup de perguntas | **Offline no `gen_data`**: mascarar literais → template → 1 por template (OmniSQL) | Captura o benefício da penalidade de repetição do R-Zero sem a complexidade online |

### 2.1 Níveis de complexidade (rubrica)

Medíveis por contagem estrutural (regex, `search/sql_utils.py`), sem modelo:

| Nível | Critério estrutural |
|---|---|
| **1 — Simple** | 1 tabela; sem join/agregação/subquery |
| **2 — Moderate** | 1 join **ou** agregação (GROUP BY / COUNT / AVG…) |
| **3 — Complex** | ≥2 joins, **ou** subquery/aninhamento/HAVING/set ops |

Uso: (a) nível **sorteado** vai no prompt do proposer e na chave `data_source` (`t2sql_zero_<level>`); (b) baseline do `grpo_batch` agrupado pelo nível solicitado (HRPO-lite); (c) no reward, mismatch solicitado-vs-real é **logado**, não penalizado. Distribuição: uniforme.

## 3. Rewards

| Decisão | Escolha | Porquê |
|---|---|---|
| Reward do solver | **3 níveis: `1.0` match · `0.1` executa-mas-erra · `0.0` inválido** | Arctic: mínimo vence composto, evita reward-hacking |
| SQL inválido | Reward `0`, **nunca negativo** | Consenso 4/4 papers (negativo desestabiliza GRPO) |
| Comparação de result-set | **Multiset, order-agnostic** salvo `ORDER BY` no gold | Convenção Spider EX |
| Reward do challenger | `0.5·format + difficulty` (dificuldade domina) | Estrutura original do Dr. Zero |
| Format-reward do challenger | tags presentes + SQL parseável + executa + resultado não-vazio + determinístico | Adaptação do format-reward de busca para SQL |
| Reward denso (QATCH/PRM/judge) | **Não** no início; QATCH é o 1º upgrade se binário ficar esparso p/ o 3B | Ganho ~+2pts EX com custo/risco; QATCH sai quase de graça do `compare_results` |

## 4. Representação de schema no prompt

| Decisão | Escolha | Porquê |
|---|---|---|
| Formato | **`CREATE TABLE` (DDL)** com PK/FK/tipos | Formato OmniSQL/Arctic; confirmado 3× (CodeS, BASE-SQL, SQL-o1) |
| Valores | **+ 3 linhas de exemplo** por tabela (em comentário) | PK/FK+valores são o que mais importa (ablação CodeS) |
| Schema linking prévio | **Não** — schema inteiro | Solver com CoT+GRPO aprende a linkar; proposer precisa do schema completo |
| Value-retrieval (BM25) | **Depois**, se BIRD virar gargalo | Otimização não-essencial |

## 5. Modelo e hiperparâmetros

| Decisão | Escolha | Porquê |
|---|---|---|
| Modelo base | **Qwen2.5-Coder-3B-Instruct** | Paridade de custo com Dr. Zero (3B) + viés p/ código; escalar p/ 7B depois |
| Tag de saída | `<think>` (raciocínio) + `<sql>` **limpo** | DAIL: "no explanation" melhora acurácia → raciocínio no think, SQL limpo |
| Dialeto | SQLite | Ambiente Spider |
| `ground_truth` no parquet | JSON `{db_id, gold_sql}` | Reward precisa do db_id p/ executar |
| Group size GRPO (solver) | **8** | Arctic/Think2SQL usam 16; 8 é o compromisso p/ 1 GPU |
| `n` rollouts de dificuldade | **5** | Dr. Zero original |
| `max_prompt_length` / response | **3072 / 1536** | Prompt carrega o schema inteiro (DDL + sample rows) |

## 6. POC — staging e critério de sucesso

| Decisão | Escolha | Status |
|---|---|---|
| **Passo 0 — Baseline** | Coder-3B **zero-shot, greedy**, mesmo prompt do treino, EX | ✅ **Spider dev 0.617** (níveis: 0.651/0.687/0.456) · **BIRD dev 0.347** (níveis: 0.345/0.393/0.215; exec-ok 71%) — ambos na faixa da literatura (CSC SC@4: 0.644 Spider / 0.447 BIRD; Think2SQL c/ gold-schema: 0.725 / 0.382) |
| **Iter0 — loop mínimo** | Proposer = modelo base só-prompt → gen_data c/ filtros → treina **só o solver** → eval | ✅ **rodou** (job 21704): gen0 2.125 pares (yield ~10%: 5.095 extraídos, 58% inválidos, 8 dup); 8 steps GRPO; ckpt `global_step_8` |
| **Iter1+ — loop completo** | Treinar challenger → gen_data → solver → convert | 🔄 challenger@50 treinado (difficulty 0.037→0.107, 3×); gen1+solver1 rodando |
| **Critério de sucesso** | **EX do solver iter0/iter1 > EX zero-shot** no Spider dev | ✅ **ATINGIDO** — 2k pares: 0.645; **10k pares: 0.703 (+8.6)**; transferência BIRD: 0.347→**0.372 (+2.5)** sem nunca ver BIRD |
| Filtragem do gold | (a) format-reward do challenger; (b) **gen_data** (validade + não-vazio + determinismo + dedup) | ✅ implementado/testado |
| Monitorar | comprimento do `<think>`, taxa executável, mismatch de nível, diversidade de templates | gen0 smoke: 62% inválido, 0 dup, 27% mismatch |

## 7. Infraestrutura (status)

| Componente | Arquivo | Status |
|---|---|---|
| Executor SQLite (`/execute`,`/compare`,`/schema` c/ sample rows) | `search/sql_exec_server.py` | ✅ testado |
| Utilitários SQL (rubrica, determinismo, template) | `search/sql_utils.py` | ✅ testado |
| Download do Spider | `scripts/download_spider.py` | ✅ usado no cluster (166 bancos) |
| Prompts challenger/solver | `verl/prompts.py` | ✅ |
| Reward do solver (3 níveis + batelado) | `verl/utils/reward_score/sql_exec.py`, `verl/custom_reward/t2sql_reward.py` | ✅ testado (local + cluster) |
| Reward do challenger (exec difficulty + HRPO-lite) | `verl/custom_reward/t2sql_reward.py` | ✅ testado local |
| Data CLI (challenger-prompts / test-set / gen-solver-data) | `t2sql_data.py` | ✅ testado (local + cluster) |
| Baseline zero-shot | `scripts/eval_spider.py` + `t2sql_baseline.sh` | ✅ rodou no cluster |
| Scripts de treino + config | `t2sql_{challenger,gen_data,solver}.sh`, `config/t2sql_grpo.yaml` | ✅ solver validado no cluster |
| Slurm/singularity | `t2sql_poc.sbatch` + `t2sql_poc_run.sh` | ✅ smoke completo |

## 8. Dimensionamento e operação

> **Nota (2026-07-07):** o destilado operacional (armadilhas de cluster, receita B200, convenções) vive agora em `CLAUDE.md` na raiz — é o runbook que o agente carrega em toda sessão. Esta seção permanece como registro histórico/detalhado.

| Decisão | Escolha | Observação |
|---|---|---|
| **Split de bancos (regra dura)** | Proposer amostra **apenas bancos do Spider train**; os 20 do dev invisíveis até a avaliação | Enforced no `t2sql_data.py` (testado) |
| Dataset sintético por iteração | ~10–25k pares pós-filtros | Yield do proposer base ≈10% → calibrar `--num_prompts` |
| Host/GPUs | **dgx-H100-03, partição `gpu-node-a`, 1× H100** (`NUM_GPUS`/`TP` parametrizados) | 3B cabe em 1 GPU (pico 57.6GB); challenger pede 2 |
| Imagem | `verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.2-te2.2` (cu126) via singularity | Tag da doc oficial não existe; esta é a real do Docker Hub |
| Smoke test | ✅ **PASSOU** (job 21698): baseline + gen0 + 1 step GRPO + val no dev completo (EX 0.632) | — |
| Critério de parada entre iterações | EX no dev estabilizar (< +1pt) | — |

### Lições do cluster (fixes do smoke)
1. `lsof` não existe na imagem verl → `fuser -k ... || true`
2. Hydra resolve `--config-path` relativo contra `main_ppo.py` → usar `$PWD/config`
8. `fuser` TAMBÉM não existe na imagem → matar servidores **por PID** (`$!` + trap) e health-check com timeout
9. Schema Spider de 3.7k tokens estourou `max_prompt_length` 3072 → 4096 + `filter_overlong_prompts: True`
3. `.gitignore` `data/` sem âncora engolia `verl/trainer/config/data/legacy_data.yaml` → ancorar com `/data/`
4. Slurm exporta `ROCR_VISIBLE_DEVICES` → `unset` antes do verl
5. 80G de RAM insuficiente p/ FSDP offload + Ray → 256G
6. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` segfaulta o engine do sglang** → nunca exportar com sglang
7. `/tmp` dentro do job Slurm é tmpfs por-job → logs do Ray morrem com o job

## Documentos do projeto

| Arquivo | Conteúdo |
|---|---|
| `docs/DECISOES_ARQUITETURA.md` (este) | Decisões travadas + status + dimensionamento + lições |
| `references/papers/RESUMOS.md` | 44 papers: problema/solução/contribuição/insights |
| `docs/PAPER_NOTES.md` | Planejamento do artigo (tese, gap, INSIGHTs) — pausado |

### 8.x Treino na B200 (Blackwell/sm_100) — como destravar (2026-07-06)
A imagem verl0.5 (cu126) não roda em sm_100. Solução validada por smoke test (job 21853, 3B, 3 steps):
1. Imagem: `verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2` (cu128) → `/raid/.../images/verl0.6-cu128.sif`.
2. `PYTHONNOUSERSITE=1` obrigatório — o singularity monta o $HOME e o `~/.local` (compartilhado entre nós via NFS) sombreia os pacotes da imagem.
3. Shims de import no verl vendorado (sglang ≥0.5 moveu `*MemoryOccupationReqInput`/`UpdateWeightsFromTensorReqInput` p/ io_struct; removeu `maybe_set_triton_cache_manager`) — try/except, compatível com as duas imagens.
4. `SGLANG_ATTENTION_BACKEND=flashinfer` — fa3 (hardcoded no verl 0.5) exige SM 80-90.
5. Cache JIT do flashinfer stale em `~/.cache/flashinfer` causa `OSError ... .so: No such file` — limpar antes do primeiro uso no nó.
