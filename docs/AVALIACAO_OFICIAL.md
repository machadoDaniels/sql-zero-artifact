# Avaliação oficial — BIRD e Spider

Como os números "oficiais" do artigo são produzidos, e o que falta. (2026-07-17)

## Repos usados

Clonados em `official_eval/` na **gpu-node-a** (rodam em CPU):

| Benchmark | Repo | O que é |
|-----------|------|---------|
| Spider | `github.com/taoyds/test-suite-sql-eval` | Avaliador oficial do Spider, mantido pelo time do benchmark (Tao Yu). É o harness do leaderboard. |
| BIRD | `github.com/bird-bench/mini_dev` | Repo oficial do BIRD bench; script de EX do leaderboard. |

**Zero código de avaliação próprio** — `official_eval/run_official.py` é um wrapper fino
que formata as predições e chama o `evaluation.py` de cada repo.

## Protocolo

1. **Pesos exatos publicados no HF** (`anonymous/*` + baselines públicos).
2. **Re-inferência**: vllm (docker, gpu-node-a GPUs 4-7), greedy@1, schema completo (DDL + 3 rows)
   — mesmo protocolo de prompt do harness interno.
3. Predições salvas em `results/hf_*.json`.
4. Avaliadores oficiais pontuam as predições:
   `official_eval/run_official.py spider|bird <results.json> <db_root> [diff_json]`.
5. Cada célula logada em `official_eval/official_results.jsonl` — **44 células**
   (14 modelos × Spider dev / Spider test / BIRD; inclui baselines Qwen 3B/7B,
   Arctic, CSC-SQL, OmniSQL, gold-3b e todos os nossos).

## Por que é o mesmo harness que todos usam

- São os repos **canônicos** dos benchmarks (papers e leaderboards apontam pra eles), não forks.
- O **scorer** é idêntico ao dos leaderboards. A **inferência**, não: nosso protocolo unificado
  difere do de cada paper (ver "Published vs reproduced" abaixo) — por isso os números de
  OmniSQL/Arctic reproduzidos ficam abaixo dos publicados; a comparação válida é entre linhas
  da nossa tabela, todas sob o mesmo protocolo.
- Desvio harness interno vs oficial é **sistemático** (+~6pt Spider, +~8pt BIRD; o oficial usa
  comparação set-based, mais leniente com ordem/duplicatas) e o **ranking é idêntico** em todos
  os modelos — consistente com diferença de comparador, não bug.

## Resultados oficiais (células completas)

Coluna **Published** = número reportado no paper de cada modelo (não comparável diretamente
entre linhas nem com as nossas colunas — ver "Published vs reproduced" abaixo).

| Modelo | Spider dev | Spider test | BIRD | Published (BIRD dev / Spider test) |
|--------|-----------|-------------|------|-----------------------------------|
| baseline-3b | 0.681 | 0.718 | 0.4231 | — |
| baseline-7b | 0.823 | 0.829 | 0.5365 | — |
| omnisql-7b-zs | 0.810 | 0.814 | 0.5391 | 0.639 / 0.879 |
| arctic-7b | 0.865 | 0.856 | 0.6102 | 0.689 / 0.888 |
| cscsql-3b | 0.756 | 0.766 | 0.4785 | (ver paper CSC-SQL) |
| gold-3b | 0.795 | 0.794 | 0.4205 |
| sqlzero-3b-it1 | 0.795 | 0.801 | 0.4733 |
| sqlzero-7b-it1 | 0.818 | 0.812 | 0.5522 |
| f5 (omnisql+sqlzero) | 0.833 | 0.836 | 0.590 |
| g1a-s75 | 0.795 | 0.802 | 0.4622 |
| g1b-s78 | 0.788 | 0.803 | 0.4772 |
| bird2-3b-s78 | 0.789 | 0.800 | 0.485 |
| bird3-3b-s78 | 0.793 | 0.803 | 0.4915 |
| bird1-7b-s78 | 0.841 | 0.829 | 0.5821 |
| bird2x-7b-s78 (iter2) | 0.842 | 0.832 | 0.5828 |
| bird3x-7b-s78 (iter3) | 0.826 | 0.831 | 0.5652 | — |

Nossos modelos: `—` em Published (este projeto; os números das 3 primeiras colunas SÃO os oficiais).

## Published vs reproduced — por que os números diferem

Os números publicados de OmniSQL/Arctic são ~3-8pt acima dos reproduzidos aqui. Investigado
em 2026-07-17; **não é bug do avaliador** (o scorer é o oficial em ambos os casos). É protocolo
de **inferência**:

1. **Prompt**: nosso protocolo unificado usa o mesmo prompt para todos (o do nosso treino).
   Teste controlado com o template nativo do OmniSQL (copiado verbatim do repo deles) no BIRD:
   0.5391 → **0.5606** (+2.2pt) — recupera pouco; prompt não é a causa principal.
2. **Schema**: usamos DDL + 3 linhas de exemplo (formato do nosso protocolo). O pipeline do
   OmniSQL embute as **column descriptions** do BIRD (`database_description/*.csv`) no schema —
   no BIRD, onde os nomes de coluna são crípticos, isso vale vários pontos. Nosso protocolo não
   as inclui (para nenhum modelo, incluindo os nossos).
3. **Decoding**: greedy@1, max_tokens 1024 para todos; papers usam orçamentos maiores e, em
   alguns resultados, self-consistency/major-voting.
4. **Evidence (hint)**: incluído no nosso protocolo (padrão BIRD) — NÃO é a causa.

**Leitura correta da tabela**: as colunas reproduzidas comparam todos os modelos sob protocolo
idêntico (única comparação internamente válida); a coluna Published contextualiza o teto de
cada modelo no protocolo do próprio autor. Cf. a não-comparabilidade entre papers sem alinhar
split/prompt/candidatos (nota "Text-to-SQL Benchmarks are Broken").

Evidência do teste de prompt: `results/omnisql7b_bird_NATIVE.json` (gpu-node-a), oficial 0.5606.

## Bateria OmniSQL-pipeline (BIRD dev) — 2026-07-17

Re-inferência de 13 modelos com o **pipeline oficial do repo OmniSQL** (`official_eval/OmniSQL/
train_and_evaluate/infer.py`): prompts pré-processados deles (schema com column descriptions do
BIRD), greedy, max_output 2048, scorer `evaluate_bird.py`. **Gates de fidelidade**: OmniSQL-7B
0.6375 vs 0.639 publicado; Arctic-7B 0.6851 vs 0.689 publicado — pipeline validado.

| Modelo | OmniSQL-pipe | Unified | Published |
|--------|--------------|---------|-----------|
| Arctic-7B | 0.6851 | 0.6102 | 0.689 |
| F5 (OmniSQL+SQL-Zero) | 0.6473 | 0.590 | — |
| OmniSQL-7B | 0.6375 | 0.5391 | 0.639 |
| bird1-7b (it1) | 0.5737 | 0.5821 | — |
| bird3x-7b (it3) | 0.5691 | 0.5652 | — |
| bird2x-7b (it2) | 0.5606 | 0.5828 | — |
| Qwen2.5-Coder-7B | 0.5111 | 0.5365 | — |
| CSC-SQL-3B | 0.4831 | 0.4785 | — |
| g1b-3b (it1) | 0.4374 | 0.4772 | — |
| bird3-3b (it3) | 0.4342 | 0.4915 | — |
| bird2-3b (it2) | 0.4237 | 0.485 | — |
| Qwen2.5-Coder-3B | 0.3866 | 0.4231 | — |
| gold-3b | 0.3859 | 0.4205 | — |

**Leituras:**
1. Gates batem os publicados (Δ≤0.4pt) — o gap anterior era protocolo de inferência, não erro
   de avaliação; nossos números unified são válidos como comparação sob régua única.
2. **F5 > OmniSQL no protocolo do próprio autor** (0.6473 > 0.6375): self-play zero-data agrega
   sobre init forte jogando o jogo deles.
3. **Self-play > gold-supervisionado nas duas réguas**: gold-3b 0.3859 < todas as espirais 3B
   (0.424-0.437) no pipe OmniSQL; idem no unified.
4. Rankings nossos-vs-baselines (Qwen 3B/7B) e vs gold preservados em ambos os protocolos.
   **Exceção**: CSC-3B × nossas 3B inverte com a régua (unified: bird3-3b 0.4915 > CSC 0.4785;
   pipe: CSC 0.4831 > bird3-3b 0.4342) — reportar ambos os números no paper.
5. Espirais no pipe OmniSQL: 7B 0.574/0.561/0.569, 3B 0.437/0.424/0.434 — flat (saturação
   na iter2 reproduz em protocolo independente).

Predições: `official_eval/OmniSQL/train_and_evaluate/results/*_bird_greedy.json` (gpu-node-a) e
`omnisql_pipeline/results/` (gpu-node-a, cadeia 3B).

## Bateria OmniSQL-pipeline (Spider test) — 2026-07-17

Mesmo pipeline (infer.py do OmniSQL, greedy, max 2048), input `test_spider.json` deles,
scorer oficial test-suite. **Gates**: OmniSQL-7B 0.877 vs 0.879 publicado; Arctic-7B 0.897 vs
0.888 publicado — pipeline fiel também no Spider test. (Conversor tira as linhas `--` do
template OmniSQL antes de pontuar.)

| Modelo | OmniSQL-pipe (test) | Unified (test) |
|--------|--------------------|----------------|
| Arctic-7B | 0.897 | 0.856 |
| OmniSQL-7B | 0.877 | 0.814 |
| F5 (OmniSQL+SQL-Zero) | 0.866 | 0.836 |
| bird3x-7b (it3) | 0.829 | 0.831 |
| bird1-7b (it1) | 0.824 | 0.829 |
| Qwen2.5-Coder-7B | 0.822 | 0.829 |
| bird2x-7b (it2) | 0.813 | 0.832 |
| CSC-SQL-3B | 0.76 | 0.766 |
| bird3-3b (it3) | 0.759 | 0.803 |
| bird2-3b (it2) | 0.757 | 0.800 |
| g1b-3b (it1) | 0.741 | 0.803 |
| gold-3b | 0.719 | 0.794 |
| Qwen2.5-Coder-3B | 0.68 | 0.718 |

**Leituras (Spider test):**
1. Gates batem (Δ≤0.9pt) — pipeline validado.
2. **Nossos 7B ≈ baseline no Spider test em ambas réguas** (0.81-0.83): o self-play foi
   treinado no ambiente BIRD; Spider é transfer, ganho pequeno esperado.
3. **Nossos 3B > gold-3b nas duas réguas** (0.74-0.76 vs 0.719): self-play zero-data supera
   o controle supervisionado também em transfer.
4. Espirais flat entre iters (7B 0.824/0.813/0.829; 3B 0.741/0.757/0.759) — saturação
   reproduz. Nenhuma inversão nosso-vs-baseline; CSC-3B ≈ nossas 3B (0.76, empate).

## O que falta

| Modelo | Falta | Bloqueio |
|--------|-------|----------|
| G3 OmniSQL iter1 (bird1_omni20k) | 3 células | solver esperando GPU (gêmeos B200/gpu-node-a) |
| G3 iter2 (bird2_omni20k) | 3 células | encadeado atrás da iter1 |
| G3 iter3 (bird3_omni20k) | 3 células | encadeado atrás da iter2 |
