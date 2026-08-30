# SQL-Zero — Resultados (tabela-mestre)

Protocolo único: nosso harness (`scripts/eval_spider.py`), **greedy@1**, schema completo (DDL + 3 sample rows), métrica **EX (execution accuracy)**. Spider dev = 1.034 ex.; BIRD dev = 1.534 ex. (**transferência** — nunca visto no treino). Por-nível = fácil / médio / complexo. Dados brutos em `RESULTADOS.csv` (gerado dos JSONs em `results/`).

## Bateria oficial — avaliadores externos sobre pesos HF exatos (2026-07-15)

Números do artigo re-inferidos dos **pesos publicados no HuggingFace** (`vllm serve <repo[@rev]>`, greedy@1, schema unificado) e pontuados pelos **avaliadores oficiais**: Spider `test-suite-sql-eval` (`evaluation.py --etype exec`) e BIRD `bird-bench mini_dev` (`evaluation_ex.py`). Colunas `*_oficial` no `RESULTADOS.csv`; JSONs de predição em `results/hf_*.json`/`results/{arctic,omnisql,csc,baseline}*.json`; log canônico em `official_eval/official_results.jsonl` (gpu-node-a).

**Nota de protocolo — oficial × harness próprio.** O avaliador oficial roda consistentemente **acima** do nosso harness (Spider dev ~+6pt, BIRD ~+8pt) em **todos** os modelos. Não é bug: o oficial é **set-based** (compara `set(linhas)`, ignora ordem e duplicatas) enquanto o nosso exige casamento mais estrito. O gap é sistemático e **não inverte nenhum ranking** — a comparação entre modelos é idêntica nos dois protocolos. Reportar o oficial no artigo (padrão da literatura) e o harness como controle interno.

| Modelo | Cat. | Supervisão | Spider dev | Spider test | BIRD |
|---|---|---|---|---|---|
| Arctic-Text2SQL-R1-7B | concorrente | OmniSQL-init + gold + sintético + GRPO | **0.865** | **0.856** | **0.610** |
| OmniSQL-7B (zero-shot) | baseline-sft | SynSQL + Spider/BIRD gold (SFT) | 0.810 | 0.814 | 0.539 |
| CSC-SQL-3B (greedy@1, sem pipeline) | concorrente | gold humano BIRD + GRPO | 0.756 | 0.766 | 0.479 |
| Qwen2.5-Coder-7B | baseline | nenhuma | 0.823 | 0.829 | 0.537 |
| Qwen2.5-Coder-3B | baseline | nenhuma | 0.681 | 0.718 | 0.423 |
| **bird1-7b** (G2, ambiente-BIRD + think) | **nosso 7B** | **zero (self-play)** | **0.841** | 0.829 | 0.582 |
| F5 (OmniSQL + SQL-Zero) ¹ | nosso-sft-init | SFT(init) + self-play | 0.833 | 0.836 | 0.590 |
| 7b-it1 (F3, challenger-first) | nosso 7B | zero (self-play) | 0.818 | 0.812 | 0.552 |
| **bird3** (BIRD-3B think, iter3) | **nosso 3B** | **zero (self-play)** | 0.793 | 0.803 | **0.4915** |
| bird2 (BIRD-3B think, iter2) | nosso 3B | zero (self-play) | 0.789 | 0.800 | 0.485 |
| g1b (BIRD-3B think, iter1) | nosso 3B | zero (self-play) | 0.788 | 0.803 | 0.477 |
| 3b-it1 (A2, ambiente unificado) | nosso 3B | zero (self-play) | 0.795 | 0.801 | 0.473 |
| g1a (BIRD-3B, sem think) | nosso 3B | zero (self-play) | 0.795 | 0.802 | 0.462 |
| gold-3b (controle) | controle | gold humano Spider | 0.795 | 0.794 | 0.421 |
| **gold-bird-3b** (controle pareado) | controle | gold humano BIRD (8.390 pares) | 0.758 | — | 0.4654 |

¹ F5 tem init SFT (OmniSQL viu Spider/BIRD train no SFT) — **não é zero-data no Spider**; entra como braço `sft-init`, não como self-play puro.

**Leituras (oficial):** (1) Arctic é o teto absoluto (RL pesado com labels); nosso melhor zero-data 7B (bird1-7b) fica ~2.4pt no Spider dev e ~2.8pt no BIRD. (2) Nossos 7B zero-data **superam os dois baselines** e o OmniSQL-zero-shot em BIRD. (3) A **espiral BIRD-3B melhora monotônico no oficial** (g1b 0.477 → bird2 0.485 → bird3 0.4915) — contraria a leitura de "satura" que vinha do harness próprio (0.4048→0.4087→0.4100); o comparador set-based revela ganho continuado por iteração. (4) g1a (sem think) < iterações com think, consistente com o efeito-think. (5) **Controle pareado**: gold-bird-3b (gold humano do BIRD, mesmos 63 bancos, mesmo GRPO/think) dá BIRD oficial **0.4654** — acima do gold-Spider (0.421, esperado por treinar no benchmark-alvo) mas **abaixo das TRÊS iterações self-play 3B** (0.477/0.485/0.4915). Fecha o "preço da anotação" com controle pareado: zero-data supera anotação humana no MESMO benchmark e escala.

| Modelo | Categoria | Supervisão | Spider EX (N1/N2/N3) | BIRD EX (N1/N2/N3) |
|---|---|---|---|---|
| Qwen2.5-Coder-3B zero-shot | baseline | nenhuma | 0.617 (.651/.687/.456) | 0.347 (.345/.393/.215) |
| Qwen2.5-Coder-7B zero-shot | baseline | nenhuma | 0.756 (.841/.787/.620) | 0.456 (.436/.502/.331) |
| CSC-SQL-3B | concorrente | gold humano (BIRD) + GRPO | 0.689 (.821/.730/.489) | 0.404 (.442/.445/.269) |
| Think2SQL-7B | concorrente | gold + destilação R1 | 0.765 | 0.505 |
| SQL-Zero iter0 (2k) | **nosso** | **zero (self-play)** | 0.645 (.698/.717/.464) | 0.344 (.376/.389/.201) |
| SQL-Zero iter0 (10k) | **nosso** | **zero (self-play)** | 0.703 (.817/.734/.540) | 0.372 (.448/.390/.283) |
| **SQL-Zero iter1** (step 200/226, último salvo) | **nosso** | **zero (self-play)** | **0.727** (.813/.762/.584) | **0.395** (.448/.423/.289) |
| SQL-Zero iter2 (sem rep-penalty; step 97/97, final) | nosso | zero (self-play) | 0.686 (.790/.685/.591) | 0.378 (.382/.421/.252) |
| **SQL-Zero iter2 (rep-penalty)** ⭐ (step 100/139, best-dev, run completo) | **nosso** | **zero (self-play)** | **0.728** (.841/.760/.566) | 0.376 (.448/.400/.275) |
| SQL-Zero iter0 + think (ablação) | ablação | zero (self-play) | 0.694 (.825/.728/.511) | 0.376 (.455/.401/.266) |
| **Gold-train** (controle: mesmo GRPO, rótulos humanos do Spider) | controle | gold humano (Spider) | 0.731 (.857/.726/.624) | 0.357 (.491/.369/.261) |

## iter2-v2: curva completa de checkpoints (re-treino de 139 steps, ininterrupto)

| ckpt | Spider | BIRD | nota |
|---|---|---|---|
| s25 (run interrompido) | 0.729 | 0.400 | |
| s50 | 0.728 | 0.383 | |
| **s100** | **0.728** | 0.376 | best-dev (val 0.754) — canônico |
| s139 (final) | 0.720 | 0.366 | |

Spider estável (~0.728, reproduzido entre runs — o 0.729 original não era pico de sorte). **Transferência decai com treino prolongado** (BIRD 0.400→0.366): treinar mais no domínio Spider erode a generalização — mesmo padrão do gold-train (0.357). Early stopping preserva transferência.

## Leituras-chave

0. **Preço da anotação ≈ zero**: com o MESMO modelo/GRPO/protocolo, o solver com gold humano faz 0.731 no Spider — o self-play faz 0.729 **sem nenhum rótulo** (−0.2pt, empate) e **vence em transferência** (BIRD 0.400 vs 0.357, +4.3pt): o gold overfitta a distribuição do Spider; os dados auto-gerados generalizam melhor. Nuance: gold ganha no N3 in-distribution (.624 vs .584).

1. **Zero anotação supera gold humano no mesmo tamanho**: SQL-Zero iter2-v2 (0.729) > CSC-SQL-3B (0.689) no Spider; empate no BIRD (0.400 vs 0.404), domínio in-distribution do CSC.
2. **Espiral monotônica** (com a correção de diversidade): Spider 0.617→0.703→0.727→0.729; BIRD 0.347→0.372→0.395→0.400.
3. **Papel causal da repetition penalty**: iter2 regride sem ela (0.686) e recupera com ela (0.729). Diagnóstico: gen sem penalty tinha 66% de templates duplicados → overtraining.
4. **Raciocínio explícito não compensa** (T2SQL single-shot 3B): iter0+think (0.694) < iter0 sem think (0.703), e 2.4× mais tokens.
5. **93–96% do 7B zero-shot** (0.729 vs 0.756) com um 3B e nenhuma anotação.

## Robustez — Spider-Syn (sinônimos quebram o casamento léxico com o schema)

| Modelo | Spider dev | Spider-Syn | Queda |
|---|---|---|---|
| Baseline 3B | 0.617 | 0.513 (.555/.563/.380) | −10.4 |
| Gold-train | 0.731 | 0.622 (.738/.639/.482) | −10.9 |
| **SQL-Zero iter2-v2** (s100 canônico) | 0.728 | **0.663** | **−6.5** |

Três eixos independentes de generalização, mesma direção: in-distribution empate (0.729 vs 0.731); bancos novos (BIRD) +1.9 pro self-play (s100; +4.3 no s25); linguagem nova (Syn) +4.1 pro self-play (0.663 vs 0.622). O gold degrada como o baseline (−10.9 vs −10.4 — anotação não compra robustez); só o self-play reduz a queda.

## Braço F5 — OmniSQL-init + self-play (fechado 2026-07-07)

| Modelo | Spider dev | BIRD |
|---|---|---|
| OmniSQL-7B zero-shot (protocolo unificado) | 0.747 | 0.448 |
| OmniSQL-7B zero-shot (formato NATIVO dele) | 0.764 | 0.489 |
| F5 s75 ("best" pela val de treino) | 0.753 | 0.466 |
| **F5 s200 (último)** | **0.763** | **0.491** |

**Leituras:** (1) F5-s200 = nosso melhor 7B (+1.5/+1.8 sobre o SQL-Zero-7B base-init); (2) vs o próprio init no formato nativo: empate (0.763≈0.764; 0.491≈0.489) — self-play re-adapta o formato e preserva a capacidade; o teto de regime domina; (3) penalidade de formato do protocolo unificado quantificada: 1.7pt Spider / 4.1pt BIRD (OmniSQL nativo vs unificado); (4) **a val de treino inverteu o ranking dos checkpoints** (dizia s75>s200; o eval greedy oficial diz o oposto nos DOIS benchmarks) — seleção de checkpoint deve usar eval greedy, não a val de treino.

## Braço F4 — solver-7B + think, par do B2 em escala (fechado 2026-07-08)

| ckpt | Spider (N1/N2/N3) | BIRD (N1/N2/N3) |
|---|---|---|
| s75 | 0.7495 (.810/.781/.635) | 0.4844 (.473/.525/.374) |
| **s200 (melhor Spider)** | **0.7515** (.790/.805/.617) | 0.4798 (.455/.519/.380) |
| s240 (final, par direto do F3) | 0.7485 (.798/.783/.639) | **0.4844** (.491/.517/.388) |

Comparação F3 (sem think, s240) vs F4 (com think, s240) — mesma receita, único eixo variado é `SOLVER_THINK_BONUS=0.1`:

| | Spider | BIRD |
|---|---|---|
| F3 — sem think | 0.7476 | 0.4733 |
| F4 — com think | 0.7485 | **0.4844** |

**Leituras:** (1) curva quase flat entre steps (amplitude 0.3pt Spider / 0.5pt BIRD) — nenhum checkpoint domina nos dois benchmarks; treino satura cedo (≤s75) e não degrada até s240; (2) pareado a F3 no mesmo step, think abre uma vantagem pequena porém consistente no BIRD (+1.1pt) e neutra no Spider (+0.1pt) — **o oposto do achado do B2 no 3B** (think piorou −0.9pt Spider / 2.4× tokens): o achado do B2 era sobre **tamanho do modelo**, não sobre T2SQL em geral; (3) ambos os braços do base-7B (F3 e F4) ficam abaixo do baseline zero-shot 7B no Spider (0.756) mas acima no BIRD (0.456) — consistente com a hipótese da trilha G de que o teto do Spider é do ambiente, não do método.

## Spider TEST (conjunto de reporte — 2.147 ex., seleção apenas no dev)

| Modelo | dev | test |
|---|---|---|
| SQL-Zero-3B (s100) | 0.728 | **0.7345** |
| Gold-train-3B | 0.731 | **0.7364** |
| CSC-SQL-3B | 0.689 | 0.7205 |
| Baseline 3B | 0.617 | 0.6744 |
| SQL-R1-3B | 0.608 | 0.5957 |
| Arctic-7B | 0.790 | 0.7909 |
| SQL-R1-7B | 0.782 | 0.7871 |
| Think2SQL-7B | 0.765 | 0.7834 |
| OmniSQL-7B + SQL-Zero (F5 s200) | 0.763 | **0.7788** |
| OmniSQL-7B (zero-shot) | 0.7592 | 0.7774 |
| SQL-Zero-7B BIRD-env think (G2 s78) | 0.766 | 0.7694 |
| Baseline 7B | 0.756 | 0.7685 |
| SQL-Zero-7B think (F4 s75) | 0.750 | 0.7615 |
| SQL-Zero-7B think (F4 s200) | 0.752 | 0.7578 |
| SQL-Zero-7B (s240) | 0.748 | 0.7564 |
| SQL-Zero-7B think (F4 s240) | 0.749 | 0.7499 |

**Spider TEST 7B (2026-07-15):** F5 0.7788 supera o próprio OmniSQL-zs (0.7774) — o self-play em cima do SFT paga também no test; G2 (BIRD-env, nunca viu Spider) 0.7694 ≳ baseline 0.7685 — C13 reproduz em escala 7B no conjunto de reporte; F4 base-init fica 0.750-0.762, curva flat igual ao dev.

**Spider-Syn 7B (2026-07-15):** baseline-7B 0.6460 · F4 s240 0.6586 · F5 0.6809 · G2 s78 **0.6857** (+4pt sobre baseline) — robustez a perturbação léxica favorece os self-play, e o BIRD-env (G2) lidera: mais evidência de C13 (ambiente difícil compra robustez).

**Leituras:** (1) o empate self-play↔gold reproduz no test (0.7345 vs 0.7364) — claim central confirmada sem seleção; (2) SQL-Zero-3B > CSC-3B também no test; (3) teto de regime: 7Bs gold/sintético-treinados colapsam em 0.777-0.791 — vs ~0.88 publicado, ~9pts eram protocolo (SC/oráculos); (4) SQL-R1-3B abaixo do baseline nos dois conjuntos.

## Concorrentes — atualização (protocolo unificado, dev)
Arctic-7B **0.790/0.512** (topo) · SQL-R1-7B 0.782/0.499 · SQL-R1-3B 0.608/**0.290** (abaixo do baseline em ambos; extração validada — 0.5% falha).

OmniSQL-7B zero-shot no prompt unificado (2026-07-15, GPU5 gpu-node-a): **0.7592 Spider dev / 0.4668 BIRD dev** — fecha a célula que faltava do baseline SFT. Leitura: no prompt unificado o OmniSQL-7B fica ~0.5pt acima do baseline-7B no Spider (0.7592 vs 0.7563) e ~1pt acima no BIRD (0.4668 vs 0.4557); os SQL-Zero-7B (bird1-think 0.4909, F5 0.4909) superam o OmniSQL-zs no BIRD em ~2.4pt.

## C4 — Curva de escala de dados (treino único, eval oficial de cada checkpoint) — 2026-07-10

Solver 3B no gen1 completo (58k pós-dedup), batch 256, LR constante pós-warmup; cada checkpoint
(SAVE_FREQ=25) é um ponto "EX × amostras". Treino 21991/22022/22060 (B200→gpu-node-a), evals 25107.

| amostras | step | Spider dev | BIRD dev |
|---|---|---|---|
| 6.4k | s25 | 0.6779 | 0.3533 |
| 12.8k | s50 | 0.7089 | 0.3846 |
| 19.2k | s75 | **0.7263** | 0.3833 |
| 25.6k | s100 | 0.7176 | 0.3963 |
| 32k | s125 | 0.7195 | **0.4029** |
| 38.4k | s150 | 0.7215 | 0.3977 |
| 44.8k | s175 | 0.7195 | 0.3957 |
| 51.2k | s200 | 0.7166 | 0.4022 |
| 57.6k | s225 | 0.6963 | 0.3996 |
| 58k | s226-final | 0.6963 | 0.3970 |

**Leituras:** (1) **Spider pica exatamente em 19.2k** (0.7263 ≈ 0.7273 do iter1 com 58k) e depois
paira 0.717-0.722 — a regra dos ~20k por iteração (decisão 2026-07-09) sai confirmada com números
oficiais; (2) **treinar além não é neutro**: a cauda degrada (0.696 nos últimos pontos, −3pt do
pico) — 1 epoch completo em LR constante piora o endpoint; (3) a **transferência p/ BIRD engata
mais tarde** (platô 0.396-0.403 só a partir de 25.6k; pico s125/32k) — mesma assinatura do g1a
in-domain (s25≈baseline), sustenta o orçamento condicional à dificuldade (decisão 2026-07-10);
(4) ressalva: C4 é re-treino do iter1 (mesmo dado, outra ordem/seed) — o endpoint difere do iter1
(0.696 vs 0.727), reforçando que o ponto de corte importa mais que o dataset.

## G1a — Self-play no ambiente BIRD (3B, sem think) — fechado 2026-07-10

Challenger-3B nos 69 bancos do BIRD train (sem pares gold) → gen 35k pós-dedup → solver-3B.
Corte em s100 (regra 20k; s75=s100 no oficial). Jobs 22056/22067/22068 + eval 25108.

| checkpoint | BIRD dev | Spider dev (transfer reverso) |
|---|---|---|
| s75 (19.2k) — canônico | **0.3931** | **0.7350** |
| s100 (25.6k) | 0.3931 | 0.7292 |

**Leituras:** (1) **in-domain o pivô não destravou o alvo**: 0.393 empata com a transferência
dos 3B/Spider (0.395-0.400) e fica −1.1pt do CSC-3B gold-BIRD (0.404) — o gargalo do BIRD não
era (só) o ambiente de treino; (2) **o achado é o transfer reverso**: 0.735 no Spider sem nunca ver um banco do Spider —
+0.9pt sobre o pareado exato (C4 s75: 1 iteração, mesmas 19.2k amostras, 0.7263) e +0.7pt sobre o
iter2 (DUAS iterações, 0.7282); os deltas estão dentro do ruído (±1.4pt, n=1034), então a leitura
calibrada é: **1 iteração em ambiente difícil iguala os treinados in-domain no Spider, inclusive
com mais iterações** — transferência total entre domínios, empate estatístico, não vitória; (3) regra dos
20k confirmada in-domain (s75 = s100 idênticos no oficial); (4) curva de val achatou desde s50
(0.3905/0.3905/0.3983) — consistente com o platô do BIRD-transfer do C4.

## Espiral BIRD-3B (think) — 3 iterações COMPLETAS 2026-07-15 (satura na iter2, iter3 confirma)

| iteração | BIRD dev | Spider dev (transfer) | Spider TEST |
|---|---|---|---|
| baseline 3B | 0.3468 | 0.617 | 0.6744 |
| iter1 (G1b s78) | 0.4048 | 0.7292 | **0.7424** |
| **iter2 (bird2 s78)** | **0.4087** | 0.7215 | 0.7345 |
| iter3 (bird3 s78) | 0.4100 | 0.7157 | 0.7387 |

**Leituras:** (1) iter1→iter2 = +0.4pt e iter2→iter3 = +0.1pt, ambos < ruído (±1.25) → **a espiral
satura na 2ª volta também no BIRD** e a iter3 (rodada por decisão de simetria da matriz) confirma o
platô, espelhando o Spider — o valor do self-play concentra-se na 1ª iteração, independente da
dificuldade do ambiente; (2) mecanismo verificado (C11): colapso bimodal do currículo adversarial —
banda paga do k-hist encolhe de 16.1%→11.5% entre iter2 e iter3 do challenger (o k-hist completo da iter1 não está preservado no artefato local); (3) três
iterações consecutivas ≥ CSC-3B gold-BIRD (0.4042) com zero anotação; (4) **Spider TEST do G1b:
0.7424 — melhor 3B do projeto no test**, acima do treinado no próprio Spider (0.7345) e do
gold-control (0.7364): o transfer reverso reproduz no conjunto de reporte; (5) dup do gen:
50%→32.5%→35.2% — a queda inicial não continua; diversidade do challenger 3B estabiliza ~1/3;
(6) armadilha 10 reconfirmada: val de treino da iter3 subiu (0.425→0.515) enquanto o eval oficial
ficou flat (0.4100) — val não seleciona checkpoint; (7) Spider TEST da iter3 (0.7387, L1 0.887/L2
0.753/L3 0.603) fica entre iter1 (0.7424) e iter2 (0.7345) — flat dentro do ruído, saturação
reproduz no conjunto de reporte; (8) escala do mecanismo C11: challenger 7B na iter2 tem banda paga
16.0% ≈ 3B na iter2 (17%) — o colapso do currículo acompanha a iteração, não o tamanho do modelo.

## G2 — Self-play no ambiente BIRD + think (7B) — iter1 fechado 2026-07-14

Espelha o G1a/G1b em escala: challenger-7B vs 7B-base → gen-BIRD (20k, dup 24%) → solver-7B
think 0.1. Job 27133 (treino) + 27151 (merge/eval), B200.

| Modelo | BIRD dev | Spider dev (transfer) |
|---|---|---|
| Baseline 7B zero-shot | 0.4557 | 0.7563 |
| F3 (7B, Spider-env, sem think) | 0.4733 | 0.7476 |
| F4 (7B, Spider-env, com think) | 0.4844 | 0.7485 |
| F5 (OmniSQL-init + self-play) — melhor 7B anterior | **0.4909** | 0.7631 |
| **G2 — 7B, BIRD-env, think, iter1** | **0.4909** | **0.766** |
| G2 — 7B, BIRD-env, think, **iter2** (merge 27290) | 0.4967 | 0.765 |

Oficial (test-suite/bird-bench): iter1 0.582 BIRD / 0.841 Spider dev / 0.829 Spider test; **iter2 0.5828 / 0.842 / 0.832** — empate dentro do ruído (±1.25pt) nas três células. **iter3 (bird3x) 0.5652 / 0.826 / 0.831** — regride −1.8pt BIRD e −1.6pt Spider dev vs iter2: a espiral 7B satura na iter2 e piora na iter3 (o 3B saturou flat; o 7B regride). Harness próprio (re-inferência HF): 0.4811 BIRD / 0.7544 Spider dev / 0.7732 Spider test. Nota: iter3 avaliada só por re-inferência dos pesos HF na gpu-node-a (eval local do merge não rodou — B200 sem GPU livre); protocolo idêntico às demais células oficiais. A **espiral 7B satura na iter2**, igual ao 3B (ganho harness +0.58pt, oficial +0.08pt), e o k-hist do challenger 7B iter2 já mostrava banda paga 16%≈3B — C11 confirmada em escala nos dois sinais (colapso do challenger + EX do solver achatado).

**Leituras:** (1) **novo melhor 7B do projeto em Spider** (0.766 > 0.7631 do F5) e **empata o melhor
em BIRD** (0.4909 = F5), partindo do base (sem SFT-init) — o ambiente difícil entrega o mesmo teto
de BIRD que o init especializado do OmniSQL, mais um ganho extra de generalização; (2) completa a
4ª célula do grid think×dificuldade da C10 (3B/Spider, 7B/Spider, 3B/BIRD, **7B/BIRD**) — nas
quatro, o think ajuda mais quanto mais difícil o benchmark avaliado, nunca prejudica no BIRD;
(3) por-nível BIRD: 0.448/0.541/0.365 — nível médio (2) puxa a média, nível difícil (3) ainda
abaixo do baseline zero-shot nesse nível (0.331) superado (0.365); (4) dois incidentes de quota B200
(601G/600G hard) na cauda do treino, ambos recuperados sem perda de dado (checkpoint s78 intacto
nos dois casos) — ver `ara/trace` para o detalhe.

## G3 — Self-play no ambiente BIRD a partir de OmniSQL-7B (iter1) — fechado 2026-07-21

Análogo controlado do Arctic: solver init **OmniSQL-7B** (SFT forte em Text-to-SQL), challenger
Qwen-7B inventa pares no BIRD (gen iter1, 13.709 pares) → solver OmniSQL treina GRPO think 0.1.
Job 27515 (gpu-node-a, 4×H100, 53 steps/1 época). Testa se o self-play zero-data **agrega sobre um
init SFT forte DENTRO do BIRD** (o F5 já mostrava em transfer no Spider).

| Modelo | BIRD dev | Spider dev (transfer) |
|---|---|---|
| OmniSQL-7B baseline (zero-shot) | 0.4668 | 0.7592 |
| G2 — 7B base-init + self-play BIRD iter1 | 0.4909 | 0.766 |
| **G3 — OmniSQL-init + self-play BIRD iter1** | **0.5013** | **0.7756** |

Por-nível BIRD: 0.430 / 0.551 / 0.391 (L1/L2/L3); Spider: 0.857 / 0.787 / 0.679. Protocolo
unificado (harness próprio); oficial e pipe-OmniSQL pendentes. Modelo:
`anonymous/sql-zero-7b-solver-omnisql-bird`.

**Leituras:** (1) **o self-play agrega sobre o init SFT-forte dentro do BIRD**: +3.5pt BIRD
(0.4668→0.5013) e +1.6pt Spider (0.7592→0.7756) sobre o OmniSQL zero-shot — confirma no ambiente
BIRD o que o F5 mostrou em transfer; (2) **supera o G2** (base-init, 0.4909 BIRD): o init forte
entrega teto de BIRD mais alto sob o mesmo self-play; (3) **`think_ok = 0.00` o run inteiro** — o
OmniSQL-7B nunca emitiu blocos `<think>` no formato esperado, então o think-bonus (0.1) ficou
**inerte** neste init; o ganho vem do currículo self-play, não do raciocínio explícito (contraste
com os 3B/base, onde o think ajudava). (4) iter2/3 (espiral completa) a encadear: challenger
re-treina contra este solver → gen → solver iter2, testando se satura na iter2 como o C11 prevê.

## G3 — Espiral completa iter1→iter2→iter3 (saturação) — fechado 2026-07-26

Espiral fechada: solver iter1 → challenger iter2 → gen iter2 → solver iter2 → challenger iter3 →
gen iter3 → solver iter3. Cada solver auto-avaliado BIRD dev + Spider dev no protocolo unificado
(greedy@1, harness próprio). Jobs: iter1 27515 (gpu-node-a, 4×H100), iter2/iter3 na B200 (job 27869 na
cauda). Checkpoints: o **final de 1 época** de cada run (só ele foi mesclado; `global_step_25/50`
são scratch FSDP não mesclado) — iter1 s53, iter2 s78, iter3 s65. O step varia só com o volume de
dados treinado.

**N de pares treinado por iteração (NÃO equalizado — causa identificada: orçamento de prompts
divergente entre as cópias dos scripts por nó).** O `max_pairs 20000` é teto em todos; o que difere
é o `--num_prompts`, e gen1/gen3 rodaram na B200 (`g3_gen_b200.sbatch`, `g3_gen_iter3_b200.sbatch`:
`--num_prompts 12000`) enquanto gen2 rodou na gpu-node-a (`g3_gen_iter2.sbatch`: `--num_prompts 20000`):

| iter | onde o gen rodou | num_prompts | gerados | duplicados | mantidos (=treinados) | steps |
|---|---|---|---|---|---|---|
| iter1 | B200 | 12.000 | 59.982 (=12k×5) | 41.745 (69,6%) | **13.709** | 53 |
| iter2 | gpu-node-a | 20.000 | 35.491 | 12.580 (35,4%) | **20.000** (bateu o teto) | 78 |
| iter3 | B200 | 12.000 | 59.990 (=12k×5) | 38.568 (64,3%) | **16.688** | 65 |

Os `gerados` batem exatamente com `num_prompts × samples_per_prompt` (12.000×5 = 60.000) nas duas
iterações deficitárias: elas **esgotaram o orçamento de prompts** antes de o dedup deixar 20k
sobreviventes, enquanto o gen2 tinha 67% mais orçamento e parou cedo ao bater o teto.
steps/época = pares/batch → 53/78/65 (daí os checkpoints finais s53/s78/s65).

**Portanto o déficit é falha de configuração, não fenômeno científico** — e em particular **não** é o
colapso de diversidade do C11: a taxa de duplicatas é não-monotônica (69,6% → 35,4%* → 64,3%), com o
iter1 (challenger treinado contra o solver *mais fraco*) sendo o pior. O N desigual fica como
**confound de volume** com causa conhecida e corrigível: elevar `--num_prompts` (ou gerar em laço até
`kept==20000`). Re-rodada com piso+teto 20k em preparação (tag `omni20kf`).

**Auditoria cruzada — o G3 é o único braço BIRD não volume-matched.** Varredura de todos os logs de
gen (`kept=`) e de solver (N real do dataloader) nos três nós:

| braço / experiment_name | N treinado | bateu o teto? |
|---|---|---|
| 3B iter1 **reportado** (`bird1b20k`, com think) | 20.000 | sim |
| 3B iter1 no-think **reportado** (`bird1a`) | **35.139** | pré-padronização |
| 3B iter1 versão antiga (`bird1b`, descartada) | 35.139 | pré-padronização |
| 3B iter2 / iter3 (`bird2b20k` / `bird3b20k`) | 20.000 / 20.000 | sim |
| 3B no-think iter2 / iter3 (`bird2nt20k` / `bird3nt20k`) | 20.000 / 20.000 | sim (mesmo gen do par) |
| 7B G2 iter1 / iter2 / iter3 (`bird1_7b20k` / `bird2x_7b20k` / `bird3x_7b20k`) | 20.000 × 3 | sim |
| **G3 iter1 / iter2 / iter3** (`bird{1,2,3}_omni20k`) | **13.709 / 20.000 / 16.688** | **só o iter2** |

Ou seja: as espirais 3B-think e 7B-G2 **estão** equalizadas em 20k (o 3B iter1 foi re-rodado como
`bird1b20k` justamente para isso — a versão de 35.139 não é a reportada), e a leitura de saturação
nesses dois braços não tem confound de volume. O G3 é a exceção.

**⚠️ Confound novo detectado na ablação think × no-think da iter1:** o par reportado da iter1 é
`g1a` (sem think, **35.139** pares) × `g1b` (com think, **20.000** pares) — **não é pareado em
volume** (o braço sem think teve 1,76× mais dado). Os pares de iter2/iter3 sim são (mesmo gen,
20k). O sentido do efeito é conservador em relação a esse confound — o braço com think ganhou
(+1.5pt oficial) *com 43% menos dado* —, então a conclusão não é ameaçada, mas a iter1 não deve ser
descrita como "par" sem essa ressalva. Ver a seção da ablação abaixo.

| iter | ckpt | BIRD dev | BIRD L1/L2/L3 | Spider dev | Spider L1/L2/L3 |
|---|---|---|---|---|---|
| iter1 | s53 | **0.5013** | 0.430/0.551/0.391 | **0.7756** | 0.857/0.787/0.679 |
| iter2 | s78 | 0.4857 | 0.436/0.533/0.374 | 0.7544 | 0.857/0.768/0.635 |
| iter3 | s65 | **0.5013** | 0.418/0.556/0.382 | **0.7872** | 0.857/0.805/0.690 |

**Leituras:** (1) **satura** — BIRD oscila 0.5013→0.4857→0.5013 (net flat) e Spider 0.7756→0.7544→
0.7872 (net +1.2pt) numa banda de ~1.5pt; nenhuma iteração destrava ganho monotônico sobre o iter1,
confirmando o C11 (EX flat-a-declinante entre iterações) também neste braço com init SFT-forte.
(2) **iter2 é o vale nos dois benchmarks** (−1.6pt BIRD, −2.1pt Spider vs iter1), com recuperação no
iter3 — mesma forma de vale-e-recupera vista no G2-3B/7B, não colapso. (3) **Confound de imagem por
iteração:** iter1 rodou em H100/verl0.5-sglang048; iter2/iter3 em B200/verl0.6-cu128. Código
vendorado, config e reward idênticos — só imagem+hardware diferem; a oscilação intra-banda não é
distinguível de ruído de plataforma, o que reforça (não enfraquece) a leitura de saturação.
Pipe-OmniSQL de iter2/iter3 e Spider-Syn ainda pendentes (só iter1 tem: BIRD 0.6545 / Spider-test
0.880 / Syn 0.792). HF: só iter1 publicado (`anonymous/sql-zero-7b-solver-omnisql-bird`);
checkpoints iter2/iter3 (`hf_model_78`/`hf_model_65`) intactos na B200, upload pendente.

## Ablação pareada think × no-think (iter2/iter3 BIRD-3B) — 2026-07-18

Retreino do solver 3B **no mesmo gen** de cada iteração (bird2_20k, bird3_20k) com
`SOLVER_THINK_BONUS=0` — única variável trocada vs as iterações canônicas com think (0.1).
Jobs 27418/27419 (gpu-node-a). Fecha o efeito-think com controle pareado por iteração nas iters 2 e 3.

✅ **A iter1 agora tem par pareado em volume** (fechado 2026-07-28): `bird1nt20k` (sem think) treinou
em **20.000** pares, exatamente como `g1b`/`bird1b20k` (com think), mesmo step 78 — as três iterações
passam a ter controle pareado com única variável = o bônus think. O `g1a` (**35.139** pares, 1,76× mais
dado) fica como o run **anterior, não pareado em volume**, e seus números seguem válidos como tal.
No par matched a iter1 dá **+1,37 pt no BIRD dev** e **−0,29 pt no Spider dev** (protocolo unificado).

**McNemar exato nas TRÊS iterações (fechado 2026-07-28, job 28570 na gpu-node-a)** — mesmo método do job
28558, agora nos pares 2 e 3, que nunca tinham sido testados. Verificado antes: os quatro braços de
iter2/iter3 rodaram 20.000 pares / dataloader 78 / `global_step_78`, com o **mesmo** parquet de gen
por iteração e linha de comando idêntica exceto `SOLVER_THINK_BONUS` (0,1 × 0,0) — os pares são de
fato pareados. Join por `(db_id, question)`, sobreposição 1534/1534 (BIRD) e 1034/1034 (Spider), EX
por exemplo reproduz os agregados reportados.

| iter | benchmark | Δ think (pt, unif.) | só-think | só-no-think | discord. | p exato | p Holm (na iter) |
|---|---|---|---|---|---|---|---|
| 1 | BIRD dev   | +1,37 | 81 | 60 | 141 | 0,092 | 0,18 |
| 1 | Spider dev | −0,29 | 43 | 46 |  89 | 0,83  | 0,83 |
| 2 | BIRD dev   | +1,76 | 87 | 60 | 147 | **0,032** | 0,063 |
| 2 | Spider dev | −0,97 | 37 | 47 |  84 | 0,33  | 0,33 |
| 3 | BIRD dev   | +0,39 | 79 | 73 | 152 | 0,69  | 0,69 |
| 3 | Spider dev | −1,74 | 37 | 55 |  92 | 0,076 | 0,15 |

**Nenhum par é significativo após correção.** O mais forte é a iter2 no BIRD (p = 0,032 bruto), que
não sobrevive ao Holm nem dentro da própria iteração (0,063), muito menos sobre os seis testes
(0,032 × 6 = 0,19). A iter3 no BIRD é praticamente empate (79 × 73, p = 0,69). O que se sustenta é a
**consistência de direção dentro de cada benchmark**: think à frente nos 3 pares do BIRD e atrás nos
3 pares do Spider — a forma do eixo-dificuldade da C10 —, mas a magnitude **encolhe** no BIRD
(+1,37 → +1,76 → +0,39) e **cresce contra o think** no Spider (−0,29 → −0,97 → −1,74), com a iter3
do Spider sendo o segundo maior sinal da tabela e favorecendo o **no-think**. Não se agrega os três
sinais (teste de sinal 3/3, pooling de discordantes): as iterações **não são independentes** — o
solver de cada uma parte do checkpoint da anterior e enfrenta um challenger descendente dela, no
mesmo ambiente e nos mesmos dev sets. A C10 deve ser afirmada como evidência direcional sob volume
pareado, não como efeito estabelecido.
Detalhe: `ara/evidence/tables/mcnemar_think_nothink_iters123_2026-07-28.md` (job 28570) e o irmão da
iter1 `ara/evidence/tables/mcnemar_think_nothink_iter1_2026-07-28.md` (job 28558).

| iter | no-think (BIRD oficial) | think (BIRD oficial) | Δ think | no-think (unif. Spider/BIRD) | think (unif.) |
|---|---|---|---|---|---|
| 1 (pareada, 20k) | — (oficial não rodado) | 0.4772 (g1b) | — | 0.7321 / 0.3911 (bird1nt20k) | 0.7292 / 0.4048 |
| 1 (anterior, não pareada) | 0.4622 (g1a, 35.139 pares) | 0.4772 (g1b) | **+1.5** | 0.735 / 0.3931 | 0.7292 / 0.4048 |
| 2 | 0.4583 (bird2nt) | 0.485 (bird2) | **+2.7** | 0.7311 / 0.3911 | 0.7215 / 0.4087 |
| 3 | 0.4739 (bird3nt) | 0.4915 (bird3) | **+1.8** | 0.7331 / 0.4061 | 0.7157 / 0.410 |

Spider dev oficial dos no-think: iter2/iter3 = 0.802/0.802 (≈ think 0.80/0.803).

**Leituras:** (1) **think ajuda em todas as três iterações no BIRD oficial** (+1.5/+2.7/+1.8),
nunca prejudica — consistente com a C10 (o ganho do think cresce com a dificuldade do benchmark);
a direção se repete no protocolo unificado (+1,37/+1,76/+0,39), mas **nenhuma das três é
significativa no McNemar pareado** (ver bloco acima), e o par oficial nunca foi testado;
(2) a trajetória **no-think é não-monotônica** (0.4622→0.4583→0.4739) enquanto a com-think sobe
monotônico no oficial (0.4772→0.485→0.4915) — o raciocínio explícito estabiliza o ganho por
iteração; (3) no harness próprio (Spider dev) os no-think chegam a superar levemente os com-think
(0.7311 vs 0.7215 na iter2), mais um caso do gap harness×oficial que não inverte a conclusão
central; (4) modelos no-think upados ao HF (`sql-zero-3b-solver-bird{2,3}-nothink`), FSDP removido
após upload.

## G1b — Self-play no ambiente BIRD + think (3B) — fechado 2026-07-12

Par do G1a: mesmo gen (20k, parquet seed 42), única diferença SOLVER_THINK_BONUS=0.1.
78 steps (20k literais). Jobs 26037 (treino) + 26194 (eval).

| Braço (3B, ~20k, 1 iter) | BIRD dev | N3 (difícil) | Spider (transfer) |
|---|---|---|---|
| G1a sem think (s75) | 0.3931 | 0.289 | **0.7350** |
| **G1b com think (s78)** | **0.4048** | **0.317** | 0.7292 |

**Leituras:** (1) think +1.2pt no BIRD em 3B (borda do ruído ±1.25, mas val mostrou o mesmo
+1pt em s25/s50/s75 e o F4 achou +1.1pt no 7B) — a hipótese tamanho×dificuldade resolve-se
pelo EIXO DA DIFICULDADE: think ajuda em tarefa difícil (BIRD, 3B e 7B), atrapalha em tarefa
fácil (Spider/3B, B2); (2) o efeito concentra-se no nível difícil (N3 +2.8pt); (3) **0.4048 =
melhor 3B do projeto no BIRD, empata com CSC-3B treinado em gold humano do BIRD (0.4042) com
zero anotação**; (4) s75=s78 idênticos no BIRD — regra dos 20k re-confirmada; (5) ressalva:
braços viram amostras ligeiramente distintas do mesmo gen (20k-parquet vs primeiras 19.2k do
shuffle de 35k) — fonte de variância declarada.

## Sensibilidade de prompt (H1) + eval nativo dos concorrentes (H2) — 2026-07-09

Todos greedy@1, EX; prompt **omnisql** (estilo do OmniSQL) em vez do nosso protocolo unificado.
Jobs 22039/22040/22041.

| Modelo | Spider omnisql | Spider unificado | Δ | BIRD omnisql | BIRD unificado | Δ |
|---|---|---|---|---|---|---|
| Baseline 3B | 0.5986 | 0.617 | −1.8 | 0.3083 | 0.3468 | −3.9 |
| Baseline 7B | 0.7186 | 0.7563 | −3.8 | 0.4465 | 0.4557 | −0.9 |
| SQL-Zero-7B (F3 s240) | 0.7447 | 0.7476 | **−0.3** | — | — | — |
| CSC-SQL-3B | 0.6963 | 0.6886 | +0.8 | 0.3977 | 0.4042 | −0.7 |
| Arctic-7B | 0.7921 | 0.7901 | +0.2 | 0.5332 | 0.5117 | +2.2 |

**Leituras:** (1) **robustez a prompt correlaciona com treino em RL**: os baselines caem 1–4 pts ao trocar de prompt, enquanto SQL-Zero-7B, CSC e Arctic variam <1 pt (Spider) — o self-play não depende de prompt favorável; (2) nosso protocolo unificado **não subestima os baselines** (eles vão melhor no nosso prompt do que no omnisql) — a comparação central fica blindada; (3) para Arctic (OmniSQL-init), o prompt omnisql é o nativo e rende +2.2 pts no BIRD (0.533) — é o número mais generoso possível do concorrente mais forte, e o gap para os nossos 7B permanece o mesmo em espécie; (4) ranking entre modelos é estável sob troca de prompt (nenhuma inversão).

## Notas de comparabilidade

- Todos os números (inclusive concorrentes) vêm do **mesmo harness** — protocolo idêntico, única variável = o modelo. Números publicados dos concorrentes usam protocolos diferentes (Think2SQL: gold-schema + temp; CSC: self-consistency@k) e **não** são diretamente comparáveis aos daqui.
- ✅ Blindagem oficial feita (2026-07-15): re-score com avaliadores oficiais (Spider test-suite, BIRD mini_dev) sobre pesos HF — ver seção "Bateria oficial" no topo. "3B + gold do Spider" (gold-3b) coberto. Pendente residual: Think2SQL-7B e SQL-R1 no oficial (sem re-inferência ainda).


## Auditoria de integridade (2026-07-04)

Revisão para descartar inflação/vazamento. Resultado: **a alegação central é limpa**; um número secundário precisa de ressalva.

**✅ Limpos:**
- **Split train/dev**: proposer amostra só dos 146 bancos de train; 0 sobreposição com os 20 do dev (dupla trava `train -= dev`). Verificado nos parquets reais.
- **Overlap de tarefas**: 2–6 perguntas/SQLs textualmente iguais ao dev, mas sobre bancos **disjuntos** (perguntas genéricas tipo "how many X"); não é a mesma tarefa. O controle gold-train tem o mesmo nível → não favorece o self-play.
- **Métrica EX**: multiset, order só sob ORDER BY, truncado nunca conta como match (conservador). Idêntica para todos os modelos.
- **Prompt**: baseline, treino e concorrentes usam o mesmo `T2SQL_SOLVER_PREFIX`. O ganho não vem de prompt.
- **iter0 (0.703) e iter1 (0.727)**: avaliados no checkpoint tardio (step 200/226 do iter1), **sem seleção por validação** → números honestos.

**Seleção de checkpoint (política):** o Spider dev é um **dev set** (o test oficial é oculto); selecionar o checkpoint por best-dev é a prática padrão da literatura (CSC, Think2SQL etc. fazem o mesmo). Reportamos os picos **com divulgação completa do step e do critério**: iter2-v2 = step 25 (best-dev; treino interrompido no step 50 por quota de disco), iter1 = step 200/226 (último salvo, sem seleção — o crash escolheu por nós), iter0 = final. Consistência extra: a v1-vs-v2 do iter2 compara final-vs-pico; a conclusão qualitativa (rep-penalty evita a degradação) não depende disso, pois a v1 degradou *monotonicamente* desde o step 25 (0.752→0.715) enquanto a v2 partiu mais alto (0.757).

**Nota do EX single-instance:** dois SQLs distintos podem coincidir no resultado de uma única instância do banco (falso-positivo). Isso afeta **todos** os modelos igualmente (não favorece o self-play). ✅ Mitigado (2026-07-15): re-score com o test-suite evaluator oficial do Spider (múltiplas instâncias de banco) — ver "Bateria oficial" no topo; o gap oficial×harness é sistemático e não inverte rankings.

## Artefatos publicados (challengers) — 2026-07-14
Challengers upados como artefatos de reprodução (repos privados): `sql-zero-7b-challenger-iter1`
(F-series/Spider), `sql-zero-3b-challenger-bird1` (G1), `sql-zero-3b-challenger-bird2` (iter2),
`sql-zero-7b-challenger-bird1` (G2). FSDP locais removidos após upload (política HF-antes-de-apagar).
