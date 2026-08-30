# SQL-Zero — Tabela-mestre de experimentos

Status ao vivo. **EX** = execution accuracy no nosso harness (greedy@1, schema completo DDL+3 rows, prompt idêntico para todos). Spider dev = 1.034 ex.; BIRD dev = 1.534 ex. (transferência — nunca visto em treino). Por-nível = (fácil/médio/complexo). Números finais canônicos em `RESULTADOS.md`.

Convenções: **challenger** = proposer treinado por RL para gerar pares (pergunta, SQL gold) difíceis-mas-solúveis; **solver** = modelo T2SQL treinado por GRPO com reward de execução 3-tier (1.0 match / 0.1 executa-errado / 0.0 inválido); **genN** = dataset gerado na iteração N, validado por execução e deduplicado por template.

---

## Trilha A — A espiral principal (3B, zero anotação)

### A0 — Baselines zero-shot
**O quê:** Qwen2.5-Coder-3B/7B-Instruct sem nenhum treino, avaliados no protocolo unificado. **Pergunta:** ponto de partida que qualquer treino precisa superar; réguas de tamanho.
**Resultado:** 3B **0.617** / 0.347 · 7B **0.756** / 0.456.

### A1 — iter0: primeira volta, proposer é o próprio base (sem RL no proposer)
**O quê:** o 3B base gera pares via prompt (SQL-first→NL, condicionado a nível de complexidade); pares validados por execução (single SELECT, determinístico, resultado não-vazio, sem vazamento de SQL na pergunta) e deduplicados; solver 3B treina do base por GRPO.
**Pergunta:** dados auto-gerados sem nenhum treino do proposer já ensinam algo? Quanto importa a escala do dataset?
**Config:** gen0 com yield ~10% (validade 41% do base); braço 2k pares vs braço 10k pares; 1 epoch.
**Resultado:** 2k → 0.645/0.344 (+2.8 Spider, BIRD parado); 10k → **0.703/0.372** (+8.6/+2.5).
**Conclusão:** funciona e **escala importa** — 5× mais dados quintuplicou o ganho; a transferência (BIRD) só emerge com volume.

### A2 — iter1: entra o challenger (primeira volta completa da espiral)
**O quê:** challenger 3B treinado por RL contra o solver0 congelado (reward = 0.5·formato + dificuldade, onde dificuldade = (n−k)/(n−1) com pico em k=1 acerto do solver em n=5 tentativas — "difícil-mas-solúvel"); depois gen1 (58k pares, yield 10%→58%, validade 41%→83%); solver1 treina continual do solver0.
**Pergunta:** a pressão adversarial do challenger melhora os dados e o solver além do iter0?
**Resultado:** **0.727** (0.813/0.762/0.584) / **0.395** — melhor 3B da nossa tabela; supera o CSC-SQL-3B treinado com gold humano (0.689). Challenger: formato 0.56→0.93, dificuldade 0.028→0.11.
**Alerta observado:** duplicatas no gen1 subiram para 24% (0.8% no gen0) — primeiro sinal de colapso de modo do challenger; o dedup offline segurou.

### A3-v1 — iter2 sem controle de diversidade (resultado negativo, diagnóstico)
**O quê:** challenger2 (2 epochs, SEM repetition penalty) vs solver1; gen2; solver2 continual.
**Resultado:** challenger2 ficou excelente em validade (92%) mas **colapsou em diversidade: 66% de templates duplicados** → gen2 com só 25k pares únicos; solver2 **regrediu monotonicamente** na validação (0.752@25 → 0.715@97) e fechou em **0.686** — overfit no nível 3 (0.591, o maior de todos os runs) à custa de esquecer N1/N2.
**Conclusão:** a espiral tem um modo de falha próprio — **colapso de diversidade do challenger** — que aparece quando ele é treinado sem penalidade de repetição. Duas voltas de RL no proposer sem esse controle destroem a cobertura do dataset.

### A3-v2 — iter2 com repetition penalty (a correção, novo melhor)
**O quê:** mesma receita do A3-v1 com 3 correções: (1) **repetition penalty no reward do challenger** (estilo R-Zero: clustering por template SQL mascarado dentro do batch, penalidade λ·|cluster|/batch, λ=0.5); (2) challenger 1 epoch; (3) save_freq=25.
**Resultado:** duplicatas 66%→57%, gen2v2 com 36k pares (+43%); solver **0.729/0.400** — recupera e supera o iter1. Re-treino completo de 139 steps (iter2v2-full) reproduziu: **0.728** estável nos checkpoints 50 e 100 → o número não é pico de sorte. Canônico: step 100 (best-dev), 0.728/0.376.
**Conclusão (par com A3-v1):** o par regressão→recuperação estabelece o **papel causal** da repetition penalty: a espiral continua além da 1ª volta *se e somente se* a diversidade do challenger for controlada.
**Achado extra do re-treino:** a transferência decai com treino prolongado (BIRD 0.400@s25 → 0.366@s139) — treinar demais in-domain erode generalização; early stopping preserva.

---

## Trilha B — Ablação de raciocínio explícito (think) no 3B

**Contexto:** o Qwen2.5-Coder não tem think nativo; o solver emergente **podou** o `<think>` sozinho (respostas 160→34 tokens) porque o reward de execução não paga por ele. A literatura R1-style assume que think é necessário. Testamos.

### B1 — Emergente (controle) = A1b/A2
Solver treina sem nenhum incentivo a raciocinar; o think colapsa e o modelo responde só `<sql>...</sql>`. **0.703** (iter0) / **0.727** (iter1).

### B2 — Think via reward de formato (pareado perfeito com A1b)
**O quê:** solver do **base** no **mesmo gen0-10k** do A1b, única variável: reward = 0.9·execução + 0.1·think_ok (think_ok = `<think>` não-vazio antes do SQL, com anti-hacking de tamanho mínimo).
**Resultado:** o mecanismo funciona (think_ok 0.93→1.00 — 100% das respostas raciocinam), mas EX **0.694** (0.825/0.728/0.511) vs 0.703 do controle — −0.9pt agregado, **−2.9pt no nível complexo** (onde a hipótese previa ganho), e **2.4×** o custo em tokens.
**Conclusão (portão B):** em T2SQL single-shot com 3B em dificuldade nível Spider, CoT explícito **não paga o custo**. Config sem-think adotada para a espiral. Ressalva honesta: é UM ponto da matriz (modelo pequeno, tarefa média) — ver Trilha F para o teste em 7B.

### B2-falho — Lição: formato não ressuscita think colapsado
**O quê:** primeira tentativa do B2 partiu do checkpoint iter0scale (think já colapsado). think_ok ficou **0.00 em 4096 amostras** — P(⟨think⟩)≈0 significa zero exploração, logo zero gradiente para o componente de formato.
**Conclusão:** reward de formato **reforça** raciocínio existente mas **não ressuscita** um colapsado; prevenção (treinar com o reward desde o base) funciona, cura (re-elicitar depois) não. Publicável como nota de treino.

### B3 — Sem think em nada (challenger incluso) — 🔲 backlog, baixa prioridade.

---

## Trilha C — Currículo e dados — 🔲 backlog
C1: mesmos pares do solver, ordenados fácil→difícil vs aleatório (hipótese do usuário; `data.shuffle=False` torna trivial). C2: mistura de níveis 3:2:1 vs uniforme. C3: challenger treinado vs solver-base (fidelidade ao Dr. Zero original).

**C4 — Curva de escala de dados via TREINO ÚNICO (desenho final do usuário, 2026-07-08):**
com 1 epoch e LR CONSTANTE pós-warmup (verificado: default `warmup_style=constant` do verl),
steps ≡ dados vistos e checkpoints intermediários ≈ treinos dedicados menores. Desenho: UM
solver do base no gen1 completo (58k, 228 steps), SAVE_FREQ=25, eval greedy OFICIAL de cada
checkpoint (s25..s228 → N = 6.4k..58k) → curva EX×N com ~9 pontos por 1 treino (~8h vs ~16h
do desenho de subsamples, e mais pontos). O ponto s39 (~10k) desconfunde qualidade×volume vs
iter0-10k. Saída: N* de saturação (teto de dificuldade do challenger da geração). Roda na
B200 pós-G1b. Extensão: mesma leitura já sai DE GRAÇA do G1a (BIRD, vals a cada 25).
[Subsamples 5-40k criados antes do redesenho ficam como backup.]

**✅ FECHADO 2026-07-10** (treino 21991→22022→22060, evals 25107; 10 pontos oficiais em
RESULTADOS.md §C4): Spider pica em **19.2k** (0.7263 ≈ iter1@58k) e degrada na cauda
(0.696 final, −3pt); BIRD-transfer platô 0.396-0.403 só a partir de 25.6k (pico 32k).
Consequências operacionais (decisões do usuário): **orçamento padrão = 20k amostras
pós-dedup/iteração** (gens futuros com `--max_pairs 20000`; solvers ~78 steps) e
**orçamento condicional para ambiente difícil** (G1a estendido a s100 p/ calibrar o do BIRD).

---

## Trilha D — Comparabilidade externa (protocolo unificado)

**Motivação:** números publicados dos concorrentes usam protocolos incomparáveis (gold schema linking, self-consistency@k, test sets diferentes). Baixamos os modelos e re-avaliamos TODOS no nosso harness — única variável restante: o modelo.

| ID | Modelo | O que representa | Spider | BIRD | Status |
|----|--------|------------------|--------|------|--------|
| D1 | CSC-SQL-3B | concorrente do NOSSO tamanho, GRPO + gold humano do BIRD | 0.689 | 0.404 | ✅ |
| D2 | Think2SQL-7B | teto "com recursos": 7B + gold + destilação R1 (publicado ~0.82 c/ oráculo de schema → 0.765 sem) | 0.765 | 0.505 | ✅ |
| D3 | Gold-train | **controle interno**: NOSSO modelo, NOSSO GRPO, NOSSO protocolo, única mudança = treinar nos 8.659 pares humanos do Spider train em vez dos auto-gerados. Mede o "preço da anotação" isolando uma variável | 0.731 | 0.357 | ✅ |
| D4 | SQL-Zero SC@4 | paridade de test-time com CSC | | | 🔲 |
| D5 | Re-score com avaliadores oficiais (Spider test-suite, BIRD) — blindagem contra "seu EX difere" | | | 🔲 CPU |

**Leitura do D3 (resultado central do paper):** self-play (0.729) **empata** o gold humano (0.731) no Spider e o **vence em transferência** (BIRD 0.400 vs 0.357). A anotação compra um pouco de complexidade in-distribution (N3 0.624 vs 0.584) e perde generalização — os 58k pares auto-gerados cobrem mais espaço que os 8.7k humanos.

---

## Trilha E — Robustez e generalização

### E1 — Spider-Syn (perturbação de sinônimos)
**O quê:** as mesmas 1.034 perguntas do dev reescritas à mão substituindo palavras que coincidem com nomes de tabela/coluna por sinônimos ("singers"→"vocalists") — quebra o casamento léxico; mede se o modelo faz schema linking semântico ou decorou correspondência de strings.
**Resultado:**

| Modelo | Spider dev | Spider-Syn | Queda |
|--------|-----------|-----------|-------|
| Baseline 3B | 0.617 | 0.513 | −10.4 |
| Gold-train | 0.731 | 0.622 | **−10.9** |
| **SQL-Zero (s100)** | 0.728 | **0.663** | **−6.5** |

**Conclusão:** o gold degrada na mesma taxa do baseline (anotação humana não compra robustez); só o self-play reduz a queda (+4.1 sobre o gold). Com o BIRD, são **3 eixos independentes** onde self-play ≥ gold: in-distribution (empate), bancos novos (+1.9 a +4.3), linguagem nova (+4.1).

### E2 — Spider-Realistic — 🔲 (download oficial quebrado; fonte alternativa pendente)

---

## Trilha F — Escala 7B e conhecimento prévio (EM CURSO)

**Contexto:** a espiral funcionou no 3B. O 7B parte de 0.756 (perto do teto in-distribution) — testa se o método sobrevive a pouco headroom, e se as conclusões do 3B (think não paga) generalizam. Hipótese em teste (do usuário): o valor do reasoning cresce com tamanho/dificuldade; todos os trabalhos R1-style garantem thinking por *alguma* via (format reward no SQL-R1, init CoT-SFT no Arctic) — o único braço "sem nada" da literatura é o nosso.

### F1 — iter0-7B ingênuo (resultado negativo, diagnóstico) ❌
**O quê:** réplica exata do A1b em 7B: o 7B base gera 54k pares (gen0-7B, 7.8% dup — base é diverso), solver-7B treina neles.
**Resultado:** validação **degrada monotonicamente** (0.769@25 → 0.713@50 → 0.711@75; EX estimado ~0.68 < baseline 0.756). Cancelado no step 75.
**Diagnóstico:** os dados do próprio 7B são **fáceis demais para ele** — grupos GRPO com 8/8 acertos têm vantagem zero; o gradiente restante é ruído destrutivo. O 3B tinha headroom sobre os próprios dados; o 7B não.
**Conclusão:** "iter0 ingênuo falha em 7B" ≠ "self-play falha em 7B". O mecanismo de dificuldade do challenger fica **mais** necessário com escala, não menos.

### F2 — challenger-7B (a correção do F1) ✅
**O quê:** challenger-7B treinado contra o 7B base congelado, **com repetition penalty λ=0.5 desde o início** (lição do A3 aplicada preventivamente) e servidor do solver em GPU dedicada (7B: servidor + rollout + FSDP não cabem juntos em 80GB — ver DECISOES §8.x).
**Resultado:** difficulty **0.24** (2× o pico do challenger-3B — modelo maior propõe difícil-mas-válido com muito mais eficiência); gen1-7B: **62k pares, 19% dup, 19% inválidos** — o melhor dataset da série (grande, difícil E diverso). A penalty preventiva funcionou: 19% dup vs 66% (3B sem penalty) e 57% (3B com penalty tardia).

### F3 — solver-7B sem-think (controle da matriz) 🔄
**O quê:** solver do 7B base nos 62k do gen1-7B, config vencedora do 3B (sem think).
**Parcial:** step 233/240; a degradação do F1 sumiu (platô 0.765–0.775 por 170 steps) com **subida tardia a 0.780** nos steps finais. Eval oficial (Spider/BIRD) ao terminar — veredito vs baseline 0.756/0.456.

### F4 — solver-7B + think (braço B) ✅
**O quê:** idêntico ao F3, única variável `SOLVER_THINK_BONUS=0.1` — o pareado do B2, agora em 7B. **Pergunta:** reasoning paga em modelo maior? (Se sim, o achado do B2 é sobre *tamanho*, não sobre T2SQL.)
**Resultado:** treino completo (240 steps, val final 0.796, best val 0.803@175-200). Eval triplo oficial (s75/s200/s240 × Spider/BIRD): s75 0.7495/0.4844, s200 0.7515/0.4798, s240 0.7485/0.4844 — curva quase flat entre checkpoints. Pareado a F3 no mesmo step (s240): think abre +1.1pt BIRD / +0.1pt Spider sobre sem-think — **o oposto do B2 no 3B** (think piorou lá). Veredito: o achado do B2 era sobre tamanho de modelo, não sobre T2SQL em geral (ver `docs/RESULTADOS.md` §F4).

### F5 — OmniSQL-7B + SQL-Zero (braço C) 🔄 na B200
**O quê:** solver inicializado do **OmniSQL-7B** (SFT em 2.5M pares sintéticos com CoT — a receita de init do Arctic; thinking "de fábrica", sem format reward) treinado nos mesmos 62k do gen1-7B. **Pergunta:** self-play **empilha** sobre conhecimento internalizado por SFT? (Se sim, o método é complementar à supervisão, não só substituto — e o problema de headroom se resolve por cima.)
**Nota:** primeiro treino na B200 (infra destravada — 5 correções em DECISOES §8.x); ~250s/step (1.9× mais rápida que a H100 no 7B).

**Matriz de leitura (quando F3/F4/F5 fecharem):** F3 vs F4 responde "think × escala"; F3 vs F5 responde "base × SFT-init"; F5 vs Arctic publicado contextualiza "quanto do Arctic é o init".

## Trilha G — Ambiente BIRD (hipótese: o teto era do ambiente, não do método) — PLANEJADO

**Motivação (usuário, 2026-07-07):** todos os métodos convergem a ~0.73-0.77 no Spider (teto do
ambiente); a variância está no BIRD (0.35-0.50, headroom real). Nenhum trabalho treina self-play nos
bancos do BIRD — todos usam os pares gold. A série G treina a espiral nos 69 bancos do BIRD train
(SEM os pares), avaliando em BIRD dev + Spider dev/test (transfer reverso).

**Protocolo padrão (decisão do usuário): challenger-first** (fiel ao Dr. Zero; robusto aos dois
modos de falha — fácil demais [7B/Spider] e difícil demais [risco BIRD]; resolve o yield do gen).
Rep-penalty λ=0.5 desde o início em todo challenger. Adaptações de engenharia declaradas:
max_prompt_length=8192, sample_rows=2, bancos filtrados em duas etapas (69→64 de BIRD-train):
pré-filtro pelo tamanho do `.sqlite` (~≥4 GB descartado: 4 bancos) + censo de schema (works_cycles,
66 tabelas, fora; ~≤6k tokens em sample_rows=2). Nenhum dos dois limiares ficou registrado
exatamente — ver `ara/evidence/tables/bird_db_census_2026-07-28.md`.
/dev/shm p/ SQLite, timeout no loop, cache do result-set do gold no parquet (vantagem estrutural
do self-play: o gold é executado na geração; o reward só executa a predição).

| ID | Braço | Init solver | Think | Status |
|----|-------|-------------|-------|--------|
| G0 | Censo dos 69 bancos do BIRD train (schema tokens, tamanho, latência) — go/no-go + figura | — | — | 🔄 download train.zip |
| G1a | challenger-3B vs 3B-base → gen-BIRD → solver-3B | 3B base | sem | ✅ 0.3931/0.735 |
| G1b | = G1a, solver com think reward 0.1 (mesmo gen-BIRD → pareado perfeito) | 3B base | com | ✅ 0.4048 (think +1.2) |
| G2 | challenger-7B vs 7B-base → gen-BIRD → solver-7B | 7B base | **só com** (decisão do usuário) | 🔄 solver na fila B200 (ch+gen ✅) |
| G3 | challenger vs OmniSQL-7B → gen-BIRD → solver-OmniSQL | OmniSQL-7B | think 0.1 | 🔄 RUNNING (2026-07-15): ch 27311 → gen 27312 → solver 27313. Bug do hold corrigido (reward do challenger extraía só `<sql>`; OmniSQL emite ```sql → k=0; trocado p/ `extract_sql` format-agnóstico). k-hist válida (banda paga ~25%, > base-init). |

Nota: G1a/G1b repetem o pareado B2 no ambiente BIRD — testa se "think não paga" se inverte quando a
tarefa é difícil (a hipótese tamanho×dificuldade, agora no eixo dificuldade). G1b reusa o gen-BIRD do
G1a (challenger é compartilhado; só o solver difere) — custo marginal de 1 treino de solver.

Matriz completa resultante: {Spider, BIRD} × {self-play, gold} × {3B, 7B, OmniSQL-init} — sem
análogo na literatura. Comparação-alvo: G-series vs CSC/Think2SQL/Arctic (todos gold-BIRD-trained)
no BIRD dev, agora in-domain dos dois lados.

## Backlog: re-rodar TODOS os experimentos no padrão de facto da área (decisão do usuário, 2026-07-08)
Protocolo unificado (nosso template) mantido em TODAS as baterias até o fim das conclusões.
DEPOIS, se possível, refazer os experimentos adotando o padrão dos trabalhos comparados:
1. **Template OmniSQL** (verificado verbatim em OmniSQL/Arctic/CSC — o de facto standard):
   persona + Database Engine + instruções + saída em bloco ```sql (manter nossa extração
   format-agnóstica); nossa instrução de minimalidade já é equivalente à deles.
2. **Value retrieval** (OmniSQL/Arctic `build_contents_index.py`): índice BM25/FTS sobre os
   valores das células por banco; na inferência, injetar na serialização os valores mais
   similares à pergunta (sem oráculo; automático; ataca o modo de falha de literais no BIRD).
3. Reavaliar tudo (nossos + concorrentes) sob esse padrão → tabela "de facto protocol".
Medições que dimensionam o fator (✅ concluídas 2026-07-09, jobs 22039/22040/22041 — os originais
21978/21979/21982 morreram na crise de quota de 2026-07-08):
- **Prompt-sensitivity (H1)**: baselines caem com o template omnisql (3B 0.599/0.308; 7B 0.719/0.447)
  vs unificado (0.617/0.347; 0.756/0.456) → nosso protocolo NÃO subestima os baselines. SQL-Zero-7B
  quase indiferente (0.7447 vs 0.7476, −0.3pt) → ganho do self-play não é artefato de prompt.
- **Eval nativo CSC/Arctic (H2)**: CSC-3B 0.696/0.398 (≈ unificado); Arctic-7B 0.792/0.533
  (+2.2pt BIRD no prompt nativo dele). Ranking estável sob troca de prompt — nenhuma inversão.
Detalhe em RESULTADOS.md §"Sensibilidade de prompt".
