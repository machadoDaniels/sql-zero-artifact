# SQL-Zero — anonymized artifact

Code, configurations and prompts for *SQL-Zero: Self-Evolving Text-to-SQL*.
Anonymized for double-blind review: author names, institution, cluster paths,
hostnames and model-hub accounts have been removed.

**Start with [`MANIFEST.md`](MANIFEST.md)** — it maps every claim, table row and
pointer in the paper to the file here that backs it.

## Layout

| Path | Contents |
|---|---|
| `t2sql_solver.sh`, `t2sql_challenger.sh` | the two GRPO training entry points |
| `t2sql_data.py`, `t2sql_gen_data.sh` | prompt construction and challenger generation |
| `t2sql_baseline.sh`, `scripts/eval_spider.py` | evaluation harness |
| `verl/custom_reward/t2sql_reward.py` | solver and challenger rewards |
| `search/sql_utils.py` | SQL validity, determinism filter, template masking |
| `config/` | trainer configuration |
| `bird/` | the training-database gate (64 of BIRD-train's 69) |
| `scripts/` | one launcher per reported run, plus the analysis scripts |
| `tests/` | unit tests for the rewards, the masking and the evaluators |

`verl/` is a vendored fork of the verl trainer; only `verl/custom_reward/` is
specific to this work.

## Reproducing a spiral iteration

Each iteration runs challenger training, generation with execution filtering and
template deduplication, then solver training, and each role starts from the base
checkpoint. `MANIFEST.md` names the launcher for every reported row.

## Data

BIRD and Spider are not redistributed here; `scripts/download.py` and
`scripts/download_spider.py` fetch them from their original sources. Only BIRD
*training* databases are used for training.
