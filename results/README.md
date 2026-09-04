# Per-query predictions

Raw model output for every reported checkpoint, one entry per evaluation question, gzipped.
These are what the paired tests consume and what makes the reported accuracies auditable
without re-running inference: `scripts/mcnemar_chained.sbatch` reads the files in this
directory directly.

## `pipe_omnisql/` — the Table 1 protocol

Greedy decoding through the OmniSQL inference pipeline, the protocol behind every number in
Table 1. Each file is a JSON list with one object per question: `input_seq` (the prompt as the
model saw it), `output_seq` / `responses` (the raw generation) and `pred_sqls` (the extracted
SQL).

| File prefix | Checkpoint |
|---|---|
| `bird2c_*` | SQL-Zero-3B iter2 |
| `bird3c_*` | SQL-Zero-3B iter3 |
| `bird2xc_*` | SQL-Zero-7B iter2 |
| `bird3xc_*` | SQL-Zero-7B iter3 |

Suffixes name the benchmark: `_bird_greedy` (BIRD dev), `_spidertest_greedy` (Spider test),
`_syn_greedy` (Spider-Syn).

## Top level — the official-evaluator protocol

Output of `t2sql_baseline.sh` under the unified schema prompt (full DDL + 3 sample rows),
scored with the official BIRD evaluator. Each file carries the exact invocation in `args`, the
aggregate in `summary` and the per-question records in `results`. Named
`<tag>_s<step>_<benchmark>.json.gz`, where the step is the solver update the checkpoint was
taken at.

The two protocols are kept separate on purpose and their numbers are not interchangeable.
