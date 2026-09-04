# Challenger-generated training data

Each file holds the execution-filtered, template-deduplicated pairs that one challenger
produced and that the corresponding solver was trained on. The generation budget is the same
everywhere (12,000 prompts x 5 samples, capped at 20,000 retained pairs); every run reaches the
cap except SQL-Zero-3B iter3, which exhausted the budget at 7,084 pairs after template
deduplication removed 86% of its candidates. These are the exact
inputs behind the reported rows; no human NL--SQL annotation is used anywhere in them.

| File | Used by |
|---|---|
| `t2sql_zero_bird.parquet` | challenger prompts (schema + target complexity level) |
| `t2sql_solver_bird1_20k.parquet` | SQL-Zero-3B iter1 |
| `t2sql_solver_bird1_7b_20k.parquet` | SQL-Zero-7B iter1 |
| `t2sql_solver_bird2c_20k.parquet` | SQL-Zero-3B iter2 |
| `t2sql_solver_bird3c_20k.parquet` | SQL-Zero-3B iter3 |
| `t2sql_solver_bird2xc_7b_20k.parquet` | SQL-Zero-7B iter2 |
| `t2sql_solver_bird3xc_7b_20k.parquet` | SQL-Zero-7B iter3 |

Columns follow the verl convention: `prompt` is the chat-formatted question with the full
schema, `reward_model.ground_truth` is the challenger's candidate gold SQL, and `extra_info`
carries the database id and the complexity level. Databases come from BIRD-train only.

The files named without the `c` (`t2sql_solver_bird2_20k.parquet`,
`t2sql_solver_bird3_20k.parquet`, `t2sql_solver_bird2x_7b_20k.parquet`,
`t2sql_solver_bird3x_7b_20k.parquet`) come from the earlier cold-start variant and back no
number in the paper.
