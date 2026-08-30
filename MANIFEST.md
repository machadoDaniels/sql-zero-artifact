# Artifact manifest — SQL-Zero

Maps every pointer in the paper to the file in this repository that backs it.
Paths are relative to the repository root.

## Method (Section 3)

| Paper element | File |
|---|---|
| Solver reward `r_sol`, three tiers (Eq. 1) | `verl/custom_reward/t2sql_reward.py` |
| Reasoning term, `think_bonus` | `t2sql_solver.sh` (`SOLVER_THINK_BONUS`), `verl/custom_reward/t2sql_reward.py` |
| Challenger reward `r_ch`: format, difficulty, repetition (Eqs. 3–5) | `verl/custom_reward/t2sql_reward.py` |
| Masked-SQL-template clustering (repetition penalty and dedup) | `search/sql_utils.py` (`sql_template`), `tests/test_sql_utils.py` |
| Difficulty term `d(k)`, `n = 5` frozen-solver samples | `verl/custom_reward/t2sql_reward.py`, `scripts/score_gen_k.py` |
| Complexity-grouped advantage baseline (HRPO-lite) | `t2sql_challenger.sh`, `config/t2sql_grpo.yaml` |
| Schema representation (full DDL + 3 sample rows) | `t2sql_data.py` |
| Execution oracle and result-set comparison | `verl/custom_reward/t2sql_reward.py`, `tests/test_sql_exec_reward.py` |

## Training budget and hyperparameters (Section 4)

| Paper element | File |
|---|---|
| Optimizer settings, group sizes, temperatures, token budgets | `config/t2sql_grpo.yaml`, `t2sql_solver.sh`, `t2sql_challenger.sh` |
| Solver: batch 256, group 8, one epoch → 78 updates on 20,000 pairs | `t2sql_solver.sh` |
| Challenger: batch 256, `rollout.n = 1`, `reward_rollout_n = 5` | `t2sql_challenger.sh` |
| Database gate — 64 of BIRD-train's 69 databases | `bird/train_dbs_keep.txt`, `bird/census_train.csv`, `bird/census_recheck_2026-07-28.csv`, `scripts/census_recheck.py` |

## Reported runs

Every reported arm uses `SOLVER_THINK_BONUS=0.1`. The `*nt*` runs set it to 0 and are the
ablation of Appendix B, not the reported spiral.

| Table 1 row | Launcher |
|---|---|
| SQL-Zero-3B iter1 | `scripts/g1_solver.sbatch` |
| SQL-Zero-3B iter2 | `scripts/bird2_3b_solver_eval.sbatch` |
| SQL-Zero-3B iter3 | `scripts/bird3_3b_solver_eval.sbatch` |
| SQL-Zero-7B iter1 / iter2 / iter3 | `scripts/bird1_7b_solver_b200.sbatch`, `scripts/bird2x_7b_solver_eval_b200.sbatch`, `scripts/bird3x_7b_solver_eval_b200.sbatch` |
| Gold-BIRD-control-3B | `scripts/gold_bird_3b_solver.sbatch` |
| Challenger of each iteration | `scripts/bird2_3b_challenger.sbatch`, `scripts/bird3_3b_challenger.sbatch` (the `SOLVER` variable is the frozen difficulty oracle, i.e. the previous iteration's solver) |
| Appendix A, OmniSQL-initialized arm | `scripts/g3_solver_iter{1,2,3}*.sbatch` |
| Appendix B, `think_bonus = 0` ablation | `scripts/birdnt_iter{1,2,3}*_3b_solver.sbatch`, `scripts/nothink_pipe_infer.sbatch`, `scripts/nothink_pipe_score.sbatch` |

Both roles are initialized from the base checkpoint in every iteration; no solver or
challenger is warm-started from the previous turn. The scripts named `iter1_solver.sh`,
`iter2_solver.sh`, `iter3_solver.sh` and `iter*_challenger.sh` at the repository root are
**vendored upstream Dr. Zero code for the retrieval/QA task**, not the Text-to-SQL pipeline,
and are not used to produce any number in this paper.

## Analysis and statistics (Section 5, Appendix B)

| Paper element | File |
|---|---|
| McNemar exact paired test against the gold control | `scripts/mcnemar_paired.py` |
| McNemar for the `think_bonus` ablation | `scripts/mcnemar_think_nothink.py` |
| Coverage / template-region diagnostic | `scripts/coverage_descriptive.py` |
| `k`-histograms and the hard-yet-solvable band | `scripts/score_gen_k.py`; `k_hist` logging in `verl/custom_reward/t2sql_reward.py` |
| Training-dynamics histories, including interrupted runs | `scripts/export_wandb_training_dynamics.py`, `scripts/plot_wandb_training_dynamics.py` |
| Evaluation harness and the two protocols | `t2sql_baseline.sh`, `scripts/eval_spider.py` |

## Known gaps

- The W&B run histories themselves are not committed here; the export and plotting
  scripts are, and require a `WANDB_API_KEY` for the project.
- Model checkpoints are omitted to preserve anonymity.
- Per-run result tables and internal notes are omitted; the paper reports the
  numbers they contain.
