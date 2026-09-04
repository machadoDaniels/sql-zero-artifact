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
| Challenger-generated training sets, 20,000 pairs per reported run | `data/` (see `data/README.md`) |
| Database gate — 64 of BIRD-train's 69 databases | `bird/train_dbs_keep.txt`, `bird/census_train.csv`, `bird/census_recheck_2026-07-28.csv`, `scripts/census_recheck.py` |

## Reported runs

Every reported arm uses `SOLVER_THINK_BONUS=0.1`. The `*nt*` runs set it to 0 and are the
ablation of Appendix B, not the reported spiral.

| Table 1 row | Launcher |
|---|---|
| SQL-Zero-3B iter1 | `scripts/g1_solver.sbatch` (challenger: `scripts/g1a_challenger.sbatch`) |
| SQL-Zero-3B iter2 and iter3 | `scripts/chain/submit_chain.sh 3b` |
| SQL-Zero-7B iter1 | `scripts/bird1_7b_solver_b200.sbatch` (challenger: `scripts/g2_challenger_b200.sbatch`) |
| SQL-Zero-7B iter2 and iter3 | `scripts/chain/submit_chain.sh 7b` |
| Gold-BIRD-control-3B | `scripts/gold_bird_3b_solver.sbatch` |
| Re-scoring of every Table 1 row under the OmniSQL pipeline | `scripts/chain/submit_pipe_eval.sh`, which chains `scripts/nothink_pipe_infer.sbatch` and `scripts/nothink_pipe_score.sbatch` |
| Appendix A, OmniSQL-initialized arm (iteration 1 only) | `scripts/g3_solver_iter1_b200_2gpu.sbatch` |
| Appendix B, `think_bonus = 0` ablation | `scripts/birdnt_iter{1,2,3}*_3b_solver.sbatch`, `scripts/nothink_pipe_infer.sbatch`, `scripts/nothink_pipe_score.sbatch` |

From iteration 2 on, **both roles inherit the weights of the previous turn**: the
challenger of iteration N starts from the challenger of iteration N-1 (`CH_INIT`) and the
solver of iteration N from the solver of iteration N-1 (`SOLVER_INIT`), which is what
`scripts/chain/submit_chain.sh` orchestrates. The previous solver additionally serves as the
frozen difficulty oracle, as it does in iteration 1. Neither loss carries a KL term, so this
weight initialization is equivalent to `resume_from_path` in the original Dr. Zero.

`scripts/bird{2,3}_3b_solver_eval.sbatch`, `scripts/bird{2,3}x_7b_solver_eval_b200.sbatch` and
`scripts/bird{2,3}_3b_challenger.sbatch` are an earlier, **cold-start** variant in which every
iteration restarted from the base checkpoint. They are kept for completeness and back no number
in the paper. The scripts named `iter1_solver.sh`,
`iter2_solver.sh`, `iter3_solver.sh` and `iter*_challenger.sh` at the repository root are
**vendored upstream Dr. Zero code for the retrieval/QA task**, not the Text-to-SQL pipeline,
and are not used to produce any number in this paper.

## Analysis and statistics (Section 5, Appendix B)

| Paper element | File |
|---|---|
| McNemar exact paired test against the gold control | `scripts/mcnemar_paired.py`, driven by `scripts/mcnemar_chained.sbatch` for the reported (chained) solvers |
| McNemar for the `think_bonus` ablation | `scripts/mcnemar_think_nothink.py` |
| Coverage / template-region diagnostic | `scripts/coverage_descriptive.py` |
| `k`-histograms and the hard-yet-solvable band | `scripts/score_gen_k.py`; `k_hist` logging in `verl/custom_reward/t2sql_reward.py` |
| Training-dynamics histories, including interrupted runs | `scripts/export_wandb_training_dynamics.py`, `scripts/plot_wandb_training_dynamics.py` |
| Evaluation harness and the two protocols | `t2sql_baseline.sh`, `scripts/eval_spider.py` |
| Per-query predictions behind every reported accuracy | `results/` (see `results/README.md`) |

## Known gaps

- The W&B run histories themselves are not committed here; the export and plotting
  scripts are, and require a `WANDB_API_KEY` for the project.
- Model checkpoints are omitted: a single 3B merge is ~6.8 GB and a 7B one ~15 GB, well past
  what this repository can host, and they cannot be released anonymously. They are on the model
  hub and the de-anonymized version links them directly.
- Internal working notes are omitted. The per-query predictions are not: `results/`
  ships the raw output of every reported checkpoint under both protocols, so the
  accuracies and the paired tests can be recomputed without running inference.
