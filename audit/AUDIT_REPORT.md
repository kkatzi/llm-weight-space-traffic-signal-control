# Reproducibility audit

Run 2026-11-09 20:15 on Supercomputer; stages A,B,C,D,E,F,G; 659 s. Software: Python 3.11.15, numpy 2.4.4, scipy 1.16.2, torch 2.6.0+cu124, gymnasium 1.2.3, stable-baselines3 2.8.0, matplotlib 3.10.7, libsumo 1.24.0.

**37 of 37 checks passed.**

## A. Static checks

| | Check | Detail |
|---|---|---|
| PASS | all Python files compile |  |
| PASS | all 23 wsp modules import from the copy alone (no PYTHONPATH) |  |
| PASS | no code refers to a path on the machine the study ran on (except the recorded-path map in wsp/paths.py) |  |
| PASS | every recorded policy file resolves to a shipped file (394 references) |  |
| PASS | every agent evaluation log equals its out-of-folder copy (30 logs) |  |
| PASS | every launch prompt equals the prompt archived with its run (54 runs) |  |
| PASS | archived transcript audits: right model, exact prompt, no messages after launch, simulator calls within budget and matching the mirror log (54 runs) |  |

## B. Simulation inputs

| | Check | Detail |
|---|---|---|
| PASS | the SUMO networks of L2 and L2W rebuild identically (10 files; net.net.xml compared after netconvert's header comment, which records the build date and file paths) |  |
| PASS | route files of all 222 study episodes (test, dev, val, recorded, cost; L2 and L2W) regenerate with the hashes of the files used in the study |  |

## C. Episodes

| | Check | Detail |
|---|---|---|
| PASS | 287 stored evaluations (8670 episodes) re-run from scratch with identical TTS, delay, unserved and served vehicles (tolerance 1e-9; 261 s) |  |

## D. Analysis, tables and figures

| | Check | Detail |
|---|---|---|
| PASS | `python -m wsp.analysis` runs |  |
| PASS | `python -m wsp.analysis --cap 3600` runs |  |
| PASS | `python -m wsp.secondary` runs |  |
| PASS | `python -m wsp.figures` runs |  |
| PASS | results/paper_numbers.json regenerates identically |  |
| PASS | results/paper_numbers_cap3600.json regenerates identically |  |
| PASS | results/secondary.json regenerates identically |  |
| PASS | results/RESULTS.md regenerates identically |  |
| PASS | results/RESULTS_cap3600.md regenerates identically |  |
| PASS | results/SECONDARY.md regenerates identically |  |
| PASS | all 16 paper tables regenerate identically |  |
| PASS | all 10 figures regenerate (identical text content) |  |

## E. Training

| | Check | Detail |
|---|---|---|
| PASS | L2/ppo_s100: 100k-step retraining reproduces the shipped checkpoint parameter for parameter (12 tensors) and the first 130 training episodes' returns and lengths |  |
| PASS | L2/init_r1_s400: 100k-step retraining reproduces the shipped checkpoint parameter for parameter (12 tensors) and the first 134 training episodes' returns and lengths |  |
| PASS | L2W/transfer_net0_s500: 100k-step retraining reproduces the shipped checkpoint parameter for parameter (12 tensors) and the first 132 training episodes' returns and lengths |  |
| PASS | L2/ft_net0_s600: 100k-step retraining reproduces the shipped checkpoint parameter for parameter (12 tensors) and the first 133 training episodes' returns and lengths |  |
| PASS | L2_rules/ppo_s300: 100k-step retraining reproduces the shipped checkpoint parameter for parameter (12 tensors) and the first 132 training episodes' returns and lengths |  |
| PASS | L2_fault/ft_net0_s800: 100k-step retraining reproduces the shipped checkpoint parameter for parameter (12 tensors) and the first 132 training episodes' returns and lengths |  |

## F. Baselines and tuning

| | Check | Detail |
|---|---|---|
| PASS | tuning of fixed-time (270 plans) and actuated control (180 settings) on L2 reproduces results/tuned/L2.json |  |
| PASS | (1+1)-ES on L2 (Net 0, sigma0 0.3, seed 0): all 10 development evaluations, the accepted steps and the submitted network reproduce |  |
| PASS | (1+1)-ES on L2_fault (Net 0, sigma0 0.3, seed 0): all 10 development evaluations, the accepted steps and the submitted network reproduce |  |
| PASS | baseline `meanimp` regenerates its 3 outputs identically |  |
| PASS | baseline `pad` regenerates its 3 outputs identically |  |
| PASS | baseline `cart` regenerates its 9 outputs identically |  |
| PASS | DAgger distillation of Net 0 (3 depths x 6 iterations with simulator rollouts) regenerates the same selected rule files |  |

## G. Agent task folders and scoring

| | Check | Detail |
|---|---|---|
| PASS | all 54 agent task folders rebuild identically from the generator (303 files; pack ids, the Python interpreter path and the evaluator's pipeline path excepted) |  |
| PASS | all 54 agent deliverables re-score identically (edit size, cap, trainability, comparisons, fidelity, predictions) |  |
