# Can Large Language Models Program Traffic Signal Controllers in Weight Space?

Code, data, trained networks and complete agent records for the paper of the same title.
Repository: https://github.com/kkatzi/llm-weight-space-traffic-signal-control

**Authors.** Konstantinos Katzilieris, Emmanouil Kampitakis and Eleni I. Vlahogianni, Department of Transportation
Planning and Engineering, National Technical University of Athens.

The study asks whether an LLM coding agent (Claude Opus 5.5, in Claude Code) can construct, edit, transfer, explain
and supervise the weights of a PPO traffic-signal controller by reasoning alone, with no gradient fitting and no LLM
at run time. Everything runs on a single signalised junction in SUMO, **L2**, where converged PPO beats tuned actuated
control by 3.3 % and tuned fixed-time control by 7.3 % in total time spent. A wider variant, **L2W**, is the transfer
target. Five uses are studied, each against non-LLM baselines:
- Initialise;
- Patch, on converged networks and after a detector failure;
- Transfer;
- Explain;
- Constrain.

Every number in the paper is computed by the scripts here from the stored evaluations. Every agent run is archived
with its task folder, prompt, transcript and deliverables.

## Quick start

```bash
conda env create -f environment.yml && conda activate wsp-traffic
python -m pytest tests/ -q          # about 30 s on 8+ cores: nets, routes, sample episodes, the main analysis
                                    # (works on temporary copies; nothing in the repository is written)
python audit/run_audit.py           # the full reproducibility audit, about 25 min on 40 cores
```

The latest audit report is [audit/AUDIT_REPORT.md](audit/AUDIT_REPORT.md). Run every command from the repository root.
The figures use the TeX Gyre Heros font when TeX Live provides it, and a default font otherwise.

## What the audit checks

`audit/run_audit.py` copies the repository to a temporary folder elsewhere on disk and regenerates everything it can
there. It then compares the result with the files shipped here.

| Stage | What is regenerated and compared |
|---|---|
| A. Static | All modules compile and import from the copy alone. No code refers to the original machine. Every policy recorded in a result resolves to a shipped file. Every agent's evaluation log equals its out-of-folder copy. Every launch prompt equals the archived one. The transcript audits are consistent. |
| B. Inputs | The SUMO networks rebuild identically. The route files of all 222 study episodes regenerate with the hashes of the files used in the study ([docs/route_hashes.json](docs/route_hashes.json)). |
| C. Episodes | Every stored test evaluation, including the 60-min-cap re-scores and the fixed-time cost sets, re-runs from scratch. TTS, delay, unserved and served vehicles must be identical in every episode. |
| D. Analysis | `paper_numbers*.json`, `secondary.json`, `RESULTS*.md`, `SECONDARY.md` and every paper table regenerate identically. The figures regenerate with identical text content. |
| E. Training | Six PPO runs are retrained for 100k steps: from scratch, warm start, transfer warm start, fine-tuning, under the rules, and under the fault. Each must reproduce the shipped checkpoint parameter for parameter, and its training episodes. |
| F. Baselines | These must regenerate identical outputs: the tuning of fixed-time and actuated control on L2, one (1+1)-ES run per problem, mean imputation, positional padding, the CART trees, and DAgger on Net 0. |
| G. Agents | Every agent task folder rebuilds identically from the generator. Every deliverable re-scores identically: edit size, trainability, comparisons, fidelity and predictions. |

## Layout

| Path | Contents |
|---|---|
| `wsp/` | The pipeline, as a Python package; each module documents its command line (`python -m wsp.<module>`). |
| `nets/` | The SUMO networks of L2 and L2W (`python -m wsp.junctions` rebuilds them). |
| `models/<problem>/<run>/` | Every PPO run (35): `final.zip`, the checkpoints, `monitor.csv` (training episodes) and `curve_val.json` (validation curve). Also `models/*/baselines/`: ES networks, mean imputation, padding, and the CART and DAgger rule files. |
| `results/eval/<problem>/<label>/<set>.json` | Every evaluation, episode by episode: 419 files, 291 controllers. |
| `results/` | Also `tuned/` (selected reference controllers), `baselines/` (ES, DAgger and wrapper logs), `fault*.json` (the failed-detector screen), `paper_numbers*.json`, `RESULTS*.md`, `secondary.json` and `SECONDARY.md`. |
| `packs/<case>/<id>/` | The 54 analysed agent runs. Each holds the task folder exactly as the agent received it (`TASK.md`, `tools.py`, `spec.json`, `states.npz`, starting weights, `eval.py` in closed-loop arms), plus `work/` (the agent's scripts, notes, predictions and deliverables), `eval_log.jsonl` (its simulator calls), `scored.json` (our scoring) and `provenance/` (launch prompt, transcript, audit). |
| `runs/agent_launch/` | Launch records: the exact prompt, model, times, exit code and session id of every run, and the launch plans. |
| `runs/agent_eval_log/` | The out-of-folder copies of the agents' simulator logs. |
| `runs/agent_runs.json` | One audit row per run. |
| `runs/eval/**/record.npz` | Recorded observations that analyses and task folders read. |
| `runs/baselines/` | ES candidates and DAgger iterations. |
| `docs/EXPERIMENT_PLAN.md` | The study protocol, pre-registered analysis plan, amendments and recorded outcomes. |
| `docs/PROVENANCE.md` | Launch protocol, audit and adjudication of flagged calls. |
| `paper/figs/`, `paper/tables/` | The paper's figures and tables, as generated by `python -m wsp.figures`. |
| `queue/jobs.jsonl` | Every job that produced the stored models and results, with its exact command. |
| `queue/state.json` | The jobs' final states. |
| `scripts/` | The queue runner and the pack-scoring helper. |
| `audit/` | The audit, its latest report and `MANIFEST.sha256` (hash of every file). |

## Reproducing the study from scratch

Commands as run in the study; `queue/jobs.jsonl` has every one with its arguments. `scripts/run_queue.sh` starts the
core-budgeted runner (`python -m wsp.queue`); start it inside the environment of `environment.yml`, because jobs run
with the interpreter that runs the queue. `python -m wsp.status` reports progress. Times are for 40 cores.

| Step | Command | Time |
|---|---|---|
| Networks | `python -m wsp.junctions` | seconds |
| Reference controllers | `python -m wsp.tune --problem L2` (likewise `L2W`, `L2_rules`; `L2_fault` reuses L2's tuning) | minutes each |
| PPO | `python -m wsp.train --problem L2 --name ppo_s100 --seed 100 --steps 3000000` (and the other runs in `queue/jobs.jsonl`) | about 40 min per 3M-step run on 4 cores |
| Curves | `python -m wsp.curves --run models/L2/ppo_s100 --workers 4` | minutes per run |
| Evaluation | `python -m wsp.evaluate --problem L2 --policy sb3:models/L2/ppo_s100/final.zip --set test --label ppo_s100` | — |
| Baselines | `python -m wsp.baselines {es, es_fault, meanimp, wrapper, pad, cart, dagger} --targets 0,1,2` | — |
| Agent task folders | `python -m wsp.packs --case patch_fault --run 0 --budget 10 [--rep 2]` | — |
| Launch | `python -m wsp.launch --plan runs/agent_launch/plan_fault_rep.json` | — |
| Score | `python -m wsp.agents_eval packs/<case>/<id>` | — |
| Archive | `python -m wsp.archive headless` (needs the Claude Code session store of the launching machine) | — |
| Analysis | `python -m wsp.analysis [--evaluate] [--cap 3600]`, `python -m wsp.secondary [--evaluate]`, `python -m wsp.figures` | minutes |

`wsp.launch` runs each agent as an independent non-interactive Claude Code session (`claude -p`), started inside its
task folder with an explicit model id and only the file and shell tools. The `claude` executable is found on `PATH`,
or set `CLAUDE_CLI`. The launcher never relaunches a recorded run: to run agents anew, build fresh task folders into
another folder (`WSP_PACK_ROOT=<dir> python -m wsp.packs ...`) and launch with the same `WSP_PACK_ROOT` and a new
record folder (`WSP_LAUNCH_LOG=<dir>`), so that the shipped task folders and records stay untouched.

Simulation and training are deterministic: the same command reproduces the same episodes and the same network
parameters (audit stages C and E).

## Records and paths

The stored records are kept as written during the study: evaluation files, agent logs, launch records and transcripts
(with the account e-mail address and account/organisation identifiers in the transcripts replaced by `[redacted ...]`
markers). They contain absolute paths of the machine the study ran on. `wsp/paths.py` maps those paths to this
repository, so cached evaluations are reused wherever the repository lives. A task folder's `eval.py` still points to
the original pipeline: rebuild the folder with `wsp.packs` to run an agent task anew.

**Outside git**
- About 1,530 intermediate PPO checkpoints (about 318 MB) are excluded from git by `.gitignore` and are meant to be
  distributed as a separate data archive.
- Sixteen checkpoints stay in git: the 300k checkpoints of the fine-tuning runs (stored evaluations use them) and the
  seven the audit needs (the 100k checkpoints of the six retraining checks and the 600k checkpoint that the
  Initialise task files quote). The audit therefore also runs from a plain git checkout.

## Licence and citation

- **Code** (`wsp/`, `tests/`, `audit/*.py`, `scripts/`): [MIT License](LICENSE). Every source file carries an SPDX
  header. The exceptions are `wsp/pack_tools.py` and `wsp/pack_eval.py`, which are copied verbatim into the agents'
  task folders and must stay byte-identical to what the agents received.
- **Data, models and records** (everything else): [Creative Commons Attribution 4.0](LICENSE-DATA).
- **Third-party content.** The agent transcripts also contain text written by Anthropic and third-party skill/MCP
  authors: the Claude Code system prompt, tool and skill descriptions, and harness messages. That text is not covered
  by these licences.
- **Dependencies.** SUMO and the other dependencies are not distributed here; they keep their own licences.

If you use this repository, please cite the paper and the repository; [CITATION.cff](CITATION.cff) has the metadata.
GitHub shows it as "Cite this repository".

```bibtex
@misc{katzilieris2026wsp,
  title  = {Can Large Language Models Program Traffic Signal Controllers in Weight Space?},
  author = {Katzilieris, Konstantinos and Kampitakis, Emmanouil and Vlahogianni, Eleni I.},
  year   = {2026},
  note   = {Code and data: https://github.com/kkatzi/llm-weight-space-traffic-signal-control}
}
```

The venue and the DOI will be added when available, in CITATION.cff as well.
