# Study protocol and analysis plan

Specification of the experimental design, agent-run protocol, pre-registered hypotheses, and the three study
amendments (convergence retrain of L2W seed 203, Amendment P-F for failed-detector Patch, and Amendment P-F-R
for failed-detector replicate sessions), together with the recorded hypothesis outcomes.

## Context

The study evaluates all five weight-space use cases on junction **L2** (and transfer target **L2W**).
- **The junction.** Four approaches, each with a short (about 3-car) left-turn pocket (`wsp/junctions.py`).
- **Why L2.** On it, PPO beats tuned fixed-time and tuned actuated control.
- **Network.** The PPO policies use 46 inputs and a 46-64-64-4 tanh actor with 7,428 parameters.
- **Training.** The reward is the per-second increment of total delay. Training episodes drain like evaluation
  episodes (`wsp/env.py`).

## Common rules

**Episode sets.** All episode seeds are disjoint across development, validation, recorded states, training-cost reference, and held-out test sets:

| set | episodes | used for |
|---|---|---|
| `dev` | low / nominal / high × seeds 1101–1103 (9) | baseline tuning; the agents' `eval.py`; wrapper and ES evaluations |
| `val` | low / nominal / high × seeds 1301–1302 (6) | learning curves, convergence, ES step-size choice, Explain held-out states |
| `rec` | training family × seeds 1201–1206 (6) | recorded states in task packs; DAgger rollouts |
| `cost` | training family × seeds 1501–1560 (60) | the reference controllers' training-cost distribution |
| `test` | low / nominal / high × seeds 19001–19010 (30) | final scoring only |

**PPO seeds.**

| run family | seeds |
|---|---|
| L2 from scratch | 100–104 (Nets 0–4; the agent targets are Nets 0–2) |
| L2W from scratch | 200–204 |
| L2 + rules from scratch | 300–302 |
| Initialise warm starts | 400–401 |
| Transfer warm starts | 500–501 |
| Patch fine-tuning | 600–602 |
| Constrain fine-tuning | 700–702 |

**KPI.**
- **Primary: total time in system (TTS, veh·h).** Every second, count vehicles in the network plus vehicles
  waiting to enter. Sum from t = 0 until the network is empty after 1 h of arrivals, with a cap of 30 min after
  arrivals end. Teleporting is disabled.
- **Secondary: mean delay per vehicle** (time loss plus departure delay, from SUMO's statistics).
- **Unserved vehicles:** those still present at the cap.

**Comparisons.**
- **Paired by test episode.** We report:
  - the mean difference in TTS;
  - the change in mean TTS relative to the reference, in %;
  - a two-sided 95 % t-interval (29 degrees of freedom);
  - the number of the 30 episodes on which the first policy is better.
- **Several targets or seeds.** Each is reported. Pooled statistics average the per-episode differences over
  targets before the interval is computed.
- **Independent families of runs** (e.g. PPO from different initialisations): Welch intervals over per-run means.
- **"Significant"** means the 95 % interval excludes 0.

**Training.**
- Final checkpoint always; no checkpoint selection.
- Convergence criterion (6 `val` episodes, at every checkpoint): the mean validation TTS over the last 20 % of
  steps must be no more than 2 % below the mean over the preceding 20 %. Otherwise the run is retrained with 1.6×
  the steps, repeatedly.
- No network becomes an agent target before it meets the criterion.

**Reporting.**
- Every run of every condition is reported.
- An agent run that breaks the protocol (detected by the transcript audit) is archived with the reason and rerun
  with an identical prompt; it never enters an analysis.

**Baselines in the paper:** tuned fixed-time and tuned actuated control. No other rule-based controller is
evaluated.

## R. Reference controllers

- **Fixed-time.** Cycle 0→1→2→3. Greens (NS through, NS left, EW through, EW left) ∈ {10, 15, 20, 25, 30} ×
  {10, 15, 20} × {10, 15, 20, 25, 30, 40} × {10, 15, 20} s. Chosen by mean `dev` TTS.
- **Actuated.**
  - After the minimum green, the controller extends the green while at least `ext` vehicles move on its lanes
    and the green is shorter than `max_green`. Otherwise it switches to the phase with the most halting vehicles.
  - Grid: `ext` ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 16} × `max_green` ∈ {10, 15, 20, 25, 30, 40, 50, 60, 90} s ×
    detector layout ∈ {stop-line lanes; stop-line plus upstream lanes}.
  - Chosen by mean `dev` TTS. The same grids are used on L2W and, under the rules, on L2 + rules.
- **PPO.** L2: 5 seeds × 3M steps. L2W: 5 seeds × 3M.
- **H-R1.** On L2, PPO has lower test TTS than tuned actuated control: the pooled paired difference over the 5
  seeds is < 0 with a CI excluding 0. The same comparison is reported against fixed-time, and on L2W
  (secondary).

## I. Initialise (write a network from scratch)

- **Runs.** 3 blind runs (Claude Opus 5.5), no simulator. Each pack starts from a different random initialisation
  (seeds 1–3). The deliverable is a 46-64-64-4 network meeting the trainability requirements:
  - median greedy probability ≤ 0.95;
  - share of states with greedy probability > 0.99 ≤ 0.30;
  - |logit| ≤ 15.

  Reference values from the L2 PPO network after 600k steps and at convergence
  are given in the task file.
- **Follow-on.** Each network is scored zero-shot on test, then used as the starting actor for PPO: 3M steps,
  seeds 400–401, critic warm-up 50k steps with the actor frozen.
- **I1, zero-shot.** For each network, paired test TTS against fixed-time and against actuated control.
  - **H-I1.** At least 2 of 3 agent networks have lower test TTS than fixed-time (CI excluding 0).
- **I2, cost of learning.** Total delay per training episode (veh·h), from each run's training log. Two windows:
  the first 50 episodes, and the first 1M steps. Warm starts (6 runs) are compared with PPO from random
  initialisation (the 5 L2 reference runs), with a Welch interval.
  - **H-I2.** Warm starts incur less delay in the first 50 episodes.
  - Also reported: episodes above the 95th percentile of fixed-time's delay on the `cost` episodes.
- **I3, final level (descriptive).** Test TTS at 3M for warm starts against PPO from random initialisation.

## P. Patch (edit a trained network)

- **Targets.** L2 Nets 0–2. At most **371 of 7,428 parameters (5 %)** may differ from the target. No training
  afterwards.
- **Arms.**
  - Budget B = 0 (blind), B = 5 and B = 10 evaluations through `eval.py`; each evaluation covers the 9 `dev`
    episodes.
  - One run per target and arm: 9 runs by Claude Opus 5.5.
  - Packs are identical across arms except the budget lines.
- **Pre-specified non-LLM baselines.**
  - **(1+1)-ES with the agents' budget.**
    - Target parameters: the output layer (W3, b3; 260 parameters, within the cap).
    - Moves: additive Gaussian steps of σ · std(W3); the 1/5 success rule (×1.5 after a success,
      ×1.5^(−1/4) after a failure).
    - Budget: 10 `dev` evaluations, all spent on candidates.
    - Submission: the best network evaluated, which is the target itself if nothing improves.
    - Step size: σ0 ∈ {0.03, 0.1, 0.3} × 2 ES seeds. The σ0 with the best mean `val` TTS is selected, and both
      of its seeds are reported.
  - **PPO fine-tuning.** Each target continues training for 1M steps on the training family (actor and critic;
    seeds 600–602). Reported at 300k and 1M steps, with the simulation each used.
- **Primary outcome.** Change in mean test TTS against the target, per target and pooled.
- **Hypotheses.**
  - **H-P1.** With B = 10, the pooled change is < 0 (CI excluding 0), and at least 2 of 3 targets improve
    significantly.
  - **H-P2 (descriptive).** The pooled change is ordered by budget: B = 0 ≥ B = 5 ≥ B = 10. All three arms and
    every run are reported, and no trend is claimed from overlapping intervals.
  - **H-P3.** With B = 10, patches beat the selected ES and 1M fine-tuning: pooled paired difference < 0, CI
    excluding 0.
- **Exploratory, post hoc, labelled as such.** An execution-time wrapper that implements each closed-loop agent's
  stated rule outside the weights, with the same 10-evaluation `dev` budget for its thresholds. This tests whether
  writing the rule into the weights adds value.
- **Secondary.**
  - Parameters changed, by layer and unit.
  - Rounds used.
  - Unserved vehicles.
  - Truncation of the B = 10 runs: the test TTS of the best-so-far candidate after k rounds.

## T. Transfer (to a junction with a different input size)

- **Design.** Sources: L2 Nets 0–2. Target: **L2W**, with 54 inputs. The agent designs W1ᴮ (64 × 54) and b1ᴮ;
  W2, b2, W3 and b3 must equal the source's. There are 3 blind runs (Opus 5.5), no simulator.
- **Baselines.**
  - Positional padding: copy the 46 source columns by index and set the remaining 8 to zero.
  - PPO from scratch on L2W (5 seeds × 3M).
- **T1, zero-shot.** Paired test TTS on L2W against fixed-time, actuated control and padding; Welch against
  scratch PPO.
  - **H-T1.** At least 2 of 3 transferred networks beat fixed-time and padding (CI excluding 0).
- **T2, warm start.** Each transferred network starts PPO on L2W (3M steps, seeds 500–501, critic warm-up 50k).
  Training-episode delay (first 50 episodes; first 1M steps) and final test TTS are compared with scratch PPO.
  - **H-T2.** Warm starts incur less delay in the first 50 episodes (Welch CI excluding 0).

## E. Explain (network to readable rules)

- **Design.** Targets: Nets 0–2; 3 runs (Opus 5.5), no simulator.
- **Deliverables.**
  - `rules_small.py`, with at most 8 comparisons;
  - `rules_full.py`;
  - `EXPLANATION.md`, `NOTES.md` and `predictions.json`.
- **Metrics.**
  - Fidelity: agreement with the network's greedy phase on held-out states (the network's own trajectories on the
    6 `val` episodes), counting only steps where the minimum green has elapsed.
  - Closed-loop test TTS relative to the network (paired).
  - Size: comparisons and numeric constants.
- **Baselines.**
  - CART trees (Gini; depth 3, 5 and 8; min leaf 10), fitted on the pack's recorded states labelled with the
    network's greedy phase.
  - DAgger-style iterative distillation: 5 iterations, each rolling out the current tree on the 6 `rec` episodes
    and refitting. The iteration is selected by its own rollout TTS.
- **H-E1 (descriptive, per target).** The agents' rule sets reach higher fidelity than trees with at least as many
  comparisons, and smaller closed-loop TTS changes.

## C. Constrain (operating rules)

- **Environment.** L2 + rules, with the shield (`wsp/env.py`). Targets: Nets 0–2.
- **Design.**
  - Blind arm and closed-loop arm (B = 10 `dev` evaluations under the rules); 3 runs each (Opus 5.5).
  - Deliverable: `supervisor.py` with `decide(x, logits, state) -> phase`. It may use the network only through
    its logits.
- **Baselines.**
  - Naive shield (network plus shield).
  - Masked logits (best legal phase).
  - Actuated and fixed-time control, tuned under the rules.
  - PPO fine-tuned under the rules: 1M steps from each target, seeds 700–702.
  - PPO trained from scratch under the rules: seeds 300–302, 3M steps.
- **Primary outcome.** Change in test TTS against the naive shield, per target and pooled.
- **Hypotheses.**
  - **H-C1.** Closed-loop supervisors reduce test TTS against the naive shield (pooled CI excluding 0).
  - **H-C2 (descriptive).** Supervisors compared with fine-tuned and from-scratch PPO under the rules.

## K. Calibration

Each agent's `predictions.json` gives test TTS per scenario (and fidelity for Explain). We report the signed
error against the measured value, per arm. Descriptive.

## Agent-run protocol

- **Launch.**
  - Every run is an independent headless Claude Code session (`claude -p` with the approved launch prompt, an explicit
    model id, and tools Bash, Read, Write, Edit, Glob and Grep). Each session starts with its pack folder as the
    working directory (`~/wsp_v3_agent_runs/`, outside the repository, so there is no parent session, no memory and no
    git context).
  - Launch prompts and task files are fixed before the first run (`wsp/prompts.py`, `wsp/packs.py`).
  - Arms differ only in budget lines.
- **Audit.** Every transcript is archived and screened for:
  - paths outside the pack;
  - writes outside `work/`;
  - tools other than file and shell tools;
  - simulator use outside `eval.py`;
  - fitting or optimisation calls;
  - messages received after launch.
- **Model.** The model identifier is verified from each transcript.

## Amendments (logged as made)

- **2026-11-08 16:41 EET, convergence rule (as pre-specified, recorded here).** L2W PPO seed 203 failed the criterion at 3M steps
  (−5.0 %). It is retrained with 4.8M steps (`ppo_s203_r1`), and the 3M run is not used.
- **2026-11-08 18:28 EET, P-F: Patch under a failed detector (added after the converged-target Patch result; before any of the
  runs below).**
  - **Why.** On the converged targets the pre-registered H-P1 is not supported. No method moves them, neither the
    agents nor ES nor fine-tuning: a converged network leaves little to repair. The owners therefore add a Patch study on
    a network with a specific, realistic defect.
  - **Scenario.** A converged network (Nets 0-2) is deployed. One lane detector then fails: both of its readings
    (queue and density) stay at 0. The agent is told which detector failed.
  - **Selection of the failed detector (rule fixed now).** Among the 20 lane detectors (12 stop-line, 8 upstream),
    choose the one whose failure gives the largest increase in mean `val` TTS, averaged over Nets 0-2 (the worst single
    fault). If no single failure costs at least 3 %, the owners are consulted before any agent run.
  - **Environment.** Problem `L2_fault`: L2 with that detector's two inputs set to 0. The actuated controller loses the
    same detector (it reads 0 halting and 0 vehicles on that lane). Fixed time is unaffected.
  - **Arms.** Budgets 0, 5 and 10 `dev` evaluations under the fault; one run per target; 9 runs by Claude Opus 5.5.
    Same generator, launch prompt and protocol (independent headless sessions) as Patch. The task file adds a
    description of the fault, and its reference table shows the network with and without the fault, fixed time, and
    actuated with the same failed detector.
  - **Baselines.**
    - Mean imputation: b1 += W1[:, failed] x mean of those inputs over the target's recorded `rec` states. This is
      the standard fix for a missing input.
    - (1+1)-ES with the agents' budget, as in P, under the fault.
    - PPO fine-tuning under the fault: 1M steps from each target, seeds 800-802, reported at 300k and 1M.
  - **Outcomes.**
    - Paired test change against the faulty network (per target and pooled).
    - Recovered share = (faulty − patched) / (faulty − healthy), per target, from the mean test TTS.
  - **Hypotheses.**
    - **H-F1.** With 10 calls, the pooled change is < 0 (CI excluding 0), and at least 2 of 3 targets improve
      significantly.
    - **H-F2.** With 10 calls, the patches beat mean imputation and ES (pooled CI excluding 0).
    - **H-F3 (descriptive).** Ordered by budget.

## Outcomes against the hypotheses (recorded 2026-11-08 19:20 EET, after every run had been scored)

Source: `results/paper_numbers.json` (`python -m wsp.analysis`). Each entry gives the change in mean test TTS with
its 95 % interval. A hypothesis counts as supported only when every pre-specified condition holds.

- **H-R1: supported.** PPO vs actuated on L2 −3.3 % [−5.2, −1.5]; all 5 seeds significant (−2.9 % to −3.9 %).
  - vs fixed time: −7.3 % [−7.9, −6.7].
  - L2W (secondary): vs actuated −8.7 % [−12.0, −5.5], vs fixed −6.9 % [−7.9, −5.9].
- **H-I1: supported.** 2 of 3 networks beat fixed time significantly: r1 −5.3 % [−6.4, −4.3], r3 −4.5 % [−5.7, −3.3].
  r2 does not: −1.4 % [−3.3, +0.4].
- **H-I2: supported.** First-50-episode mean delay: warm starts 20.4 veh·h, from scratch 49.4 (−58.8 % [−85.5, −32.0],
  Welch). First 1M steps: −6.3 % [−10.4, −2.3]. Final level (descriptive): +0.5 % [−0.2, +1.1].
- **H-P1: not supported.** With 10 calls the pooled change is −0.35 % [−0.74, +0.05]. Only Net0 improves significantly:
  −0.76 % [−1.44, −0.07].
- **H-P2 (descriptive): no ordering.** Pooled changes: B = 0 +0.02 %, B = 5 +0.09 %, B = 10 −0.35 %, all with
  overlapping intervals.
- **H-P3: not supported.** 10 calls vs ES seed 0: −0.60 % [−1.21, +0.01]; vs ES seed 1: −0.27 % [−0.65, +0.11]; vs 1M
  fine-tuning: −0.22 % [−0.55, +0.11].
- **H-T1: supported.** All 3 transferred networks beat fixed time (−4.6 % to −6.7 %) and padding (−93 % to −97 %);
  every interval excludes 0.
- **H-T2: supported.** First-50-episode mean delay: −65.6 % [−100.7, −30.6], Welch. First 1M steps: −10.5 %
  [−15.6, −5.3]. Final level (descriptive): −0.5 % [−1.2, +0.1].
- **H-E1 (descriptive): holds on all 3 targets.**
  - Fidelity: on each target the agent's rule sets agree with the network more often than every tree with at least
    as many comparisons. The narrowest margin is Net2, 68.1 % vs 64.3 % for the 192-comparison DAgger tree.
  - Closed loop: the agents' rules cost +0.1 % to +1.1 % against the network; the trees cost +2.9 % to +437 %.
- **H-C1: not supported.** Closed-loop supervisors vs the naive shield: −0.31 % [−0.63, +0.02]. Net0 improves
  significantly (−1.15 % [−1.90, −0.40]).
- **H-C2 (descriptive).**
  - vs 1M fine-tuning under the rules: +0.22 % [−0.16, +0.61].
  - vs PPO trained from scratch under the rules: +0.73 % [+0.05, +1.41] (Welch), i.e. the supervisors are worse.
- **H-F1: supported.**
  - With 10 calls the pooled change against the faulty network is −59.9 % [−91.2, −28.6].
  - All 3 targets improve significantly: −6.4 %, −81.4 %, −4.5 %.
  - Recovered share of the fault's cost: 94 %, 100 %, 78 %.
- **H-F2: supported against ES, not against mean imputation.**
  - vs ES: pooled −4.5 % [−6.2, −2.9] (seed 0) and −3.5 % [−4.6, −2.3] (seed 1); significant on every target.
  - vs mean imputation: pooled −26.1 % [−54.5, +2.3], so the pre-specified pooled condition fails. Net1, where mean
    imputation still gridlocks in some episodes, dominates the interval.
  - Per target vs mean imputation: Net0 −2.4 % [−3.3, −1.6] and Net2 −1.1 % [−1.9, −0.4] are significant; Net1
    −50.6 % [−107.6, +6.4] is not.
- **H-F3 (descriptive): no ordering.**
  - Pooled changes are equal to within 0.2 points (B = 0 −59.8 %, B = 5 −60.0 %, B = 10 −59.9 %).
  - Recovered share on Net2 by budget: 62 %, 86 %, 78 %.
- **Not hypothesised, reported:** the agents with 10 calls vs 1M steps of fine-tuning under the fault: +0.13 %
  [−0.21, +0.47]; no target differs significantly.

**Robustness to the drain cap (not pre-registered, run 2026-11-08 19:40 EET).**
- **Why.** Some controllers never serve a few lone vehicles during the drain (starvation). These vehicles are still
  present at any cap: re-running with a 60-min cap leaves exactly the same vehicles unserved.
- **What was done.** Every test evaluation with unserved vehicles was re-scored with a 60-min cap; evaluations without
  unserved vehicles are identical under any cap. The full analysis was then repeated
  (`results/paper_numbers_cap3600.json`).
- **Hypotheses.** No pre-registered outcome changes.
- **Descriptive comparisons that change significance:**
  - Transfer warm starts vs scratch, final level: −0.8 % [−1.4, −0.2] (was −0.5 % [−1.2, +0.1]).
  - Explain, small rule sets vs the network: Net0 +1.5 % [+0.0, +2.9] and Net1 +1.2 % [+0.3, +2.2] (were not
    significant).
  - P-F, one-shot agents vs 1M fine-tuning: +0.5 % [−0.1, +1.1] (was +0.5 % [+0.1, +0.9]).

## Amendment P-F-R: replicate sessions of the failed-detector Patch (2026-11-09 11:07 EET; launched 11:08)

- **Why.** Every P-F condition (target × budget) rests on a single agent session, so run-to-run variation is not
  estimated. The authors add replicates of the one-shot and 10-call arms.
- **Timing (disclosed).** This amendment is written after the first P-F results were known: the outcomes are recorded
  above.
- **Runs.**
  - Arms B = 0 and B = 10, Nets 0-2, 2 new sessions each (replicates r2 and r3): 12 runs by Claude Opus 5.5.
  - The B = 5 arm is not replicated.
- **Packs.** The packs are rebuilt by the same generator (`python -m wsp.packs --case patch_fault --run k --budget B
  --rep r`). They are byte-identical to the first-run packs except for the pack id in `spec.json`, which names the
  simulator's mirror log.
- **Protocol.** Same launch template (only the pack path differs), same model, tools and headless-session protocol,
  and the same audit. A run that breaks the protocol is archived and rerun, as before.
- **Analysis.**
  - The pre-registered tests H-F1 to H-F3 stay on the first runs, exactly as registered.
  - Added, descriptive: for each target and arm, the 3 sessions' recovered shares (mean and range), and their test
    changes against the faulty network.
  - The comparisons of H-F1 and H-F2 are repeated with each target's episode-wise mean over its 3 sessions. This
    treats "an agent session" as the method, and it is reported next to the first-run result. If the two disagree,
    both are reported and the disagreement is discussed.
  - Every replicate is reported.

**Outcome of amendment P-F-R (recorded 2026-11-09 11:45 EET, after all 12 replicate sessions had been scored).**
- **Audit.** All 12 sessions pass. Three were flagged, and each was adjudicated by hand:
  - `itertools.product` enumerates synthetic probe states.
  - `np.linalg.solve` is a projection on W3 that uses no data.
  - `polyfit` computes N2/S2 slopes on the recorded states; the coefficient 0.45 was then set by hand within the
    0.3-0.5 slope range. This one is borderline and is disclosed.
- **Recovered share per session (sessions 1 / 2 / 3).**
  - One shot: Net0 97 / 98 / 89 %; Net1 100 / 100 / 100 %; Net2 62 / 71 / 78 %.
  - 10 calls: Net0 94 / −579 / 101 %; Net1 100 / 100 / 100 %; Net2 78 / 80 / 80 %.
- **The −579 % session (Net0, 10 calls, r2).** It gridlocks one high-demand test episode (seed 19010: 717 veh-h,
  985 vehicles unserved). On the other 29 episodes it recovers 98 %. It left no vehicle unserved on its 9 dev
  episodes. It is the only gridlock among the 21 failed-detector agent patches.
- **Session means, one shot.**
  - vs faulty: pooled −59.8 %, all 3 networks significant.
  - vs ES: every network significant, both ES seeds.
  - vs mean imputation: pooled −25.9 % [−54.2, +2.5]; Net0 significant.
  - vs 1M fine-tuning: +0.4 % [−0.1, +0.9], no network significant.
- **Session means, 10 calls.**
  - vs faulty: pooled −57.7 % [−88.4, −27.1]; Nets 1 and 2 significant, Net0 not (+8.7 %).
  - vs ES seed 0: pooled +0.6 % [−10.0, +11.2], not significant; Nets 1 and 2 significant.
  - vs 1M fine-tuning: +5.5 % [−5.7, +16.7].
- **Agreement with the first-run tests.** The session means agree with the first-run tests for H-F1 (both arms). For
  H-F2 they agree for the one-shot arm. They disagree for the 10-call arm against ES, because of the gridlocking
  session; the paper reports both.
