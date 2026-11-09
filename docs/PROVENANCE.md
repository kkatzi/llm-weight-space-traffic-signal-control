# Provenance of the agent runs

## Launch protocol

The 54 analysed runs were launched on 2026-11-08 and 2026-11-09 by `wsp/launch.py`:

- **Main runs:** 33 runs between 17:15 and 18:04 EET, from `runs/agent_launch/plan.json`. Of these, 24 are by Claude
  Opus 5.5, and 3 each by Claude Sonnet 5.5, Claude Haiku 5.5 and Claude Fable 5.1 (closed-loop Patch, 10 calls).
- **Failed-detector Patch (amendment P-F):** 9 runs by Claude Opus 5.5, launched between 18:31:47 and 18:32:03 EET from
  `runs/agent_launch/plan_fault.json`. They finished between 18:38 and 18:50, all with exit code 0.
- **Replicate sessions of the failed-detector Patch (amendment P-F-R):** 12 runs by Claude Opus 5.5, launched
  2026-11-09 11:08 EET from `runs/agent_launch/plan_fault_rep.json` (one-shot and 10-call arms, sessions r2 and r3
  per network). They finished by 11:29, all with exit code 0. Their packs were rebuilt by the same generator and
  are byte-identical to the first-run packs except for the pack id.
- **Isolation:** each run is an independent headless Claude Code session (`claude -p`), started with its pack folder in
  `~/wsp_v3_agent_runs/` as the working directory. That folder is outside the repository, so the session has no git
  context, no parent session and no memory entries.
- **Prompt:** the launch prompt is `python -m wsp.prompts --pack <pack>`, i.e. the approved text, with the pack's path.
- **Tools allowed:** Bash, Read, Write, Edit, Glob, Grep.
- **Per-run records:** `runs/agent_launch/<case>__<id>.json` holds the exact prompt, model id, times, exit code and session
  id.

## Audit (`python -m wsp.archive headless`; per run `packs/<case>/<id>/provenance/audit.json`; table `runs/agent_runs.json`)

All 54 runs pass these checks (the 9 first-run failed-detector runs raised no flag of any kind):

- **Model:** every assistant message is from the requested model (verified from each transcript).
- **Prompt:** the first message equals the launch prompt exactly.
- **Isolation:**
  - no file path outside the pack in any tool call, apart from the Python interpreter, the closed-loop `eval.py`
    harness, and two harmless uses of the session's `/tmp` folder (three temporary `.npz` probe files in
    `patch_fault/net1_b10_r2` and a background task output log in `patch_models/net0_fable`);
  - no write to the repository or to other packs;
  - no tools other than Bash, Read, Write, Edit, Glob and Grep;
  - no messages received after launch;
  - no permission denials.
- **Simulator:**
  - no simulator use outside `eval.py`;
  - simulator calls within budget;
  - the pack's evaluation log and the out-of-pack mirror log agree for every run.
- **Rules:**
  - every Patch edit, including the failed-detector Patch, is within the 371-parameter cap;
  - every Transfer keeps W2, b2, W3, b3 bit-identical;
  - every Initialise network meets the trainability bounds;
  - every rules_small.py has at most 8 comparisons.
- **Outside context:** no NOTES.md mentions any external repository or project notes.

## Flags adjudicated (screen `fit_or_search_keyword`)

All seven flagged runs were reviewed and confirmed to be compliant with the mechanistic non-fitting rule:

- **explain/net0, explain/net1, explain/net2: `np.linalg.lstsq` (and in net0 `itertools.product`).**
  - **The lstsq.** All three agents derived the rule files' phase offsets the same way. For each current phase and each
    pair of candidate phases, they took the median score difference at the network's own tie points: recorded honoured
    states where the two phases' logits are the top two and nearly equal. Then a 3-equation, 2- to 4-unknown weighted
    least-squares solve, with a sum-to-zero constraint, reconciled the three pairwise medians into one offset per phase.
  - **Why it is allowed.** The medians are the statistic the task explicitly allows ("quantiles or conditional means of
    features around the network's switching decisions"). The small solve combines three statistics into consistent
    offsets. It is not a regression of decisions on features, and no objective involving fidelity is optimised. In
    net1 (`explore26.py`) the solve is a diagnostic of a candidate score.
  - **The itertools.product in net0** enumerates synthetic binary lane occupancies for forward-pass probes of the
    network, which the task allows.
  - **Verdict:** compliant.
- **patch/net1_b0: `np.linalg.lstsq` / `np.linalg.pinv`.**
  - **What they do.** They regress recorded outcomes (later queue or delay indicators) on the network's choice and on
    controls, to estimate the effect of a decision, with a standard error, on the recorded data.
  - **Why it is allowed.** This is a statistic of the recorded data, which rule 4 allows. No weight value is fitted;
    the agent's NOTES.md describes the weight edit as built by construction.
  - **Verdict:** compliant.

- **patch_fault replicates (net1_b0_r2 `itertools.product`; net1_b10_r2 `np.linalg.solve`; net2_b10_r3 `np.polyfit`).**
  - `itertools.product` enumerates synthetic one-vehicle probe states for forward-pass checks.
  - `np.linalg.solve` projects one W3 row onto the orthogonal complement of the other three. It is linear algebra
    on the weights and uses no data.
  - `np.polyfit` computes N2/S2 pocket-reading slopes on the recorded states (0.3-0.5). The stand-in coefficient
    was then set by hand to 0.45; 0.25 and 0.7 were tried in the simulator.
  - **Verdict:** compliant (disclosed in Appendix A of the paper).

## Observations worth reporting

- **Low effort by two writer models.** Sonnet 5.5 delivered output-bias nudges of 1-2 parameters after 3-4 simulator calls
  (about 1 min of activity each). Haiku 5.5 delivered the unchanged network once (net1_haiku, 0 parameters).
- **Bytecode caches.** The packs of the analysed runs ship a pre-compiled `tools.py` bytecode cache so that importing
  `tools.py` never writes outside `work/`.
