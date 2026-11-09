# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Task packs for the five uses. Every number in a TASK.md is generated from the pipeline's own results.

    (from the repository root)
    python -m wsp.packs --case init      --run 1                  # run = random-initialisation seed (1-3)
    python -m wsp.packs --case patch     --run 0 --budget 10      # run = target index (Net 0-2); budget 0, 5 or 10
    python -m wsp.packs --case transfer  --run 0                  # run = source index
    python -m wsp.packs --case explain   --run 0
    python -m wsp.packs --case constrain --run 0 --budget 10      # budget 0 (blind) or 10

Packs go to packs/<case>/<id>/ and contain TASK.md, tools.py, spec.json, the weights the task starts from,
states.npz, an empty work/ folder and, in closed-loop arms, eval.py. Refuses to touch an existing pack.
Targets: Net k = models/L2/ppo_s{100 + k}/final.zip (must meet the convergence criterion).
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

from wsp import junctions as J
from wsp.env import N_PHASES, RULES, SignalEnv
from wsp.evaluate import SETS, V3, evaluate
from wsp.policies import ACTOR_KEYS, load_net, tuned
from wsp.train import CONFIG

PY = os.environ.get("WSP_PACK_PYTHON", sys.executable)
CAP = 0.05
WARMUP = 50_000
WARM_STEPS = 3_000_000
CALL_S = 20
TARGET_SEEDS = [100, 101, 102]
SC = ("low", "nominal", "high")
PACK_ROOT = os.environ.get("WSP_PACK_ROOT", os.path.join(V3, "packs"))
PREVIEW_TARGET = None      # tests only: a stand-in network path for rendering draft packs before the targets exist


# ---------------------------------------------------------------- data
def target_path(k, problem="L2"):
    if PREVIEW_TARGET:
        return PREVIEW_TARGET
    return os.path.join(V3, "models", problem, f"ppo_s{TARGET_SEEDS[k]}", "final.zip")


def check_converged(path):
    if PREVIEW_TARGET:
        return 2_000_000
    cv = json.load(open(os.path.join(os.path.dirname(path), "curve_val.json")))
    if not cv["converged"]:
        sys.exit(f"{path}: not converged ({cv['change_pct']:+.2f} %); it cannot be an agent target")
    return cv["total_steps"]


def concrete(problem, policy):
    return tuned(problem, policy) if policy in ("fixed", "actuated") else policy


def dev_ref(problem, policy, label):
    d = evaluate(problem, concrete(problem, policy), "dev", label, workers=9)
    s = d["summary"]
    return {"all": s["all"]["tts_veh_h"], **{sc: s[sc]["tts_veh_h"] for sc in SC},
            "switches": float(np.mean([e["switches"][0] for e in d["episodes"]]))}


def record(problem, policy, label):
    """The policy's decisions on the 6 rec episodes -> dict of arrays (obs, act, episode, time_s[, rule_state])."""
    evaluate(problem, concrete(problem, policy), "rec", label, workers=6, record=True)
    out = {k: [] for k in ("obs", "act", "episode", "time_s", "rule_state")}
    for i, (sc, s) in enumerate(SETS["rec"]):
        z = np.load(os.path.join(V3, "runs", "eval", problem, label, "rec", f"{sc}_{s}", "record.npz"))
        out["obs"].append(z["obs"]), out["act"].append(z["act"][:, 0])
        out["episode"].append(np.full(len(z["obs"]), i, np.int16))
        out["time_s"].append(np.arange(len(z["obs"]), dtype=np.int32) * 5)
        if "rule_state" in z.files:
            out["rule_state"].append(z["rule_state"])
    return {k: np.concatenate(v) for k, v in out.items() if v}


def save_states(path, parts):
    names = list(parts)
    out = {k: np.concatenate([parts[n][k] for n in names]) for k in parts[names[0]]}
    out["controller"] = np.concatenate([np.full(len(parts[n]["obs"]), i, np.int8) for i, n in enumerate(names)])
    out["controller_names"] = np.array(names)
    np.savez_compressed(path, **out)
    return {n: len(parts[n]["obs"]) for n in names}


def fresh_init(problem, seed):
    """The actor as PPO initialises it (SB3 default orthogonal initialisation), for the given seed."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    n = SignalEnv(problem, workdir=os.path.join(V3, "runs", "tmp", "init")).observation_space.shape[0]

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(0.0, 1.0, (n,), np.float32)
        action_space = gym.spaces.MultiDiscrete([N_PHASES])

        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, a):
            return self.observation_space.sample(), 0.0, True, False, {}
    h = CONFIG["hidden"]
    m = PPO("MlpPolicy", Dummy(), seed=seed, device="cpu", policy_kwargs=dict(net_arch=dict(pi=[h, h], vf=[h, h])))
    sd = m.policy.state_dict()
    return {k: sd[v].double().numpy() for k, v in ACTOR_KEYS.items()}


def trainability_of(w, X):
    a1 = X @ w["W1"].T + w["b1"]
    h1 = np.tanh(a1)
    a2 = h1 @ w["W2"].T + w["b2"]
    h2 = np.tanh(a2)
    lg = h2 @ w["W3"].T + w["b3"]
    p = np.exp(lg - lg.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    mp = p.max(axis=1)
    return {"median_max_prob": float(np.median(mp)), "share_max_prob_above_0.99": float((mp > 0.99).mean()),
            "max_abs_logit": float(np.abs(lg).max()), "sat_l1": float((np.abs(a1) > 3).mean()),
            "sat_l2": float((np.abs(a2) > 3).mean()), "dead_l1": int((h1.std(axis=0) < 0.01).sum()),
            "dead_l2": int((h2.std(axis=0) < 0.01).sum())}


# ---------------------------------------------------------------- text
def system_md(problem):
    t = J.load_meta(problem)["tls"][0]
    g = J.JUNCTIONS[J.base_of(problem)]
    L = dict(zip(t["in_lanes"], t["lane_length"]))
    up_len = t["up_length"][0]

    def stop_len(a):
        return L[f"U{a}_J0_0"]
    main, other = g["base"]
    tr = g["turn"]
    tot = sum(tr.values())
    pct = {k: round(100 * v / tot) for k, v in tr.items()}
    groups = [("N and S", "N"), ("E and W", "E")] if g["lanes"]["N"] != g["lanes"]["E"] else [("All", "N")]
    lines = [f"One signalised junction J0 with four approaches (N, E, S, W), each {g['length']:.0f} m long, speed "
             f"limit 50 km/h. Each approach is a road that widens {g['pocket']:.0f} m before the junction, where a "
             "left-turn pocket opens:"]
    for label, a in groups:
        n, u = g["lanes"][a], g["up"][a]
        stop = ", ".join(["lane 0 = right + straight"] + [f"lane {i} = straight" for i in range(1, n - 1)]
                         + [f"lane {n - 1} = left-turn pocket"])
        feeds = "; ".join(["upstream lane 0 = right + straight, continues into lane 0"]
                          + [f"upstream lane {i} = straight, continues into lane {i}" for i in range(1, u - 1)]
                          + [f"upstream lane {u - 1} = straight + left, continues into lanes {u - 1} and {n - 1}"])
        lines.append(f"- {label} approaches: at the stop line {n} lanes, {stop}; these lanes are "
                     f"{stop_len(a):.1f} m long (about {round(stop_len(a) / 7.5)} cars each). Upstream {u} lanes, "
                     f"{up_len:.0f} m long: {feeds}. Exits have {g['out'][a]} lanes.")
    lines.append(f"\nDemand at nominal level: {main} veh/h per E and W approach (the main road) and {other} veh/h "
                 f"per N and S approach. From every approach, {pct['s']} % go straight, {pct['l']} % turn left, "
                 f"{pct['r']} % turn right.")
    return "\n".join(lines)


SIGNALS = """**Signals.** The junction has 4 protected green phases:

| phase | green movements |
|---|---|
| 0 NS_through | from N and S: straight + right |
| 1 NS_left | from N and S: left |
| 2 EW_through | from E and W: straight + right |
| 3 EW_left | from E and W: left |

Every 5 s the controller picks a phase. If the pick differs from the current green and the current green has
lasted at least 10 s, the junction shows 3 s of yellow and then the new green (so at the next decision the new green
is 2 s old). A different pick before 10 s of green is ignored. Any phase may follow any phase; there is no maximum
green.

**Demand over time.** One hour of traffic in four 15-minute intervals with nominal multipliers 0.75, 1.0, 1.15, 0.9,
then no new traffic until the network is empty (capped at 30 more minutes). Arrivals are random.

**KPI.** Total time in system, TTS (vehicle-hours): every vehicle's time in the network plus its time waiting to
enter, summed over the episode. Lower is better. Teleporting is disabled, so vehicles that are not served stay in
the network and keep adding to TTS."""


def network_md(n):
    h = CONFIG["hidden"]
    total = n * h + h + h * h + h + 4 * h + 4
    return (f"An SB3 PPO actor (MlpPolicy): {n} inputs -> {h} tanh -> {h} tanh -> 4 logits ({total} parameters). "
            "The decision is `argmax(logits)`, deterministic. Forward pass, exactly as evaluated:\n\n"
            "    h1 = tanh(W1 @ x + b1);  h2 = tanh(W2 @ h1 + b2);  logits = W3 @ h2 + b3\n\n"
            f"Weight shapes (npz keys, torch convention [out, in]): W1 ({h}, {n}), b1 ({h},), W2 ({h}, {h}), "
            f"b2 ({h},), W3 (4, {h}), b3 (4,).")


def obs_md(problem):
    env = SignalEnv(problem, workdir=os.path.join(V3, "runs", "tmp", "obs"))
    t = env.meta["tls"][0]
    nm = env.obs_names()
    n, u = len(env.lanes[0]), len(env.up_lanes[0])
    dirname = {"rs": "right + straight", "s": "straight", "l": "left pocket"}
    stop = ", ".join(f"{a}{ln.rsplit('_', 1)[1]} ({dirname[d]})"
                     for a, ln, d in zip(t["lane_approach"], t["in_lanes"], t["lane_dirs"]))
    upd = []
    for ln in env.up_lanes[0]:
        fed = [f"{t['lane_approach'][t['in_lanes'].index(x)]}{x.rsplit('_', 1)[1]}" for x in t["feeds"][ln]]
        upd.append(f"{ln[1]}{ln.rsplit('_', 1)[1]}u (feeds {' and '.join(fed)})")
    return (f"## Observation x ({len(nm)} values, all in [0, 1])\n\n"
            f"- `queue` {0}-{n - 1}: halting vehicles (speed < 0.1 m/s) on each stop-line lane / lane capacity;\n"
            f"- `density` {n}-{2 * n - 1}: vehicles on each stop-line lane / lane capacity;\n"
            f"- `up_queue` {2 * n}-{2 * n + u - 1} and `up_density` {2 * n + u}-{2 * n + 2 * u - 1}: the same for the "
            "upstream lanes;\n"
            f"- `phase` {2 * n + 2 * u}-{2 * n + 2 * u + 3}: one-hot of the current green phase;\n"
            f"- `ok` {env.ok_index()}: 1 if the current green has lasted ≥ 10 s (a switch request would be honoured);\n"
            f"- `t_green` {env.ok_index() + 1}: time in the current green / 60 s, clipped to 1.\n\n"
            "Lane capacity = lane length / 7.5 m; every value is clipped to 1.\n"
            f"- Stop-line lanes, in order: {stop}.\n"
            f"- Upstream lanes, in order: {', '.join(upd)}.\n\n"
            "Lane names: approach letter + lane index (lane 0 is the rightmost); `u` marks an upstream lane. "
            "`tools.names()` lists all entries; `tools.block(x)` shows one observation by name.")


def training_md(steps):
    c = CONFIG
    lines = [f"- configuration: clip {c['clip_range']}, learning rate {c['lr']:g}, entropy coefficient {c['ent_coef']:g}, "
             f"discount {c['gamma']}, GAE lambda {c['gae_lambda']}, {c['n_envs']} parallel episodes per update "
             f"({c['n_steps']} decisions each), minibatch {c['batch_size']}, {c['n_epochs']} epochs, normalised rewards;",
             "- reward per decision: -(delay over the 5 s)/10, where the delay counts, every second, 1 - speed / speed "
             "limit for each vehicle in the network plus 1 for each vehicle waiting to enter, averaged over the 5 s;",
             "- training demand: overall level 0.7-1.25 x nominal, a random 15-minute profile (multipliers 0.6-1.25) and "
             "a random directional bias (E and W entries x e^b, N and S entries x e^-b, b between -0.2 and +0.3);",
             "- training episodes run like the test episodes, until the network is empty" + (";" if steps else ".")]
    if steps:
        lines.append(f"- training length: {steps:,} environment steps (one step = one decision).")
    return "Stable-Baselines3 PPO:\n" + "\n".join(lines)


def refs_md(rows, title):
    head = "| controller | low | nominal | high | all 9 |\n|---|---|---|---|---|\n"
    body = "\n".join(f"| {n} | " + " | ".join(f"{r[s]:.1f}" for s in SC) + f" | {r['all']:.1f} |" for n, r in rows)
    return f"## {title}\n\n" + head + body + "\n"


CONTROLLERS = """- **Fixed-time:** a cycle 0 -> 1 -> 2 -> 3 with green times tuned on the dev episodes.
- **Actuated:** after the minimum green, extends the current green while vehicles still move on its lanes, up to
  a maximum green; otherwise it switches to the phase with the most halting vehicles. Its thresholds and detector
  lanes are tuned on the dev episodes."""


def test_md(what):
    return ("## How your result will be tested (after you finish; you will not see these runs)\n\n"
            f"{what} runs on 30 held-out episodes with seeds you have no access to, 10 each of\n"
            "- **low**: nominal x 0.8;\n- **nominal**: the demand described above;\n- **high**: nominal x 1.15.\n\n"
            "Score: mean TTS over the 30 episodes, also per scenario.\n")


def sim_rule(budget, kind):
    if budget == 0:
        return "You may NOT run the simulator at all. You get one shot: what you deliver is tested as it is."
    f = "work/supervisor.py" if kind == "py" else "work/<file>.npz"
    return (f"You may run the simulator on YOUR {'supervisor' if kind == 'py' else 'weights'} through "
            f"`python eval.py {f}`, at most {budget} times in total. Each call runs the 9 dev episodes (3 seeds each "
            "of low, nominal and high demand; seeds differ from the test), prints TTS per episode and per scenario, "
            "leftover vehicles and switch counts" + (" and shield overrides" if kind == "py" else "") +
            ", and is logged.")


def rules_weights(budget, extra):
    return f"""## Python

Use `{PY}`. It has numpy, which is all `tools.py` needs.

## Rules

1. Work only inside this folder. Read only files in this folder; write your files to `work/`. Do not read any other
   file on this machine: no project code, models, results or other experiments.
2. Do not run SUMO, TraCI or libsumo, any training, or any simulator other than what rule 3 allows.
3. {sim_rule(budget, 'npz')}
4. **No fitting of weights to data.** No gradient steps on any loss (SGD, Adam, ...), no least-squares or other
   regression fit of weights, no random or evolutionary search over weights, and no imitation fit of a controller's
   actions. Set the values by construction. You may compute statistics of the recorded data (means, quantiles,
   correlations, activations) to choose numbers such as thresholds, and you may run forward passes freely, on recorded
   or synthetic inputs.
{extra}
## Deliverables (in `work/`)

- `weights.npz`: written with `tools.save(w, "work/weights.npz")`, which checks the shapes.
- `NOTES.md`:
  - what you built and why, in plain words;
  - which units or weights do what;
  - the checks you ran{{trainability}};
  - what you expect to go wrong.
- `predictions.json`: your predicted mean test TTS (veh-h) per scenario for YOUR network as delivered:
  `{{{{"low": x, "nominal": y, "high": z}}}}`.
- Keep your scripts in `work/` too.
"""


def rules_explain(ok):
    return f"""## Python

Use `{PY}`. Rule files may use only numpy and math.

## Rules

1. Work only inside this folder. Read only files in this folder; write your files to `work/`. Do not read any other
   file on this machine: no project code, models, results or other experiments.
2. Do not run SUMO, TraCI or libsumo, any training, or any simulator. You get one shot.
3. **No fitting.** Do not fit decision trees, regressions, classifiers or any other model to the network's decisions,
   and do not run automated searches over rules or thresholds (e.g. loops that maximise fidelity). Choose the
   structure and the numbers by reasoning:
   - inspect the weights;
   - run forward passes on recorded or synthetic inputs;
   - compute statistics such as quantiles or conditional means of features around the network's switching decisions.
   You may check fidelity with `tools.fidelity` as often as you like.
4. Rules must stay readable: no weight matrices or long coefficient vectors. Each comparison compares a feature, or a
   small expression of a few named features (sums, differences, simple scalings), with a threshold or with another
   such expression.

## Deliverables (in `work/`)

- `rules_small.py`: at most 8 comparisons in total (`tools.count_comparisons` counts them).
- `rules_full.py`: as faithful as you can make it, any size, still readable rules.
  - Both define `decide(x) -> int`, where x is the observation (numpy array, layout above). They return the phase, 0-3.
- `EXPLANATION.md`: a plain-language explanation of the policy for a traffic engineer:
  - when it switches, and to which phase;
  - what it ignores;
  - where it may perform poorly, if anywhere.
- `NOTES.md`: how you derived the rules and the checks you ran.
- `predictions.json`:
  - `{{"fidelity_small": a, "fidelity_full": b, "tts_small": c, "tts_full": d}}`;
  - fidelity is your predicted agreement with the network on held-out states where the minimum green has elapsed
    (x[{ok}] = 1), between 0 and 1;
  - TTS is your predicted mean test TTS in veh-h over the 30 test episodes, for each rule file used as the controller.
- Keep your scripts in `work/` too.
"""


def rules_constrain(budget):
    return f"""## Python

Use `{PY}`. `supervisor.py` may use only numpy and math.

## Rules

1. Work only inside this folder. Read only files in this folder; write your files to `work/`. Do not read any other
   file on this machine: no project code, models, results or other experiments.
2. Do not run SUMO, TraCI or libsumo, any training, or any simulator other than what rule 3 allows.
3. {sim_rule(budget, 'py')}
4. **No fitting or automated search.** Do not fit models to data, and do not run loops that try many supervisor
   variants offline and keep the best. Choose the logic and the numbers by reasoning. You may compute statistics of
   the recorded data and run the network freely.
5. The supervisor must use the network through the `logits` it receives. Do not copy, modify or re-implement the
   network's weights.

## Deliverables (in `work/`)

- `supervisor.py` with `decide(x, logits, state) -> int` (the phase, 0-3).
- `NOTES.md`:
  - what your supervisor does and why;
  - when it overrides the network;
  - the checks you ran;
  - what you expect to go wrong.
- `predictions.json`: your predicted mean test TTS (veh-h) per scenario under the rules:
  `{{"low": x, "nominal": y, "high": z}}`.
- Keep your scripts in `work/` too.
"""


TRAINABLE = """### Make it trainable (required)

PPO will continue training from your network, so it must be a good *starting point*, not only a good controller.
During training PPO samples phases from the softmax of your logits.
- If the network is right but extremely confident (probabilities of 1.0, logits of ±50), it does not explore. Tiny
  weight changes then flip decisions wholesale, and the first PPO updates wreck it.
- Units whose weights are all zero get no gradient and stay dead.

`tools.trainability(w)` measures this on the recorded states. For reference, on this junction:
- a PPO network partway through training ({mid_steps:,} steps) measures median_max_prob {a[median_max_prob]:.2f}, share_max_prob_above_0.99 {a[share_max_prob_above_0.99]:.2f}, max_abs_logit {a[max_abs_logit]:.1f}, sat_l1 {a[sat_l1]:.2f}, sat_l2 {a[sat_l2]:.2f}, dead_l1 {a[dead_l1]}, dead_l2 {a[dead_l2]}. That is the kind of starting point PPO handles well;
- a fully trained one ({end_steps:,} steps) is near-deterministic: median_max_prob {b[median_max_prob]:.2f}, share_max_prob_above_0.99 {b[share_max_prob_above_0.99]:.2f}, max_abs_logit {b[max_abs_logit]:.1f}. That is fine for deployment, but a poor starting point.

**Required** for your delivered weights, on `states.npz`:
- median_max_prob ≤ 0.95;
- share_max_prob_above_0.99 ≤ 0.30;
- max_abs_logit ≤ 15.

**Recommended:**
- sat_l1 and sat_l2 ≤ 0.05;
- few dead units: leave units you do not need at their initial values instead of zeroing them.
"""

TRANSFER_TRAINABILITY = """### Trainability (reported, not required here)

PPO will continue training from your network. Because W2, b2, W3 and b3 are the source network's trained layers,
your network inherits its confidence: a fully trained PPO network is near-deterministic. So the trainability limits
used elsewhere are not required for this task. `tools.trainability(w)` still reports the numbers; include them in
NOTES.md. Do not make the network sharper than the source network, for example with very large W1 values that
saturate layer 1.
"""

TITLES = {"init": "write a network that PPO will start training from",
          "patch": "improve a trained signal-control network by editing a few of its weights",
          "patch_fault": "repair a trained signal-control network after one of its detectors failed",
          "transfer": "transfer a trained network to a junction with a different observation",
          "explain": "translate a trained network into readable rules",
          "constrain": "deploy a trained network under operating rules with as little loss as possible"}


# ---------------------------------------------------------------- formatting
WIDTH = 116


def normalise(md, width=WIDTH):
    """Join manually wrapped lines into paragraphs / list items and re-wrap them at `width`. Tables, headings and
    4-space code lines are kept verbatim."""
    import re
    import textwrap
    item = re.compile(r"^(\s*)(- |\d+\. )")
    out, cur, prev_blank, prev_code = [], None, True, False

    def flush():
        nonlocal cur
        if cur is not None:
            first, cont, text = cur
            out.append(textwrap.fill(text, width, initial_indent=first, subsequent_indent=cont,
                                     break_long_words=False, break_on_hyphens=False))
            cur = None
    for ln in md.split("\n"):
        if not ln.strip():
            flush()
            out.append("")
            prev_blank, prev_code = True, False
            continue
        if ln.startswith("    ") and (prev_blank or prev_code) and not item.match(ln):
            flush()
            out.append(ln)
            prev_blank, prev_code = False, True
            continue
        if ln.startswith("|") or ln.startswith("#"):
            flush()
            out.append(ln)
        elif item.match(ln):
            flush()
            m = item.match(ln)
            first = m.group(1) + m.group(2)
            cur = [first, " " * len(first), ln[len(first):].strip()]
        elif cur is not None:
            cur[2] += " " + ln.strip()
        else:
            cur = ["", "", ln.strip()]
        prev_blank, prev_code = False, False
    flush()
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------- build
def precompile(d):
    """Compile tools.py with the pack's interpreter, so that importing it never writes a cache outside work/."""
    import py_compile
    py_compile.compile(os.path.join(d, "tools.py"), doraise=True)


def pack_id(case, run, budget, rep=1):
    """rep >= 2: a replicate of the same condition (EXPERIMENT_PLAN.md, amendment P-F-R); identical pack, new id."""
    if case == "init":
        pid = f"r{run}"
    elif case in ("patch", "constrain", "patch_fault"):
        pid = f"net{run}_b{budget}"
    else:
        pid = f"net{run}"
    return pid if rep == 1 else f"{pid}_r{rep}"


def build(case, run, budget, rep=1):
    pid = pack_id(case, run, budget, rep)
    d = os.path.join(PACK_ROOT, case, pid)
    if os.path.exists(d) and any(files for _, _, files in os.walk(d)):
        sys.exit(f"{d} exists")
    problem = {"transfer": "L2W", "constrain": "L2_rules", "patch_fault": "L2_fault"}.get(case, "L2")
    base = J.base_of(problem)
    env = SignalEnv(base, workdir=os.path.join(V3, "runs", "tmp", "pack"))
    n_obs = env.observation_space.shape[0]
    fx, ac = dev_ref(base, "fixed", "fixed"), dev_ref(base, "actuated", "actuated")
    ref_rows = [("fixed-time (tuned)", fx), ("actuated (tuned)", ac)]
    states = {}
    files = ["- `TASK.md`: this file.",
             "- `tools.py`: load, save, forward, hidden, greedy, probs, states, names, block, edit_size, trainability, "
             "fidelity and count_comparisons helpers (read its docstring).",
             "- `spec.json`: the observation layout, the lane names and the dev references."]
    extra_rules, trainability_note, start, src = "", "", None, None
    kind = "npz"
    if case == "init":
        start = fresh_init(base, seed=run)
        states = {"fixed": record(base, "fixed", "fixed"), "actuated": record(base, "actuated", "actuated")}
        X = np.concatenate([states[k]["obs"] for k in states]).astype(np.float64)
        end = target_path(0)
        end_steps = check_converged(end)
        import glob
        import re
        cks = {int(re.search(r"ckpt_(\d+)_steps", q).group(1)): q
               for q in glob.glob(os.path.join(os.path.dirname(end), "ckpt_*_steps.zip"))}
        mid_steps = min((s for s in cks if s > 0), key=lambda s: abs(s - 600_000))     # the checkpoint nearest 600k
        tr = TRAINABLE.format(mid_steps=mid_steps, end_steps=end_steps, a=trainability_of(load_net(cks[mid_steps]), X),
                              b=trainability_of(load_net(end), X))
        goal = ("Write the weights of a fresh network that will be the **starting point for PPO training** on this "
                "junction. `start_weights.npz` is the network as PPO would initialise it (random); you may keep or "
                "overwrite any of it. Your network is judged on two things:\n"
                "1. its own test TTS, before any training;\n"
                f"2. how fast PPO improves from it over {WARM_STEPS:,} steps (two training seeds), compared with PPO "
                "from a random start.\n\n"
                f"How PPO continues from your network: {training_md(None)}\n\nThe critic (value network) starts random "
                f"and is trained alone for the first {WARMUP:,} environment steps while your actor is frozen; then both "
                f"train, for {WARM_STEPS:,} steps in total.\n\n" + tr)
        files.append("- `start_weights.npz`: the network you start from (random initialisation).")
        trainability_note = ", including `tools.trainability`"
        test_what = "Your network"
    elif case == "patch":
        target = target_path(run)
        steps = check_converged(target)
        start = load_net(target)
        lab = ("preview_" if PREVIEW_TARGET else "") + f"target_net{run}"
        tg = dev_ref(base, f"sb3:{target}", lab)
        ref_rows.append(("the trained network (start_weights.npz)", tg))
        states = {"start_network": record(base, f"sb3:{target}", lab), "fixed": record(base, "fixed", "fixed"),
                  "actuated": record(base, "actuated", "actuated")}
        total = sum(v.size for v in start.values())
        cap = int(CAP * total)
        goal = (f"`start_weights.npz` is a network trained with PPO on this junction. Improve it so that the edited "
                f"network has a lower test TTS than the original, by changing **at most {cap} of its {total} "
                f"parameters ({CAP:.0%})**. Any value that differs from the original counts as changed. No further "
                "training happens: your edited network is tested exactly as delivered.\n\n"
                f"How the network was trained: {training_md(steps)}")
        extra_rules = (f"5. At most {cap} parameters may differ from `start_weights.npz` (check with "
                       "`tools.edit_size`). More than that disqualifies the edit.\n")
        files.append("- `start_weights.npz`: the trained network.")
        test_what = "Your network"
    elif case == "patch_fault":
        target = target_path(run)
        steps = check_converged(target)
        start = load_net(target)
        lab = ("preview_" if PREVIEW_TARGET else "") + f"target_net{run}"
        fault = json.load(open(os.path.join(V3, "results", "fault.json")))
        fenv = SignalEnv(problem, workdir=os.path.join(V3, "runs", "tmp", "pack"))
        idx = [fenv.obs_names()[i] for i in fenv._zero_idx]
        healthy = dev_ref(base, f"sb3:{target}", lab)
        faulty = dev_ref(problem, f"sb3:{target}", lab)
        ref_rows = [("fixed-time (tuned)", fx), ("actuated (tuned), with the same failed detector",
                                                  dev_ref(problem, "actuated", "actuated")),
                    ("the trained network with every detector working (for reference)", healthy),
                    ("the trained network (start_weights.npz) with the failed detector", faulty)]
        states = {"start_network": record(problem, f"sb3:{target}", lab), "fixed": record(problem, "fixed", "fixed"),
                  "actuated": record(problem, "actuated", "actuated")}
        total = sum(v.size for v in start.values())
        cap = int(CAP * total)
        lane_desc = {"2": "left-turn pocket", "0": "right + straight lane", "1": "straight lane"}[fault["name"][1]] \
            if not fault["name"].endswith("u") else "upstream lane"
        goal = (f"`start_weights.npz` is a network trained with PPO on this junction, with every detector working. It is "
                f"now deployed, and one detector has failed: the detector of lane {fault['name']} (the "
                f"{ {'N': 'north', 'E': 'east', 'S': 'south', 'W': 'west'}[fault['name'][0]] } approach's {lane_desc}) always "
                f"reports 0 halting vehicles and 0 vehicles. So x[{fenv._zero_idx[0]}] ({idx[0]}) and "
                f"x[{fenv._zero_idx[1]}] ({idx[1]}) are always 0; every other input works. Traffic is unchanged: vehicles "
                f"still use lane {fault['name']}.\n\nImprove the network so that it controls the junction as well as "
                f"possible with the failed detector, by changing **at most {cap} of its {total} parameters ({CAP:.0%})**. "
                "Any value that differs from the original counts as changed. No further training happens: your edited "
                "network is tested exactly as delivered, with the same failed detector.\n\n"
                f"How the network was trained (with every detector working): {training_md(steps)}")
        extra_rules = (f"5. At most {cap} parameters may differ from `start_weights.npz` (check with "
                       "`tools.edit_size`). More than that disqualifies the edit.\n")
        files.append("- `start_weights.npz`: the trained network.")
        test_what = "Your network, with the same failed detector,"
    elif case == "explain":
        target = target_path(run)
        steps = check_converged(target)
        start = load_net(target)
        lab = ("preview_" if PREVIEW_TARGET else "") + f"target_net{run}"
        tg = dev_ref(base, f"sb3:{target}", lab)
        ref_rows.append(("the trained network (start_weights.npz)", tg))
        states = {"start_network": record(base, f"sb3:{target}", lab), "fixed": record(base, "fixed", "fixed"),
                  "actuated": record(base, "actuated", "actuated")}
        goal = ("`start_weights.npz` is a network trained with PPO on this junction. **Translate it into readable "
                "Python rules that make the same decisions**, and explain the policy in plain language. Faithfulness "
                "is the goal: do not improve the policy. If you see a flaw, describe it in EXPLANATION.md, but keep the "
                "rules faithful.\n\nHow your rule files will be judged:\n"
                "- **Fidelity:** agreement with the network's decisions on held-out states: the network's own runs on "
                "episodes of the three test scenarios, which you will not see. Only states where the minimum green has "
                f"elapsed (x[{env.ok_index()}] = 1) count, because before that a request does not change the signal. "
                "`tools.fidelity(\"work/rules_small.py\")` measures this on the recorded states you have.\n"
                "- **Closed-loop performance:** we run each rule file as the controller on the 30 test episodes and "
                "compare its TTS with the network's. A faithful rule set should perform like the network.\n"
                "- **Size:** the number of comparisons.\n\n"
                f"How the network was trained: {training_md(steps)}")
        files.append("- `start_weights.npz`: the trained network.")
        test_what = "Each rule file"
    elif case == "constrain":
        target = target_path(run)
        steps = check_converged(target)
        start = load_net(target)
        lab = ("preview_" if PREVIEW_TARGET else "") + f"target_net{run}"
        unc = dev_ref(base, f"sb3:{target}", lab)
        naive = dev_ref(problem, f"sb3:{target}", lab)
        acr, fxr = dev_ref(problem, "actuated", "actuated"), dev_ref(problem, "fixed", "fixed")
        ref_rows = [("the network WITHOUT the rules (for reference)", unc),
                    ("the network under the rules, naive (shield only)", naive),
                    ("actuated, tuned under the rules", acr), ("fixed-time, tuned under the rules", fxr)]
        st = record(problem, f"sb3:{target}", lab)
        states = {"start_network_under_rules": st}
        r = RULES
        goal = (f"`start_weights.npz` is a network trained with PPO on this junction. It must now be deployed under "
                "three operating rules that it was never trained with (below). A shield enforces the rules no matter "
                "what: when the controller asks for a phase the rules forbid, the shield replaces it with the legal "
                "phase that has the most halting vehicles. Deployed naively under the shield, the network loses "
                f"efficiency: dev TTS {naive['all']:.1f} veh-h under the rules, against {unc['all']:.1f} without them.\n\n"
                "**Write `work/supervisor.py`, a decision layer on top of the network:**\n\n"
                "    def decide(x, logits, state) -> int   # the phase, 0-3\n\n"
                "- `x`: the observation (numpy array, layout below).\n"
                "- `logits`: the network's 4 output scores for x (higher = preferred; the network's own choice is the "
                "argmax).\n"
                "- `state`: a dict with the rule state:\n"
                "  - `green`: the current phase;\n  - `t_green`: seconds of the current green;\n"
                "  - `since_served[p]`: seconds since phase p last had green (0 for the current phase);\n"
                "  - `phase_halting[p]`: halting vehicles on phase p's stop-line lanes;\n"
                "  - `legal[p]`: whether the rules allow choosing p now;\n"
                "  - `overdue[p]`: whether p must be served next;\n  - `rules`: the rule parameters.\n\n"
                "**Use the trained behaviour.** Keep the network in charge wherever the rules allow, and step in to "
                "respect or anticipate the rules so that as little efficiency as possible is lost. Your supervisor is "
                "judged on test TTS under the rules (lower is better). We also report how often its choice differs "
                "from the network's own choice.\n\n### The rules\n\n"
                f"1. **Pedestrian minimum green:** through phases (0 and 2) must stay green at least {r['min_green'][0]} s, "
                f"and left phases (1 and 3) at least {r['min_green'][1]} s.\n"
                f"2. **Maximum green:** a green may not continue past {r['max_green']} s while another phase has halting "
                "vehicles.\n"
                f"3. **Maximum wait:** a phase with halting vehicles must get green within {r['max_wait']} s of its last "
                "green. An overdue phase must be served next, as soon as the current green may end.\n\n"
                "Decisions are taken every 5 s, and a new green is 2 s old at the next decision (after 3 s of yellow). "
                f"So through greens can end at {r['min_green'][0] + 2} s at the earliest, left greens at "
                f"{r['min_green'][1] + 2} s, and a max-green change happens at the first decision after "
                f"{r['max_green']} s.\n\nHow the network was trained (without the rules): {training_md(steps)}")
        files.append("- `start_weights.npz`: the trained network.")
        kind = "py"
        test_what = "Your supervisor, on top of the network and under the shield,"
    else:  # transfer
        src = target_path(run, "L2")
        steps = check_converged(src)
        A = load_net(src)
        lab = ("preview_" if PREVIEW_TARGET else "") + f"target_net{run}"
        ra = dev_ref("L2", f"sb3:{src}", lab)
        states = {"fixed": record(base, "fixed", "fixed"), "actuated": record(base, "actuated", "actuated")}
        states_src = {"source_network": record("L2", f"sb3:{src}", lab)}
        n_src = SignalEnv("L2", workdir=os.path.join(V3, "runs", "tmp", "pack")).observation_space.shape[0]
        goal = ("Junction **L2W** (this problem) is a wider, busier version of junction **L2**: its E and W approaches "
                "have one more upstream lane, one more straight lane at the stop line and one more exit lane, and carry "
                "more traffic. A network was trained with PPO on L2 (`source_weights_L2.npz`). L2W's observation has "
                f"{n_obs} values instead of L2's {n_src}, so the L2 network cannot be applied to L2W. **Transfer it**: "
                f"build an L2W network ({n_obs} -> 64 -> 64 -> 4) in which **W2, b2, W3 and b3 are exactly the L2 "
                f"network's** (copied unchanged), and **you design W1 (64 x {n_obs}) and b1**. Your network is judged on "
                "two things:\n1. its own test TTS on L2W, before any training;\n"
                f"2. how fast PPO improves from it on L2W over {WARM_STEPS:,} steps (two training seeds), compared with "
                "PPO on L2W from a random start.\n\n### Junction L2, for reference\n\n" + system_md("L2") + "\n\n"
                + obs_md("L2").replace("## Observation x", "#### L2's observation x") +
                f"\n\nDev TTS of the trained L2 network on L2: {ra['all']:.1f} veh-h (9 dev episodes).\n\n"
                f"How the L2 network was trained: {training_md(steps)}\n\n"
                f"How PPO will continue from your network: the same PPO, on L2W. The critic starts random and is trained "
                f"alone for the first {WARMUP:,} environment steps while your actor is frozen; then both train.\n\n"
                + TRANSFER_TRAINABILITY)
        extra_rules = ("5. W2, b2, W3, b3 of your L2W network must equal those of `source_weights_L2.npz` exactly. "
                       "Any difference disqualifies the transfer.\n")
        trainability_note = ", including the `tools.trainability` numbers"
        test_what = "Your network"
        files.append(f"- `source_weights_L2.npz`: the PPO-trained network of junction L2 ({n_src} -> 64 -> 64 -> 4).")

    # ---- write the pack
    os.makedirs(os.path.join(d, "work"), exist_ok=True)
    spec = {"pack_id": f"{case}_{pid}", "case": case, "problem": problem, "obs_dim": n_obs, "hidden": CONFIG["hidden"],
            "obs_names": env.obs_names(), "ok_index": env.ok_index(), "eval_budget": budget,
            "dev_refs": {n: r["all"] for n, r in ref_rows}}
    json.dump(spec, open(os.path.join(d, "spec.json"), "w"), indent=1)
    shutil.copy(os.path.join(V3, "wsp", "pack_tools.py"), os.path.join(d, "tools.py"))
    if start is not None:
        np.savez(os.path.join(d, "start_weights.npz"), **{k: v.astype(np.float32) for k, v in start.items()})
    counts = save_states(os.path.join(d, "states.npz"), states)
    if case == "constrain":
        files.append(f"- `states.npz`: the network running under the rules in 6 training-like episodes "
                     f"({sum(counts.values())} decisions).\n"
                     "  - Keys: obs, act (the network's request, before the shield), episode, time_s, controller, "
                     "controller_names.\n"
                     "  - rule_state columns: since_served[0-3], legal[4-7], overdue[8-11], phase_halting[12-15], "
                     "t_green (16), green (17).")
    else:
        files.append(f"- `states.npz`: recorded observations of THIS junction from 6 training-like episodes per "
                     f"controller ({', '.join(f'{k}: {v} decisions' for k, v in counts.items())}). Keys: obs, act (the "
                     "decision taken), episode, time_s, controller (index into controller_names).")
    if case == "transfer":
        np.savez(os.path.join(d, "source_weights_L2.npz"), **{k: v.astype(np.float32) for k, v in A.items()})
        cs = save_states(os.path.join(d, "states_L2.npz"), states_src)
        files.append(f"- `states_L2.npz`: {sum(cs.values())} recorded L2 observations with the L2 network's own "
                     "decisions (L2's layout; same keys as states.npz).")
    if budget > 0:
        txt = open(os.path.join(V3, "wsp", "pack_eval.py")).read()
        txt = txt.replace('"@@PIPELINE@@"', repr(V3)).replace('"@@CALL_S@@"', str(CALL_S))
        open(os.path.join(d, "eval.py"), "w").write(txt)
        files.append("- `eval.py`: runs the simulator on your " + ("supervisor" if kind == "py" else "weights")
                     + " (budgeted, see rule 3).")
    title_refs = ("Reference controllers on the 9 dev episodes, under the rules unless noted (mean TTS, veh-h)"
                  if case == "constrain" else "Reference controllers on the 9 dev episodes (mean TTS, veh-h)")
    if case == "explain":
        tail = rules_explain(env.ok_index())
    elif case == "constrain":
        tail = rules_constrain(budget)
    else:
        tail = rules_weights(budget, extra_rules).replace("{trainability}", trainability_note).replace("{{", "{").replace("}}", "}")
    txt = (f"# Task: {TITLES[case]}\n\n## Goal\n\n{goal}\n\n"
           f"## The traffic system (SUMO microsimulation, 1 s steps)\n\n{system_md(problem)}\n\n{SIGNALS}\n\n"
           f"## The network\n\n{network_md(n_obs)}\n\n{obs_md(base)}\n\n"
           + refs_md(ref_rows, title_refs) + "\n" + CONTROLLERS + "\n\n## Files\n\n" + "\n".join(files) + "\n\n"
           + test_md(test_what) + "\n" + tail)
    open(os.path.join(d, "TASK.md"), "w").write(normalise(txt))
    precompile(d)
    print("wrote", d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(TITLES))
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--rep", type=int, default=1, help="replicate number (>= 2 builds an identical pack under a new id)")
    a = ap.parse_args()
    if a.case in ("init", "transfer", "explain"):
        assert a.budget == 0, "init, transfer and explain are blind"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    build(a.case, a.run, a.budget, a.rep)


if __name__ == "__main__":
    main()
