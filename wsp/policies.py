# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Controllers behind one interface: act(obs, env) -> int array of K phases.

    MLPPolicy(w)          greedy actor of an SB3 MlpPolicy (tanh, 2 hidden layers); w = dict W1 b1 W2 b2 W3 b3
                          (W: [out, in], as in torch). Trained models and agent-written npz files go through the
                          same code.
    FixedTime(greens)     cycle 0 -> 1 -> 2 -> 3; greens[p] = target green of phase p (s)
    Actuated(ext, max_green, layout)
                          after min green: extend the green while at least `ext` vehicles are moving (on the network,
                          not halting) on its lanes and the green is shorter than max_green; otherwise switch to the
                          phase whose lanes hold the most halting vehicles (keep the green if nothing waits elsewhere).
                          layout "stop": the stop-line lanes; "up": the stop-line lanes plus the upstream lanes that
                          feed them. These are the lanes the PPO observation covers.
    MaskedPolicy(w)       the network's highest-scoring legal phase under the rules (Constrain baseline)
    SupervisorPolicy      a supervisor file decide(x, logits, state) -> phase on top of a network (Constrain)
    RulesPolicy           a rules file decide(x) -> phase (Explain)
    HoldWrapper           execution-time hold rule on top of a network (exploratory Patch analysis)

make_policy(spec, problem):
    "npz:PATH" | "sb3:PATH" | "fixed" | "fixed:g0,g1,g2,g3" | "actuated" | "actuated:ext,max_green,layout" |
    "masked:NET" | "super:FILE.py|NET" | "rules:FILE.py" | "hold:k,max_hold|NET"
    "fixed" / "actuated" without parameters use the tuned settings in results/tuned/<problem>.json.
"""
import io
import json
import os
import zipfile

import numpy as np

from wsp.env import MIN_GREEN_S, N_PHASES
from wsp.junctions import V3
from wsp.paths import norm_spec

ACTOR_KEYS = {"W1": "mlp_extractor.policy_net.0.weight", "b1": "mlp_extractor.policy_net.0.bias",
              "W2": "mlp_extractor.policy_net.2.weight", "b2": "mlp_extractor.policy_net.2.bias",
              "W3": "action_net.weight", "b3": "action_net.bias"}


def forward(w, X):
    """Logits for a batch of observations X [N, obs]."""
    h = np.tanh(X @ w["W1"].T + w["b1"])
    h = np.tanh(h @ w["W2"].T + w["b2"])
    return h @ w["W3"].T + w["b3"]


def load_npz(path):
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in ACTOR_KEYS}


def extract_actor(sb3_zip):
    import torch
    with zipfile.ZipFile(sb3_zip) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
    return {k: sd[v].double().numpy() for k, v in ACTOR_KEYS.items()}


def load_net(path):
    return load_npz(path) if path.endswith(".npz") else extract_actor(path)


class MLPPolicy:
    def __init__(self, w):
        self.w = w

    def act(self, obs, env=None):
        lg = forward(self.w, np.asarray(obs, dtype=np.float64)[None])[0]
        return lg.reshape(-1, N_PHASES).argmax(axis=1)


class FixedTime:
    def __init__(self, greens):
        self.greens = [float(g) for g in greens]

    def act(self, obs, env):
        out = env.green.copy()
        for k in range(env.K):
            if env.t_green[k] >= max(self.greens[env.green[k]], MIN_GREEN_S):
                out[k] = (env.green[k] + 1) % N_PHASES
        return out


class Actuated:
    def __init__(self, ext=2.0, max_green=60.0, layout="stop"):
        assert layout in ("stop", "up"), layout
        self.ext, self.max_green, self.layout = float(ext), float(max_green), layout

    def act(self, obs, env):
        lane = env._sumo.lane
        out = env.green.copy()
        for k in range(env.K):
            lanes = env.phase_lanes[k] if self.layout == "stop" else env.phase_lanes_ext[k]
            g = int(env.green[k])
            if env.t_green[k] < MIN_GREEN_S:
                continue
            failed = getattr(env, "failed_lanes", set())       # a failed detector reads 0
            halt = {ln: (0 if ln in failed else lane.getLastStepHaltingNumber(ln)) for p in range(N_PHASES)
                    for ln in lanes[p]}
            moving = sum((0 if ln in failed else lane.getLastStepVehicleNumber(ln)) - halt[ln] for ln in lanes[g])
            if moving >= self.ext and env.t_green[k] < self.max_green:
                continue
            q = np.array([sum(halt[ln] for ln in lanes[p]) for p in range(N_PHASES)], float)
            q[g] = -1.0
            if q.max() > 0:
                out[k] = int(q.argmax())
        return out


class MaskedPolicy:
    def __init__(self, w):
        self.w = w

    def act(self, obs, env):
        lg = forward(self.w, np.asarray(obs, dtype=np.float64)[None])[0]
        legal = np.array(env.shield_state(0)["legal"])
        return np.array([int(np.argmax(np.where(legal, lg, -np.inf)))])


def _load_module(path, tag):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"{tag}_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SupervisorPolicy:
    """decide(x, logits, state) -> phase; state = env.shield_state(0). The env's shield still applies afterwards."""

    def __init__(self, path, w):
        self.decide, self.w = _load_module(path, "sup").decide, w

    def act(self, obs, env):
        x = np.asarray(obs, dtype=np.float64)
        lg = forward(self.w, x[None])[0]
        return np.array([int(self.decide(x, lg.copy(), env.shield_state(0)))])


class RulesPolicy:
    def __init__(self, path):
        self.decide = _load_module(path, "rules").decide

    def act(self, obs, env=None):
        return np.array([int(self.decide(np.asarray(obs, dtype=np.float64)))])


class HoldWrapper:
    """The network decides, except that a request to leave the current green is refused while the green has lasted
    less than max_hold seconds and at least k vehicles halt on the stop-line lanes it serves (read from the network's
    own observation: queue x lane capacity)."""

    def __init__(self, w, k, max_hold):
        self.w, self.k, self.max_hold = w, float(k), float(max_hold)
        self._idx = None

    def act(self, obs, env):
        x = np.asarray(obs, dtype=np.float64)
        a = forward(self.w, x[None])[0].reshape(-1, N_PHASES).argmax(axis=1)
        if self._idx is None:
            lanes = env.lanes[0]
            self._idx = [[lanes.index(ln) for ln in env.phase_lanes[0][p]] for p in range(N_PHASES)]
            self._cap = np.asarray(env.caps[0], float)
        g, tg = int(env.green[0]), float(env.t_green[0])
        if a[0] != g and MIN_GREEN_S <= tg < self.max_hold:
            if sum(x[i] * self._cap[i] for i in self._idx[g]) >= self.k - 1e-9:
                a[0] = g
        return a


def tuned(problem, kind):
    path = os.path.join(V3, "results", "tuned", f"{problem}.json")
    with open(path) as f:
        return json.load(f)[kind]["spec"]


def make_policy(spec, problem=None):
    if spec in ("fixed", "actuated"):
        spec = tuned(problem, spec)
    spec = norm_spec(spec)
    if spec.startswith("npz:") or spec.startswith("sb3:"):
        return MLPPolicy(load_net(spec[4:]))
    if spec.startswith("fixed:"):
        return FixedTime([float(x) for x in spec[6:].split(",")])
    if spec.startswith("actuated:"):
        e, m, layout = spec[9:].split(",")
        return Actuated(float(e), float(m), layout)
    if spec.startswith("masked:"):
        return MaskedPolicy(load_net(spec[7:]))
    if spec.startswith("super:"):
        sup, net = spec[6:].split("|")
        return SupervisorPolicy(sup, load_net(net))
    if spec.startswith("rules:"):
        return RulesPolicy(spec[6:])
    if spec.startswith("hold:"):
        par, net = spec[5:].split("|")
        k, mh = (float(v) for v in par.split(","))
        return HoldWrapper(load_net(net), k, mh)
    raise ValueError(spec)
