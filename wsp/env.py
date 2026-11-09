# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""SignalEnv for junctions L2 and L2W: one controller choosing the green phase of junction J0.

Control: one decision every 5 s; 4 protected phases (0 NS through, 1 NS left, 2 EW through,
3 EW left); if the requested phase differs from the current green and the green has run >= 10 s (min green),
the junction switches through 3 s of yellow followed by 2 s of the new green; a switch request before min green
is ignored.

Observation ("stdup", the paper's layout with stop-line and upstream lanes; all values in [0, 1]):
    queue[n]       halting vehicles / lane capacity (length / 7.5 m), per stop-line lane
    density[n]     vehicles / lane capacity, per stop-line lane
    up_queue[u]    the same for the upstream lanes (the two-lane road before the pocket)
    up_density[u]
    phase[4]       one-hot of the current green
    ok             1 if the minimum green has elapsed (a switch request would be honoured)
    t_green        time in the current green / 60 s, clipped to 1
Lane order: approach N, E, S, W; lane 0 (rightmost) first. L2: n = 12, u = 8, 46 values; L2W: n = 14, u = 10, 54.
"std" is the same without the upstream block.

Reward ("delay"): -(sum over network lanes of n * (1 - mean speed / speed limit) + vehicles waiting to enter),
averaged over the 5 s and divided by 10. Its episode sum is the total delay; TTS differs from it only by the
free-flow travel time of the vehicles served, which no controller can change.

Episodes: 3600 s of arrivals, then (drain=True, used for training and evaluation alike) until the network is empty,
capped at 3600 + drain_cap s. Teleporting is disabled. With drain=True the env accumulates TTS (vehicles in the
network plus vehicles waiting to enter, every second).

Rules (problem "<junction>_rules", the Constrain study; enforced by a shield inside the env): through phases need >= 15 s of green, left phases >= 10 s; a green may not run past 45 s while another
phase has halting vehicles; a phase with halting vehicles must get green within 90 s of its last green. An illegal
request is replaced by the legal phase with the most halting vehicles (on its stop-line lanes).

Failed detector (problem "<junction>_fault", the Patch study under a fault, EXPERIMENT_PLAN.md amendment P-F): one
lane detector reads 0, i.e. that lane's queue and density inputs are 0 in the observation, and controllers that read
the detectors (wsp.policies.Actuated) see 0 halting and 0 vehicles on it. The failed lane is results/fault.json
"lane" unless given explicitly (fault=...).
"""
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from wsp import junctions as J

DECISION_S, YELLOW_S, MIN_GREEN_S = 5, 3, 10
N_PHASES = 4
LANE_CAP_M = 7.5
REWARD_SCALE = 10.0
HORIZON = 3600
RULES = dict(min_green=[15, 10, 15, 10], max_green=45, max_wait=90)


class SignalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, problem, scenario="train", seed=0, drain=True, drain_cap=1800, statistic_output=None,
                 workdir=None, obs="stdup", reward="delay", fault=None):
        assert obs in ("std", "stdup") and reward in ("delay", "halting"), (obs, reward)
        self.problem, self.scenario, self.drain, self.drain_cap = problem, scenario, drain, drain_cap
        self.obs_kind, self.reward_kind = obs, reward
        self.rules = problem.endswith("_rules")
        self.meta = J.load_meta(problem)
        self.net = J.net_path(problem)
        T = self.meta["tls"]
        self.K = len(T)
        self.lanes = [t["in_lanes"] for t in T]
        self.lane_len = [np.array(t["lane_length"]) for t in T]
        self.caps = [L / LANE_CAP_M for L in self.lane_len]
        self.all_lanes = [ln for t in T for ln in t["in_lanes"]]
        self.phase_lanes = [[sorted({l["in_lane"] for l, c in zip(t["links"], t["green"][p]) if c == "G"})
                             for p in range(N_PHASES)] for t in T]
        self.up_lanes = [t["up_lanes"] for t in T]
        self.up_caps = [np.array(t["up_length"]) / LANE_CAP_M for t in T]
        # phase lanes extended with the upstream lanes that feed them (a shared upstream lane counts for each)
        self.phase_lanes_ext = []
        for t, pls in zip(T, self.phase_lanes):
            feeds = t.get("feeds", {})
            self.phase_lanes_ext.append([sorted(set(pl) | {u for u, ins in feeds.items() if set(ins) & set(pl)})
                                         for pl in pls])
        self.failed_lanes, self._zero_idx = set(), []
        if problem.endswith("_fault"):
            if fault is None:
                import json as _json
                fault = _json.load(open(os.path.join(J.V3, "results", "fault.json")))["lane"]
            self.failed_lanes = {fault}
            n, u = len(self.lanes[0]), len(self.up_lanes[0])
            if fault in self.lanes[0]:
                i = self.lanes[0].index(fault)
                self._zero_idx = [i, n + i]
            else:
                j = self.up_lanes[0].index(fault)
                self._zero_idx = [2 * n + j, 2 * n + u + j]
        self.net_lanes = sorted(self.meta["lanes"])
        self.net_speed = np.array([self.meta["lanes"][ln]["speed"] for ln in self.net_lanes])
        self.pend_app = J.pending_approach(problem)
        size = sum(2 * len(ls) + (2 * len(u) if obs == "stdup" else 0) + N_PHASES + 2
                   for ls, u in zip(self.lanes, self.up_lanes))
        self.observation_space = spaces.Box(0.0, 1.0, (size,), np.float32)
        self.action_space = spaces.MultiDiscrete([N_PHASES] * self.K)
        self.rng = np.random.default_rng(seed)
        self.statistic_output = statistic_output
        self.workdir = workdir or os.path.join(J.V3, "runs", "tmp", f"env_{os.getpid()}")
        os.makedirs(self.workdir, exist_ok=True)
        self._sumo = None
        self.fixed_sumo_seed = None

    # ---- SUMO lifecycle
    def _start(self, sumo_seed, routes):
        import libsumo
        self._sumo = libsumo
        args = ["sumo", "-n", self.net, "-r", routes, "--seed", str(sumo_seed), "--no-step-log", "true",
                "--no-warnings", "true", "--time-to-teleport", "-1", "--step-length", "1"]
        if self.statistic_output:
            args += ["--statistic-output", self.statistic_output, "--duration-log.statistics", "true"]
        libsumo.start(args)

    def close(self):
        if self._sumo is not None:
            self._sumo.close()
            self._sumo = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.close()
        sc = J.sample_train(self.problem, self.rng) if self.scenario == "train" else J.scenario(self.problem, self.scenario)
        self.scenario_params = sc
        routes = os.path.join(self.workdir, "routes.rou.xml")
        J.write_routes(self.problem, sc, routes)
        sumo_seed = int(self.rng.integers(0, 2**31 - 1)) if self.fixed_sumo_seed is None else self.fixed_sumo_seed
        self._start(sumo_seed, routes)
        self.green = np.zeros(self.K, dtype=int)
        self.t_green = np.zeros(self.K)
        for t in self.meta["tls"]:
            self._sumo.trafficlight.setRedYellowGreenState(t["id"], t["green"][0])
        self.time = 0
        self.teleports = 0
        self.halting_sum = 0.0
        self.tts = 0.0
        self.delay_sum = 0.0
        self.switches = np.zeros(self.K, dtype=int)
        self.ignored = np.zeros(self.K, dtype=int)
        self.overrides = np.zeros(self.K, dtype=int)
        self.last_served = np.zeros((self.K, N_PHASES))
        return self._obs(), {}

    # ---- deployment rules (shield)
    def phase_halting(self, k=0):
        lane = self._sumo.lane
        return np.array([sum(lane.getLastStepHaltingNumber(ln) for ln in self.phase_lanes[k][p])
                         for p in range(N_PHASES)], float)

    def shield_state(self, k=0):
        """What the rules allow right now at junction k (meaningful when self.rules)."""
        g, tg = int(self.green[k]), float(self.t_green[k])
        since = self.time - self.last_served[k].copy()
        since[g] = 0.0
        halt = self.phase_halting(k)
        overdue = np.array([p != g and halt[p] > 0 and since[p] >= RULES["max_wait"] for p in range(N_PHASES)])
        legal = np.ones(N_PHASES, bool)
        if tg < max(RULES["min_green"][g], MIN_GREEN_S):
            legal[:] = False
            legal[g] = True
        else:
            others_waiting = any(halt[p] > 0 for p in range(N_PHASES) if p != g)
            if tg >= RULES["max_green"] and others_waiting:
                legal[g] = False
            if overdue.any():
                p = int(np.argmax(np.where(overdue, since, -1)))
                legal[:] = False
                legal[p] = True
        return {"green": g, "t_green": tg, "since_served": since.tolist(), "phase_halting": halt.tolist(),
                "legal": legal.tolist(), "overdue": overdue.tolist(), "rules": RULES}

    def _shield(self, a):
        for k in range(self.K):
            st = self.shield_state(k)
            legal = np.array(st["legal"])
            if not legal[a[k]]:
                if self.t_green[k] >= MIN_GREEN_S:      # a request that would otherwise take effect
                    self.overrides[k] += 1
                halt = np.array(st["phase_halting"])
                cand = np.where(legal)[0]
                a[k] = int(cand[np.argmax(halt[cand])]) if halt[cand].max() > 0 else int(cand[0])
        return a

    # ---- dynamics
    def step(self, action):
        a = np.asarray(action, dtype=int).reshape(-1).copy()
        assert a.shape == (self.K,), a.shape
        if self.rules:
            a = self._shield(a)
        tl, lane, sim = self._sumo.trafficlight, self._sumo.lane, self._sumo.simulation
        T = self.meta["tls"]
        switching = []
        for k, t in enumerate(T):
            if a[k] != self.green[k]:
                if self.t_green[k] >= MIN_GREEN_S:
                    switching.append(k)
                    tl.setRedYellowGreenState(t["id"], t["yellow"][self.green[k]])
                else:
                    self.ignored[k] += 1
        halt = delay = 0.0
        for s in range(DECISION_S):
            if s == YELLOW_S:
                for k in switching:
                    tl.setRedYellowGreenState(T[k]["id"], T[k]["green"][a[k]])
            self._sumo.simulationStep()
            self.teleports += sim.getStartingTeleportNumber()
            halt += sum(lane.getLastStepHaltingNumber(ln) for ln in self.all_lanes)
            pend = len(sim.getPendingVehicles())
            if self.drain:
                self.tts += self._sumo.vehicle.getIDCount() + pend
            if self.reward_kind == "delay":
                d = 0.0
                for ln, vmax in zip(self.net_lanes, self.net_speed):
                    c = lane.getLastStepVehicleNumber(ln)
                    if c:
                        d += c * (1.0 - lane.getLastStepMeanSpeed(ln) / vmax)
                delay += d + pend
        self.time += DECISION_S
        self.halting_sum += halt
        self.delay_sum += delay
        for k in range(self.K):
            if k in switching:
                self.green[k], self.t_green[k] = a[k], DECISION_S - YELLOW_S
                self.switches[k] += 1
            else:
                self.t_green[k] += DECISION_S
            self.last_served[k, self.green[k]] = self.time
        cost = delay if self.reward_kind == "delay" else halt
        reward = -(cost / DECISION_S / self.K) / REWARD_SCALE
        terminated = False
        if self.time >= HORIZON:
            if not self.drain:
                terminated = True
            elif sim.getMinExpectedNumber() == 0 or self.time >= HORIZON + self.drain_cap:
                terminated = True
        info = self.episode_info() if terminated else {}
        return self._obs(), float(reward), terminated, False, info

    def episode_info(self):
        return {"time": self.time, "teleports": int(self.teleports), "tts_veh_h": self.tts / 3600.0,
                "halting_veh_h": self.halting_sum / 3600.0, "delay_veh_h": self.delay_sum / 3600.0,
                "remaining": int(self._sumo.simulation.getMinExpectedNumber()),
                "switches": self.switches.tolist(), "ignored": self.ignored.tolist(),
                "overrides": self.overrides.tolist()}

    def _obs(self):
        lane = self._sumo.lane
        parts = []
        for k, L in enumerate(self.lanes):
            parts.append(np.array([lane.getLastStepHaltingNumber(ln) for ln in L]) / self.caps[k])
            parts.append(np.array([lane.getLastStepVehicleNumber(ln) for ln in L]) / self.caps[k])
            if self.obs_kind == "stdup" and self.up_lanes[k]:
                up = self.up_lanes[k]
                parts.append(np.array([lane.getLastStepHaltingNumber(ln) for ln in up]) / self.up_caps[k])
                parts.append(np.array([lane.getLastStepVehicleNumber(ln) for ln in up]) / self.up_caps[k])
            ph = np.zeros(N_PHASES)
            ph[self.green[k]] = 1.0
            parts += [ph, [float(self.t_green[k] >= MIN_GREEN_S), min(self.t_green[k] / 60.0, 1.0)]]
        x = np.clip(np.concatenate(parts), 0, 1).astype(np.float32)
        if self._zero_idx:
            x[self._zero_idx] = 0.0
        return x

    def obs_names(self, k=0):
        """Human-readable names of the observation entries (task packs, Explain baselines)."""
        t = self.meta["tls"][k]
        lane_tag = [f"{a}{ln.rsplit('_', 1)[1]}" for a, ln in zip(t["lane_approach"], t["in_lanes"])]
        up_tag = [f"{ln[1]}{ln.rsplit('_', 1)[1]}u" for ln in self.up_lanes[k]] if self.obs_kind == "stdup" else []
        return ([f"q_{x}" for x in lane_tag] + [f"d_{x}" for x in lane_tag] + [f"q_{x}" for x in up_tag]
                + [f"d_{x}" for x in up_tag] + [f"phase{p}" for p in range(N_PHASES)] + ["ok", "t_green"])

    def ok_index(self, k=0):
        return self.obs_names(k).index("ok")
