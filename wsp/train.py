# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""PPO training on junctions L2, L2W, L2_rules and L2_fault (the paper's algorithm and hyper-parameters).

    python -m wsp.train --problem L2 --name ppo_s100 --seed 100 --steps 3000000
    # start from a given actor (agent-written, transferred), critic warm-up first:
    python -m wsp.train --problem L2 --name init_r1_s400 --seed 400 --init-actor net.npz --critic-warmup 50000
    # continue a trained model (actor + critic), e.g. fine-tuning or under the rules:
    python -m wsp.train --problem L2_rules --name ft_s700 --seed 700 --init-model models/L2/ppo_s100/final.zip

Configuration (fixed): SB3 PPO, separate actor and critic MLPs 64-64 (tanh), lr 3e-4, entropy 0.01,
gamma 0.99, GAE lambda 0.95, clip 0.2, 4 parallel envs x 720 steps per update, minibatch 720, 10 epochs,
value coefficient 0.5, max grad norm 0.5, rewards normalised (VecNormalize, observations raw). Observation "stdup",
reward "delay"; training episodes drain like evaluation episodes (cap +30 min). Env seeds: seed x 1000 + rank.

--critic-warmup N freezes the actor for the first N environment steps, so that the randomly initialised critic learns
the given policy's values before policy-gradient updates start; those steps count toward --steps.

Writes models/<problem>/<name>/: ckpt_<steps>_steps.zip every --ckpt-every steps (plus ckpt_0_steps.zip when an
initial actor or model is given), final.zip, monitor.csv (raw training-episode returns and lengths), config.json.
Refuses to reuse a folder.
"""
import argparse
import json
import os
import sys

V3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, V3)
sys.path.insert(0, os.path.dirname(V3))

CONFIG = dict(lr=3e-4, ent_coef=0.01, gamma=0.99, gae_lambda=0.95, clip_range=0.2, n_envs=4, n_steps=720,
              batch_size=720, n_epochs=10, vf_coef=0.5, max_grad_norm=0.5, hidden=64, obs="stdup", reward="delay",
              drain=True, drain_cap=1800)


def make_env(problem, seed, rank, out):
    def _f():
        from wsp.env import SignalEnv
        return SignalEnv(problem, scenario="train", seed=seed * 1000 + rank, drain=CONFIG["drain"],
                         drain_cap=CONFIG["drain_cap"], obs=CONFIG["obs"], reward=CONFIG["reward"],
                         workdir=os.path.join(V3, "runs", "train", f"{os.path.basename(out)}_{rank}_{os.getpid()}"))
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--ckpt-every", type=int, default=100_000)
    ap.add_argument("--init-actor", default=None, help="npz with W1 b1 W2 b2 W3 b3 for the actor")
    ap.add_argument("--init-model", default=None, help="SB3 zip to continue (actor + critic)")
    ap.add_argument("--critic-warmup", type=int, default=0)
    a = ap.parse_args()

    import numpy as np
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
    from wsp.policies import ACTOR_KEYS

    out = os.path.join(V3, "models", a.problem, a.name)
    if os.path.exists(os.path.join(out, "config.json")):
        sys.exit(f"{out} exists; pick another --name")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(dict(vars(a), **CONFIG), f, indent=1)
    c = CONFIG
    venv = SubprocVecEnv([make_env(a.problem, a.seed, i, out) for i in range(c["n_envs"])], start_method="fork")
    venv = VecMonitor(venv, os.path.join(out, "monitor.csv"))
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=c["gamma"])
    if a.init_model:
        model = PPO.load(a.init_model, env=venv, device="cpu", seed=a.seed)
    else:
        model = PPO("MlpPolicy", venv, n_steps=c["n_steps"], batch_size=c["batch_size"], n_epochs=c["n_epochs"],
                    learning_rate=c["lr"], gamma=c["gamma"], gae_lambda=c["gae_lambda"], clip_range=c["clip_range"],
                    ent_coef=c["ent_coef"], vf_coef=c["vf_coef"], max_grad_norm=c["max_grad_norm"], seed=a.seed,
                    device="cpu", verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[c["hidden"]] * 2, vf=[c["hidden"]] * 2)))
    if a.init_actor:
        w = np.load(a.init_actor)
        sd = model.policy.state_dict()
        for k, key in ACTOR_KEYS.items():
            t = torch.as_tensor(w[k], dtype=sd[key].dtype)
            assert t.shape == sd[key].shape, (k, t.shape, sd[key].shape)
            sd[key] = t
        model.policy.load_state_dict(sd)
    if a.init_actor or a.init_model:
        model.save(os.path.join(out, "ckpt_0_steps.zip"))
    actor = [model.policy.mlp_extractor.policy_net, model.policy.action_net]

    class CriticWarmup(BaseCallback):
        def __init__(self, steps):
            super().__init__()
            self.steps, self.done = steps, steps <= 0

        def _set(self, flag):
            for m in actor:
                for p in m.parameters():
                    p.requires_grad_(flag)

        def _on_training_start(self):
            if not self.done:
                self._set(False)
                print(f"critic warm-up: actor frozen for {self.steps} steps", flush=True)

        def _on_step(self):
            if not self.done and self.num_timesteps >= self.steps:
                self._set(True)
                self.done = True
                print(f"critic warm-up over at {self.num_timesteps} steps", flush=True)
            return True

    every = max(a.ckpt_every // c["n_envs"], 1)
    model.learn(total_timesteps=a.steps, reset_num_timesteps=True,
                callback=CallbackList([CriticWarmup(a.critic_warmup), CheckpointCallback(every, out, name_prefix="ckpt")]))
    model.save(os.path.join(out, "final.zip"))
    venv.close()
    print("done", out)


if __name__ == "__main__":
    main()
