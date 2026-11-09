"""Analog test v2: hide N2 or S2, rebuild it with hand-written schemes q_hat = c + a*q_opp + k*q_up + tau*t_green,
compare the start network's decisions with real vs rebuilt values (honoured states, drain-deadlock states removed)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
S = tools.states(); X = S["obs"].astype(float); I = {k: i for i, k in enumerate(tools.names())}
w = tools.load("start_weights.npz")
dead = np.zeros(len(X), bool)
for c, e in [(0, 0), (0, 5), (2, 5)]:
    dead |= (S["controller"] == c) & (S["episode"] == e) & (S["time_s"] > 3700)
ok = (X[:, 44] > 0.5) & ~dead
rate = 0.6  # N/S left arrivals relative to E/W
schemes = {  # name: (c_q, c_d, a, k, tau)
    "zero": (0, 0, 0, 0, 0), "half": (0.15, 0.2, 0.5, 0, 0), "A-like": (0.05, 0.1, 0.5, 3, 0.4),
    "A-noTau": (0.1, 0.15, 0.5, 3, 0), "a.75": (0.05, 0.1, 0.75, 3, 0.4), "copy+k3+tau": (0, 0.05, 1.0, 3, 0.4),
    "copy": (0, 0, 1, 0, 0), "a.75-noTau": (0.08, 0.12, 0.75, 3, 0), "A-k6": (0.05, 0.1, 0.5, 6, 0.4),
}
for T, O, U in [("N2", "S2", "N1u"), ("S2", "N2", "S1u")]:
    for c in [0, 1, 2]:
        m = ok & (S["controller"] == c); Y = X[m]; ref = tools.greedy(w, Y)
        out = []
        for nm, (cq, cd, a, k, tau) in schemes.items():
            Z = Y.copy()
            Z[:, I["q_" + T]] = cq + a * Y[:, I["q_" + O]] + k * Y[:, I["q_" + U]] + tau * rate * Y[:, 45]
            Z[:, I["d_" + T]] = cd + a * Y[:, I["d_" + O]] + k * Y[:, I["q_" + U]] + tau * rate * Y[:, 45]
            out.append(f"{nm}={(tools.greedy(w, Z) == ref).mean():.3f}")
        print(T, c, m.sum(), " ".join(out))
