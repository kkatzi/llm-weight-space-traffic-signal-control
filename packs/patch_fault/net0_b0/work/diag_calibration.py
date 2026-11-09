"""Calibration check of the mirror proxy on the N/S pair (both pockets observed): does replacing S2 (or N2) by the
mirror pocket's halting count keep the share of NS_left choices of the original network, or inflate it?"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import tools
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(float); t = S["time_s"]
ok = (X[:, 44] > 0.5) & (t >= 300) & (t <= 3600)
I = tools.names().index
for tgt, prx in [("S2", "N2"), ("N2", "S2")]:
    out = {}
    for mode in ["true", "zero", "proxy(q)"]:
        Y = X.copy()
        if mode == "zero":
            Y[:, I("q_" + tgt)] = 0; Y[:, I("d_" + tgt)] = 0
        if mode == "proxy(q)":
            Y[:, I("q_" + tgt)] = X[:, I("q_" + prx)]; Y[:, I("d_" + tgt)] = X[:, I("q_" + prx)]
        a = tools.greedy(w0, Y[ok])
        out[mode] = np.bincount(a, minlength=4) / ok.sum()
    print("hide %s: share of choices (in-hour honoured)  " % tgt, {k: np.round(v, 3) for k, v in out.items()})
