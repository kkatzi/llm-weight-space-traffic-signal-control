import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, tools
from collections import Counter
from build import build, ideal_logits, H_coef, LAM
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > .5; ctl = S["controller"]
H = X @ H_coef().T; ph = X[:, 40:44].argmax(1); I = np.arange(len(X))
lg0 = tools.forward(w0, X); g0 = lg0.argmax(1)
for G in [0, 5, 10]:
    for alpha, M in [(0.5, 2.5), (0.75, 2.5), (1.0, 2.5), (1.0, 3.5)]:
        w = build(w0, alpha=alpha, M=M, G=G, X_mean=X[ok], k_units=(31, 27, 59), j_units=(25, 35, 41))
        g1 = tools.greedy(w, X)
        m = ok & (ctl == 0); ch = m & (g1 != g0)
        low = ch & (H.sum(1) <= 2)
        dH = H[I, g1] - H[I, g0]
        print(f"G={G:2d} a={alpha} M={M}: changed {ch.sum():3d} ({ch.sum()/m.sum():.3f}) low-traffic changes {low.sum():2d}  "
              f"dH<0 changes {(ch & (dH < 0)).sum():2d}  mean dH {dH[ch].mean():.2f}  holds {(g1[m]==ph[m]).sum()} (orig {(g0[m]==ph[m]).sum()})  picks {np.bincount(g1[m], minlength=4)}")
