"""Statistic used to set P_BLOCK: on recorded in-hour blockage states (phase 2 green, ok, a halted vehicle on W1u;
W2 is then full with p~0.99 by the mirror-side statistics), how much does the ORIGINAL network's EW_left logit gain
relative to its best other logit when W2 goes from the proxy value (q_E2) to full (1)? Forward passes only."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import tools
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(float); t = S["time_s"]
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
bs = (ph == 2) & ok & (X[:, 31] > 0) & (t <= 3600)
Yp = X[bs].copy(); Yp[:, 11] = Yp[:, 5]; Yp[:, 23] = Yp[:, 5]
Yf = X[bs].copy(); Yf[:, 11] = 1; Yf[:, 23] = 1
lp, lf = tools.forward(w0, Yp), tools.forward(w0, Yf)
d3 = lf[:, 3] - lp[:, 3]
dmargin = (lf[:, 3] - np.delete(lf, 3, axis=1).max(1)) - (lp[:, 3] - np.delete(lp, 3, axis=1).max(1))
print("n=%d  mean dl3=%.2f median=%.2f | mean d(l3-l2)=%.2f median=%.2f | mean d(margin)=%.2f median=%.2f" %
      (bs.sum(), d3.mean(), np.median(d3), (d3 - (lf[:, 2] - lp[:, 2])).mean(), np.median(d3 - (lf[:, 2] - lp[:, 2])),
       dmargin.mean(), np.median(dmargin)))
# split by the proxy's value: the push only matters where the proxy under-reports W2
for lv in [0, 1, 2, 3]:
    m = np.round(X[bs, 5] * 2.987).astype(int) == lv
    if m.sum():
        print("  q_E2 = %d cars: n=%3d  mean d(margin)=%.2f  median=%.2f  q25=%.2f q75=%.2f" %
              (lv, m.sum(), dmargin[m].mean(), np.median(dmargin[m]), np.quantile(dmargin[m], .25), np.quantile(dmargin[m], .75)))
