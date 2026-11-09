"""Spill-back picture: upstream queues per lane over the recorded run, by episode and by time in the episode."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import tools

S = tools.states()
X = S["obs"].astype(np.float64)
ep = S["episode"]; t = S["time_s"]
rs = S["rule_state"]; ph = rs[:, 12:16]
CAP_U = 256 / 7.5
names = tools.names()
up = X[:, 24:32] * CAP_U
print("upstream halting vehicles, mean per lane:", dict(zip(names[24:32], up.mean(0).round(2))))
print("share of states with any upstream halting > 3 veh:", (up.max(1) > 3).mean().round(3))
for e in np.unique(ep):
    m = ep == e
    tot = ph[m].sum(1)
    print(f"ep {e}: mean total stop-line halting {tot.mean():.1f}, max upstream {up[m].max():.1f}, "
          f"share upstream>3: {(up[m].max(1) > 3).mean():.3f}, end {t[m].max()} s")
# by quarter-hour
for k in range(5):
    m = (t >= k * 900) & (t < (k + 1) * 900)
    print(f"t {k*15}-{(k+1)*15} min: mean stop-line halting {ph[m].sum(1).mean():.1f}, mean upstream halting {up[m].sum(1).mean():.1f}")
