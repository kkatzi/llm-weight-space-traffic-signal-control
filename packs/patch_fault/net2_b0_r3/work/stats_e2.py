import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]
ph = X[:, 40:44].argmax(1)
for ci, cn in enumerate(S["controller_names"]):
    m = c == ci
    o = m & (ph != 3); t3 = m & (ph == 3)
    print(f"{cn:13s} phases0-2: qE2 {X[o,5].mean():.3f} dE2 {X[o,17].mean():.3f} | phase3: qE2 {X[t3,5].mean():.3f} dE2 {X[t3,17].mean():.3f}"
          f" | phase3 by t_green: " + " ".join(f"{tg:.2f}:{X[t3 & np.isclose(X[:,45],tg,atol=0.01),5].mean():.2f}/{X[t3 & np.isclose(X[:,45],tg,atol=0.01),17].mean():.2f}" for tg in [0.033,0.117,0.2,0.283,0.367]))
