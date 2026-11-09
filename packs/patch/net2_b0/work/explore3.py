import numpy as np, sys
sys.path.insert(0, '.')
import tools
from collections import Counter
np.set_printoptions(linewidth=200, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); act = S["act"]; ctl = S["controller"]; ep = S["episode"]; t = S["time_s"]
print(np.unique(np.round(X[:,0]*2.987,2)))
print(np.unique(np.round(X[:,24]*256/7.5,2))[:20])
print("t_green unique", np.unique(np.round(X[:,45]*60,1))[:40])
print("ok vs tgreen", Counter(zip(X[:,44].round(), (X[:,45]*60).round())).most_common(30))
