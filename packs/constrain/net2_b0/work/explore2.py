import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
lg = tools.forward(w, X)
act = S["act"].astype(int)
rs = S["rule_state"]
since = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; overdue = rs[:, 8:12] > 0.5; halt = rs[:, 12:16]
tg = rs[:, 16]; green = rs[:, 17].astype(int)
ep = S["episode"]; t = S["time_s"]
N = len(X)
# consistency
print("phase onehot matches green:", (X[:, 40:44].argmax(1) == green).mean())
print("t_green obs vs state:", np.abs(X[:, 45]*60 - np.minimum(tg, 60)).max())
# requested legal?
req_legal = legal[np.arange(N), act]
print("request legal share:", req_legal.mean(), " illegal count:", (~req_legal).sum())
# classify illegal requests
switch_req = act != green
print("switch requests:", switch_req.sum())
ok = X[:, 44] > 0.5
print("ok share", ok.mean())
# reasons
cur_legal = legal[np.arange(N), green]
n_legal = legal.sum(1)
any_over = overdue.any(1)
print("states with any overdue:", any_over.sum())
print("states where current green illegal (max green or overdue forced):", (~cur_legal).sum())
# illegal by type
il = ~req_legal
cats = {}
for i in np.where(il)[0]:
    if act[i] != green[i] and n_legal[i] == 1 and cur_legal[i] and not any_over[i]:
        c = "min_green_hold"
    elif act[i] == green[i] and not any_over[i]:
        c = "max_green_forced"
    elif any_over[i]:
        c = "overdue_" + ("stay" if act[i] == green[i] else "switch_other")
    else:
        c = "other"
    cats[c] = cats.get(c, 0) + 1
print(cats)
# min green hold: by green and tg and ok
mg = il & (act != green) & cur_legal & (n_legal == 1)
print("min green blocked by (green,tg):")
from collections import Counter
print(Counter(zip(green[mg], tg[mg].astype(int), ok[mg])))
# overdue patterns
print("overdue patterns counter:", Counter(map(tuple, overdue[any_over].astype(int))))
print("legal patterns when overdue:", Counter(map(tuple, legal[any_over].astype(int))))
print("tg distribution when any overdue:", Counter(tg[any_over].astype(int)))
# legal pattern overall
print("legal patterns:", Counter(map(tuple, legal.astype(int))).most_common(20))
