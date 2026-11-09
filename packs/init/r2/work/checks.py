import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, tools, rule, build
np.set_printoptions(linewidth=200, precision=3, suppress=True)
S = tools.states(); X = S["obs"].astype(float); A_ = S["act"]; C = S["controller"]; T = S["time_s"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
w = tools.load(os.path.join(HERE, "weights.npz"))
print("trainability:", tools.trainability(w))
print("edit size:", tools.edit_size(w, tools.load(os.path.join(ROOT, "start_weights.npz"))))
net = tools.greedy(w, X); r = rule.scores(X).argmax(1)
print(f"net vs linear rule: ok {np.mean(net[ok]==r[ok]):.4f} all {np.mean(net==r):.4f}")
print("fidelity via tools:", tools.fidelity(os.path.join(HERE, "rule.py"), w=w))
for c in range(2):
    m = ok & (C == c)
    print(f"ctrl{c} ok-states: net agrees with recorded controller {np.mean(net[m]==A_[m]):.3f}; net holds {np.mean(net[m]==ph[m]):.3f}")
pr = tools.probs(w, X)
mp = pr.max(1)
print("max prob quantiles all:", np.quantile(mp, [.1, .25, .5, .75, .9]))
print("max prob quantiles ok :", np.quantile(mp[ok], [.1, .25, .5, .75, .9]))
lg = tools.forward(w, X)
print("logit range", lg.min(), lg.max())
