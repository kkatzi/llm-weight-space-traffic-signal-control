import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=220)
St, X = load_states(); c = St["controller"]; ep = St["episode"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
w = tools.load(os.path.join(HERE, "work", sys.argv[1] if len(sys.argv) > 1 else "v2.npz")); w1 = tools.load(os.path.join(HERE, "work", "v1.npz"))
good = (c > 0) | (ep < 2)
sig = good & (ph == 2) & (X[:, 27] > 0) & (X[:, 16] == 0) & ok
Y = mirror(X[sig])
h1, h2 = tools.hidden(w, Y)
print("spill fires", (h1[:, 31] > 0).mean(), "relay fires", (h2[:, 25] > 0).mean())
L1 = tools.forward(w1, Y); L2 = tools.forward(w, Y)
print("v1 gap (max - logit3) quantiles", np.quantile(L1.max(1) - L1[:, 3], [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).round(2))
print("new gap quantiles", np.quantile(L2.max(1) - L2[:, 3], [0.1, 0.5, 0.75, 0.9, 1.0]).round(2))
print("logit3 increase new-v1", np.quantile(L2[:, 3] - L1[:, 3], [0, 0.5, 1]).round(2))
print("picks", np.bincount(L2.argmax(1), minlength=4))
i = np.where(L2.argmax(1) != 3)[0][:3]
for k in i:
    print(tools.block(Y[k])); print(L1[k], L2[k])
