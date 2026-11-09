"""What would the trained network (detectors working) do in W-side spillback states if it saw W2 full?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
St, X = load_states(); c = St["controller"]; ep = St["episode"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
good = (c > 0) | (ep < 2)
sig = good & (ph == 2) & (X[:, 27] > 0) & (X[:, 16] == 0) & ok
Y = mirror(X[sig])
for e2 in [0.0, 0.335, 0.67, 1.0]:
    Z = Y.copy(); Z[:, 11] = 1.0; Z[:, 23] = 1.0; Z[:, 5] = e2; Z[:, 17] = max(e2, Z[:, 17].max() * 0) if False else e2
    print(f"w0 with W2 full, E2={e2:.2f}: picks", np.bincount(tools.greedy(w0, Z), minlength=4))
Z = Y.copy(); Z[:, 11] = 1.0; Z[:, 23] = 1.0
print("w0 with W2 full, E2 as observed (=0):", np.bincount(tools.greedy(w0, Z), minlength=4))
# also: unmirrored E-side states, w0 with real E2 and W2 := E2 (= v1)
for nm, wf in [("v1", "v1.npz"), ("v2", "v2.npz")]:
    w = tools.load(os.path.join(HERE, "work", wf))
    print(nm, "on mirrored:", np.bincount(tools.greedy(w, Y), minlength=4))
