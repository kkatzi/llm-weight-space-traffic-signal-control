"""Logit margins of the stall breaker in the ON state (phase 2 held >= 60 s), synthetic states."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
from make_final import build
capS = 22.4 / 7.5
def state(lanes):
    x = np.zeros(46); x[42] = 1; x[44] = 1; x[45] = 1.0
    for i, v in lanes: x[i] = v
    return x
def margin(lg, want):
    o = np.delete(lg, want); return lg[want] - o.max()
print('rows vW, cols vN: (margin empty->3, min margin single N/S-through car->0)')
vNs = [0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
for vW in [0.5, 0.67, 0.8, 1.0, 1.2, 1.5]:
    row = []
    for vN in vNs:
        w = build(vW, vN)
        me = margin(tools.forward(w, state([]))[0], 3)
        mn = min(margin(tools.forward(w, state([(l, 1 / capS), (l + 12, 1 / capS)]))[0], 0) for l in [0, 1, 6, 7])
        row.append(f'({me:+.2f},{mn:+.2f})')
    print(vW, ' '.join(row))
