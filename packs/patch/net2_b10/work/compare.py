import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import numpy as np, tools
from build import build
from collections import Counter
w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
cap, capu = 22.4/7.5, 256/7.5
q = np.round(X[:, 0:12]*cap).astype(int); qu = np.round(X[:, 24:32]*capu).astype(int)
g0 = tools.greedy(w0, X)
left = {0: 1, 1: 3, 2: 1, 3: 3}; thr = {0: 0, 1: 2, 2: 0, 3: 2}
anyup = (qu > 0).any(1)

def report(name, w):
    g = tools.greedy(w, X)
    ch = ok & (g != g0)
    # spillback service: in honoured states with left spillback (pocket full & up1>=1), share picking that left phase
    sl = []; st = []
    for a in range(4):
        m = ok & (qu[:, 2*a+1] >= 1) & (q[:, 3*a+2] == 3) & (ph != left[a])
        sl.append(((g[m] == left[a]).sum(), (g0[m] == left[a]).sum(), m.sum()))
        m = ok & (qu[:, 2*a] >= 1) & (ph != thr[a])
        st.append(((g[m] == thr[a]).sum(), (g0[m] == thr[a]).sum(), m.sum()))
    sl = np.array(sl).sum(0); st = np.array(st).sum(0)
    # through-blocked: up1 halted & lane1 full & pocket not full -> picks left (bad)
    tb = []
    for a in range(4):
        m = ok & (qu[:, 2*a+1] >= 1) & (q[:, 3*a+1] == 3) & (q[:, 3*a+2] < 3) & (ph != left[a])
        tb.append(((g[m] == left[a]).sum(), (g0[m] == left[a]).sum(), m.sum()))
    tb = np.array(tb).sum(0)
    print(f"{name:34s} edit {tools.edit_size(w, w0)['changed']:4d} | honoured changed {ch.mean():.3f} (net eps {(ch & (ctrl==0)).sum()}/{(ok&(ctrl==0)).sum()}) "
          f"changed w/o upq {(ch & ~anyup).sum()} | left-spill picks left {sl[0]}/{sl[2]} (orig {sl[1]}) | thr-spill picks thr {st[0]}/{st[2]} (orig {st[1]}) | thr-blocked picks left {tb[0]}/{tb[2]} (orig {tb[1]})")
    return g

for args in [dict(), dict(rho_thr=0), dict(rho_left=0), dict(rho_left=0.5, rho_thr=0.5), dict(mode_left="avg", mode_thr="avg"),
             dict(topk_left=48, topk_thr=32), dict(topk_left=40, topk_thr=40), dict(topk_left=32, topk_thr=32),
             dict(mode_left="avg", mode_thr="avg", topk_left=48, topk_thr=32), dict(rho_left=1.5, rho_thr=1.0, topk_left=48, topk_thr=32)]:
    report(str(args), build(**args))
