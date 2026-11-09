import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"]; ctrl = S["controller"]
names = tools.names(); ix = {n:i for i,n in enumerate(names)}
L = tools.forward(w, X); net = L.argmax(1)
ok = X[:,44] > 0.5; cur = X[:,40:44].argmax(1)
f = lambda *ns: sum(X[:,ix[n]] for n in ns)
D = np.stack([f('d_N0','d_N1','d_S0','d_S1'), f('d_N2','d_S2'), f('d_E0','d_E1','d_W0','d_W1'), f('d_E2','d_W2')],1)
Q = np.stack([f('q_N0','q_N1','q_S0','q_S1'), f('q_N2','q_S2'), f('q_E0','q_E1','q_W0','q_W1'), f('q_E2','q_W2')],1)
U = np.stack([f('d_N0u','d_N1u','d_S0u','d_S1u'), f('d_N1u','d_S1u'), f('d_E0u','d_E1u','d_W0u','d_W1u'), f('d_E1u','d_W1u')],1)
for p in range(4):
    m = ok & (cur==p)
    print(f"=== cur {p}")
    for a in range(4):
        mm = m & (net==a)
        if mm.sum()==0: continue
        print(f"  -> {a} n={mm.sum():4d} D={D[mm].mean(0)} Q={Q[mm].mean(0)} U={U[mm].mean(0)}")
