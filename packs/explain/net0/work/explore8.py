import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; T=S["time_s"]; ep=S["episode"]
net = tools.greedy(w, X)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
dead = (ctrl==0)&(ep==2)&(T>=3625)
H = np.round(X[:,0:12]*2.985).astype(int)   # halted vehicles per stop-line lane
D = np.round(X[:,12:24]*2.985).astype(int)  # vehicles per stop-line lane
U = np.round(X[:,32:40]*34.13).astype(int)  # vehicles upstream
UQ = np.round(X[:,24:32]*34.13).astype(int)
print("check quantization H", np.unique(X[:,0]), )
print("upstream max", U.max(0))
N0,N1,N2,E0,E1,E2,S0,S1,S2,W0,W1,W2 = range(12)
def stats(m, label):
    print(label, "n", m.sum())
    for p in range(4):
        mm = m & (net==p)
        if mm.sum()==0: continue
        print(f"  pick {p} n={mm.sum():4d} H:", np.round(H[mm].mean(0),2), " D:", np.round(D[mm].mean(0),2), "U:", np.round(U[mm].mean(0),1))
for c in range(4):
    stats(ok&(cur==c)&(ctrl==0)&~dead, f"own cur={c}")
