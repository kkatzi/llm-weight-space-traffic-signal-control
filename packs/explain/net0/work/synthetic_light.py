"""Validation only: random light-traffic states (0-5 vehicles, halted or moving, any stop-line lane,
a few upstream vehicles), compare rule files with the network. Not used to choose numbers."""
import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
w = tools.load("start_weights.npz")
rng = np.random.default_rng(0)
def sample(n=4000):
    Xs=[]
    for _ in range(n):
        x=np.zeros(46); c=rng.integers(4); x[40+c]=1; x[44]=1
        x[45] = 12/60 if rng.random()<0.7 else min(1.0, rng.choice([17,22,32,47,60])/60)
        k = rng.integers(0,6)
        for _ in range(k):
            ln = rng.integers(12)
            if x[12+ln] >= 0.99: continue
            x[12+ln] += 1/2.985
            if rng.random()<0.7: x[ln] += 1/2.985
        for j in range(8):
            u = rng.poisson(0.7); x[32+j] = u/34.13; x[24+j] = rng.binomial(u,0.3)/34.13
        Xs.append(np.minimum(x,1.0))
    return np.array(Xs)
Xs = sample()
net = tools.greedy(w,Xs); cur = Xs[:,40:44].argmax(1)
for f in sys.argv[1:]:
    dec = tools._load_rules(f); r = np.array([dec(x) for x in Xs])
    print(f, "synthetic light-traffic agreement %.3f"%(r==net).mean(), "per cur", [round(float((r[cur==c]==net[cur==c]).mean()),3) for c in range(4)])
