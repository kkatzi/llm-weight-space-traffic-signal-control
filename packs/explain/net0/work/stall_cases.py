"""Validation only: the single- and two-vehicle cases that decide whether EW_through stalls at the end
of an episode. Shows network vs both rule files at several green durations."""
import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
w = tools.load("start_weights.npz")
LANES = {"N0":0,"N1":1,"N2":2,"E2":5,"S0":6,"S1":7,"S2":8,"W2":11}
small = tools._load_rules("work/rules_small.py"); full = tools._load_rules("work/rules_full.py")
cfgs = [("N0",),("N1",),("S0",),("S1",),("N2",),("S2",),("E2",),("W2",),("N0","N1"),("N0","S0"),("S0","S1"),("N1","S1"),
        ("N0","W2"),("S1","W2"),("N2","S1"),("S2","N1"),("S2","W2"),("N2","W2"),("N2","S2"),("N0","N1","W2"),("N0","N1","S1")]
ts = [12,17,32,57,60]
print("cells: net/small/full", ts)
for cfg in cfgs:
    row=[]
    for t in ts:
        x=np.zeros(46); x[42]=1; x[44]=1; x[45]=t/60
        for ln in cfg: x[LANES[ln]]+=1/2.985; x[12+LANES[ln]]+=1/2.985
        row.append(f"{tools.greedy(w,x)[0]}/{small(x)}/{full(x)}")
    print(str(cfg).ljust(22), "  ".join(row))
# robustness on random inputs
rng=np.random.default_rng(1); R=rng.random((2000,46)); R[:,40:44]=np.eye(4)[rng.integers(0,4,2000)]
for f,d in [("small",small),("full",full)]:
    out=[d(x) for x in R]; assert all(o in (0,1,2,3) for o in out)
    print(f, "random-input outputs ok; agreement on uniform-random inputs %.3f"%np.mean(np.array(out)==tools.greedy(w,R)))
