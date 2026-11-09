"""Validation only: EW_through held for a long time (t_green 17..60 s) with 0-4 side-street vehicles,
the situation in which the network stalls. Compare rule files with the network."""
import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools, itertools
w = tools.load("start_weights.npz")
LANES = {"N0":0,"N1":1,"N2":2,"E2":5,"S0":6,"S1":7,"S2":8,"W2":11}
Xs=[]
for t in [12,17,27,42,60]:
    for k in range(0,5):
        for combo in itertools.combinations_with_replacement(list(LANES),k):
            x=np.zeros(46); x[42]=1; x[44]=1; x[45]=t/60
            for ln in combo: x[LANES[ln]]+=1/2.985; x[12+LANES[ln]]+=1/2.985
            if (x[:24]>1.0001).any(): continue
            Xs.append(np.minimum(x,1))
Xs=np.array(Xs); net=tools.greedy(w,Xs); nveh=np.round(Xs[:,12:24].sum(1)*2.985)
tg = np.round(Xs[:,45]*60)
for f in sys.argv[1:]:
    dec = tools._load_rules(f); r=np.array([dec(x) for x in Xs])
    print(f, "overall %.3f"%(r==net).mean(), "| by #veh", {int(k): round(float((r[nveh==k]==net[nveh==k]).mean()),2) for k in range(5)},
          "| by green s", {int(t): round(float((r[tg==t]==net[tg==t]).mean()),2) for t in [12,17,27,42,60]})
