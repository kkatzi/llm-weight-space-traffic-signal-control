import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools, itertools
w = tools.load("start_weights.npz")
LANES = {"N0":0,"N1":1,"N2":2,"E0":3,"E1":4,"E2":5,"S0":6,"S1":7,"S2":8,"W0":9,"W1":10,"W2":11}
SIDE = ["N0","N1","N2","S0","S1","S2","E2","W2"]
def sparse_states(cur=2, tgs=(0.2,1.0), maxveh=3):
    out=[]
    for tg in tgs:
        for k in range(0,maxveh+1):
            for combo in itertools.combinations(SIDE,k):
                x=np.zeros(46); x[40+cur]=1; x[44]=1; x[45]=tg
                for ln in combo:
                    x[LANES[ln]]=1/2.985; x[12+LANES[ln]]=1/2.985
                out.append((tg,combo,x))
    return out
def check(dec, cur=2, verbose=False, maxveh=3):
    st = sparse_states(cur, maxveh=maxveh)
    Xs = np.array([s[2] for s in st]); nt = tools.greedy(w,Xs)
    rr = np.array([dec(x) for x in Xs])
    agree = (nt==rr)
    byk = {}
    for (tg,combo,_),a in zip(st,agree):
        byk.setdefault((tg,len(combo)),[]).append(a)
    summ = {k: f"{np.mean(v):.2f}" for k,v in byk.items()}
    if verbose:
        for (tg,combo,_),a,n,r_ in zip(st,agree,nt,rr):
            if not a and len(combo)<=2: print("   mismatch", tg, combo, "net",n,"rule",r_)
    return summ
