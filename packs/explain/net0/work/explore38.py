import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore35.py").read().split("for nm,sf in")[0]
exec(src)
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def jac(x):
    a1 = w["W1"]@x+w["b1"]; h1=np.tanh(a1)
    a2 = w["W2"]@h1+w["b2"]; h2=np.tanh(a2)
    return (w["W3"]*(1-h2**2)) @ (w["W2"]*(1-h1**2)) @ w["W1"]
LN = "N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2".split()
GR = {"nsT":["N0","N1","S0","S1"],"nsL":["N2","S2"],"ewT":["E0","E1","W0","W1"],"ewL":["E2","W2"]}
def gfeat(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985; v = (q+d)/2
    l = dict(zip(LN, v))
    return np.array([sum(l[k] for k in GR[g]) for g in GR] + [upEW(x)])
GF = np.array([gfeat(x) for x in X])
def slopes(J):
    cols=[]
    for g,ls in GR.items():
        cols.append(np.mean([(J[:,LN.index(k)]+J[:,12+LN.index(k)])/2.985 for k in ls],axis=0))
    cols.append(np.mean([(J[:,i]+J[:,i+8])/34.13 for i in [26,27,30,31]],axis=0))
    return np.array(cols).T   # 4 x 5
SL = {}
for c in range(4):
    mm = ok&(ctrl==0)&~dead&(cur==c)
    J = np.array([jac(x) for x in X[mm]]).mean(0)
    S_ = slopes(J); S_ = S_ - S_[[2]]   # relative to phase 2
    SL[c] = np.round(S_/2.0, 1)          # scale to ~vehicle units, round
    print("cur",c,"slopes (rows phase, cols nsT nsL ewT ewL upEW)\n", SL[c])
RS = np.array([SL[c]@f for c,f in zip(cur,GF)])
B = tie_biases(RS, verbose=True)
def dec(x):
    c = int(np.argmax(x[40:44]))
    if c==2 and hold(x): return 2
    s = SL[c]@gfeat(x) + B[c]
    if c!=2: s[c]=-99
    return int(np.argmax(s))
r = report(dec, "per-cur slopes + tie biases")
m = ok&(ctrl==0)&~dead
print("   per-cur own:", [round(float((r[m&(cur==a)]==net[m&(cur==a)]).mean()),3) for a in range(4)])
