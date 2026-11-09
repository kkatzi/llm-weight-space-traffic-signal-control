import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore24.py").read()
src = src.split("variants = [")[0]; exec(src)
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def jac(x):
    a1 = w["W1"]@x+w["b1"]; h1=np.tanh(a1)
    a2 = w["W2"]@h1+w["b2"]; h2=np.tanh(a2)
    return (w["W3"]*(1-h2**2)) @ (w["W2"]*(1-h1**2)) @ w["W1"]
LN = "N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2".split()
# feature definitions as lane-weight vectors over load L (halted + 0.5 moving)
FD = {"nsT": {"N0":1.2,"S0":1.2,"N1":0.8,"S1":0.8}, "nsL": {"N2":1,"S2":1},
      "ewT": {"E0":1.33,"E1":1.33,"W1":1.33}, "ewL": {"E2":1,"W2":1}}
def fvec(x):
    l = L(x); return np.array([sum(v*l[k] for k,v in FD[f].items()) for f in FD] + [upEW(x)])
def dlogit_dfeature(J):
    # direction in x for +1 unit of feature: spread across lanes proportional to weights / sum(w^2) (min-norm)
    cols=[]
    for f,lw in FD.items():
        ww = np.array(list(lw.values())); dirn = ww/np.sum(ww**2)
        g = np.zeros(4)
        for (k,_),a in zip(lw.items(),dirn):
            i = LN.index(k); g += a*(J[:,i]+J[:,12+i])/2.985
        cols.append(g)
    cols.append(np.mean([(J[:,i]+J[:,i+8])/34.13 for i in [26,27,30,31]],axis=0))
    return np.array(cols).T
per = {}
for c in range(4):
    mm = ok&(ctrl==0)&~dead&(cur==c)
    Js = np.array([jac(x) for x in X[mm]]).mean(0)
    Wf = dlogit_dfeature(Js)
    F = np.array([fvec(x) for x in X[mm]]); Lg = tools.forward(w,X[mm])
    b = (Lg - F@Wf.T).mean(0)
    per[c]=(Wf,b)
    Wrel = Wf - Wf[[2]]; brel = b-b[2]
    print("cur",c,"bias rel2", np.round(brel,2)); print(" weights rel2 (rows phase; cols nsT nsL ewT ewL upEW)\n", np.round(Wrel,2))
def dec(x):
    c = int(np.argmax(x[40:44])); Wf,b = per[c]
    s = Wf@fvec(x)+b
    if c!=2: s[c]=-99
    return int(np.argmax(s))
report(dec, "per-cur feature Taylor (avg jacobian, mean intercept)")
r = np.array([dec(x) for x in X])
for nm,m in [("own",ok&(ctrl==0)&~dead),("other",ok&(ctrl>0))]:
    C = np.zeros((4,4,4),int)
    for a,b,c in zip(cur[m],net[m],r[m]): C[a,b,c]+=1
    print(nm)
    for a in range(4): print(" cur",a,"agree %.3f"%((r[m&(cur==a)]==net[m&(cur==a)]).mean()),"rows=net cols=rule", C[a].tolist())
