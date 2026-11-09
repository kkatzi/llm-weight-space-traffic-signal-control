import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore30.py").read()
src = src.split("for nm,d in")[0]; exec(src)
def hold(x):
    l = L(x)
    nsT = l["N0"]+l["N1"]+l["S0"]+l["S1"]; nsL = l["N2"]+l["S2"]
    return (l["E2"] < 0.25) and (nsL < 0.25) and (l["W2"] <= 1) and (nsT <= 2 + l["W2"])
H = np.array([hold(x) for x in X])
for nm,mm in [("own",ok&(ctrl==0)&~dead&(cur==2)),("dead",ok&dead),("other",ok&(ctrl>0)&(cur==2))]:
    sel = mm&H
    print(nm, "cur2 states", mm.sum(), "hold-rule fires", sel.sum(), "net holds there", (net[sel]==2).sum(), "net holds elsewhere", (net[mm&~H]==2).sum())
    if sel.sum(): print("   net choices where rule fires:", np.bincount(net[sel],minlength=4))
def with_hold(d):
    def d2(x):
        if np.argmax(x[40:44])==2 and hold(x): return 2
        return d(x)
    return d2
for nm,d in [("succ 0.8", mk()), ("E2 1.25 W2 0.75", mk(1.25,0.75))]:
    report(with_hold(d), nm+" + hold"); print("   sparse:", check(with_hold(d), cur=2))
d = with_hold(mk())
r = np.array([d(x) for x in X])
for nm,m in [("own",ok&(ctrl==0)&~dead),("other",ok&(ctrl>0))]:
    C = np.zeros((4,4,4),int)
    for a,b,c in zip(cur[m],net[m],r[m]): C[a,b,c]+=1
    print(nm)
    for a in range(4): print(" cur",a,"agree %.3f"%((r[m&(cur==a)]==net[m&(cur==a)]).mean()),"rows=net cols=rule", C[a].tolist())
