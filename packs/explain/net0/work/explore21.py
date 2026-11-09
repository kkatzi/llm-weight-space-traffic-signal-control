import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore18.py").read()
src = "\n".join(l for l in src.splitlines() if not l.startswith("report("))
exec(src)
d = make(nsT1,ewT1,ewL0)
r = np.array([d(x) for x in X])
def loads4(x):
    l = L(x); return np.array([l["N0"]+l["N1"]+l["S0"]+l["S1"], l["N2"]+l["S2"], l["E0"]+l["E1"]+l["W0"]+l["W1"], l["E2"]+l["W2"]])
FL = np.array([loads4(x) for x in X]); tot = FL.sum(1)
m = ok&(ctrl==0)&~dead
for lo,hi in [(0,3),(3,6),(6,9),(9,12),(12,15),(15,20),(20,40)]:
    mm = m&(tot>=lo)&(tot<hi)
    print(f"own tot [{lo},{hi}) n={mm.sum():4d} agree={(r[mm]==net[mm]).mean():.3f}")
m2 = ok&(ctrl>0)
for lo,hi in [(0,3),(3,6),(6,9),(9,12),(12,15),(15,20),(20,40)]:
    mm = m2&(tot>=lo)&(tot<hi)
    print(f"other tot [{lo},{hi}) n={mm.sum():4d} agree={(r[mm]==net[mm]).mean():.3f}")
# per-cur
for c in range(4):
    mm = m&(cur==c); print("own cur",c,"agree",(r[mm]==net[mm]).mean().round(3), mm.sum())
