import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore36.py").read().split("variants = [")[0]
exec(src)
from sparsecheck import check
sf = mkS(); RS = np.array([sf(x) for x in X]); B = tie_biases(RS)
KEEP = {0:-1.5, 1:-1.0, 3:-0.6}
def build_keep(sf, B, keep):
    def dec(x):
        c = int(np.argmax(x[40:44]))
        if c==2 and hold(x): return 2
        b = B[c].copy()
        if c!=2: b[c] = keep[c]
        return int(np.argmax(sf(x) + b))
    return dec
m = ok&(ctrl==0)&~dead
for nm,d in [("no keep", build(sf,B)), ("keep handicaps", build_keep(sf,B,KEEP))]:
    r = report(d, nm)
    print("   per-cur own:", [round(float((r[m&(cur==a)]==net[m&(cur==a)]).mean()),3) for a in range(4)])
    for c in range(4): print("   sparse cur",c, check(d,cur=c))
d = build_keep(sf,B,KEEP); r = np.array([d(x) for x in X])
GF = np.array([[ (lambda l: [l["N0"]+l["N1"]+l["S0"]+l["S1"], l["N2"]+l["S2"], l["E0"]+l["E1"]+l["W0"]+l["W1"], l["E2"]+l["W2"]])(L(x)) ] for x in X])[:,0,:]
tot = GF.sum(1)
for c in range(4):
  for lo,hi in [(0,3),(3,5),(5,8),(8,99)]:
    mm = ok&~dead&(cur==c)&(tot>=lo)&(tot<hi)
    print(f"cur {c} tot[{lo},{hi}) n={mm.sum():4d} agree={(r[mm]==net[mm]).mean():.3f}  net choices {np.bincount(net[mm],minlength=4)} rule {np.bincount(r[mm],minlength=4)}")
