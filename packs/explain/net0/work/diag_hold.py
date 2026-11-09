import sys
exec(open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/synthetic_hold.py").read().split("for f in sys.argv")[0])
from collections import Counter
dec = tools._load_rules("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/rules_full.py"); r=np.array([dec(x) for x in Xs])
names = list(LANES)
c = Counter()
for x,n,rr,k,t in zip(Xs,net,r,nveh,tg):
    if n!=rr and k<=3:
        cfg = tuple(sorted(sum([[ln]*int(round(x[LANES[ln]]*2.985)) for ln in names],[])))
        c[(cfg, int(n), int(rr))] += 1
for (cfg,n,rr),v in sorted(c.items(), key=lambda z:-z[1])[:40]: print(cfg, "net",n,"rule",rr, "x",v,"green values")
