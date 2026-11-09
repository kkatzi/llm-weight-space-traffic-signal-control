import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore18.py").read()
src = "\n".join(l for l in src.splitlines() if not l.startswith("report("))
exec(src)
from sparsecheck import check
for nm,d in [("base",make(nsT0,ewT0,ewL0)),("all",make(nsT1,ewT1,ewL1,0.14)), ("nsT1,ewT1,ewL1", make(nsT1,ewT1,ewL1))]:
    for c in range(4):
        print(nm, "cur",c, check(d, cur=c, verbose=(c==2 and nm=="all")))
