import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
exec(open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/synthetic_light.py").read().split("for f in sys.argv")[0])
import importlib.util
spec = importlib.util.spec_from_file_location("rf", "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/rules_full.py"); rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
spec = importlib.util.spec_from_file_location("rs", "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/rules_small.py"); rs = importlib.util.module_from_spec(spec); spec.loader.exec_module(rs)
from collections import Counter
cnt = Counter(); 
for x,n in zip(Xs[cur==2], net[cur==2]):
    v = 1.5*(x[0:12]+x[12:24]); nsT = v[[0,1,6,7]].sum(); nsL = v[[2,8]].sum()
    lt = rf.light_traffic_from_ew_through(v, nsT, nsL, 60*x[45])
    branch = "none" if lt is None else f"light->{lt}"
    rfull = rf.decide(x); rsmall = rs.decide(x)
    cnt[(branch, "full ok" if rfull==n else "full bad", "small ok" if rsmall==n else "small bad")] += 1
for k,v in sorted(cnt.items()): print(k, v)
