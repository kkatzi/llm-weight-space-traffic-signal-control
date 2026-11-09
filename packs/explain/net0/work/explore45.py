import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
import importlib.util
from evalh import *
spec = importlib.util.spec_from_file_location("rf", "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/rules_full_v1.py"); rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
N0, N1, N2, E0, E1, E2, S0, S1, S2, W0, W1, W2 = range(12)
def light_calls(v, green_s):
    t = (min(green_s,60) - 12)/48
    ewT = v[E0]+v[E1]+v[W0]+v[W1]
    c0 = 1.1*(v[N0]+v[S0]) + 0.6*(v[N1]+v[S1]) + 0.6*(v[N2]+v[S2]) - 0.5*v[W2] - 0.2*v[E2] - 0.3*ewT - 2.2 - 0.15*t
    c1 = 1.9*v[N2] + 1.3*v[S2] - 0.5*(v[N1]+v[S1]) - 0.2*v[S0] + 0.2*v[N0] - 0.4*ewT - 1.3 - 0.25*t
    c3 = 1.5*v[E2] + 0.9*v[W2] + 0.8*v[N2] - 0.4*(v[N1]+v[S1]) - 0.6*ewT - 1.15 + 0.27*t
    return {0:c0, 1:c1, 2:0.0, 3:c3}
def make(limit):
    def dec(x):
        x = np.asarray(x,float); c = int(np.argmax(x[40:44]))
        v = 1.5*(x[0:12]+x[12:24])
        if c==2 and v.sum() < limit:
            calls = light_calls(v, 60*x[45])
            return int(max(calls, key=calls.get))
        return rf.decide(x)
    return dec
for lim in [3.25, 4.25, 5.25]:
    d = make(lim); r = report(d, f"light model below {lim} veh")
