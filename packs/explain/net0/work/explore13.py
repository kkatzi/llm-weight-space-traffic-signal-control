import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
V = 2.985
def groups(x, mode="q"):
    if mode=="q": a = x[0:12]*V
    elif mode=="qd": a = (x[0:12]+x[12:24])*V/2
    nsT = a[0]+a[1]+a[6]+a[7]; nsL = a[2]+a[8]; ewT = a[3]+a[4]+a[9]+a[10]; ewL = a[5]+a[11]
    return nsT, nsL, ewT, ewL
def model_sparse(mode, keepall=True):
    def dec(x):
        nsT,nsL,ewT,ewL = groups(x,mode)
        s = np.array([-4.0 + 3.6*nsT - 1.2*ewL + 0.4*ewT,
                      -0.8 + 3.8*nsL,
                       3.8 + 1.1*ewT - 1.9*nsL,
                      -0.24 + 4.1*ewL - 0.75*ewT])
        c = int(np.argmax(x[40:44]))
        if not keepall and c!=2: s[c] = -99
        return int(np.argmax(s))
    return dec
report(model_sparse("q"), "sparse q")
report(model_sparse("qd"), "sparse qd")
report(model_sparse("q",False), "sparse q, no keep except 2")
report(model_sparse("qd",False), "sparse qd, no keep except 2")
