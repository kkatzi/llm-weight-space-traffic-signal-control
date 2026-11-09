import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def jac(x):
    a1 = w["W1"]@x+w["b1"]; h1=np.tanh(a1)
    a2 = w["W2"]@h1+w["b2"]; h2=np.tanh(a2)
    return (w["W3"]*(1-h2**2)) @ (w["W2"]*(1-h1**2)) @ w["W1"]
G = {"nsT":[0,1,6,7],"nsL":[2,8],"ewT":[3,4,9,10],"ewL":[5,11]}
def loads(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985; L=(q+d)/2
    return np.array([L[ix].sum() for ix in G.values()])
FL = np.array([loads(x) for x in X]); tot = FL.sum(1)
m = ok&~dead
bins = [0,3,6,9,12,15,20,40]
for lo,hi in zip(bins[:-1],bins[1:]):
    mm = m&(tot>=lo)&(tot<hi)
    if mm.sum()<10: continue
    Js = np.array([jac(x) for x in X[mm]]).mean(0)
    rows=[]
    for g,ix in G.items():
        v = np.mean([(Js[:,i]+Js[:,12+i])/2.985 for i in ix],axis=0)
        rows.append(v)
    R = np.array(rows)  # groups x phases
    # own-phase weight relative to mean of others
    own = [R[0,0]-R[0,[1,2,3]].mean(), R[1,1]-R[1,[0,2,3]].mean(), R[2,2]-R[2,[0,1,3]].mean(), R[3,3]-R[3,[0,1,2]].mean()]
    print(f"tot [{lo},{hi}) n={mm.sum():4d} own-group weight (nsT,nsL,ewT,ewL):", np.round(own,2))
