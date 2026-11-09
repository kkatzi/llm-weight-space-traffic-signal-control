import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def loads(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985
    L = (q + d)/2
    nsT = L[[0,1,6,7]].sum(); nsL = L[[2,8]].sum(); ewT = L[[3,4,9,10]].sum(); ewL = L[[5,11]].sum()
    upEW = (x[[26,27,30,31]].sum()+x[[34,35,38,39]].sum())*34.13/2
    return nsT,nsL,ewT,ewL,upEW
BIAS = {0:[-1.2,-2.3,0,-2.1], 1:[-2.4,-0.25,0,-1.5], 2:[-2.8,-1.9,0,-0.7], 3:[-1.9,-0.56,0,-0.05]}
def dec(x):
    c = int(np.argmax(x[40:44]))
    nsT,nsL,ewT,ewL,upEW = loads(x)
    s = np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW, ewL]) + BIAS[c]
    if c!=2: s[c]=-99
    return int(np.argmax(s))
r = report(dec, "rounded shared model")
m = ok&(ctrl==0)&~dead
C = np.zeros((4,4,4),int)
for a,b,c in zip(cur[m],net[m],r[m]): C[a,b,c]+=1
for a in range(4): print("cur",a,"rows=net cols=rule\n",C[a])
print("deadlock state decision:", r[dead][:3])
L = tools.forward(w, X)
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
err = m & (r!=net)
print("gap quantiles at errors", np.round(np.percentile(gap[err],[10,25,50,75,90]),2))
print("gap quantiles all own", np.round(np.percentile(gap[m],[10,25,50,75,90]),2))
# what's the network's logit for the rule's choice vs best
lr = L[np.arange(len(L)), r]
print("net logit deficit of rule choice at errors", np.round(np.percentile((L.max(1)-lr)[err],[10,25,50,75,90]),2))
# total load at errors vs correct
FL = np.array([loads(x) for x in X]); tot = FL[:,:4].sum(1)
print("total load errors", np.round(np.percentile(tot[err],[10,50,90]),1), "correct", np.round(np.percentile(tot[m&~err],[10,50,90]),1))
