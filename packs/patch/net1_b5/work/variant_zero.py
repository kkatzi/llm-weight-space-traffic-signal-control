from common import *
from build_gate import build
G=(31,0,59,54); R=(25,35,39,15)
wG=build(W0)
# variant: cut the gates' outgoing weights to non-relay layer-2 units; b2 absorbs old mean of the replaced units
wZ={k:v.copy() for k,v in W0.items()}
wZ['W1']=wG['W1'].copy(); wZ['b1']=wG['b1'].copy(); wZ['W3']=wG['W3'].copy(); wZ['b3']=wG['b3'].copy()
wZ['W2']=W0['W2'].copy(); wZ['b2']=W0['b2'].copy()
X0=X[ctl==0]; H1old=np.tanh(X0@W0['W1'].T+W0['b1'])
for p in range(4):
    i=G[p]
    for j in range(64):
        if j in R: continue
        wZ['b2'][j]+=W0['W2'][j,i]*H1old[:,i].mean()
        wZ['W2'][j,i]=0.0
    wZ['W2'][R[p],i]=40.0
print('wZ size',tools.edit_size(wZ,W0))
for w,l in [(wG,'gate'),(wZ,'gate+cut')]:
    r=[]
    for c in range(3):
        m=(ctl==c)&ok
        r.append(round((tools.greedy(w,X[m])!=tools.greedy(W0,X[m])).mean(),4))
    # low-demand-ish states: total vehicles small
    print(l,'change at ok per controller',r)
tools.save(wZ,'/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wZ.npz')
from deadlock import check
check(tools.load('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wZ.npz'),'wZ')
print(tools.edit_size(tools.load('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wZ.npz'),W0))
