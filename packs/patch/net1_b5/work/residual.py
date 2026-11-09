from common import *
g=tools.greedy(W0,X)
lanes={0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
ups={0:[24,25,28,29],1:[25,29],2:[26,27,30,31],3:[27,31]}
for c in [0]:
    for p in range(4):
        m=(ctl==c)&ok&(cur==p)
        rq=q[m][:,lanes[p]].sum(1); rd=d[m][:,lanes[p]].sum(1)
        ru=qu[m][:,[i-24 for i in ups[p]]].sum(1)
        stay=(g[m]==p)
        print('phase',p,'n',m.sum(),'residual halting stop-line pct50/90',np.percentile(rq,[50,75,90]),'dens',np.percentile(rd,[50,90]),'up halting >0 frac',(ru>0.5).mean(),'stay',stay.mean())
        # stay rate vs residual
        for lo,hi in [(0,1),(1,3),(3,6),(6,13)]:
            mm=(rq>=lo-1e-6)&(rq<hi-1e-6)
            if mm.sum(): print('    resid',lo,hi,'n',mm.sum(),'stay',stay[mm].mean().round(2), 'others max load', np.median(np.delete(Lq[m][mm],p,axis=1).max(1)))
