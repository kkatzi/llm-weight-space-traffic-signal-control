from common import *
for c in [0,2]:
    gaps={p:[] for p in range(4)}
    for e in range(6):
        idx=np.where((ctl==c)&(ep==e))[0]
        ph=cur[idx]; tt=t[idx]
        last={p:None for p in range(4)}
        for i in range(len(idx)):
            p=ph[i]
            if i>0 and ph[i-1]!=p:
                # phase p begins: gap since previous end of p
                if last[p] is not None: gaps[p].append(tt[i]-last[p])
            if i+1<len(idx) and ph[i+1]!=p: last[p]=tt[i+1]
    print(S['controller_names'][c])
    for p in range(4):
        v=np.array(gaps[p]); print('  phase',p,'n',len(v),'gap pct 50/90/99/max',np.percentile(v,[50,90,99]),v.max())
