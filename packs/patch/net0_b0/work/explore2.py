import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']; act = S['act']
names = tools.names()
# Per controller: episode lengths, phase green durations
for c, cn in enumerate(S['controller_names']):
    m = ctrl==c
    print('=====', cn)
    lens=[]; durs={p:[] for p in range(4)}; seqs=[]
    for e in np.unique(ep[m]):
        me = m & (ep==e)
        Xe = X[me]; te = t[me]
        lens.append(te.max())
        ph = Xe[:,40:44].argmax(1)
        # green durations: t_green at the last obs before phase change
        seq=[]
        for i in range(1,len(ph)):
            if ph[i]!=ph[i-1]:
                durs[ph[i-1]].append(Xe[i-1,45]*60+5)  # approx
                seq.append(ph[i-1])
        seqs.append(seq)
    print('episode end times', lens)
    for p in range(4):
        d=np.array(durs[p]); print(' phase',p,'n',len(d),'mean dur',d.mean().round(1),'q',np.percentile(d,[10,50,90]).round(1))
    # transitions
    T=np.zeros((4,4),int)
    for s in seqs:
        for a,b in zip(s[:-1],s[1:]): T[a,b]+=1
    print(' transitions\n',T)
