import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']
cs, cu = 22.4/7.5, 256/7.5
groups = {'NS thru':['N0','N1','S0','S1'], 'NS left':['N2','S2'], 'EW thru':['E0','E1','W0','W1'], 'EW left':['E2','W2']}
ix = {n:i for i,n in enumerate(tools.names())}
upg = {'N0u':'NS','N1u':'NS','S0u':'NS','S1u':'NS','E0u':'EW','E1u':'EW','W0u':'EW','W1u':'EW'}
for c in range(3):
    m = (ctrl==c)&(t<3600)
    r = {g: (sum(X[m, ix['q_'+l]] for l in ls)*cs).mean() for g, ls in groups.items()}
    r['NS up'] = (sum(X[m, ix['q_'+u]] for u in upg if upg[u]=='NS')*cu).mean()
    r['EW up'] = (sum(X[m, ix['q_'+u]] for u in upg if upg[u]=='EW')*cu).mean()
    tot = (X[m,0:12].sum(1)*cs + X[m,24:32].sum(1)*cu).mean()
    vehs = (X[m,12:24].sum(1)*cs + X[m,32:40].sum(1)*cu).mean()
    print(S['controller_names'][c], 'mean halted veh by group:', {k: round(v,2) for k,v in r.items()}, 'total halted %.2f  total veh %.2f'%(tot, vehs))
    # per episode
    print('   per-episode mean veh:', [round(((X[(ctrl==c)&(ep==e),12:24].sum(1)*cs + X[(ctrl==c)&(ep==e),32:40].sum(1)*cu).sum()*5/3600),2) for e in range(6)])
