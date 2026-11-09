import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from build import build, demand
np.set_printoptions(linewidth=220, precision=2, suppress=True)
S=tools.states(); w0=tools.load('start_weights.npz'); w=build()
X=S['obs'].astype(np.float64); C=S['controller']; ep=S['episode']; T=S['time_s']
ok=X[:,44]>0.5; E=demand(X)==0
norm=ok&~E.any(1)
lg0=tools.forward(w0,X); lg1=tools.forward(w,X)
g0=lg0.argmax(1); g1=lg1.argmax(1)
dis=norm&(g0!=g1)
gap0=lg0[np.arange(len(X)),g0]-lg0[np.arange(len(X)),g1]
print('n disagreements',dis.sum(),'orig gap between its pick and new pick: quantiles',np.percentile(gap0[dis],[10,50,90,100]).round(2))
print('logit change rms on normal ok states',np.sqrt(((lg1-lg0)[norm]**2).mean()).round(3), 'after removing uniform shift', np.sqrt((((lg1-lg0)-(lg1-lg0).mean(1,keepdims=True))[norm]**2).mean()).round(3))
# flags sanity: layer-2 flag units
h1,h2=tools.hidden(w,X)
K=(37,0,11,5)
for p,k in enumerate(K):
    print('phase',p,'flag unit',k,'empty: min',h2[E[:,p],k].min().round(4) if E[:,p].any() else None,' nonempty: max',h2[~E[:,p],k].max().round(4))
