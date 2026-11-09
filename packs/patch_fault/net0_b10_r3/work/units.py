import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'work')
import tools, numpy as np
from build2 import build
S = tools.states(); X = S['obs'].astype(np.float64)
w0=tools.load("start_weights.npz")
print(tools.trainability(w0))
h1,h2 = tools.hidden(w0, X)
a1 = X@w0['W1'].T + w0['b1']
o=np.argsort(h1.std(0))
print("h1 lowest std units:", [(int(j), round(float(h1[:,j].std()),3), round(float(h1[:,j].mean()),3), round(float(a1[:,j].min()),2), round(float(a1[:,j].max()),2)) for j in o[:8]])
# importance of h1 unit = std(h1_j) * norm(W3 @ diag(1-h2^2) @ W2[:,j]) approx -> simpler: ablate to mean, count decision changes
a0 = tools.greedy(w0,X)
imp=[]
for j in range(64):
    w = {k:v.copy() for k,v in w0.items()}
    m = h1[:,j].mean()
    w['b2'] = w['b2'] + w['W2'][:,j]*m; w['W2'] = w['W2'].copy(); w['W2'][:,j]=0
    imp.append((a0!=tools.greedy(w,X)).mean())
imp=np.array(imp)
o=np.argsort(imp)
print("h1 units with least ablation impact:", [(int(j), round(float(imp[j]),4), round(float(h1[:,j].std()),3)) for j in o[:10]])
a2 = X.copy()
o2=np.argsort(h2.std(0))
print("h2 lowest std:", [(int(j), round(float(h2[:,j].std()),3), round(float(h2[:,j].mean()),3)) for j in o2[:8]])
