import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=2, suppress=True)
S=tools.states(); w0=tools.load('start_weights.npz'); w=tools.load('work/weights.npz')
X=S['obs'].astype(np.float64); C=S['controller']; ep=S['episode']; T=S['time_s']
i=np.where((C==0)&(ep==2)&(T==3625))[0][0]
x=X[i].copy()
print('recorded deadlock state (phase 2 green, cars on N0,N1,W2): orig logits',tools.forward(w0,x)[0],'-> new logits',tools.forward(w,x)[0],'pick',tools.greedy(w,x)[0])
# follow-up: phase 3 served W2 -> only N0,N1 left, phase 3 green, ok
y=np.zeros(46); y[[0,1,12,13]]=0.3348; y[43]=1; y[44]=1; y[45]=0.2
print('follow-up (phase 3 green, cars on N0,N1): orig pick',tools.greedy(w0,y)[0],'new pick',tools.greedy(w,y)[0])
y2=y.copy(); y2[43]=0; y2[42]=1
print('variant (phase 2 green, cars on N0,N1): orig pick',tools.greedy(w0,y2)[0],'new pick',tools.greedy(w,y2)[0])
