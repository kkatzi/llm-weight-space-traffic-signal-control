# diagnostic only: how much would per-current-phase group weights gain over shared group weights?
from harness import *
GG = np.load("GG.npy"); Gavg = np.load("Gavg.npy")
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
groups = {'nsT':['N0','S0','S1'], 'N1':['N1'], 'nsL':['N2','S2'], 'ewT':['E0','E1','W1'], 'W0':['W0'], 'ewL':['E2','W2']}
ug = {'nsUp':['N0u','N1u','S0u','S1u'], 'ewUp':['E0u','E1u','W0u','W1u']}
def feats(X):
    F = []
    for g, ls in groups.items(): F.append(sum(1.5*(X[:,ix['q_'+l]]+X[:,ix['d_'+l]]) for l in ls))
    for g, ls in ug.items(): F.append(sum(34.1*X[:,ix['d_'+l]] for l in ls))
    return np.stack(F,1)
F = feats(X)
def coef(G):
    C = np.zeros((4, 8))
    for k,(g, ls) in enumerate(groups.items()):
        C[:,k] = np.mean([(G[:,ix['q_'+l]]+G[:,ix['d_'+l]])/3 for l in ls],0)
    for k,(g, ls) in enumerate(ug.items()):
        C[:,6+k] = np.mean([G[:,ix['d_'+l]]/34.1 for l in ls],0)
    return C
top2 = np.argsort(-L,1)[:,:2]
def run(Cfun, label):
    Bs = {}
    for c in range(4):
        C = Cfun(c); S = F @ C.T
        # pairwise residual least squares (diagnostic)
        rows, rhs, wts = [], [], []
        for a in range(4):
            for b in range(a+1,4):
                m = ok & (cur==c) & (((top2[:,0]==a)&(top2[:,1]==b))|((top2[:,0]==b)&(top2[:,1]==a))) & (gap<2)
                if m.sum() < 3: continue
                r = ((L[m,a]-L[m,b]) - (S[m,a]-S[m,b])).mean()
                e = np.zeros(4); e[a]=1; e[b]=-1
                rows.append(e*np.sqrt(m.sum())); rhs.append(r*np.sqrt(m.sum()))
        rows.append(np.array([1,1,1,1])); rhs.append(0)
        B = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        if c == 0: B[0] = -99
        Bs[c] = (C, B)
    def dec(x):
        c = int(np.argmax(x[40:44])); C, B = Bs[c]
        return int(np.argmax(C @ feats(x[None])[0] + B))
    report(dec, label)
run(lambda c: coef(Gavg), "shared group weights")
run(lambda c: coef(GG[c]), "per-cur group weights")
