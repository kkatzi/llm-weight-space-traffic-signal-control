from harness import *
Gavg = np.load("Gavg.npy")
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
lanes = ['N0','N1','N2','E0','E1','E2','S0','S1','S2','W0','W1','W2']
own = {'N0':0,'N1':0,'S0':0,'S1':0,'N2':1,'S2':1,'E0':2,'E1':2,'W0':2,'W1':2,'E2':3,'W2':3}
def mkdec(G, label, nostay=False, bandgap=2.0):
    Bs = {}
    for p in range(4):
        m = ok & (cur==p) & (gap<bandgap)
        Bs[p] = (L[m] - X[m] @ G.T).mean(0)
    def dec(x):
        p = int(np.argmax(x[40:44])); l = Bs[p] + G @ x
        if nostay: l[p] = -1e9
        return int(np.argmax(l))
    r = report(dec, label)
    return Bs, r
mkdec(Gavg, "shared")
G = Gavg.copy(); a = (G[:,0:12]+G[:,12:24])/2; G[:,0:12]=a; G[:,12:24]=a
mkdec(G, "shared, q=d weights")
Gm = G.copy()
# own-only for stop-line lanes: weight = mean pairwise push (own minus each other), placed on own row only
G2 = Gm.copy(); G2[:,0:24] = 0
for l,a in own.items():
    for pre in ['q_','d_']:
        i = ix[pre+l]
        pw = np.mean([Gm[a,i]-Gm[b,i] for b in range(4) if b!=a])
        G2[a,i] = pw
mkdec(G2, "own-only stop lanes, full upstream")
G3 = G2.copy(); G3[:,24:40]=0
mkdec(G3, "own-only stop lanes, no upstream")
G4 = Gavg.copy(); G4[:,24:40] = 0
mkdec(G4, "shared (q,d separate), no upstream")
