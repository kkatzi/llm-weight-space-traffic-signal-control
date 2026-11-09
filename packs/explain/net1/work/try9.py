from harness import *
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
top2 = np.argsort(-L,1)[:,:2]
K = 2.3
def scores9(x):
    v = lambda l: 1.5*(x[ix['q_'+l]] + x[ix['d_'+l]])
    u = lambda *ls: 34.1*sum(x[ix['d_'+l]] for l in ls)
    nsT = v('N0')+v('S0')+v('S1'); nsL = v('N2')+v('S2'); ewT = v('E0')+v('E1')+v('W1'); ewL = v('E2')+v('W2')
    shared = 0.55*(v('N1')+v('W0'))
    ewUp = u('E0u','E1u','W0u','W1u'); nsUp = u('N0u','N1u','S0u','S1u')
    return np.array([v('N0') + v('S0') + 0.7*v('S1') + shared + 0.2*(nsL+ewL) + 0.03*nsUp,
                     1.4*v('N2') + 1.8*v('S2') + 0.04*ewUp,
                     0.65*v('E0') + 1.05*v('E1') + 0.8*v('W1') + shared + 0.3*(nsT+ewL) + 0.09*ewUp,
                     1.3*ewL + 0.2*(ewT+nsT+nsL) + 0.09*ewUp])
def pair_table(scorefun, K=K):
    S_all = np.array([scorefun(x) for x in X])
    for c in range(4):
        print(f"cur {c}: r_ab")
        for a in range(4):
            row = []
            for b in range(4):
                m = ok & (cur==c) & (((top2[:,0]==a)&(top2[:,1]==b)) | ((top2[:,0]==b)&(top2[:,1]==a))) & (gap<2)
                if a!=b and m.sum()>=3:
                    r = (L[m,a]-L[m,b])/K - (S_all[m,a]-S_all[m,b]); row.append(f"{r.mean():6.2f}[{m.sum():3d}]")
                else: row.append("     .     ")
            print("   ", " ".join(row))
if __name__ == "__main__":
    pair_table(scores9)
