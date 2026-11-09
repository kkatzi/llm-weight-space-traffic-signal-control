from harness import *
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]; top2 = np.argsort(-L,1)[:,:2]
def sc_small(x, lw=1.6, sh=0.5, up=0.1, cross=0.0):
    v = lambda l: 1.5*(x[ix['q_'+l]] + x[ix['d_'+l]])
    nsT = v('N0')+v('S0')+v('S1'); nsL = v('N2')+v('S2'); ewT = v('E0')+v('E1')+v('W1'); ewL = v('E2')+v('W2')
    shared = sh*(v('N1')+v('W0'))
    ewUp = 34.1*sum(x[ix['d_'+l]] for l in ['E0u','E1u','W0u','W1u'])
    s = np.array([nsT + shared, lw*nsL, ewT + shared + up*ewUp, 1.3*ewL + up*ewUp])
    return s
def pair_table(scorefun, K=2.3):
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
    pair_table(sc_small)
