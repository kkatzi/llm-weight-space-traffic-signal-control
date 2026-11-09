from try7 import *
np.set_printoptions(precision=2, suppress=True)
S_all = np.array([scores(x) for x in X])
top2 = np.argsort(-L,1)[:,:2]
for c in range(4):
    print(f"cur {c}: mean residual r_ab = (La-Lb)/K - (Sa-Sb) over states whose top-2 are a,b and gap<2  [n]")
    for a in range(4):
        row = []
        for b in range(4):
            m = ok & (cur==c) & (((top2[:,0]==a)&(top2[:,1]==b)) | ((top2[:,0]==b)&(top2[:,1]==a))) & (gap<2)
            if a!=b and m.sum()>=3:
                r = (L[m,a]-L[m,b])/K - (S_all[m,a]-S_all[m,b])
                row.append(f"{r.mean():6.2f}[{m.sum():3d}]")
            else: row.append("     .     ")
        print("   ", " ".join(row))
