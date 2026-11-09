from harness import *
GG = np.load("GG.npy")
groups = {'nsT':['N0','S0','S1'], 'N1':['N1'], 'nsL':['N2','S2'], 'ewT':['E0','E1','W1'], 'W0':['W0'], 'ewL':['E2','W2']}
ug = {'nsUp':['N0u','N1u','S0u','S1u'], 'ewUp':['E0u','E1u','W0u','W1u']}
def show(G, label, pairs):
    print(label)
    print("          "+" ".join(f"{g:>6s}" for g in list(groups)+list(ug)))
    for a,b in pairs:
        row = []
        for g, ls in groups.items():
            row.append(np.mean([(G[a,ix['q_'+l]]+G[a,ix['d_'+l]]-G[b,ix['q_'+l]]-G[b,ix['d_'+l]])/3 for l in ls]))
        for g, ls in ug.items():
            row.append(np.mean([(G[a,ix['d_'+l]]-G[b,ix['d_'+l]])/34.1 for l in ls]))
        print(f" L{a}-L{b}: "+" ".join(f"{v:6.2f}" for v in row))
show(np.load("Gavg.npy"), "shared", [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])
show(GG[0], "cur 0", [(1,2),(1,3),(2,3)])
show(GG[1], "cur 1", [(0,2),(0,3),(2,3)])
show(GG[2], "cur 2", [(0,1),(0,3),(1,3)])
show(GG[3], "cur 3", [(0,1),(0,2),(1,2)])
