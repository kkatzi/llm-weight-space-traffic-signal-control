from harness import *
Gavg = np.load("Gavg.npy")
groups = {'nsT':['N0','S0','S1'], 'N1':['N1'], 'nsL':['N2','S2'], 'ewT':['E0','E1','W1'], 'W0':['W0'], 'ewL':['E2','W2']}
ug = {'nsUp':['N0u','N1u','S0u','S1u'], 'ewUp':['E0u','E1u','W0u','W1u']}
print("pairwise push per halted vehicle, averaged over lanes in each group (q+d)/3; upstream per vehicle (d)/34.1")
print("          "+" ".join(f"{g:>6s}" for g in list(groups)+list(ug)))
for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
    row = []
    for g, ls in groups.items():
        row.append(np.mean([(Gavg[a,ix['q_'+l]]+Gavg[a,ix['d_'+l]]-Gavg[b,ix['q_'+l]]-Gavg[b,ix['d_'+l]])/3 for l in ls]))
    for g, ls in ug.items():
        row.append(np.mean([(Gavg[a,ix['d_'+l]]-Gavg[b,ix['d_'+l]])/34.1 for l in ls]))
    print(f" L{a}-L{b}: "+" ".join(f"{v:6.2f}" for v in row))
