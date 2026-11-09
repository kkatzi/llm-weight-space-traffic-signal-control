from try9 import *
def scores11(x, ew_up=0.12, ns_up=0.05):
    v = lambda l: 1.5*(x[ix['q_'+l]] + x[ix['d_'+l]])
    u = lambda *ls: 34.1*sum(x[ix['d_'+l]] for l in ls)
    nsT = v('N0')+v('S0')+v('S1'); nsL = v('N2')+v('S2'); ewT = v('E0')+v('E1')+v('W1'); ewL = v('E2')+v('W2')
    shared = 0.55*(v('N1')+v('W0'))
    ewUp = u('E0u','E1u','W0u','W1u'); nsUp = u('N0u','N1u','S0u','S1u')
    return np.array([v('N0') + v('S0') + 0.7*v('S1') + shared + 0.2*(nsL+ewL) + ns_up*nsUp,
                     1.4*v('N2') + 1.8*v('S2') + 0.05*ewUp,
                     0.65*v('E0') + 1.05*v('E1') + 0.8*v('W1') + shared + 0.3*(nsT+ewL) + ew_up*ewUp,
                     1.3*ewL + 0.2*(ewT+nsT+nsL) + ew_up*ewUp])
if __name__ == "__main__":
    pair_table(scores11)
