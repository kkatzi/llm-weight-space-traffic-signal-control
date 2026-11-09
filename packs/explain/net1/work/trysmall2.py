from trysmall import *
def sc_small2(x):
    v = lambda l: 1.5*(x[ix['q_'+l]] + x[ix['d_'+l]])
    nsT = v('N0')+v('S0')+v('S1'); nsL = v('N2')+v('S2'); ewT = v('E0')+v('E1')+v('W1'); ewL = v('E2')+v('W2')
    shared = 0.5*(v('N1')+v('W0'))
    T = nsT + nsL + ewT + ewL
    ewUp = 34.1*sum(x[ix['d_'+l]] for l in ['E0u','E1u','W0u','W1u'])
    return np.array([nsT + shared + 0.2*(T-nsT), 1.6*nsL, ewT + shared + 0.2*(T-ewT) + 0.1*ewUp, 1.3*ewL + 0.2*(T-ewL) + 0.1*ewUp])
pair_table(sc_small2)
