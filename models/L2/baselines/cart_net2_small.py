"""CART distillation of Net 2, depth 3"""


def decide(x):
    if x[10] <= 0.1674:  # q_W1
        if x[17] <= 0.5022:  # d_E2
            if x[7] <= 0.1674:  # q_S1
                return 1
            else:
                return 0
        else:
            if x[18] <= 0.8348:  # d_S0
                return 3
            else:
                return 0
    else:
        if x[8] <= 0.5022:  # q_S2
            if x[23] <= 0.8348:  # d_W2
                return 2
            else:
                return 3
        else:
            if x[1] <= 0.5022:  # q_N1
                return 2
            else:
                return 0
