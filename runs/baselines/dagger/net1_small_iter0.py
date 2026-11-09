"""DAgger distillation of Net 1, depth 3, iteration 0; 5095 labelled states"""


def decide(x):
    if x[4] <= 0.1674:  # q_E1
        if x[5] <= 0.1674:  # q_E2
            if x[19] <= 0.1674:  # d_S1
                return 1
            else:
                return 0
        else:
            if x[8] <= 0.5022:  # q_S2
                return 3
            else:
                return 1
    else:
        if x[6] <= 0.5022:  # q_S0
            if x[2] <= 0.5022:  # q_N2
                return 2
            else:
                return 1
        else:
            if x[41] <= 0.5000:  # phase1
                return 0
            else:
                return 2
