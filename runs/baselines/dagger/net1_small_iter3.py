"""DAgger distillation of Net 1, depth 3, iteration 3; 14209 labelled states"""


def decide(x):
    if x[2] <= 0.5022:  # q_N2
        if x[10] <= 0.1674:  # q_W1
            if x[5] <= 0.1674:  # q_E2
                return 0
            else:
                return 3
        else:
            if x[11] <= 0.8348:  # q_W2
                return 2
            else:
                return 3
    else:
        if x[6] <= 0.5022:  # q_S0
            if x[4] <= 0.5022:  # q_E1
                return 1
            else:
                return 2
        else:
            if x[12] <= 0.8348:  # d_N0
                return 0
            else:
                return 0
