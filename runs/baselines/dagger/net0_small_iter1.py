"""DAgger distillation of Net 0, depth 3, iteration 1; 9146 labelled states"""


def decide(x):
    if x[11] <= 0.5022:  # q_W2
        if x[19] <= 0.5022:  # d_S1
            if x[2] <= 0.1674:  # q_N2
                return 2
            else:
                return 1
        else:
            if x[41] <= 0.5000:  # phase1
                return 0
            else:
                return 2
    else:
        if x[4] <= 0.1674:  # q_E1
            if x[19] <= 0.8348:  # d_S1
                return 3
            else:
                return 0
        else:
            if x[4] <= 0.5022:  # q_E1
                return 2
            else:
                return 2
