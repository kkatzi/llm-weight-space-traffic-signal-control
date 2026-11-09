"""DAgger distillation of Net 1, depth 3, iteration 4; 17512 labelled states"""


def decide(x):
    if x[40] <= 0.5000:  # phase0
        if x[11] <= 0.5022:  # q_W2
            if x[5] <= 0.1674:  # q_E2
                return 0
            else:
                return 3
        else:
            if x[16] <= 0.5022:  # d_E1
                return 3
            else:
                return 2
    else:
        if x[4] <= 0.1674:  # q_E1
            if x[23] <= 0.1674:  # d_W2
                return 1
            else:
                return 1
        else:
            if x[2] <= 0.5022:  # q_N2
                return 2
            else:
                return 1
