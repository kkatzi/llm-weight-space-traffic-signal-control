"""DAgger distillation of Net 1, depth 3, iteration 5; 19447 labelled states"""


def decide(x):
    if x[40] <= 0.5000:  # phase0
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
        if x[9] <= 0.1674:  # q_W0
            if x[23] <= 0.1674:  # d_W2
                return 1
            else:
                return 1
        else:
            if x[2] <= 0.5022:  # q_N2
                return 2
            else:
                return 1
