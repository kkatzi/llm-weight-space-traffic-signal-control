"""DAgger distillation of Net 2, depth 3, iteration 1; 8121 labelled states"""


def decide(x):
    if x[10] <= 0.1674:  # q_W1
        if x[11] <= 0.1674:  # q_W2
            if x[1] <= 0.1674:  # q_N1
                return 1
            else:
                return 0
        else:
            if x[7] <= 0.5022:  # q_S1
                return 3
            else:
                return 0
    else:
        if x[14] <= 0.8348:  # d_N2
            if x[3] <= 0.5022:  # q_E0
                return 2
            else:
                return 2
        else:
            if x[20] <= 0.5022:  # d_S2
                return 2
            else:
                return 1
