"""DAgger distillation of Net 1, depth 3, iteration 1; 8000 labelled states"""


def decide(x):
    if x[11] <= 0.5022:  # q_W2
        if x[6] <= 0.1674:  # q_S0
            if x[10] <= 0.1674:  # q_W1
                return 1
            else:
                return 2
        else:
            if x[3] <= 0.5022:  # q_E0
                return 0
            else:
                return 2
    else:
        if x[16] <= 0.5022:  # d_E1
            if x[8] <= 0.5022:  # q_S2
                return 3
            else:
                return 3
        else:
            if x[17] <= 0.8348:  # d_E2
                return 2
            else:
                return 3
