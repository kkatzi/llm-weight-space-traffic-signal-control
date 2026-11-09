"""DAgger distillation of Net 1, depth 3, iteration 2; 10919 labelled states"""


def decide(x):
    if x[10] <= 0.1674:  # q_W1
        if x[5] <= 0.1674:  # q_E2
            if x[11] <= 0.5022:  # q_W2
                return 0
            else:
                return 3
        else:
            if x[8] <= 0.5022:  # q_S2
                return 3
            else:
                return 1
    else:
        if x[11] <= 0.8348:  # q_W2
            if x[6] <= 0.5022:  # q_S0
                return 2
            else:
                return 0
        else:
            if x[4] <= 0.5022:  # q_E1
                return 3
            else:
                return 2
