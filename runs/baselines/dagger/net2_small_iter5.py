"""DAgger distillation of Net 2, depth 3, iteration 5; 18983 labelled states"""


def decide(x):
    if x[8] <= 0.5022:  # q_S2
        if x[10] <= 0.1674:  # q_W1
            if x[11] <= 0.1674:  # q_W2
                return 0
            else:
                return 3
        else:
            if x[45] <= 0.7417:  # t_green
                return 2
            else:
                return 1
    else:
        if x[6] <= 0.5022:  # q_S0
            if x[5] <= 0.5022:  # q_E2
                return 1
            else:
                return 1
        else:
            if x[14] <= 0.8348:  # d_N2
                return 0
            else:
                return 1
