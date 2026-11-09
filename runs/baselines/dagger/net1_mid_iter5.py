"""DAgger distillation of Net 1, depth 5, iteration 5; 18680 labelled states"""


def decide(x):
    if x[10] <= 0.1674:  # q_W1
        if x[11] <= 0.1674:  # q_W2
            if x[1] <= 0.1674:  # q_N1
                if x[6] <= 0.1674:  # q_S0
                    if x[45] <= 0.9083:  # t_green
                        return 1
                    else:
                        return 1
                else:
                    if x[45] <= 0.5750:  # t_green
                        return 0
                    else:
                        return 0
            else:
                if x[17] <= 0.8348:  # d_E2
                    if x[8] <= 0.5022:  # q_S2
                        return 0
                    else:
                        return 0
                else:
                    if x[2] <= 0.5022:  # q_N2
                        return 3
                    else:
                        return 1
        else:
            if x[8] <= 0.5022:  # q_S2
                if x[11] <= 0.5022:  # q_W2
                    if x[2] <= 0.8348:  # q_N2
                        return 3
                    else:
                        return 1
                else:
                    if x[0] <= 0.5022:  # q_N0
                        return 3
                    else:
                        return 3
            else:
                if x[17] <= 0.5022:  # d_E2
                    if x[2] <= 0.1674:  # q_N2
                        return 3
                    else:
                        return 1
                else:
                    if x[14] <= 0.5022:  # d_N2
                        return 3
                    else:
                        return 1
    else:
        if x[2] <= 0.5022:  # q_N2
            if x[18] <= 0.5022:  # d_S0
                if x[11] <= 0.5022:  # q_W2
                    if x[8] <= 0.5022:  # q_S2
                        return 2
                    else:
                        return 2
                else:
                    if x[16] <= 0.5022:  # d_E1
                        return 3
                    else:
                        return 2
            else:
                if x[4] <= 0.5022:  # q_E1
                    if x[5] <= 0.1674:  # q_E2
                        return 0
                    else:
                        return 3
                else:
                    if x[7] <= 0.8348:  # q_S1
                        return 2
                    else:
                        return 0
        else:
            if x[18] <= 0.5022:  # d_S0
                if x[8] <= 0.5022:  # q_S2
                    if x[16] <= 0.5022:  # d_E1
                        return 1
                    else:
                        return 2
                else:
                    if x[4] <= 0.8348:  # q_E1
                        return 1
                    else:
                        return 2
            else:
                if x[0] <= 0.5022:  # q_N0
                    if x[4] <= 0.5022:  # q_E1
                        return 1
                    else:
                        return 2
                else:
                    if x[7] <= 0.1674:  # q_S1
                        return 1
                    else:
                        return 0
