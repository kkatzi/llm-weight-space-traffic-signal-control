"""DAgger distillation of Net 1, depth 5, iteration 0; 5095 labelled states"""


def decide(x):
    if x[4] <= 0.1674:  # q_E1
        if x[5] <= 0.1674:  # q_E2
            if x[19] <= 0.1674:  # d_S1
                if x[20] <= 0.5022:  # d_S2
                    if x[0] <= 0.1674:  # q_N0
                        return 2
                    else:
                        return 0
                else:
                    if x[1] <= 0.1674:  # q_N1
                        return 1
                    else:
                        return 0
            else:
                if x[6] <= 0.5022:  # q_S0
                    if x[20] <= 0.8348:  # d_S2
                        return 0
                    else:
                        return 1
                else:
                    if x[20] <= 0.8348:  # d_S2
                        return 0
                    else:
                        return 0
        else:
            if x[8] <= 0.5022:  # q_S2
                if x[0] <= 0.5022:  # q_N0
                    if x[23] <= 0.1674:  # d_W2
                        return 3
                    else:
                        return 3
                else:
                    if x[5] <= 0.5022:  # q_E2
                        return 0
                    else:
                        return 3
            else:
                if x[14] <= 0.5022:  # d_N2
                    if x[11] <= 0.5022:  # q_W2
                        return 1
                    else:
                        return 3
                else:
                    if x[19] <= 0.5022:  # d_S1
                        return 1
                    else:
                        return 0
    else:
        if x[6] <= 0.5022:  # q_S0
            if x[2] <= 0.5022:  # q_N2
                if x[11] <= 0.8348:  # q_W2
                    if x[10] <= 0.1674:  # q_W1
                        return 2
                    else:
                        return 2
                else:
                    if x[16] <= 0.8348:  # d_E1
                        return 3
                    else:
                        return 2
            else:
                if x[20] <= 0.5022:  # d_S2
                    if x[4] <= 0.5022:  # q_E1
                        return 2
                    else:
                        return 2
                else:
                    if x[3] <= 0.8348:  # q_E0
                        return 1
                    else:
                        return 1
        else:
            if x[41] <= 0.5000:  # phase1
                if x[12] <= 0.8348:  # d_N0
                    if x[4] <= 0.5022:  # q_E1
                        return 0
                    else:
                        return 2
                else:
                    if x[22] <= 0.8348:  # d_W1
                        return 0
                    else:
                        return 0
            else:
                if x[17] <= 0.8348:  # d_E2
                    if x[9] <= 0.5022:  # q_W0
                        return 2
                    else:
                        return 2
                else:
                    if x[22] <= 0.5022:  # d_W1
                        return 3
                    else:
                        return 2
