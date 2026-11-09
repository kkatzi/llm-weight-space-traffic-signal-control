"""CART distillation of Net 0, depth 5"""


def decide(x):
    if x[5] <= 0.5022:  # q_E2
        if x[0] <= 0.5022:  # q_N0
            if x[2] <= 0.1674:  # q_N2
                if x[6] <= 0.1674:  # q_S0
                    if x[23] <= 0.5022:  # d_W2
                        return 2
                    else:
                        return 2
                else:
                    if x[42] <= 0.5000:  # phase2
                        return 2
                    else:
                        return 3
            else:
                if x[3] <= 0.5022:  # q_E0
                    if x[8] <= 0.5022:  # q_S2
                        return 1
                    else:
                        return 1
                else:
                    if x[2] <= 0.5022:  # q_N2
                        return 2
                    else:
                        return 1
        else:
            if x[41] <= 0.5000:  # phase1
                if x[23] <= 0.5022:  # d_W2
                    if x[18] <= 0.5022:  # d_S0
                        return 0
                    else:
                        return 0
                else:
                    if x[7] <= 0.5022:  # q_S1
                        return 3
                    else:
                        return 0
            else:
                if x[22] <= 0.5022:  # d_W1
                    if x[18] <= 0.5022:  # d_S0
                        return 2
                    else:
                        return 0
                else:
                    if x[7] <= 0.5022:  # q_S1
                        return 2
                    else:
                        return 2
    else:
        if x[4] <= 0.1674:  # q_E1
            if x[19] <= 0.8348:  # d_S1
                if x[11] <= 0.5022:  # q_W2
                    if x[12] <= 0.8348:  # d_N0
                        return 3
                    else:
                        return 0
                else:
                    if x[18] <= 0.8348:  # d_S0
                        return 3
                    else:
                        return 3
            else:
                if x[23] <= 0.8348:  # d_W2
                    if x[38] <= 0.0439:  # d_W0u
                        return 0
                    else:
                        return 0
                else:
                    if x[32] <= 0.0732:  # d_N0u
                        return 3
                    else:
                        return 0
        else:
            if x[14] <= 0.8348:  # d_N2
                if x[4] <= 0.5022:  # q_E1
                    if x[23] <= 0.5022:  # d_W2
                        return 2
                    else:
                        return 3
                else:
                    if x[22] <= 0.5022:  # d_W1
                        return 2
                    else:
                        return 2
            else:
                if x[8] <= 0.8348:  # q_S2
                    if x[15] <= 0.8348:  # d_E0
                        return 3
                    else:
                        return 2
                else:
                    if x[11] <= 0.8348:  # q_W2
                        return 1
                    else:
                        return 1
