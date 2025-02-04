import pandas as pd
import functions
import time


# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #
# ————————————————————————————————————    Preparations    ———————————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


# get order's data from instances file
filename = functions.filename
df = pd.read_csv(filename, header=None)
data_list = df.values.tolist()
height_order = data_list[0]               # height of the order
square_order = data_list[1]               # bottom area of the order
volume_order = data_list[2]               # volume of the order
TR_order = data_list[3]                   # r_i of the order
TD_order = data_list[4]                   # d_i of the order
TE_order = data_list[5]                   # e_i of the order
profit_order = data_list[6]               # mu_i of the order
tardiness_penalty_order = data_list[7]    # theta_i of the order
rejection_penalty_order = data_list[8]    # eta_i of the order
n_i_order = [a/b for a, b in zip(height_order, volume_order)]  # used for volume_reject operator

# parameter setting
num_part = len(height_order)
part = range(num_part)
INT_MAX = 20000                 # M (sufficiently large number)
S_machine = 625                 # build chamber’s base area
TM_machine = 0.03               # Processing time per unit volume
TP_machine = 0.8                # Processing time per unit height
Tset_machine = 2                # setup time
CS_machine = 60                 # The cost of using additive manufacturing machines per unit time (from Li 2019)
CV_order = 2                    # The cost of additive manufacturing materials per unit volume


# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #
# ————————————————————————————————   Initialized Solution    ————————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


# Record the start time of the algorithm
time1 = time.time()


# initial solution generation


A = [[] for _ in range(len(part))]
index1 = 0                # the index of batch
square = 0                # the square of each batch
temporary_start_time = 0  # the start time of each temporary batch
temporary_end_time = 0    # the completion time of each temporary batch
timespan = []             # the start time of each batch
D = []                    # temporary batch
E = []                    # the index of order in temporary batch
G = 0                     # the produce time of last batch
Max_tr = 0                # the maximum r_i in a temporary batch


for i in range(len(A)):
    square += square_order[i]
    A[i].append(i)
    A[i].append(index1)
    D.append(A[i])
    E.append(A[i][0])

    if index1 == 0:       # The start and completion time of the first temporary batch
        for q in range(len(D)):
            if TR_order[D[q][0]] > Max_tr:
                Max_tr = TR_order[D[q][0]]
        temporary_start_time = Max_tr
        temporary_end_time = temporary_start_time + functions.span(D)
    else:                 # The start and completion time of other temporary batch
        temporary_before_end_time = timespan[index1-1] + G
        for p in range(len(D)):
            if TR_order[D[p][0]] > Max_tr:
                Max_tr = TR_order[D[p][0]]
        temporary_start_time = max(Max_tr, temporary_before_end_time)
        temporary_end_time = temporary_start_time + functions.span(D)

    for t in E:
        if TR_order[t] <= temporary_start_time and TE_order[t] >= temporary_end_time and square <= S_machine:
            continue      # If both the time and area are feasible, continue to fill the order into the current batch
        else:             # otherwise, try to fill the order into the next batch
            C = D[:len(D)-1]  # Remove the last order of D, consider C as a new batch
            F = E[:len(E)-1]  # the index of orders in C

            # Calculate the start time of each batch
            if index1 == 0:   # calculate the start time of first batch, only consider max_tr
                Max_TR = 0
                for x in F:
                    if TR_order[x] >= Max_TR:
                        Max_TR = TR_order[x]
                timespan.append(Max_TR)
                G = functions.span(C)
            else:  # Calculate the start time of the latter batch, with a value of max (the start time of the previous batch + the completion time of the previous batch, max_tr)
                Max_TR = 0
                for y in F:
                    if TR_order[y] >= Max_TR:
                        Max_TR = TR_order[y]
                span_completion_time = timespan[index1-1] + G
                G = functions.span(C)
                timespan.append(max(Max_TR, span_completion_time))
            index1 += 1
            A[i][1] = index1
            D = [A[i]]
            E = [i]

            # If the current order cannot meet the constraints when added to the next batch, it will be rejected
            single_start_time = max(TR_order[i], timespan[index1-1] + G)
            singe_end_time = single_start_time + functions.span(D)
            if singe_end_time > TE_order[i]:
                A[i][1] = -1
                D = []
                E = []

            Max_tr = 0
            square = square_order[i]
            break

A = functions.reschedule(A)
