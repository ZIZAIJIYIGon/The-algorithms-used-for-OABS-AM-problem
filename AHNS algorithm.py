import random
import pandas as pd
import functions
import copy
import numpy as np
import time
import matplotlib.pyplot as lib
import math
import initialization as ini


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
# ———————————————————————————————————    AHNS algorithm    ——————————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


# temporary variables initialization
A = ini.A
N1 = 900                                            # The value of N1
N2 = (1/10) * N1                                    # The value of N2
N3 = (1/6) * N1                                     # The value of N3
ini_wo = [130, 130, 130, 130, 24, 60, 60, 60, 60]   # The score of each operator in initialized wo
wo = copy.deepcopy(ini_wo)                          # wo
operator = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])    # the selection list of operators
iteration = 0                                       # iteration
p_pick = 0                                          # Probability of accepting inferior solutions
Not_even_improve = 0                                # NI_1
Not_improve = 0                                     # NI_2
No_improve = 0                                      # NI_3

# update temporary variables
origin_solution = copy.deepcopy(A)                                                     # IS
origin_value = functions.profit(origin_solution)  # IV
great_solution = copy.deepcopy(A)   # GOS
great_value = origin_value          # GOV
global_solution = copy.deepcopy(A)  # LOS
global_value = origin_value         # LOV
local_solution = copy.deepcopy(A)   # PS
local_value = origin_value          # PV
solution = copy.deepcopy(A)         # CS
value = copy.deepcopy(A)            # CV

# the list for recording temporary variables
greatest_value = [great_value]      # the list for recording new gov
best_value = [global_value]         # the list for recording new lov
best_solution = [global_solution]   # BL
best_iteration = [0]                # the list for recording the iteration when finding a new los
forbidden = []                      # FL

print('optimization begins')

# main loop
while Not_even_improve <= N1:               # stop criterion

    pick_prob = [i/sum(wo) for i in wo]     # Calculate the selection probability of each operator based on wo

    if Not_even_improve % 100 == 0 and Not_even_improve != 0:  # visualization of NI_1 during optimization
        print('has not optimized for %d times' % Not_even_improve)

    solution = functions.reschedule(solution)
    pick_result = np.random.choice(operator, p=pick_prob)  # pick an operator
    iteration += 1

    if pick_result == 1:  # insert operator
        solution = functions.insert(solution)

    elif pick_result == 2:  # next_insert operator
        solution = functions.next_insert(solution)

    elif pick_result == 3:  # switch operator
        solution = functions.switch(solution)

    elif pick_result == 4:  # more_batch operator
        solution = functions.more_batch(solution)

    elif pick_result == 5:  # accept operator
        solution = functions.accept(solution)

    elif pick_result == 6:  # random_reject operator
        solution = functions.random_reject(solution)

    elif pick_result == 7:  # tardiness_reject operator
        solution = functions.time_reject(solution)

    elif pick_result == 8:  # cost_reject operator
        solution = functions.cost_reject(solution)

    elif pick_result == 9:  # volume_reject operator
        solution = functions.accept(solution)

    time3 = time.time()
    value = functions.profit(solution)

    # score the operator in wo according to the comparison of CV and GOV, LOV, LV
    if value <= local_value and value/local_value < 500:  # Prevent errors in calculating e ^ (CV - LV) when the difference between CV and LV is too large
        p_pick = math.exp(round((value - local_value)/local_value))

    if value in forbidden:  # If CV is already in FL then jump into the next iteration
        continue

    if value > global_value and functions.checksquare(solution) and functions.checksquare(solution):  # CV > LOV
        wo[pick_result-1] += 50             # Psi_2
        global_solution = copy.deepcopy(solution)
        global_value = value
        best_value.append(value)
        best_solution.append(solution)
        best_iteration.append(iteration)
        No_improve = 0
        Not_improve = 0
        Not_even_improve += 1

        if value >= max(best_value):        # CV > GOV
            wo[pick_result - 1] += 150      # Psi_1
            great_solution = copy.deepcopy(solution)
            great_value = value
            greatest_value.append(great_value)
            Not_even_improve = 0
            print('has optimized for', len(greatest_value)-1, 'time')  # visualization of optimized situation
            print('GOV is', great_value)
            print('')

    elif value > local_value:               # CV > LV
        wo[pick_result-1] += 5              # Psi_3
        Not_improve += 1
        Not_even_improve += 1

    elif random.random() < p_pick:          # e^(CV-LV) > random(0，1)
        wo[pick_result-1] += 1              # Psi_4
        No_improve += 1
        Not_improve += 1
        Not_even_improve += 1

    else:                                   # Psi_5
        No_improve += 1
        Not_improve += 1
        Not_even_improve += 1

    if No_improve >= 30:                    # backtracking strategy 1
        wo = copy.deepcopy(ini_wo)
        solution = copy.deepcopy(global_solution)
        value = global_value

    # if Not_improve >= N2:                   # backtracking strategy 2
    #     Not_improve = 0
    #     index_math = round(Not_even_improve/N2)
    #     forbidden.append(global_value)
    #
    #     # the longer time for non_optimized, the earlier solution in BL is selected
    #     if index_math <= len(best_value) and index_math < 3:
    #         solution = copy.deepcopy(best_solution[-index_math])
    #         value = best_value[-index_math]
    #     else:
    #         value = random.sample(best_value, 1)[0]
    #         index_relocate = best_value.index(value)
    #         solution = copy.deepcopy(best_solution[index_relocate])
    #
    #     global_solution = copy.deepcopy(solution)
    #     global_value = value
    #
    # if Not_even_improve > N3 and Not_even_improve % N3 == 0:  # backtracking strategy 3
    #     forbidden.append(global_value)
    #     value = random.sample(best_value, 1)[0]
    #     index_relocate = best_value.index(value)
    #     solution = copy.deepcopy(best_solution[index_relocate])

    local_solution = copy.deepcopy(solution)
    local_value = value


# output visualization
mylist = [i[1] for i in great_solution]
reject_num = mylist.count(-1)
reject_percentage = 100*reject_num/num_part
time2 = time.time()
execution_time = time2-ini.time1
great_solution = functions.reschedule(great_solution)

print('')
print('GOS is                          ', great_solution)
print('GOV is                          ', great_value)
print('batch number of GOS is          ', max(uu[1] for uu in great_solution)+1)
print('total tardiness of GOS is       ', functions.tardiness(great_solution))
print('rejection rate of GOS is:        %.2f %%' % reject_percentage)
print('total CPU time is:              ', execution_time, 's')
print('total iteration is:              %d' % iteration)

lib.plot(best_iteration, best_value)
lib.show()
