import random
import pandas as pd
import functions
import copy
import numpy as np
import time
import matplotlib.pyplot as lib
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
S_machine = 625                 # build chamber’s base area
TM_machine = 0.03               # Processing time per unit volume
TP_machine = 0.8                # Processing time per unit height
Tset_machine = 2                # setup time
CS_machine = 60                 # The cost of using additive manufacturing machines per unit time (from Li 2019)
CV_order = 2                    # The cost of additive manufacturing materials per unit volume


# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #
# ——————————————————————————————————————    Main loop    ————————————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


print('TS algorithm begins')
# Parameter setting
CG = 0                                  # Current Generation CG
RG = 0                                  # Restart Generation RG
len_tabu_list = round(num_part/6)       # Length of tabu list
len_NCS = len_tabu_list + 2             # The size of the neighborhood space
tabu_list = [0] * len_tabu_list         # Initialization of tabu list TL
operator = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # index of each operator
pick_probable = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]   # selection probability of each operator


# Initial solution process
A = ini.A
output_solution = copy.deepcopy(A)                         # OS, output solution
output_value = functions.profit(A)  # The functional value of OS
Current_solution = copy.deepcopy(A)                        # CS, current solution
value = output_value                                       # The value of CS
optimal_solution_list = [output_solution]                  # OSL, optimal solution list
best_value = [output_value]                                # List for recording the output_value ,used for visualization
best_iteration = [0]                                       # List for recording the iteration everytime TS reaches a new OS, for visualization
iteration = 0                                              # Iteration


# Main loop
while CG <= 400:

    forbidden = 0                   # The parameter to judge the tabu condition
    Neighbor_CS = []                # Neighborhood solution
    value_NCS = []                  # The value of each solution in Neighbor_CS
    CG += 1
    RG += 1
    iteration += 1
    if CG % 50 == 0 and CG != 0:    # Visualization of CG
        print('has not optimized for %d iterations' % CG)

    # Neighborhood search
    while len(value_NCS) < len_NCS:
        pick = np.random.choice(operator, p=pick_probable)
        temp_solution = copy.deepcopy(Current_solution)

        if pick == 1:
            temp_solution = functions.switch(temp_solution)

        elif pick == 2:
            temp_solution = functions.insert(temp_solution)

        elif pick == 3:
            temp_solution = functions.next_insert(temp_solution)

        elif pick == 4:
            temp_solution = functions.time_reject(temp_solution)

        elif pick == 5:
            temp_solution = functions.random_reject(temp_solution)

        elif pick == 6:
            temp_solution = functions.accept(temp_solution)

        elif pick == 7:
            temp_solution = functions.more_batch(temp_solution)

        elif pick == 8:
            temp_solution = functions.cost_reject(temp_solution)

        elif pick == 9:
            temp_solution = functions.volume_reject(temp_solution)

        if temp_solution not in Neighbor_CS:
            value_NCS.append(functions.profit(temp_solution))
            Neighbor_CS.append(temp_solution)  # Add neighborhood solutions

    # Tabu list checking
    while forbidden == 0:
        good_value = max(value_NCS)  # Choose the solution with maximum functional value in neighborhood as BS
        good_solution = Neighbor_CS[value_NCS.index(good_value)]
        if good_solution not in tabu_list or good_value > value:  # The former judgement is checking the tabu list, while the latter is the principle of exception
            forbidden = 1                                         # update tabu condition
            Current_solution = copy.deepcopy(good_solution)
            Current_solution = functions.reschedule(Current_solution)
            value = functions.profit(Current_solution)

            if good_solution not in tabu_list:      # tabu list process
                tabu_list.append(Current_solution)  # add CS to TL
                tabu_list.pop(0)                    # remove the earliest recorded solution in TL

        value_NCS.remove(functions.profit(good_solution))
        Neighbor_CS.remove(good_solution)   # If BS does not meet the tabu conditions and amnesty criteria,
                                            # then remove BS from the neighborhood and search for the suboptimal
                                            # solution in the neighborhood space.

    # Update OS
    if value > output_value and functions.checksquare(Current_solution) and functions.checktime(Current_solution):
        CG = 0
        RG = 0
        output_solution = copy.deepcopy(Current_solution)
        output_value = value
        best_value.append(output_value)
        optimal_solution_list.append(output_solution)
        best_iteration.append(iteration)
        print('')
        print('has optimized for', len(best_value) - 1, 'times')
        print('the output value is:', output_value)
        print('')

    # Backtracking strategy
    if RG >= 50:
        RG = 0
        if CG <= 200:
            value = output_value
            Current_solution = copy.deepcopy(output_solution)
            tabu_list = [0] * len_tabu_list
        elif CG > 200:
            value = random.sample(best_value, 1)[0]
            index_relocate = best_value.index(value)
            Current_solution = copy.deepcopy(optimal_solution_list[index_relocate])

# Visualization
mylist = [i[1] for i in output_solution]
reject_num = mylist.count(-1)
reject_percentage = 100*reject_num/num_part
time2 = time.time()
execution_time = time2-ini.time1
global_solution = functions.reschedule(output_solution)

print('')
print('OS is:                     ', output_solution)
print('value of OS is:            ', output_value)
print('the number of batch is:    ', max(u1[1] for u1 in global_solution)+1)
print('total CPU time is:         ', execution_time, 's')
print('total tardiness is:        ', functions.tardiness(global_solution))
print('rejection rate of OS is:    %.2f %%' % reject_percentage)
print('total iteration is:         %d' % iteration)

lib.plot(best_iteration, best_value)
lib.show()
