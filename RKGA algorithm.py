import random
import pandas as pd
import functions
import copy
import numpy as np
import time
import initialization as ini
import matplotlib.pyplot as lib

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
# ————————————————————————————————   Population Initialization   ————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


print('Initial population generation procedure begins')
iter_ini = 0                                                    # iteration of initial population generation
A = ini.A                                                       # get a feasible initial solution
A1 = copy.deepcopy(A)                                           # current solution
value_A1 = functions.profit(A, functions.timeline(A))           # value of current solution
pop_ini = []                                                    # initial population
value_ini = []                                                  # the value of each solution in initial population
operator = [1, 2, 3, 4, 5, 6, 7, 8, 9]                          # correspond to each operator
probability = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]     # probability of being select of each operator

while len(pop_ini) < 10:
    if iter_ini % 20 == 0:    # For every 20 iterations, backtrack the solution
        A1 = copy.deepcopy(A)
    iter_ini += 1

    if iter_ini % 5 == 0 and iter_ini != 0:  # visualization
        print('ini_iteration is:', iter_ini)

    for x in range(40):
        pick_result = np.random.choice(operator, p=probability)
        if pick_result == 1:
            A1 = functions.switch(A1)

        elif pick_result == 2:
            A1 = functions.insert(A1)

        elif pick_result == 3:
            A1 = functions.next_insert(A1)

        elif pick_result == 4:
            A1 = functions.more_batch(A1)

        elif pick_result == 5:
            A1 = functions.random_reject(A1)

        elif pick_result == 6:
            A1 = functions.time_reject(A1)

        elif pick_result == 7:
            A1 = functions.volume_reject_rkga(A1)

        elif pick_result == 8:
            A1 = functions.cost_reject(A1)

        else:
            A1 = functions.accept(A1)

        if iter_ini <= 40:
            # Only when both time and area are feasible, the profit is positive,
            # and the solution has not been recorded in initial population
            # will the solution be added to the initial population.
            if functions.checktime(A1) and functions.checksquare(A1) and functions.profit(A1, functions.timeline(A1)) >= 0 and A1 not in pop_ini:
                pop_ini.append(A1)
                value_ini.append(functions.profit(A1, functions.timeline(A1)))
                print('has found', len(pop_ini), 'individuals')   # visualization
                break

        else:
            if iter_ini <= 80:
                # When the profit is positive and solution has not been recorded in initial population,
                # will the solution be added to the initial population.
                if functions.profit(A1, functions.timeline(A1)) >= 0 and A1 not in pop_ini:
                    pop_ini.append(A1)
                    value_ini.append(functions.profit(A1, functions.timeline(A1)))
                    break

            else:
                # When the profit is positive,
                # will the solution be added to the initial population.
                if functions.profit(A1, functions.timeline(A1)) >= 0:
                    pop_ini.append(A1)
                    value_ini.append(functions.profit(A1, functions.timeline(A1)))
                    break

for t in pop_ini:
    t = functions.reschedule(t)
time2 = time.time()


# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #
# ———————————————————————————————————    RKGA algorithm    ——————————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


# algorithm parameter setting
not_improve = 0                 # the iteration without optimization implemented
cross_probable = 0.2            # the probability of crossover
mutate_probable = 0.1           # the probability of mutation
Elite_num = 2                   # number of elite individuals in population
Non_Elite_num = 8               # number of non_elite individuals in population
pop_scale = 10                  # the scale of population

# temporary variables initialization
iteration = 0                                           # The iteration
pop = copy.deepcopy(pop_ini)                            # The current population
value_pop = copy.deepcopy(value_ini)                    # The value of each solution in pop
global_value = max(value_pop)                           # The optimal function value ever obtained
global_solution = pop[value_pop.index(global_value)]    # The solution correspond to the global_value
local_value = copy.deepcopy(global_value)               # The optimal function value in the current iteration
local_solution = copy.deepcopy(global_solution)         # The solution correspond to the local_value
best_value = [global_value]                             # The list for recording global_value (used for visualization)
iteration_best = [0]     # The list for recording the iteration when finding a new global value (used for visualization)

# main loop
print('')
print('Initial solution generation completed, RKGA begins')
while not_improve <= 200:
    iteration += 1
    pop_now = []                            # The population completely generated by mutation and crossover
    value_now = []                          # The value of each solution in pop_now
    operator = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # The index of each operator

    while len(pop_now) < 2*pop_scale:
        # Using roulette wheel to select parents
        # based on the proportion of each individual's function value to the sum of all function values.
        total_value = sum(value_pop)
        probabilities = [x / total_value for x in value_pop]
        parent = random.choices(pop, probabilities, k=2)
        dad = parent[1]
        mom = parent[0]

        a1 = random.random()        # crossover random number
        a2 = random.random()        # mutation random number
        if a1 <= cross_probable:    # crossover
            children = functions.crossover(dad, mom, pop)
        else:
            children = [dad, mom]

        if a2 <= mutate_probable:   # Randomly mutate two children
            mutate_pick = np.random.choice(operator, p=probability)  # randomly apply an operator for mutation
            if mutate_pick == 1:
                children = [functions.switch(i) for i in children]

            elif mutate_pick == 2:
                children = [functions.insert(i) for i in children]

            elif mutate_pick == 3:
                children = [functions.next_insert(i) for i in children]

            elif mutate_pick == 4:
                children = [functions.more_batch(i) for i in children]

            elif mutate_pick == 5:
                children = [functions.random_reject(i) for i in children]

            elif mutate_pick == 6:
                children = [functions.time_reject(i) for i in children]

            elif mutate_pick == 7:
                children = [functions.cost_reject(i) for i in children]

            elif mutate_pick == 8:
                children = [functions.volume_reject(i) for i in children]

            else:
                children = [functions.accept(i) for i in children]

        # Ensure population diversity
        for c1 in children:
            if c1 not in pop_now:
                pop_now.append(c1)
                value_now.append(functions.profit(c1, functions.timeline(c1)))

    # Select individuals with optimal function values as elites
    Elite_pick = sorted(range(len(value_pop)), key=lambda i: value_pop[i], reverse=True)[:Elite_num]
    for v1 in Elite_pick:
        pop.append(pop[v1])
        value_pop.append(value_pop[v1])

    # Randomly select Non_Elite individuals from pop_now
    Non_Elite_pick = random.sample(list(range(len(pop_now))), Non_Elite_num)
    for v2 in Non_Elite_pick:
        pop.append(pop_now[v2])
        value_pop.append(value_now[v2])

    # Backtracking strategy
    local_value = max(value_pop)
    local_solution = pop[value_pop.index(local_value)]
    if local_value > global_value and functions.checksquare(local_solution) and functions.checktime(local_solution):
        global_value = local_value
        global_solution = copy.deepcopy(pop[value_pop.index(local_value)])
        not_improve = 0
        best_value.append(global_value)
        iteration_best.append(iteration)
        print('')
        print('has optimized for', len(best_value) - 1, 'times')
        print('the global value is:', global_value)
        print('')
    else:
        not_improve += 1

    if not_improve % 10 == 0 and not_improve != 0:
        print('has not optimized for %d iterations' % not_improve)

# output visualization
mylist = [i[1] for i in global_solution]
reject_num = mylist.count(-1)
reject_percentage = 100*reject_num/num_part

time3 = time.time()
execution_time = time3-ini.time1
generation_time = time2-ini.time1

global_solution = functions.reschedule(global_solution)
print('')
print('output solution is:                              ', global_solution)
print('output function value is:                        ', global_value)
print('batch number is:                                 ', max(u1[1] for u1 in global_solution)+1)
print('CPU time for generating initial population is:   ', generation_time, 's')
print('total CPU time is:                               ', execution_time, 's')
print('total tardiness is:                              ', functions.tardiness(global_solution))
print('rejection rate is:                                %.2f %%' % reject_percentage)

lib.plot(iteration_best, best_value)
lib.show()