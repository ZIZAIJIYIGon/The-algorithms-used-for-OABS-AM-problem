import random
import pandas as pd
import copy


# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #
# ————————————————————————————————————    Preparations    ———————————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


# get order's data from instances file
filename = 'instances/Order scale_20/Scenario1_3.0.txt'
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
# —————————————————————————————————————    Functions    —————————————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


# calculate the objective function value of a solution
def profit(schedule):
    pro = 0
    time = timeline(schedule)
    for i in range(len(height_order)):
        if schedule[i][1] > -1:
            pro += profit_order[i]                   # income of orders
            pro -= CV_order * volume_order[i]        # material cost
            pro -= tardiness_penalty_order[i] * max((time[schedule[i][1]] + produce_time(schedule)[schedule[i][1]] - TD_order[i]), 0)  # tardiness penalty
        else:
            pro -= rejection_penalty_order[i]        # rejection penalty
    pro -= sum(produce_time(schedule)) * CS_machine  # using machine cost
    return pro


# obtain the start time of each batch in a solution
def timeline(schedule):
    index1 = max(i[1] for i in schedule)
    B = [0] * (index1 + 1)
    max_tr = 0
    for i in range(len(B)):
        for j in range(len(schedule)):
            if schedule[j][1] == i:
                if max_tr < TR_order[j]:
                    max_tr = TR_order[j]
        if i == 0:
            B[i] = max_tr
        else:
            B[i] = max((B[i-1]+produce_time(schedule)[i-1]), max_tr)
    return B


# obtain the produce time of each batch in a solution
def produce_time(schedule):
    pro_time = []
    index1 = max(i[1] for i in schedule)
    for i in range(index1+1):
        time = 0
        max_height = 0
        for j in range(len(schedule)):
            if schedule[j][1] == i:
                time += TM_machine * volume_order[j]
                if height_order[j] > max_height:
                    max_height = height_order[j]
        time += max_height * TP_machine
        time += Tset_machine
        pro_time.append(time)
    return pro_time


# check the time feasibility of all the order in a solution
def checktime(schedule):
    time = timeline(schedule)
    for i in range(len(TR_order)):
        if schedule[i][1] >= 0:  # do not take the rejected order into account
            if time[schedule[i][1]] < TR_order[i] or time[schedule[i][1]] + produce_time(schedule)[schedule[i][1]] > TE_order[i]:
                return 0         # infeasible solution return 0
            else:
                continue
    return 1                     # feasible solution return 1


# check the area feasibility of all the order in a solution
def checksquare(schedule):
    square_batch = 0
    index1 = max(i[1] for i in schedule)
    for j in range(index1+1):
        for k in schedule:
            if k[1] == j:
                square_batch += square_order[k[0]]
        if square_batch <= S_machine:
            square_batch = 0
        else:
            return 0           # infeasible solution return 0
    return 1                   # feasible solution return 1


# obtain the produce time of a singe batch (used in initial solution generation)
def span(schedule):
    produce_time = Tset_machine
    max_height = 0
    for i in range(len(schedule)):
        produce_time += TM_machine * volume_order[schedule[i][0]]
        if max_height < height_order[schedule[i][0]]:
            max_height = height_order[schedule[i][0]]
    produce_time += max_height * TP_machine
    return produce_time


# Reorder for batch index errors
def reschedule(schedule):
    temp_sche = copy.deepcopy(schedule)
    batch_re = list(set(i[1] for i in schedule))
    if -1 in batch_re:
        batch_re.remove(-1)
    list1 = list(range(len(batch_re)))
    if list1 != batch_re:
        for i in temp_sche:
            if i[1] != -1:
                index_re = batch_re.index(i[1])
                i[1] = list1[index_re]
    return temp_sche


# Calculate the total tardiness of all orders in a solution
def tardiness(schedule):
    temp_schedule = copy.deepcopy(schedule)
    timeline_tard = timeline(temp_schedule)             # start time of each batch
    produce_time_tard = produce_time(temp_schedule)     # produce time of each batch
    completion_time_tard = []                           # completion time of each batch
    tardiness_time = 0

    for le in range(len(timeline_tard)):
        completion_time_tard.append(timeline_tard[le] + produce_time_tard[le])

    batch = list(set([ii[1] for ii in temp_schedule]))  # the batch number in a batch
    if -1 in batch:
        batch.remove(-1)

    for k in temp_schedule:
        if k[1] != -1:
            tardiness_time += max(0, completion_time_tard[k[1]] - TD_order[k[0]])

    return tardiness_time


# Single point crossover operation (used in RKGA algorithm)
def crossover(dad, mom, pop):
    pop_new = []
    inter = 0
    while 1:
        temp_dad = copy.deepcopy(dad)
        temp_mom = copy.deepcopy(mom)
        len_p = len(dad)-1
        a = random.randint(0, len_p)
        pop_newa = temp_dad[0: a+1] + temp_mom[a+1: len_p+1]
        pop_newb = temp_mom[0: a+1] + temp_dad[a+1: len_p+1]
        if checktime(pop_newa) and checksquare(pop_newa):
            pop_new.append(pop_newa)

        if checktime(pop_newb) and checksquare(pop_newb):
            pop_new.append(pop_newb)

        if len(pop_new) >= 2:
            return pop_new

        inter += 1

        if inter >= 20:
            return [random.sample(pop, 1)[0], random.sample(pop, 1)[0]]


# volume_reject operator (consider the acceptance of orders, only used in RKGA)
def volume_reject_rkga(schedule):
    temp_schedule = copy.deepcopy(schedule)
    accept_index = [acc[0] for acc in temp_schedule if acc[1] != -1]
    n_i_list = [n_i_order[u] for u in accept_index]
    if n_i_list:  # only there is at least an accepted order, could this operator be applied
        least_n_i_index = n_i_list.index(min(n_i_list))
        temp_schedule[accept_index[least_n_i_index]][1] = -1
        return temp_schedule
    else:
        return schedule


# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #
# ——————————————————————————————————————    operators    ————————————————————————————————————— #
# ————————————------————————————------————————————------————————————------————————————------—— #
# ------————————————------————————————------————————————------————————————------————————————-- #


# 1.random insert operator
def insert(schedule):
    temp_sche = copy.deepcopy(schedule)
    a = random.randint(0, num_part-1)
    b = temp_sche[a][1]
    list_else = list(set([u[1] for u in schedule]))
    list_else.remove(b)
    if list_else:
        temp_sche[a][1] = random.sample(list_else, 1)[0]
    if checktime(temp_sche) and checksquare(temp_sche):
        schedule = temp_sche
    return schedule


# 2. next insert operator
def next_insert(schedule):
    temp_sche = copy.deepcopy(schedule)
    pick_list = list(set([i[1] for i in schedule]))
    max1 = pick_list[-1]
    min1 = pick_list[0]
    pick = random.randint(0, len(schedule)-1)
    pick_time_next_insert = 0

    while temp_sche[pick][1] == -1:     # If the rejected order is selected, then reselect
        pick = random.randint(0, len(schedule) - 1)
        pick_time_next_insert += 1
        if pick_time_next_insert >= 2*len(temp_sche):  # If there is no result, return the original solution
            return schedule

    if temp_sche[pick][1] == max1:      # If the last batch is selected, insert it into the previous batch
        temp_sche[pick][1] = max1 - 1

    elif temp_sche[pick][1] == min1:    # If the first batch is selected, insert it into the second batch
        temp_sche[pick][1] = min1 + 1

    else:                               # If the middle batch is selected, it will be inserted into the adjacent batch
        front_or_later = [1, -1]
        decision = random.sample(front_or_later, 1)[0]
        temp_sche[pick][1] += decision

    if checktime(temp_sche) and checksquare(temp_sche):
        return temp_sche

    else:
        return schedule


# 3.switch operator
def switch(schedule):
    iter1 = 0
    while 1:
        temp_sch = copy.deepcopy(schedule)
        i = random.randint(0, num_part-1)
        j = random.randint(0, num_part-1)
        if temp_sch[i][1] != temp_sch[j][1]:
           temp_sch[i][1], temp_sch[j][1] = temp_sch[j][1], temp_sch[i][1]
        if checktime(temp_sch) and checksquare(temp_sch):
            schedule = copy.deepcopy(temp_sch)
            return schedule
        iter1 += 1
        if iter1 >= 20:
            break
    return schedule


# 4. more_batch operator
def more_batch(schedule):
    batch = list([i[1] for i in schedule])
    duplicate = [item for item in batch if batch.count(item) >= 2]
    duplicate = list(set(duplicate))
    if -1 in duplicate:
        duplicate.remove(-1)
    if not duplicate:
        return schedule
    max_index = 0
    for y in schedule:
        if y[1] == max(duplicate):
            if y[0] > max_index:
                max_index = y[0]
    schedule[max_index][1] = max(batch) + 1
    return schedule


# 5. cost_reject operator
def cost_reject(schedule):
    target_value = profit(schedule)
    temp_sche = copy.deepcopy(schedule)
    cost_all = []
    index_all = []
    for r1 in range(len(temp_sche)):
        temp_sche = copy.deepcopy(schedule)
        if temp_sche[r1][1] == -1:  # If the part has already been rejected, skip and select another part
            continue
        else:
            temp_sche[r1][1] = -1
            now_value = profit(temp_sche)
            cut = now_value - target_value
            cost_all.append(cut)
            index_all.append(temp_sche[r1][0])
    if cost_all:
        temp_sche = copy.deepcopy(schedule)
        max_value = max(cost_all)  # Find the situation that causes the least profit loss among all rejection situations
        index3 = cost_all.index(max_value)
        re = index_all[index3]     # Order index corresponding to the minimum profit loss after rejection
        temp_sche[re][1] = -1
        return temp_sche
    else:
        return schedule


# 6.tardiness_reject operator
def time_reject(schedule):
    temp_sche = copy.deepcopy(schedule)
    temp_sche = reschedule(temp_sche)
    batch_range = list(set([u[1] for u in temp_sche if u[1] != -1]))
    tardiness_all_time = []
    tardiness_all_index = []
    if batch_range:              # The operator can only take effect when there are orders that are not rejected
        for i in batch_range:
            tardiness_best = 0   # Record the maximum tardiness in the batch
            tardiness_index = 0  # Record the order index with maximum tardiness
            tardiness_total = 0  # Record the total tardiness of the batch
            completion_time = timeline(temp_sche)[i] + produce_time(temp_sche)[i]
            for j in temp_sche:
                if j[1] == i:
                    tardiness_time = max(0, completion_time - TD_order[j[0]])
                    if tardiness_time > tardiness_best:
                        tardiness_best = tardiness_time
                        tardiness_index = j[0]
                    tardiness_total += tardiness_time
            tardiness_all_time.append(tardiness_total)
            tardiness_all_index.append(tardiness_index)
        max_value = max(tardiness_all_time)
        if max_value == 0:  # If there are no orders causing tardiness, return to the original solution
            return schedule
        else:
            index2 = tardiness_all_time.index(max_value)
            re = tardiness_all_index[index2]
            temp_sche[re][1] = -1
            return temp_sche
    else:
        return schedule


# 7.random_reject operator
def random_reject(schedule):
    temp_sche = copy.deepcopy(schedule)
    accept_schedule = [u for u in temp_sche if u[1] != -1]
    if accept_schedule:  # This operator can only be applied when there is at least an order has been accepted
        index = random.randint(0, len(accept_schedule)-1)
        temp_sche[accept_schedule[index][0]][1] = -1
        return temp_sche
    return schedule


# 8.volume_reject operator
def volume_reject(schedule):
    temp_schedule = copy.deepcopy(schedule)
    accept_index = [acc[0] for acc in temp_schedule if acc[1] != -1]
    n_i_list = [n_i_order[u] for u in accept_index]
    least_n_i_index = n_i_list.index(min(n_i_list))
    temp_schedule[accept_index[least_n_i_index]][1] = -1
    return temp_schedule


# 9.random accept operator
def accept(schedule):
    reject_part = []
    index1 = max(i[1] for i in schedule)
    batch_index = list(range(index1 + 1))

    for i in schedule:
        if i[1] == -1:
            reject_part.append(i[0])

    while reject_part and batch_index:
        if reject_part:
            a = random.choice(reject_part)
            temp_sche = copy.deepcopy(schedule)
            b = random.sample(batch_index, 1)[0]
            temp_sche[a][1] = b
            if checktime(temp_sche) and checksquare(temp_sche):
                return temp_sche
            reject_part.remove(a)
    return schedule



