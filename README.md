# Project overview
Coding with python3.9.1， this project implements three different algorithms (AHNS, RKGA, TS) to solve the order acceptance and batch scheduling problem in additive manufacturing. These algorithms accomplish their tasks by calling the functions in "**functions.py**" and the initial solution generation method in "**initialization.py**". Each algorithm file (**AHNS algorithm.py, RKGA algorithm.py, TS algorithm.py**) independently implements different optimization strategies, but they share the same initial solution generation and basic functions.

# Usage

1. Download the instances file through this link: https://github.com/ZIZAIJIYIGon/The-Instances-used-in-OABS-AM-problem.git
2. Select the example to be calculated and modify the parameter "filename" in the function.py.
 
3. Run algorithm files.

# File description

## functions.py
This file contains all the basic functions and operators required for the algorithm, mainly including:

### basic functions

profit: Calculate the objective function value for a given scheduling plan.

timeline: Calculate the start time for each batch.

produce_time: Calculate the production time for each batch.

checktime: Check the time feasibility of the scheduling solution.

checksquare: Check the area feasibility of the scheduling solution.

reschedule: Re-sort the batch index.

tardiness: Calculate the total tardiness time.

crossover: Single-point crossover operation (used in the RKGA algorithm).

### operators

insert: Random insertion operation.

next_insert: Neighboring insertion operation.

switch: Swap operation.

more_batch: Add batch operation.

cost_reject: Cost-based rejection operation.

time_reject: Delay-based rejection operation.

random_reject: Random rejection operation.

volume_reject: Volume-based rejection operation.

accept: Random acceptance operation.

## initialization.py

This file is responsible for generating the initial solution, which mainly includes:

The core logic of initial solution generation, creating a feasible initial scheduling solution based on constraints such as order height, area, arrival time, etc.

## AHNS algorithm.py

This file implements the Adaptive Hybrid Neighborhood Search Algorithm (AHNS), which mainly includes:

Adaptive operator selection: Dynamically adjust the selection probability based on the performance of the operators.

Backtracking strategy: Backtrack to a previously better solution when no optimization has occurred for a long time.

Stopping criterion: Stop the algorithm when the number of consecutive non-improvement iterations reaches the set threshold.

## RKGA algorithm.py

This file implements the Random Key Genetic Algorithm (RKGA), which mainly includes:

Initial population generation: Generate an initial population through random operations.

Crossover and Mutation: Generate new individuals using single point crossover and random mutation.

Elite selection: Retain the best individuals to enter the next generation.

Backtracking strategy: When not optimized for a long time, backtrack to the previous obtained solution.

## TS algorithm.py

This file implements the Tabu Search Algorithm (TS), which mainly includes:

Tabu list management: Record the recently visited solutions to avoid re-searching them.

Neighborhood search: Search for better solutions within the neighborhood.

Backtracking strategy: Backtrack to a previously better solution when no optimization has occurred for a long time.

# Visualization

We visualized the corresponding calculation results based on the algorithm structure during the algorithm running process, including:

1.During the initialization of population：output the current iteration count and the greatest function value ever obtained every five iterations during population initialization generation; Every time a new individual is added to the initial population, output the number of individuals in the initial population. **(only for RKGA algorithm)**

2.During mainloop: when no better solution than the currently obtained optimal solution is found every **(10 for RKGA, 50 for TS, 100 for AHNS)** iterations, output the number of iterations in which the optimal solution was not found. Whenever a better solution is found, output the objective function value of the current optimal solution and the number of times a better solution is found.

3.After the mainloop: output a two-dimensional graph consisting of the optimal function value found during each update and its corresponding iteration count.

# Output

Each algorithm file will output the following information after execution:

Optimal solution: The optimal scheduling solution.

Optimal value: The objective function value corresponding to the optimal solution.

Number of batches: The number of batches in the optimal solution.

Total tardiness: The total tardiness time in the optimal solution.

Rejection rate: The order rejection rate in the optimal solution.

CPU time: The time taken for the algorithm to run.

Number of iterations: The number of iterations the algorithm has run.

# Notes:

The testing instances is public in https://github.com/ZIZAIJIYIGon/The-Instances-used-in-OABS-AM-problem.git

Modify the file path in functions.py, and make sure that the data file path is correct.

Each algorithm file can run independently, but make sure that functions.py and initialization.py are in the same directory.

The algorithm's runtime may be long, depending on the data scale and algorithm parameter settings.
