import os
import sys
from timeit import default_timer as time
import ntpath
import numpy as np
import math

from random import shuffle, randint, random
from zadanie_1 import Solver, Instance
from zadanie_1 import parse_input_file
from parse_results import get_results_object

def timer(func):
    def function(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        stop = time()
        print("[Timer] Elapsed time: {:.3f} s".format((stop - start)))
        return result
    return function

class BetterSolver(Solver):

    def __init__(self, *args, **kwargs):
        super(BetterSolver, self).__init__(*args, **kwargs)

    def convert_list_of_tasks_to_results(self, tasks, offset=0):
        current_time_point = offset
        local_results = []

        for task in tasks:
            local_results.append([task[0], current_time_point])
            current_time_point += task[1]

        return local_results

    def convert_subsets_of_tasks_to_results(self, se, sl, h_offset=None):
         offset = h_offset if h_offset is not None else max(0, self.deadline - sum([task[1] for task in se]))
         tasks = se + sl
         return self.convert_list_of_tasks_to_results(tasks, offset=offset)

    @timer
    def solve(self, i_max=3600, t_0=50000, alpha=0.995, use_heuristic_as_first_solution=True):
        """ Simulated annealing 
        
            Args:
            -------
                i_max         - the maximum number of iterations
                t_0           - initial temperature
                alpha         - coefficient controlling the cooling schedule


        """

        def sort_left_property(tasks, reverse=True):
            """ Non-increasing order of the rations p_i/a_i """
            return sorted(tasks, key=lambda x: x[1]/x[2], reverse=reverse)

        def sort_right_property(tasks):
            """ Non-decreasing order of the rations p_i/b_i """
            return sorted(tasks, key=lambda x: x[1]/x[3])

        def generate_se_sl(tasks, deadline):
            """ Sort the jobs according to V-shape property """

            S_E, S_L = [], []
            sorted_according_to_the_left_property = sort_left_property(tasks, False)
            
            current_time_point = 0
            while(len(sorted_according_to_the_left_property)):
                task = sorted_according_to_the_left_property.pop()
                if current_time_point + task[1] <= deadline:
                    S_E.append(task)
                    current_time_point += task[1]
                else:
                    sorted_according_to_the_left_property.append(task)
                    break

            S_L = sort_right_property(sorted_according_to_the_left_property)
            return S_E, S_L

        def greedy_local_search(S_E, S_L, current_f_x):

            best_cost_value = 10 * current_f_x
            best_configuration = [S_E.copy(), S_L.copy()]

            k = np.random.choice([0, 1])
            from_set, to_set = (S_E, S_L) if not k else (S_L, S_E)
            sort_order = {"S_E": from_set, "S_L": to_set} if not k else {"S_E": to_set, "S_L": from_set}
            status = False

            # operation_type = None
            # arguments = []

            if len(from_set):
                status = True
                # Get random task
                ith = randint(0, len(from_set) - 1)
                selected_task = from_set.pop(ith)
                to_set.append(selected_task)

                # 1) MOVE TASK ONLY 
                if sort_order["S_E"] is from_set:
                    from_set = sort_left_property(from_set)
                    sort_order["S_E"] = from_set
                    to_set = sort_right_property(to_set) 
                    sort_order["S_L"] = to_set
                else:
                    from_set = sort_right_property(from_set)
                    sort_order["S_L"] = from_set
                    to_set = sort_left_property(to_set) 
                    sort_order["S_E"] = to_set

                local_cost = self.calculate_cost_on_tasks(sort_order['S_E'], sort_order['S_L'])

                if local_cost < best_cost_value:
                    best_cost_value = local_cost
                    best_configuration = (sort_order['S_E'].copy(), sort_order['S_L'].copy())

                # 2) SWAP TASKS
                for task_id in range(len(to_set)):
                    temp_sort_order = {}

                    temp_to_set = to_set.copy()
                    temp_from_set = from_set.copy()

                    temp_from_set.append(temp_to_set.pop(task_id))

                    if to_set is sort_order['S_L']:
                        temp_to_set = sort_right_property(temp_to_set)
                        temp_sort_order['S_L'] = temp_to_set

                        temp_from_set = sort_left_property(temp_from_set)
                        temp_sort_order['S_E'] = temp_from_set
                    else:
                        temp_to_set = sort_left_property(temp_to_set)
                        temp_sort_order['S_E'] = temp_to_set

                        temp_from_set = sort_right_property(temp_from_set)
                        temp_sort_order['S_L'] = temp_from_set

                    local_cost = self.calculate_cost_on_tasks(temp_sort_order['S_E'], temp_sort_order['S_L'])

                    if local_cost < best_cost_value:
                        best_cost_value = local_cost
                        best_configuration = (temp_sort_order['S_E'], temp_sort_order['S_L'])

            return status, best_configuration[0], best_configuration[1], best_cost_value
    
        # Single task structure in `tasks` list:
        # [ID, P, A, B]
        # ( 0, 2, 4, 8)
        tasks = self.instance.zipped_tasks()
        offset_determined_by_heruistic_function = None

        # Generate S_e and S_l  
        S_E, S_L = [], []

        if use_heuristic_as_first_solution:
            super(BetterSolver, self).solve()
            S_E, S_L = self.results_se, self.results_sl
            offset_determined_by_heruistic_function = self.results[0][1]
            print("HEURISTIC VAL:", self.calculate_cost(self.results))
        else:
            S_E, S_L = generate_se_sl(tasks, self.deadline)

        # Just to be sure
        self.results = []

        if len(S_E + S_L) != len(tasks):
            raise Exception("Number of the elements in S_E and S_L (%d) is not equal to the total number of tasks (%d)!" % (len(S_E + S_L), len(tasks)))

        # Algorithm
        T =  t_0        # Current temperature
        i_iter = 0      # Current iteration

        # best_x is set to X
        # best_f_x denotes the objectiv value for best_x
        
        best_x = (S_E, S_L)
        best_f_x = self.calculate_cost_on_tasks(*best_x, offset_determined_by_heruistic_function)

        # Iterations
        time_cumsum = 0
        times = [0.15] * 4
        running_mean_time = np.mean(times)

        current_iteration = 0

        f_x = best_f_x
        while time_cumsum + running_mean_time <= (len(tasks) * 0.1 - 0.145):

            start = time()

            status, new_S_E, new_S_L, new_f_x = greedy_local_search(S_E, S_L, f_x)
            new_x = [new_S_E, new_S_L]
            delta = new_f_x - f_x

            if delta > 0:
                if random() < (math.exp((-1 * delta) / T)):
                    S_E, S_L = new_S_E, new_S_L
                    f_x = new_f_x
                    
            elif delta < 0:
                S_E, S_L = new_S_E, new_S_L
                f_x = new_f_x

                if f_x < best_f_x:
                    best_f_x = f_x
                    best_x = [new_x[0].copy(), new_x[1].copy()]

            T *= alpha 
            stop = time()
            elapsed_time = stop - start
            times.append(elapsed_time)
            running_mean_time = np.mean(times[-4:])
            time_cumsum += elapsed_time
            current_iteration += 1

        self.results = best_x
        timeline = self.convert_subsets_of_tasks_to_results(*best_x)

        print(" --- AFTER: --- ")
        print("COST:", self.calculate_cost(timeline))
        print("VALID:", self.is_valid(timeline))
        print("NUMBER OF TASKS:", len(timeline))
    
        print("ITERATIONS:", current_iteration)

if __name__ == '__main__':

    input_file = 'input/sch%s.txt' % sys.argv[1]
    instance_k = 0 if len(sys.argv) < 3 else int(sys.argv[2])
    deadline = 0.4 if len(sys.argv) < 4 else float(sys.argv[3])
    instance = parse_input_file(input_file)[instance_k - 1]

    # Determinujemy czy uzywac heurystyki czy nie
    htic = False if len(sys.argv) < 5 else bool(int(sys.argv[4]))

    intput_file_name = ntpath.split(input_file)[1]
    n = int(intput_file_name.split('.')[0][3:])
    print(intput_file_name, instance_k, deadline)

    solver = BetterSolver(instance, deadline) 
    solver.solve(i_max=3600, t_0=50000, alpha=0.995, use_heuristic_as_first_solution=htic)

    """
    results = sorted(solver.results, key=lambda x: x[0])
    for task_id, start_time in results:
        print(start_time, start_time + solver.instance.p[task_id])
    """

    results = get_results_object()

    print("FINAL COST:", solver.calculate_cost_on_tasks(*solver.results))
    print("REF:", results[n][instance_k][deadline])

