import sys
import os
from math import floor

import time
import datetime
import numpy as np
from subprocess import check_output
from parse_results import get_results_object

N = [10, 20, 50, 100, 200, 500, 1000]
H = [0.2, 0.4, 0.6, 0.8]
Ks = [
    [1, 8, 5, 2],
    [2, 9, 6, 3],
    [3, 10, 7, 4],
    [4, 1, 8, 5],
    [5, 2, 9, 6],
    [6, 3, 10, 7],
    [7, 4, 1, 8]]
# K = [1, 6]
# H = [0.4, 0.6]
# N = [10]
# K = [1]
# H = [0.6]

class Instance():
    def __init__(self):
        self.p, self.a, self.b = [], [], []

    def zipped_tasks(self):
        # [ID, P, A, B]
        return list(zip(list(range(len(self.p))), self.p, self.a, self.b))

    def add_job(self, p, a, b):
        self.p.append(p)
        self.a.append(a)
        self.b.append(b)

    def __str__(self):
        return str(self.zipped_tasks())

class Check():
    def __init__(self, instance, intervals, h):
        self.h = h
        self.instance = instance
        self.deadline = floor(sum(instance.p) * h)
        self.intervals = intervals

    def check_p(self):
        return sum(self.instance.p) == sum([b - a for a, b in self.intervals])

    def is_valid(self):

        intervals = self.intervals.copy()

        def overlapping(interval_a, interval_b):
            return bool(max(0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0])))

        while(len(intervals)):
            current_interval = intervals.pop()
            for interval in intervals:
                if overlapping(current_interval, interval) or interval[0] < 0 or interval[1] < 0 or interval[1] < interval[0]:
                    print("Overlapping intervals!")
                    return False

        return True
        
    def calculate_cost(self):
        score = 0
        for task_id, interval in enumerate(self.intervals):
            delta = self.deadline - interval[1]
            if delta > 0: score += delta * self.instance.a[task_id]
            if delta < 0: score += abs(delta) * self.instance.b[task_id]
        return score

def parse_input_file(file):
    input_file = open(file, 'r').read().split('\n')[::-1]
    num_instances = int(input_file.pop().strip())
    instances = []

    for _ in range(num_instances):
        current_instance = Instance()
        num_jobs = num_instances = int(input_file.pop().strip())
        for _ in range(num_jobs):
            values = map(int, input_file.pop().split())
            current_instance.add_job(*values)
        instances.append(current_instance)

    return instances

"""
if __name__ == '__main__':

    # n k h
    # Początek_zadania Koniec_zadania
    # ...
    # Początek_zadania Koniec_zadania
    # Wartość kryterium

    # n e [10, 1000]
    # k e [1, 10] (!)
    # h e [0.0, 1.0]

    result_file = open(sys.argv[1], 'r').read().split('\n')[::-1]
    n, k, h = list(map(lambda x: float(x), result_file.pop().split()))
    intervals  = [list(map(lambda x: int(x), result_file.pop().split())) for _ in range(int(n))]
    final_cost = int(result_file.pop())

    instance = parse_input_file('sch%s.txt' % str(int(n)))[int(k) - 1]
    check = Check(instance, intervals, h)
    total_cost = check.calculate_cost()

    print("Poprawne uszeregoewnie:", check.is_valid())
    print("Zgodna suma p:", check.check_p())
    print("Zgodna wartość krytermium:", total_cost == final_cost)
    print("    Podana:", final_cost, " | obliczona:", total_cost)
"""     

if __name__ == '__main__':
    if (len(sys.argv) > 2):
        print('must pass path to the bash script')
        print('e.g. python ./validator ./bash.sh')
    elif (len(sys.argv) < 2):
        print('path to the bash script was not passed')
        print('will try using ./script.sh')
        path = './script.sh'
    else:
        path = sys.argv[1]

    total = 0
    n_inst = 28

    results = get_results_object()

    for n_idx, n in enumerate(N):
        for h_idx, h in enumerate(H):

            k = Ks[n_idx][h_idx]

            ref_str = results[n][k][h]
            ref = int(ref_str) if not '*' in ref_str else int(ref_str[:-1])

            print(f'Will do k: {k} h: {h} n: {n}')
            #original_input = parse_original_input(n, k, h)
            # print(original_input)
            output = check_output(f'{path} {str(n)} {str(k)} {str(h)}', shell=True)
            output = output.decode("utf-8")

            result_file = output.split('\n')[::-1]
            n, k, h = list(map(lambda x: float(x), result_file.pop().split()))
            intervals  = [list(map(lambda x: int(x), result_file.pop().split())) for _ in range(int(n))]
            final_cost = int(result_file.pop())

            n = int(n)
            instance = parse_input_file('sch%s.txt' % n)[int(k) - 1]
            check = Check(instance, intervals, h)
            total_cost = check.calculate_cost()

            if not check.is_valid(): print("Nie poprawne uszeregownanie")
            if not check.check_p(): print("Niepoprawne P!")
            if not total_cost == final_cost: print("Zła wartość kryterium.", "Podana:", final_cost, ", obliczona:", total_cost)
            
            print("Wartosc ref: " + ref_str)
            print("Wynik: " + str(final_cost))

            err = (final_cost - ref) / ref
            total += err
            
            #validate_output(output, original_input)
            # calculate_time

    print("Sredni blad: %s" % str(round((total / 28 ) * 100, 4)))