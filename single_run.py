import sys
import os
from math import floor
import time
import datetime
import numpy as np

class Instance():
    def __init__(self):
        self.p, self.a, self.b = [], [], []

    def zipped_tasks(self):
        # [ID, P, A, B]
        return list(zip(list(range(len(self.p))), self.p, self.a, self.b))

    def get_task_info_dict(self):
        output = [(tid, (p, a, b, p/a, p/b)) for tid, p, a, b in self.zipped_tasks()]
        return dict(output)

    def add_job(self, p, a, b):
        self.p.append(p)
        self.a.append(a)
        self.b.append(b)

    def __str__(self):
        return str(self.zipped_tasks())

class Solver():
    def __init__(self, instance, h=0.2):
        self.h = h
        self.instance = instance
        self.deadline = floor(sum(instance.p) * h)
        self.results = []

        self.jumps = {
            1000: 2,
            500: 1,
            200: 1,
            100: 1,
            50: 1,
            20: 1,
            10: 1
        }

    def solve(self):  

        def sigmoid(z):
            return 1.0/(1.0 + np.exp(-z * 3))

        self.results = []
        self.results_se = []
        self.results_sl = []

        tasks = self.instance.zipped_tasks()
        sig_h = sigmoid(self.h - 0.5)
            
        earliness_group = list(filter(lambda x: x[2] < x[3], tasks))
        total_time_of_earliness_group = sum(task[1] for task in earliness_group)    
        
        beta_coef = max(0, total_time_of_earliness_group - self.deadline) / total_time_of_earliness_group
        earliness_group = sorted(earliness_group, key=lambda x: (1 + (1 - sig_h)) * x[2]/x[3] + (1 - beta_coef + sig_h) * x[2]/x[1] - (beta_coef - sig_h) * x[3]/x[1], reverse=True)
        current_time_point = (self.deadline - total_time_of_earliness_group) if not beta_coef else 0

        while(len(earliness_group)):
            task = earliness_group.pop()
            self.results_se.append(task[0])
            next_time_point = current_time_point + task[1]
            
            if next_time_point > self.deadline:
                earliness_group.append(task)
                self.results_se.pop()
                break
 
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

        tarliness_group = sorted(list(filter(lambda x: x[2] >= x[3], tasks)) + earliness_group, key=lambda x: x[3]/x[1])
        
        """
        gap = self.deadline - (self.results[-1][1] + self.instance.p[self.results[-1][0]])
        if gap and self.results[0][1] != 0:
            for task_idx in range(len(tarliness_group)):
                if self.instance.p[tarliness_group[task_idx][0]] == gap:
                    self.results.append([tarliness_group[task_idx][0], self.deadline - gap])
                    current_time_point = self.deadline
                    del tarliness_group[task_idx]
                    break
        """

        while(len(tarliness_group)):
            task = tarliness_group.pop()
            self.results_sl.append(task[0])
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

    def is_valid(self, values=None):
    
        def overlapping(interval_a, interval_b):
            return bool(max(0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0])))

        results = values if values else self.results
        intervals = [(start_time, start_time + self.instance.p[task_id]) for task_id, start_time in results]

        while(len(intervals)):
            current_interval = intervals.pop()
            for interval in intervals:
                if overlapping(current_interval, interval) or interval[0] < 0 or interval[1] < 0:
                    return False

        return True
        
    def calculate_cost(self, values=None):
        results = values if values else self.results
        score = 0
        for task_id, start_time in results:
            delta = self.deadline - (start_time + self.instance.p[task_id])
            if delta > 0: score += delta * self.instance.a[task_id]
            if delta < 0: score += abs(delta) * self.instance.b[task_id]
        return score

    def calculate_cost_on_tasks(self, se, sl, offset=None):

        #[ID, P, A, B]
        current_time_point = offset if offset is not None else max(0, self.deadline - sum([task[1] for task in se]))
        score = 0
        tasks = se + sl
        for _, p, a, b in tasks:
            delta = self.deadline - (current_time_point + p)
            if delta > 0: score += delta *  a
            if delta < 0: score += abs(delta) * b
            current_time_point += p

        return score

    def calculate_cost_on_dict(self, se, sl, info_dict, offset=None, jump=None):
        #[ID, P, A, B]
        current_time_point = offset if offset is not None else max(0, self.deadline - sum([info_dict[task][0] for task in se]))
        
        se = se[:]
        se.extend(sl)

        score = 0
        for task_id in se:
            p, a, b, _, _ = info_dict[task_id]
            delta = self.deadline - (current_time_point + p)
            if delta > 0: score += delta * a
            if delta < 0: score += -1 * delta * b
            current_time_point += p
        return score

    def generate_timeline(self, values=None, group=True):
        """ Deprecated! """
        
        results = values if values else self.results
        if not self.is_valid(results):
            return "Invalid results!"

        sorted_results = sorted(results, key=lambda x: x[1])

        checkpoint = 0
        timeline = []
        
        for task_id, start_time in sorted_results:
            delta = abs(checkpoint - start_time)
            for _ in range(delta):
                timeline.append("_")

            for time in range(self.instance.p[task_id]):
                timeline.append(chr((ord('A') + task_id % (ord('Z') - ord('A')))))

            checkpoint = self.instance.p[task_id] + start_time
  
        timeline.insert(self.deadline, "|")
        return '[' + ''.join(timeline) + ']'

    def pretty_log(self):
        pass

def parse_input_file(filename):
    input_file = open(filename, 'r').read().split('\n')[::-1]
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
    
if __name__ == '__main__':

    input_file = 'sch%s.txt' % sys.argv[1]
    instance_k = 0 if len(sys.argv) < 3 else int(sys.argv[2])
    deadline = 0.4 if len(sys.argv) < 4 else float(sys.argv[3])

    instance = parse_input_file(input_file)[instance_k - 1]

    print(input_file[3:-4], instance_k, deadline)

    solver = Solver(instance, deadline)            
    solver.solve()

    results = sorted(solver.results, key=lambda x: x[0])
    for task_id, start_time in results:
        print(start_time, start_time + solver.instance.p[task_id])

    print(solver.calculate_cost())

                
