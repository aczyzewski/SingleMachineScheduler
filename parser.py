import sys
from math import floor
from statistics import median
# processing times p(i) 
# earliness a(i)  
# tardiness b(i)

# Sprawozdanie:
# Wszystkie pliki (n), instancje: 1, 6 (liczac od 1)
# h = [0.4, 0.6]

# 2 rankingi koncowe (wg czasu, wg. jakosci - konkretnie porownywane wartosci sa do ustalenia w grupie)
# Ograniczenie: metoda co najwyzej rzedu O(n^2)

"""

Sterna - wyklad - kryteria "due dates"?

n       - number of jobs
p_i     - processing time of job i

Step 0: B := []; A := {1, 2, ..., N + |B|_max}
Step 1: i := 1; sum := 0; stop := false.
Step 3: while(not stop):
    if sum + p_i <= d:
            sum := sum + p_i
            B := B + [i]  
            A :=  A\i
    else:
        i is stradling job
            A :=  A\i
            stop = True

    i += 1
    if i > (|B|_max + 1) or (sum < d):
        stop = True

"""  

class Instance():
    def __init__(self):
        self.p, self.a, self.b = [], [], []

    def zipped_tasks(self):
        # Structure: (ID, P, A, B)
        return list(zip(list(range(len(self.p))), self.p, self.a, self.b))

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

    def solve_without_beta(self):  
        self.results = []
        tasks = self.instance.zipped_tasks()

        # Structure: (ID, P, A, B)
        earliness_group = list(filter(lambda x: x[2] <= x[3], tasks))
        total_time_of_earliness_group = sum(task[1] for task in earliness_group)    
        earliness_group = sorted(earliness_group, key=lambda x: x[2]/x[1], reverse=True)

        current_time_point = max(0, self.deadline - total_time_of_earliness_group)

        while(len(earliness_group)):
            task = earliness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

            if current_time_point >= self.deadline:
                break

        tarliness_group = sorted(list(filter(lambda x: x[2] > x[3], tasks)) + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tarliness_group)):
            task = tarliness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

    def solve_beta_equals_weights_in_a(self):  
        self.results = []
        tasks = self.instance.zipped_tasks()

        # Structure: (ID, P, A, B)
        earliness_group = list(filter(lambda x: x[2] < x[3], tasks))
        total_time_of_earliness_group = sum(task[1] for task in earliness_group)    
        
        beta_coef = max(0, total_time_of_earliness_group - self.deadline) / total_time_of_earliness_group
        earliness_group = sorted(earliness_group, key=lambda x: min(1, 1 - beta_coef + self.h) * x[3]/x[2] - max(0, beta_coef - (self.h)) * x[3]/x[1], reverse=True)

        current_time_point = (self.deadline - total_time_of_earliness_group) if not beta_coef else 0

        while(len(earliness_group)):
            task = earliness_group.pop()
            next_time_point = current_time_point + task[1]

            if next_time_point > self.deadline:
                earliness_group.append(task)
                break
 
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

        tarliness_group = sorted(list(filter(lambda x: x[2] >= x[3], tasks)) + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tarliness_group)):
            task = tarliness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

    def solve_beta_equals_weights_in_b(self):  
        self.results = []
        tasks = self.instance.zipped_tasks()

        # Structure: (ID, P, A, B)
        earliness_group = list(filter(lambda x: x[2] < x[3], tasks))
        total_time_of_earliness_group = sum(task[1] for task in earliness_group) 

        total_time_of_earliness_group = sum(task[1] for task in earliness_group)    
        beta_coef = max(0, total_time_of_earliness_group - self.deadline) / total_time_of_earliness_group
        earliness_group = sorted(earliness_group, key=lambda x: (1 - beta_coef) * x[2]/x[1] - beta_coef * x[3]/x[1], reverse=True)

        current_time_point = (self.deadline - total_time_of_earliness_group) if not beta_coef else 0

        while(len(earliness_group)):
            task = earliness_group.pop()
            next_time_point = current_time_point + task[1]

            if next_time_point > self.deadline:
                earliness_group.append(task)
                break

            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

        tarliness_group = sorted(list(filter(lambda x: x[2] >= x[3], tasks)) + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tarliness_group)):
            task = tarliness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]
                    
    def solve_beta_equals_as_common_set(self):  
        self.results = []
        tasks = self.instance.zipped_tasks()

        # Structure: (ID, P, A, B)
        earliness_group = list(filter(lambda x: x[2] <= x[3], tasks))
        total_time_of_earliness_group = sum(task[1] for task in earliness_group) 

        equal_group = list(filter(lambda x: x[2] == x[3], tasks))
        equal_activity = {key[0]: True for key in equal_group}

        total_time_of_earliness_group = sum(task[1] for task in earliness_group)    
        beta_coef = max(0, total_time_of_earliness_group - self.deadline) / total_time_of_earliness_group
        earliness_group = sorted(earliness_group, key=lambda x: (1 - beta_coef) * x[2]/x[1] - beta_coef * x[3]/x[1], reverse=True)

        current_time_point = (self.deadline - total_time_of_earliness_group) if not beta_coef else 0

        while(len(earliness_group)):
            task = earliness_group.pop()
            if task[2] == task[3]:
                if equal_activity[task[0]]: 
                    equal_activity[task[0]] = False
                else:
                    continue

            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

            if current_time_point >= self.deadline:
                break

        tarliness_group = sorted(list(filter(lambda x: x[2] >= x[3], tasks)) + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tarliness_group)):
            task = tarliness_group.pop()
            if task[2] == task[3]:
                if equal_activity[task[0]]: 
                    equal_activity[task[0]] = False
                else:
                    continue
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

    def solve_custtom_h1(self):

        self.results = []
        tasks = self.instance.zipped_tasks()
        earliness_group, tardiness_group = [], []
        earliness_w_sum, tardiness_w_sum = 0, 0
        
        # Structure: (ID, P, A, B)
        for task in tasks:
            if earliness_w_sum < tardiness_w_sum + task[3]:
                earliness_group.append(task)
                earliness_w_sum += task[2]
            else:
                tardiness_group.append(task)
                tardiness_w_sum += task[3]

        total_time_of_earliness_group = sum(task[1] for task in earliness_group)     
        beta_coef = max(0, total_time_of_earliness_group - self.deadline) / total_time_of_earliness_group
        earliness_group = sorted(earliness_group, key=lambda x: (1 - beta_coef) * x[2]/x[1] - beta_coef * x[3]/x[1], reverse=True)
        current_time_point = (self.deadline - total_time_of_earliness_group) if not beta_coef else 0

        while(len(earliness_group)):
            task = earliness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]
            if current_time_point >= self.deadline:
                break

        tardiness_group = sorted(tardiness_group + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tardiness_group)):
            task = tardiness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

    def solve_custtom_h2(self):

        self.results = []
        tasks = self.instance.zipped_tasks()

        # 1) Poczatkowy sposob sortowania
        tasks = sorted(tasks, key=lambda x: x[3]/x[2], reverse=True)
        earliness_group, tardiness_group = [], []
        earliness_w_sum, tardiness_w_sum = 0, 0
        earliness_p_sum, tardiness_p_sum = 0, 0
        # Structure: (ID, P, A, B)
        # 2) Kryterium przydzialu

        for task in tasks:
            if earliness_w_sum < tardiness_w_sum + task[3]/task[1]: # and earliness_p_sum  self.deadline:
                earliness_group.append(task)
                earliness_w_sum += task[2]/task[1]
                earliness_p_sum += task[1]
            else:
                tardiness_group.append(task)
                tardiness_w_sum += task[3]/task[1]
                tardiness_p_sum += task[1]
        

        #earliness_group = list(filter(lambda x: x[2] < x[3], tasks))
        #tardiness_group = list(filter(lambda x: x[2] >= x[3], tasks))

        total_time_of_earliness_group = sum(task[1] for task in earliness_group)     
        beta_coef = max(0, total_time_of_earliness_group - self.deadline) / total_time_of_earliness_group
        earliness_group = sorted(earliness_group, key=lambda x: (1 - beta_coef) * x[2]/x[1] - beta_coef * x[3]/x[1], reverse=True)
        #earliness_group = sorted(earliness_group, key=lambda x: x[2]/x[1], reverse=True)
        
        current_time_point = max(0, self.deadline - total_time_of_earliness_group) if not beta_coef else 0

        while(len(earliness_group)):
            task = earliness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]
            if current_time_point >= self.deadline:
                break

        tardiness_group = sorted(tardiness_group + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tardiness_group)):
            task = tardiness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

    def solve_beta_equals_weights_in_b_gap(self):  
        self.results = []
        tasks = self.instance.zipped_tasks()

        # Structure: (ID, P, A, B)
        earliness_group = list(filter(lambda x: x[2] < x[3], tasks))
        total_time_of_earliness_group = sum(task[1] for task in earliness_group) 

        total_time_of_earliness_group = sum(task[1] for task in earliness_group)    
        beta_coef = max(0, total_time_of_earliness_group - self.deadline) / total_time_of_earliness_group
        earliness_group = sorted(earliness_group, key=lambda x: (1 - beta_coef) * x[2]/x[1] - beta_coef * x[3]/x[1], reverse=True)

        current_time_point = (self.deadline - total_time_of_earliness_group) if not beta_coef else 0

        perfect_deadline_gap = 0
        last_perfect_task_id = None
        last_perfect_task_instance_id = None 

        while(len(earliness_group)):
            task = earliness_group.pop()
            next_time_point = current_time_point + task[1]

            if next_time_point > self.deadline:
                earliness_group.append(task)
                last_perfect_task_instance_id = self.results[-1][0]
                last_perfect_task_id = len(self.results) - 1
                perfect_deadline_gap = self.deadline - (self.results[-1][1] + self.instance.p[last_perfect_task_instance_id])
                break

            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

        tarliness_group = sorted(list(filter(lambda x: x[2] >= x[3], tasks)) + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tarliness_group)):
            task = tarliness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

        if perfect_deadline_gap:
            for num_iterator in range(len(self.results) // 2):
                result_id = num_iterator
                result_instance_id = self.results[result_id][0]
                if self.instance.p[result_instance_id] == perfect_deadline_gap and self.instance.b[result_instance_id] > self.instance.b[self.results[last_perfect_task_id + 1][0]]:
                    self.results[result_id][1] = self.deadline - perfect_deadline_gap

                    for idxx in range(last_perfect_task_id + 1, len(self.results)):
                        self.results[idxx][1] += perfect_deadline_gap

                    for idxx in range(result_id):
                        self.results[idxx][1] += self.instance.p[result_instance_id]
       
                    perfect_deadline_gap = 0

    def solve_beta_equals_weights_in_b_swap_center(self):  
        self.results = []
        tasks = self.instance.zipped_tasks()

        # Structure: (ID, P, A, B)
        earliness_group = list(filter(lambda x: x[2] < x[3], tasks))
        total_time_of_earliness_group = sum(task[1] for task in earliness_group) 

        total_time_of_earliness_group = sum(task[1] for task in earliness_group)    
        beta_coef = max(0, total_time_of_earliness_group - self.deadline) / total_time_of_earliness_group
        earliness_group = sorted(earliness_group, key=lambda x: (1 - beta_coef) * x[2]/x[1] - beta_coef * x[3]/x[1], reverse=True)

        current_time_point = (self.deadline - total_time_of_earliness_group) if not beta_coef else 0

        while(len(earliness_group)):
            task = earliness_group.pop()
            next_time_point = current_time_point + task[1]

            if next_time_point > self.deadline:
                earliness_group.append(task)
                break

            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

        center_id = len(self.results) - 1

        tarliness_group = sorted(list(filter(lambda x: x[2] >= x[3], tasks)) + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tarliness_group)):
            task = tarliness_group.pop()
            self.results.append([task[0], current_time_point])
            current_time_point += task[1]

    def the_last_hope(self): 
        self.results = []
        tasks = self.instance.zipped_tasks()

        def create_b_set(tasks):
            p = sum([task[1] for task in tasks])
            results = []
            time_point = self.deadline - p
            for task in tasks:
                results.append([task[0], time_point])
                time_point += task[1]

            return results

        def create_a_set(tasks):
            results = []
            time_point = self.deadline
            for task in sorted(tasks, key=lambda x: x[1]/x[3]):
                results.append([task[0], time_point])
                time_point += task[1]

            return results
            

        earliness_tasks = []
        earliness_sum_p = 0
        tardliness_tasks = []
        tardliness_sum_p = 0

        tasks = sorted(tasks, key=lambda x: x[3]/x[2])
        next_step = False
        used_task = []

        stack = []
        while(len(tasks)):
            task = tasks.pop()
            if task[1] <= self.deadline - earliness_sum_p:
                earliness_tasks.append(task)
                earliness_sum_p += task[1]
            else:
                stack.append(task)

        b_set = create_b_set(earliness_tasks)
        a_set = create_a_set(stack)
        value = self.calculate_cost(b_set + a_set)

        while(True):
            earliness_tasks.append(stack.pop(0))
            new_b_set = create_b_set(earliness_tasks)
            new_a_set = create_a_set(stack)
            new_result = self.calculate_cost(new_b_set + new_a_set)
            if new_result < value:
                b_set = new_b_set
                a_set = new_a_set
            else:
                break

        self.results = b_set + a_set

    def is_valid(self, values=None):
    
        def overlapping(interval_a, interval_b):
            return bool(max(0, min(interval_a[1], interval_b[1]) - max(interval_a[0], interval_b[0])))

        results = values if values else self.results
        intervals = [(start_time, start_time + self.instance.p[task_id]) for task_id, start_time in results]

        while(len(intervals)):
            current_interval = intervals.pop()
            for interval in intervals:
                if overlapping(current_interval, interval):
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

    def generate_timeline(self, values=None, group=True):

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

def test():
    test_instance = Instance()
    test_instance.add_job(3, 2, 6)

    test_solver = Solver(test_instance, h=0.2)
    test_solver.deadline = 15
    test_solver.results.append((0, 12))

    print("Deadline: %d" % test_solver.deadline)
    print("Cost: %d" % test_solver.calculate_cost())
    print("Valid: %s" % str(test_solver.is_valid()))
    print("Timeline: %s" % test_solver.generate_timeline())

def instance():
    
    instances = parse_input_file(sys.argv[1])
    instance_id = int(sys.argv[3]) - 1
    deadline = float(sys.argv[2])

    #print("Instance:")
    #for record in instances[0].zipped_tasks():
    #    print("  ", record) 

    # print("Total P = %d" % sum(instances[0].p))

    
    #test_solver.solve()

    # print("Deadline: %d" % test_solver.deadline)

    #print("Cost: %d (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid())))

    instance = Instance()
    instance.add_job(3, 2, 8)
    instance.add_job(3, 4, 5)
    instance.add_job(1, 7, 5)

    test_solver = Solver(instances[instance_id], h=deadline)
    #test_solver = Solver(instance, h=0.8)
    methods = [
        test_solver.solve_without_beta,
        test_solver.solve_beta_equals_weights_in_a,
        test_solver.solve_beta_equals_weights_in_b,
        test_solver.solve_beta_equals_as_common_set,
        test_solver.solve_custtom_h1,
        test_solver.solve_custtom_h2,
        test_solver.solve_beta_equals_weights_in_b_gap,
        test_solver.the_last_hope
    ]

    for method in methods:
        method()
        print("Cost: %d (%s) (%s) (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid()), len(test_solver.results), method.__name__))

    """
    print(' --- ')
    print(test_solver.results)
    print(test_solver.generate_timeline())
    print("Cost: %d (%s) (%s) (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid()), len(test_solver.results), method.__name__))

    print(' --- ')
    test_solver.results = [[1, 12], [6, 0], [3, 18], [2, 31], [8, 44], [5, 56], [4, 68], [7, 80], [0, 83], [9, 103]]
    print(test_solver.results)
    print(test_solver.generate_timeline())
    print("Cost: %d (%s) (%s) (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid()), len(test_solver.results), method.__name__))
    """

    """
    print(test_solver.results)
    print("DEADLINE:", test_solver.deadline)
    test_solver.results = [[result[0], result[1] + 1] for result in test_solver.results]
    print(test_solver.results)
    print("Cost: %d (%s) (%s) (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid()), len(test_solver.results), method.__name__))

    test_solver.results = [[0, 1], [2, 4], [1, 5]]
    print(test_solver.results)
    print("Cost: %d (%s) (%s) (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid()), len(test_solver.results), method.__name__))
    print(test_solver.generate_timeline())

    """
    """

    test_solver.solve_without_beta()

    test_solver.solve_beta_equals_weights_in_a()
    print("Cost: %d (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid())), len(test_solver.results))

    test_solver.solve_beta_equals_weights_in_b()
    print("Cost: %d (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid())), len(test_solver.results))

    test_solver.solve_beta_equals_as_common_set()
    print("Cost: %d (%s)" % (test_solver.calculate_cost(), str(test_solver.is_valid())), len(test_solver.results))
    # print("Timeline: %s" % test_solver.generate_timeline())
    """
if __name__ == '__main__':

    #test()
    instance()