import sys
from math import floor

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
        self.instance = instance
        self.deadline = floor(sum(instance.p) * h)
        self.results = []

    def solve(self):  
        tasks = self.instance.zipped_tasks()

        # Structure: (ID, P, A, B)
        earliness_group = list(filter(lambda x: x[2] <= x[3], tasks))
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

        tarliness_group = sorted(list(filter(lambda x: x[2] > x[3], tasks)) + earliness_group, key=lambda x: x[3]/x[1])
        while(len(tarliness_group)):
            task = tarliness_group.pop()
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
        
        for task_id, start_time in results:
            delta = abs(checkpoint - start_time)
            for empty_space_id in range(delta):
                timeline.append("_")

            timeline.append("[%s" % task_id)
            for time in range(self.instance.p[task_id] - len(str(task_id))):
                timeline.append('-')
            timeline.append("]")

            checkpoint = self.instance.p[task_id] + start_time
  
        timeline.insert(self.deadline, "|")
        return ''.join(timeline)

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

    test_solver = Solver(test_instance, h=0.8)
    test_solver.deadline = 15
    test_solver.results.append((0, 12))

    print("Deadline: %d" % test_solver.deadline)
    print("Cost: %d" % test_solver.calculate_cost())
    print("Valid: %s" % str(test_solver.is_valid()))
    print("Timeline: %s" % test_solver.generate_timeline())

def instance():
    instances = parse_input_file(sys.argv[1])

    print("Instance:")
    for record in instances[0].zipped_tasks():
        print("  ", record) 

    print("Total P = %d" % sum(instances[0].p))

    test_solver = Solver(instances[9], h=0.8)
    test_solver.solve()

    print("Deadline: %d" % test_solver.deadline)

    print("Cost: %d" % test_solver.calculate_cost())
    print("Valid: %s" % str(test_solver.is_valid()))
    print("Timeline: %s" % test_solver.generate_timeline())

if __name__ == '__main__':

    #test()
    instance()