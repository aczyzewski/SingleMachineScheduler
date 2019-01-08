
import sys
import os
import ntpath

def parse_results(file):
    results_lines = open(file, 'r').read().split('\n')
    file_name = ntpath.split(file)[1]
    instance_size = int(file_name.split('.')[0][3:])
    instance_deadlines = [0.2, 0.4, 0.6, 0.8]

    local_results = {}

    for line in results_lines:
        line = line.replace(',', '')
        line = line.split()[2:]

        deadlines = dict(zip(instance_deadlines, line[1:]))
        local_results[int(line[0])] = deadlines

    return instance_size, local_results

def get_results_object(path='results'):
    
    values = {}
    results = [os.path.join(path, file) for file in os.listdir(path) if 'res' in file and file.endswith('.txt')]
    for result_file in results:
        instance_size, res = parse_results(result_file)
        values[instance_size] = res

    return values

if __name__ == "__main__":
    val = get_results_object()
    print(val[200][9][0.8])
    