import sys, glob
from subprocess import check_output
import math
import timeit
from timeit import default_timer as time

INPUT_DIR = './input'
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

# N = [10]
# H = [0.4]
# Ks = [
#     [1]
# ]

Ref = [
    [1936, 1020, 521, 615],
    [8567, 2097, 3016, 3600],
    [37641, 20048, 17715, 14105],
    [137265, 89588, 80844, 55277],
    [547953, 335714, 255029, 236160],
    [3024085, 1909304, 1520515, 1634912],
    [14160773, 7300217, 6411581, 6069658]
]
# Ks = [[1]]
# N = [20]
# H = [0.8]
# K = [1, 6]
# H = [0.4, 0.6]
# N = [10]
# K = [1]
# H = [0.6]


def parse_original_input(N, K, h):
    fileName = f'./data/sch{N}.txt'
    output = {}
    with open(fileName) as file:
        file.readline()
        for k in range(K - 1):
            for n in range(N + 1):
                file.readline()
        file.readline()
        processing_time = 0.
        for n in range(N):
            line = file.readline()
            matches = line.split()
            p, a, b = matches
            p = int(p)
            a = int(a)
            b = int(b)
            processing_time += p
            output[n] = {'p': p, 'a': a, 'b': b}
    output['deadline'] = math.floor(processing_time * h)
    # print(processing_time, h, output['deadline'])

    return output


def validate_output(out, original_input, ref):
    output = {}
    output_cost = -1;
    idx = 0
    for line in out.split('\n'):
        match = line.split(' ')
        if len(match) == 2:
            begin, end = match
            output[idx] = (int(begin), int(end))
            idx += 1
        elif len(match) == 3:
            pass
        elif match[0] != '':
            # print(match)
            output_cost = float(match[0])

    # Validate processing times are the same
    # Validate all tasks were assigned
    # Validate tasks don't overlap
    # Validate cost is equal +- epsilon

    sorted_output = dict(sorted(output.items(), key=lambda kv: kv[1][0]))
    # print(sorted_output)

    current = list(sorted_output.items())[0][1][0]
    cost_veri = 0.
    original_deadline = original_input['deadline']
    # print(f'Deadline: {original_deadline}')
    for k, v in sorted_output.items():
        begin, end = v
        processing_time = end - begin
        if original_input[k]['p'] != processing_time:
            print(
                f'Error! Processing time: {processing_time} of task {k} is different than in original definition: {original_input[k]}')
        if begin < current:
            print(f'Error! Task {k} begins before previous one has finished')
        current += processing_time
        if current != end:
            print(f'Wrong ending time: {k} {current} vs {end}')
        if current <= original_input['deadline']:
            cost = (original_input['deadline'] - current) * original_input[k]['a']
            cost_veri += cost
            # print(f'C: {current} ET: {original_deadline - current} cost: {cost} original: {original_input[k]}')
        elif current > original_input['deadline']:
            cost = (current - original_input['deadline']) * original_input[k]['b']
            cost_veri += cost
            # print(f'C: {current} TT: {current - original_deadline} cost: {cost} original: {original_input[k]}')


    epsilon = 1
    cost_diff = abs(cost_veri - output_cost)
    if cost_diff > epsilon:
        print(f'Significant difference in calculated costs: {cost_diff}')
    print(f'Calculated cost:\t{cost_veri}\t{ref}\t{(cost_veri-ref)/ref}')
    # print(f'Output cost:\t{output_cost}')

def tim(path, n, k, h):
    check_output(f'{path} {str(n)} {str(k)} {str(h)}', shell=True)

def main():
    if (len(sys.argv) > 2):
        print('must pass path to the bash script')
        print('e.g. python ./validator ./bash.sh')
    elif (len(sys.argv) < 2):
        print('path to the bash script was not passed')
        print('will try using ./script.sh')
        path = './script.sh'
    else:
        path = sys.argv[1]
    cum_sum = 0
    for n_idx, n in enumerate(N):
        for h_idx, h in enumerate(H):
            k = Ks[n_idx][h_idx]
            ref = Ref[n_idx][h_idx]
            print(f'Will do k: {k} h: {h} n: {n}')
            original_input = parse_original_input(n, k, h)
            # print(original_input)
            start = time()
            output = check_output(f'{path} {str(n)} {str(k)} {str(h)}', shell=True)
            end = time()
            part_time = end - start
            cum_sum += part_time
            print(f'Czas: {part_time}')
            output = output.decode("utf-8")
            # print(output)
            validate_output(output, original_input, ref)
            # calculate_time
    print(f'Cum_time_sum: {cum_sum}')


if __name__ == "__main__":
    main()
