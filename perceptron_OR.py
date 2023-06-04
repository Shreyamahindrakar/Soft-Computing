import numpy as np

def step_function(ip, threshold=0):
    if ip >= threshold:
        return 1
    else:
        return 0

def cal_gate(x, w, b, threshold=0):
    linear_combination = np.dot(w, x) + b
    y = step_function(linear_combination, threshold)
    return y

def OR_gate_ip(x):
    w = np.array([1, 1])
    b = -0.5
    return cal_gate(x, w, b)

input = [(0, 0), (0, 1), (1, 0), (1, 1)]

for i in input:
    print(OR_gate_ip(i))
