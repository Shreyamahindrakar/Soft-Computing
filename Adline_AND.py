import numpy as np
import matplotlib.pyplot as plt
import math
LEARNING_RATE = 0.5
def step(x):
    if (x > 0):
        return 1
    else:
        return -1
INPUTS = np.array([
    [-1,-1,1],
    [-1,1,1],
    [1,-1,1],
    [1,1,1] ])

OUTPUTS = np.array([[-1,-1,-1,1]]).T
WEIGHTS = np.array([[0],[0],[0]])
print("Random Weights {} before training".format(WEIGHTS))
errors=[]
for iter in range(1000):
    for input_item,desired in zip(INPUTS, OUTPUTS):
        ADALINE_OUTPUT = (input_item[0]*WEIGHTS[0]) + (input_item[1]*WEIGHTS[1]) + (input_item[2]*WEIGHTS[2])
        ADALINE_OUTPUT = step(ADALINE_OUTPUT)
        ERROR = desired - ADALINE_OUTPUT
        errors.append(ERROR)
        WEIGHTS[0] = WEIGHTS[0] + LEARNING_RATE * ERROR * input_item[0]
        WEIGHTS[1] = WEIGHTS[1] + LEARNING_RATE * ERROR * input_item[1]
        WEIGHTS[2] = WEIGHTS[2] + LEARNING_RATE * ERROR * input_item[2]
 
print("Random Weights {} after training".format(WEIGHTS))
for input_item,desired in zip(INPUTS, OUTPUTS):
    ADALINE_OUTPUT = (input_item[0]*WEIGHTS[0]) +(input_item[1]*WEIGHTS[1]) + (input_item[2]*WEIGHTS[2])
    ADALINE_OUTPUT = step(ADALINE_OUTPUT)
    print("Actual {} desired {} ".format(ADALINE_OUTPUT,desired))
    ax = plt.subplot(111)
    ax.plot(errors, label='Training Errors')
    ax.set_xscale("log")
    plt.title("ADALINE Errors (2,-2)")
    plt.legend()
    plt.xlabel('Error')
    plt.ylabel('Value')
    plt.show()  