import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.5

def step(x):
    if x > 0:
        return 1
    else:
        return -1

INPUTS = np.array([
    [-1, -1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, 1]
])

OUTPUTS = np.array([[-1, 1, 1, -1]]).T

WEIGHTS = np.array([[0.0], [0.0], [0.0]])  # Change data type to float64

print("Random Weights {} before training".format(WEIGHTS))

errors = []

for _ in range(1000):
    for input_item, desired in zip(INPUTS, OUTPUTS):
        ADALINE_OUTPUT = np.dot(input_item, WEIGHTS)
        ADALINE_OUTPUT = step(ADALINE_OUTPUT)
        ERROR = desired - ADALINE_OUTPUT
        errors.append(ERROR)
        WEIGHTS += LEARNING_RATE * ERROR * np.reshape(input_item, (3, 1))

print("Random Weights {} after training".format(WEIGHTS))

for input_item, desired in zip(INPUTS, OUTPUTS):
    ADALINE_OUTPUT = np.dot(input_item, WEIGHTS)
    ADALINE_OUTPUT = step(ADALINE_OUTPUT)
    print("Actual {} desired {}".format(ADALINE_OUTPUT, desired))

ax = plt.subplot(111)
ax.plot(errors, label='Training Errors')
ax.set_xscale("log")
plt.title("ADALINE Errors (XOR Gate)")
plt.legend()
plt.xlabel('Error')
plt.ylabel('Value')
plt.show()
