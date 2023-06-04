#Adline

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

OUTPUTS = np.array([[-1,1,1,1]]).T
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

# import numpy as np
# import matplotlib.pyplot as plt

# class AdaptiveLinearNeuron:
#     def __init__(self, learning_rate=0.01, num_iterations=100):
#         self.learning_rate = learning_rate
#         self.num_iterations = num_iterations
    
#     def fit(self, X, y):
#         self.weights = np.random.random_sample(1 + X.shape[1])
#         print("Random Weights", self.weights)
        
#         self.errors = []
#         for i in range(self.num_iterations):
#             error = 0
#             for xi, target in zip(X, y):
#                 update = self.learning_rate * (target - self.predict(xi))
#                 self.weights[1:] += update * xi
#                 self.weights[0] += update
#                 error += int(update != 0.0)
#             self.errors.append(error)
#         print("Trained Weights", self.weights)
#         return self
    
#     def net_input(self, X):
#         return np.dot(X, self.weights[1:]) + self.weights[0]
    
#     def predict(self, X):
#         return np.where(self.net_input(X) >= 0.0, 1, -1)

# # example usage
# X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
# y = np.array([-1, 1, 1, 1])

# model = AdaptiveLinearNeuron()
# model.fit(X, y)

# # plot errors
# plt.plot(range(1, len(model.errors) + 1), model.errors, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of errors')
# plt.show()

# # test predictions
# new_X = np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]])
# for xi in new_X:
#     prediction = model.predict(xi)
#     print("Actual", prediction, "desired", np.where(model.net_input(xi) >= 0.0, 1, -1))
