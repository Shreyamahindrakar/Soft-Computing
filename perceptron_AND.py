# Perceptron network AND



# import numpy as np
# import pandas as pd

# def step_function(ip,threshold=0):
#     if ip >= threshold:
#         return 1
#     else:
#         return 0
# def cal_gate(x, w, b, threshold=0):
#     linear_combination = np.dot(w, x) + b
#  #print(linear_combination)
#     y = step_function(linear_combination,threshold)
#  #clear_output(wait=True)
#     return y
# 23
# def AND_gate_ip(x):
#     w = np.array([1, 1])
#     b = -1.5
#  #threshold = cal_output_or()
#     return cal_gate(x, w, b)

# input=[(0, 0), (0, 1), (1, 0), (1, 1)]
# print("Activation output")
# for i in input:
#  print(AND_gate_ip(i))



import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    
    def fit(self, X, d):
        for epoch in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x

# Define the input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
d = np.array([0, 0, 0, 1])

# Create a Perceptron instance and fit the data
perceptron = Perceptron(input_size=2)
perceptron.fit(X, d)

# Print the results
print("Input 1\tInput 2\tActivation Output\tIs Correct")
for i in range(X.shape[0]):
    x = np.insert(X[i], 0, 1)
    y = perceptron.predict(x)
    is_correct_string = 'Yes' if y == d[i] else 'No'
    print("{}\t{}\t{}\t\t\t{}".format(X[i][0], X[i][1], y, is_correct_string))
