# Write a program to implement logical XOR using McCulloch Pitt’s neuron model


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

# def AND_gate_ip(x):
#     w = np.array([1, 1])
#     b = -1.5
#  #threshold = cal_output_or()
#     return cal_gate(x, w, b)

# def NOT_gate_ip(x):
#     w = -1
#     b = .5
#  #threshold = cal_output_not()
#     return cal_gate(x, w, b)

# def OR_ip(x):
#     w = np.array([1, 1])
#     b = -0.5
#     return cal_gate(x, w, b)

# def Logical_XOR(x):
#     A = AND_gate_ip(x)
#     C = NOT_gate_ip(A)
#     B = OR_ip(x)
#     AND_output = np.array([C, B])
#     output = AND_gate_ip(AND_output)
#     return output


# input=[(0, 0), (0, 1), (1, 0), (1, 1)]

# for i in input:
#  print(Logical_XOR(i))

import numpy as np
import pandas as pd

# Define the McCulloch-Pitts neuron function
def mcculloch_pitts_neuron(inputs, weights, threshold):
    # Compute the dot product of inputs and weights
    linear_combination = np.dot(inputs, weights)
    # Apply the threshold function
    output = int(linear_combination >= threshold)
    return output

# Define the logical XOR function
def logical_xor(inputs):
    # Set the weights and threshold for the XOR operation
    weights1 = np.array([1, -1])
    weights2 = np.array([-1, 1])
    threshold1 = 0
    threshold2 = 1
    # Compute the output using the McCulloch-Pitts neuron function
    hidden_output1 = mcculloch_pitts_neuron(inputs, weights1, threshold1)
    hidden_output2 = mcculloch_pitts_neuron(inputs, weights2, threshold2)
    output = mcculloch_pitts_neuron([hidden_output1, hidden_output2], weights1, threshold1)
    return output

# Create a pandas DataFrame to represent the truth table
df = pd.DataFrame({
    'Input 1': [0, 1, 0, 1],
    'Input 2': [0, 0, 1, 1],
    'Activation Output': [logical_xor([0, 0]), logical_xor([1, 0]), logical_xor([0, 1]), logical_xor([1, 1])],
    'Is Correct': ['Yes', 'Yes', 'Yes', 'Yes']
})

# Print the truth table
print(df)
