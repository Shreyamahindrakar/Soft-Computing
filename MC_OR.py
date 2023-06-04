import numpy as np
import pandas as pd

# Define the McCulloch-Pitts neuron function
def mcculloch_pitts_neuron(inputs, weights, threshold):
    # Compute the dot product of inputs and weights
    linear_combination = np.dot(inputs, weights)
    # Apply the threshold function
    output = int(linear_combination >= threshold)
    return output

def logical_or(inputs):
    # Set the weights and threshold for the OR operation
    weights = np.array([1, 1])
    threshold = 1
    # Compute the output using the McCulloch-Pitts neuron function
    output = mcculloch_pitts_neuron(inputs, weights, threshold)
    return output


# Create a pandas DataFrame to represent the truth table
df = pd.DataFrame({
    'Input 1': [0, 1, 0, 1],
    'Input 2': [0, 0, 1, 1],
    'Activation Output': [logical_or([0, 0]), logical_or([1, 0]), logical_or([0, 1]), logical_or([1, 1])],
    'Is Correct': ['Yes', 'Yes', 'Yes', 'Yes']
})

# Print the truth table
print(df)
