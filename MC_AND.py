# McCULLOCH PITTS MODEL:AND

import numpy as np
import pandas as pd
def cal_output_and(threshold=0):
    weight1 = 1
    weight2 = 1
    bias = 0
    test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    correct_outputs = [False, False, False, True]
    outputs = []
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias
        output = int(linear_combination >= threshold)
        is_correct_string = 'Yes' if output == correct_output else 'No' 
        outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])
        num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
        output_frame = pd.DataFrame(outputs, columns=['Input 1', ' Input 2', ' Linear Combination', ' Activation Output', ' Is Correct'])
    if not num_wrong:
        print('all correct for threshold {}.\n'.format(threshold))
    else:
        threshold = threshold + 1
        cal_output_and(threshold)
        print('{} wrong, for threshold {} \n'.format(num_wrong,threshold))
        print(output_frame.to_string())
    return threshold

t = cal_output_and()


# import numpy as np
# import pandas as pd

# # Define the McCulloch-Pitts neuron function
# def mcculloch_pitts_neuron(inputs, weights, threshold):
#     # Compute the dot product of inputs and weights
#     linear_combination = np.dot(inputs, weights)
#     # Apply the threshold function
#     output = int(linear_combination >= threshold)
#     return output

# # Define the logical AND function
# def logical_and(inputs):
#     # Set the weights and threshold for the AND operation
#     weights = np.array([1, 1])
#     threshold = 2
#     # Compute the output using the McCulloch-Pitts neuron function
#     output = mcculloch_pitts_neuron(inputs, weights, threshold)
#     return output

# # Create a pandas DataFrame to represent the truth table
# df = pd.DataFrame({
#     'Input 1': [0, 0, 1, 1],
#     'Input 2': [0, 1, 0, 1],
#     'Output': [logical_and([0, 0]), logical_and([0, 1]), logical_and([1, 0]), logical_and([1, 1])]
# })

# # Print the truth table
# print(df)
