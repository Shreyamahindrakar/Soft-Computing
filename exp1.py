# import pandas as pd
# test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
# correct_outputs = [False, True, True, False]
# outputs = []
# for test_input, correct_output in zip(test_inputs, correct_outputs):
 
#  output = int(test_input[0] ^ test_input[1])
#  outputs.append([test_input[0], test_input[1], output])
# output_frame = pd.DataFrame(outputs, columns=['Input A', ' Input B', 'Output'])
# print(output_frame.to_string(index=False))



import pandas as pd

# Define the XOR function
def xor(a, b):
    return int((a and not b) or (not a and b))

# Create a pandas DataFrame to represent the truth table
df = pd.DataFrame({
    'Input A': [0, 0, 1, 1],
    'Input B': [0, 1, 0, 1],
    'Output': [xor(0, 0), xor(0, 1), xor(1, 0), xor(1, 1)]
})

# Print the truth table
print(df)
