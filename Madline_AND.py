import numpy as np

class Madaline:
    def __init__(self, num_layers, num_neurons):
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.weights = [np.random.randn(num_neurons, num_neurons) for _ in range(num_layers)]
        self.biases = [np.random.randn(num_neurons) for _ in range(num_layers)]

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def feedforward(self, x):
        output = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(output, w) + b
            output = self.activation_function(z)
        return output

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        for _ in range(epochs):
            for x, target in zip(X, y):
                output = self.feedforward(x)
                error = target - output

                # Update weights and biases for each layer
                for i in range(self.num_layers):
                    delta = learning_rate * np.outer(error, self.activation_function(np.dot(x, self.weights[i]) + self.biases[i]))
                    self.weights[i] += delta
                    self.biases[i] += learning_rate * error

    def predict(self, X):
        predictions = []
        for x in X:
            output = self.feedforward(x)
            predictions.append(output)
        return predictions

# Define the AND gate inputs and outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])  # AND gate outputs

# Create a Madaline network with 1 layer and 2 neurons
madaline = Madaline(num_layers=1, num_neurons=2)

# Train the Madaline network on the AND gate data
madaline.train(X, y)

# Predict the outputs for the AND gate inputs
predictions = madaline.predict(X)

# Print the predictions
for i, pred in enumerate(predictions):
    print(f"Input: {X[i]}, Predicted Output: {pred}")
