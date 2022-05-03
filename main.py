import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1
print(synaptic_weights)

def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class Neuron:
    def __init__(self, number_of_inputs):
        self.number_of_inputs = number_of_inputs
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((number_of_inputs, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        self.graph = [[0] * training_iterations, [0] * training_iterations]

        for iteration in range(training_iterations):
            output = self.test(training_inputs)

            error = training_outputs - output

            self.graph[0][iteration]=iteration
            self.graph[1][iteration]=error[2]/training_outputs[2]

            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def test(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output
    def plot_train(self):
        plt.plot(self.graph[0], self.graph[1])


if __name__ == '__main__':
    n = Neuron(3)
    n.train(
        np.array(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 1],
             [1, 1, 1]]),
        np.array([[0, 0, 1, 1]]).T,
        1000)
    print(n.test(np.array([1, 1, 0])))
    n.plot_train()
    plt.show()
