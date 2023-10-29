import random
import math
class Layer:
    def __init__(self, num_nodes_in, num_nodes_out):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out

        self.weights = [[0 for _ in range(self.num_nodes_in)] for _ in range(self.num_nodes_out)]
        self.biases = [0 for _ in range(self.num_nodes_out)]

    def activation_function(self, weighted_input):
        """
        sigmoid function
        """
        return 1 / (1 + math.exp(-weighted_input))

    def calculate_outputs(self, inputs):
        activations = []

        for out_node in range(self.num_nodes_out):
            weighted_input = self.biases[out_node]
            for in_node in range(self.num_nodes_in):
                weighted_input += inputs[in_node] * self.weights[in_node][out_node]

            activations.append(self.activation_function(weighted_input))

        return activations

    def node_cost(self, output_activation, expected_output):
        cost = expected_output - output_activation
        return cost * cost
