import random
import math
class Layer:
    def __init__(self, num_nodes_in, num_nodes_out):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out

        self.weights = [[0 for _ in range(self.num_nodes_in)] for _ in range(self.num_nodes_out)]
        self.costGradientW = [[0 for _ in range(self.num_nodes_in)] for _ in range(self.num_nodes_out)]
        self.biases = [0 for _ in range(self.num_nodes_out)]
        self.costGradientB = [0 for _ in range(self.num_nodes_out)]

        self.stored_activations = [0 for _ in range(self.num_nodes_out)]
        self.stored_weighted_inputs = [0 for _ in range(self.num_nodes_out)]
        self.stored_inputs = []

        self.randomize_weights()

    def randomize_weights(self):

        for out_node in range(self.num_nodes_out):

            for in_node in range(self.num_nodes_in):
                
                val = random.uniform(-1, 1)
                self.weights[out_node][in_node] = val

    def clear_gradients(self):
        self.costGradientW = [[0 for _ in range(self.num_nodes_in)] for _ in range(self.num_nodes_out)]
        self.costGradientB = [0 for _ in range(self.num_nodes_out)]



    def activation_function(self, weighted_input):
        """
        sigmoid function
        """
        return 1 / (1 + math.exp(-weighted_input))
    
    def activation_derivative(self, weighted_input):
        activation = self.activation_function(weighted_input)
        return activation * (1 - activation)
    
    def apply_gradients(self, learn_rate):

        for out_node in range(self.num_nodes_out):

            self.biases[out_node] -= self.costGradientB[out_node] * learn_rate
            for in_node in range(self.num_nodes_in):
                self.weights[out_node][in_node] = self.costGradientW[out_node][in_node] * learn_rate
    
    def node_cost(self, output_activation, expected_output):
        cost = output_activation - expected_output
        return cost * cost
    
    def node_cost_derivative(self, output_activation, expected_output):
        return 2 * (output_activation - expected_output)
    

    def calculate_outputs(self, inputs):
        activations = []

        for out_node in range(self.num_nodes_out):
            weighted_input = self.biases[out_node]
            for in_node in range(self.num_nodes_in):
                weighted_input += inputs[in_node] * self.weights[out_node][in_node]

            activation_value = self.activation_function(weighted_input)
            activations.append(activation_value)

            self.stored_weighted_inputs[out_node] = weighted_input
            self.stored_activations[out_node] = activation_value

        self.stored_inputs = inputs[:]
        return activations
    
    def calculate_output_layer_node_values(self, expected_outputs):
        node_values = []

        for i in range(len(expected_outputs)):
            # Calculate the partial derivatives: cost/activation and activation/weighted_input
            cost_derivative = self.node_cost_derivative(self.stored_activations[i], expected_outputs[i])
            activation_derivative = self.activation_derivative(self.stored_weighted_inputs[i])
            node_values.append(activation_derivative * cost_derivative)

        return node_values
    
    def calculate_hidden_layer_node_values(self, oldLayer, old_node_values):
        new_node_values = []

        for new_node_index in range(self.num_nodes_out):
            new_node_value = 0
            
            for old_node_index in range(len(old_node_values)):

                weight_derivative = oldLayer.weights[old_node_index][new_node_index]
                new_node_value += weight_derivative * old_node_values[old_node_index]

            new_node_value *= self.activation_derivative(self.stored_weighted_inputs[new_node_index])
            new_node_values.append(new_node_value)

        return new_node_values
    
    def update_gradients(self, node_values):

        for out_node in range(self.num_nodes_out):

            for in_node in range(self.num_nodes_in):

                # Calculate partial derivative: cost/weight
                weight_derivative = self.stored_inputs[in_node] * node_values[out_node]
                self.costGradientW[out_node][in_node] += weight_derivative

            bias_derivative = 1 * node_values[out_node]
            self.costGradientB[out_node] += bias_derivative
