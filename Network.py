import Layer, DataPoint
class Network:
    def __init__(self, layer_sizes):
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)

    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def classify(self, inputs):
        """
        Returns the output node with the highest value
        """
        outputs = self.calculate_outputs(inputs)
        return outputs.index(max(outputs))

    def calculate_cost(self, data_point: DataPoint):
        outputs = self.calculate_outputs(data_point.inputs)
        output_layer = self.layers[len(self.layers) - 1]
        
        cost = 0
        for out_node in range(len(outputs)):
            cost += output_layer.node_cost(outputs[out_node], data_point.expected_outputs[out_node])


    def cost(self, data_points):
        total_cost = 0
        for data_point in data_points:
            total_cost += self.calculate_cost(data_point)
        return total_cost / len(data_points)
    

    # Performs one iteration of gradient descent
    def learn(self, input_data, learn_rate):
       
        for data_point in input_data:
           self.update_all_gradients(data_point)

        self.apply_all_gradients(learn_rate / len(input_data))

        # Set back to zero
        self.clear_all_gradients()
       

    def apply_all_gradients(self, learn_rate):
        """
        Applies the gradients on all the layers
        """

        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def clear_all_gradients(self):
        for layer in self.layers:
            layer.clear_gradients()

    # Backwards Propogation
    def update_all_gradients(self, data_point: DataPoint):
        # Run all the inputs through the network
        self.calculate_outputs(data_point.inputs)

        # Update gradients of output layer
        output_layer = self.layers[len(self.layers) - 1]
        node_values = output_layer.calculate_output_layer_node_values(data_point.expected_outputs)
        output_layer.update_gradients(node_values)

        # Update gradients of hidden layers
        for hidden_layer_index in range(len(self.layers)-2, 0, -1):
            hidden_layer = self.layers[hidden_layer_index]
            hidden_node_values = hidden_layer.calculate_hidden_layer_node_values(self.layers[hidden_layer_index + 1], node_values)
            hidden_layer.update_gradients(hidden_node_values)
