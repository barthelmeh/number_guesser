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

    def train(self, input_data, epochs):
        pass


