import numpy as np

class FNN:
    def __init__(self, units_per_layer):
        """ Create Feedforward Neural Network based on specifications
        units_per_layer: (list, len>=2) Number of neurons in each layer including input, hidden and output
        """
        self.units_per_layer = units_per_layer
        self.num_layers = len(units_per_layer)

        # lambdas for supported activation functions
        self.hiddenactivation = lambda x:np.maximum(0, x)
        self.outputactivation = lambda x: 1 / (1 + np.exp(-x))

        self.weightrange = 1
        self.biasrange = 1

    def setParams(self, params):
        """ Set the weights, biases, and activation functions of the neural network 
        Weights and biases are set directly by a parameter;
        The activation function for each layer is set by the parameter with the highest value (one for each possible one out of the six)
        """
        self.weights = []
        start = 0
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l]*self.units_per_layer[l+1]
            self.weights.append((params[start:end]*self.weightrange).reshape(self.units_per_layer[l],self.units_per_layer[l+1]))
            start = end
        self.biases = []
        for l in np.arange(self.num_layers-1):
            end = start + self.units_per_layer[l+1]
            self.biases.append((params[start:end]*self.biasrange).reshape(1,self.units_per_layer[l+1]))
            start = end

    def forward(self, inputs):
        """ Forward propagate the given inputs through the network """
        states = np.asarray(inputs)
        for l in np.arange(self.num_layers - 1):
            if states.ndim == 1:
                states = [states]
            if (l < self.num_layers - 1): 
                states = self.hiddenactivation(np.matmul(states, self.weights[l]) + self.biases[l]) #keep output greater than 0
            else:
                states = self.outputactivation(np.matmul(states, self.weights[l]) + self.biases[l]) #return sigmoid based result
        return states

