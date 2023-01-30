import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10):
    print (str(truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)))
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return (activation)

def transfer(activation):
    return (1.0 / (1.0 + exp(- activation)))

#def forward_propagation(network, )

class Nnetwork:
    def __init__(self, n_in_nrn, n_out_nrn, n_hidden_nrn, learning_rate):
        self.n_in_nrn = n_in_nrn
        self.n_out_nrn = n_out_nrn
        self.n_hidden_nrn = n_hidden_nrn
        self.learning_rate = learning_rate # wtf is that ?
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ INIT WEIGHT MATRICES WITH RANDOM VALUES< WILL BE USEFUL WHEN I'LL MAKE THE BACKPROPAGATION"""
        rad = 1 / np.sqrt(self.n_in_nrn)    # I still dont know why it is useful
        X = truncated_normal(mean = 0, sd = 1, low = -rad, upp = rad) # settin the random values
        self.weights_in_hidden = X.rvs((self.n_hidden_nrn, self.n_in_nrn)) # same (I think)
        rad = 1 / np.sqrt(self.n_hidden_nrn)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.n_out_nrn, self.n_hidden_nrn))
        
    def monitor(self):
        """MONITOR MEURAL NETWORK STATE"""
        print("-----------BEGIN MONITOR-----------\n")
        print("actual neural network has", str(self.n_hidden_nrn), "neurons\nin his hidden layer\n\nTHE WEIGHTS OF THOSE NEURON ARE:")
        i = 0;
        print("neuron of beetween in and hidden layer")
        for nrn in self.weights_in_hidden:
            print("neuron #" + str(i), str(nrn))
            i += 1
        print("\nneuron of beetween hidden and out layer")
        i = 0;
        for nrn in self.weights_hidden_out:
            print("neuron #" + str(i), str(nrn))
            i += 1
        print("\n------------END MONITOR------------")

    def train(self, input_vector, target_vector):
        """ LEARNING METHOD (BACKPROPAGATION) """
        pass # flemme

    def run(self, input_vector):
        """ RUNNING THE NETWORK WITH AN IMPUT VECTOR, CAN BE A TUPLE, LIST, OR NDARRAY """
#        input_vector = np.array(input_vector, ndmin=2).#T
#        input_hidden = activation_function(self.weights_in_hidden @ input_vector)
#        output_vector = activation_function(self.weights_hidden_out @    input_hidden)
        print("nrn_hidden[1] =", self.weights_in_hidden[1])
#        for nrn in self.nrn_hidden:
            
#        return (output_vector)


simple_network = Nnetwork(n_in_nrn = 3, n_out_nrn = 2, n_hidden_nrn = 4, learning_rate = 0.6)

simple_network.monitor()

print(str(simple_network.run([(3, 4, 5)])))
