import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class Nnetwork:
    def __init__(self, n_in_nrn, n_out_nrn, n_hidden_nrn, learning_rate):
        self.n_in_nrn = n_in_nrn
        self.n_out_nrn = n_out_nrn
        self.n_hidden_nrn = n_hidden_nrn
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ INIT WEIGHT MATRICES"""
        rad = 1 / np.sqrt(self.n_in_nrn)
        X = truncated_normal(mean = 0, sd = 1, low = -rad, upp = rad)
        self.weights_in_hidden = X.rvs((self.n_hidden_nrn, self.n_in_nrn))
        rad = 1 / np.sqrt(self.n_hidden_nrn)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.n_out_nrn, self.n_hidden_nrn))
        
    def train(self, input_vector, target_vector):
        """ LEARNING METHOD """
        pass # flemme

    def run(self, input_vector):
        """ RUNNING THE NETWORK WITH AN IMPUT VECTOR, CAN BE A TUPLE, LIST, OR NDARRAY """
        input_vector = np.array(input_vector, ndmin=2).T
        input_hidden = activation_function(self.weights_in_hidden @ input_vector)
        output_vector = activation_function(self.weights_out_hidden @ output_vector)
        return (output_vector)


simple_network = Nnetwork(n_in_nrn = 2, n_out_nrn = 4, n_hidden_nrn = 2, learning_rate = 0.6)


simple_network.run([(3, 4)])
