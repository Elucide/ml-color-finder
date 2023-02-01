import numpy as np
from scipy.special import expit as activation_function
from scipy.stats import truncnorm

def truncated_normal(mean = 0, sd = 1, low = 0, upp = 10):
    print (str(truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)))
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def activate(weights, inputs):
    activation = np.empty([0], dtype=np.float64 , like=None)
#    activation = np.insert(activation, 0, weights[-1])
    activation = np.append(activation, weights[-1])
#    print("len of input is ", str(len(inputs)))
    print ("activation is", str(activation))
    for i in range(len(weights) - 1):
#        print("i is", str(i))
        activation = np.insert(activation, 0, activation[0] + weights[i] * np.float64(inputs[i]))
    return (activation)

def transfer(activation):
    return (1.0 / (1.0 + np.exp(- activation)))

#def forward_propagation(network, )

class Nnetwork:
    def __init__(self, n_in_nrn, n_out_nrn, n_hidden_nrn, learning_rate):
        self.n_in_nrn = n_in_nrn
        self.n_out_nrn = n_out_nrn
        self.n_hidden_nrn = n_hidden_nrn
        self.learning_rate = learning_rate # wtf is that ?
        self.create_weight_matrices()

    def __del__(self):
        print ("neural network destroyed !")

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
                # Turn the input vector into a column vector:
        input_vector = np.array(input_vector, ndmin=2).T
        # activation_function() implements the expit function,
        # which is an implementation of the sigmoid function:
        input_hidden = activation_function(self.weights_in_hidden @   input_vector)
        output_vector = activation_function(self.weights_hidden_out @ input_hidden)
#i        return output_vector 
#        input_vector = np.array(input_vector, ndmin=2)
#        input_hidden = activation_function(self.weights_in_hidden @ input_vector)
#        output_vector = activation_function(self.weights_hidden_out @    input_hidden)
        print("type of output_vector[0] is", str(type(output_vector[0])))
        return (output_vector)

    def lcd_run(self, input_vector):
#        print("input vector is =", input_vector)
        output_vector = np.array([], ndmin=2)
        mid_vector = np.array([], ndmin=2)
        for i in range(self.n_hidden_nrn):
            np.append(mid_vector, transfer(activate(self.weights_in_hidden[i], input_vector)))
        for i in range(self.n_out_nrn):
            np.append(output_vector, transfer(activate(self.weights_hidden_out[i], mid_vector)))
        return (output_vector)


    

simple_network = Nnetwork(n_in_nrn = 3, n_out_nrn = 3, n_hidden_nrn = 4, learning_rate = 0.6)

#simple_network.monitor()

print("scipy output", str(simple_network.run([(3, 4, 5)])))
print("lcd output", str(simple_network.lcd_run([(3, 4, 5)])))
