import numpy as np
import random
import nnfs
from nnfs.datasets import spiral_data

class Dense_layer:
    def __init__(self,n_input,n_neuron):
        self.weights = 0.01 * np.random.randn(n_input,n_neuron)
        self.biases = np.zeros((1,n_neuron))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
     

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
        
class Activation_softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities
        
        

X,y = spiral_data(samples=100, classes = 3)
# create dense layer with 2 input and 3 output
dense1 = Dense_layer(2,3)
# create Relu activation function
activation1 = Activation_ReLU()
#create dense layer with 3 input and 3 output
dense2 = Dense_layer(3,3)
#create softmax activation function
activation2 = Activation_softmax()

# Forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
        
        
        
        