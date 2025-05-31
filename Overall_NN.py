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
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        print(y_true.shape)
        if len(y_true.shape) ==1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_logloss = -np.log(correct_confidences)
        return negative_logloss        
        

X,y = spiral_data(samples=100, classes = 3)
# create dense layer with 2 input and 3 output
dense1 = Dense_layer(2,3)
# create Relu activation function
activation1 = Activation_ReLU()
#create dense layer with 3 input and 3 output
dense2 = Dense_layer(3,3)
#create softmax activation function
activation2 = Activation_softmax()
loss_function = Loss_CategoricalCrossEntropy()
# Forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output,y)
print(f'Loss:{loss}')

predictions = np.argmax(activation2.output,axis=1)
if len(y.shape)==2:
    y = np.argmax(y,axis=1)
accuracy = np.mean(predictions==y)
print(f'acc: {accuracy}')        
        
        
        