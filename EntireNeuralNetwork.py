import numpy as np

#Dense layer

class Layer_Dense:
    def __init__(self,n_inputs,n_neuron):
        self.weights = np.random.randn(n_inputs,n_neuron)
        self.biases = np.zeros((1,n_neuron))
        
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
        
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)
        
# ReLU class

class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
        
    def backward(self,dvalues):
        self.dinputs = dvalues
        self.dinputs[dvalues <=0] = 0
        
# Softmax activation
class Activation_softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values/np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities
        
# class Loss
class Loss:
    def calculate(self,output,y):
        sample_lossess = self.forward(output,y)
        data_loss = np.mean(sample_lossess)
        return data_loss
# cross entropy loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        
        if len(y_true.shape) ==1:
            correct_confidence = y_pred_clipped[range(samples),y_true]
            
        elif len(y_true.shape) ==2:
            correct_confidence = np.sum(y_pred_clipped*y_true,axis=1)
        
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood
    
    def backward(self,dvalues,y_true):
        
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/samples
        
# combine softmax and categorical entropy loss for last layer
class Activation_softmax_loss_categoricalcrossentropy:
    def __init__(self):
        self.activation = Activation_softmax()
        self.loss = Loss_CategoricalCrossentropy()
        
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape)==2:
            y_true = np.argmax(y_true,axis=1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -=1
        self.dinputs = self.dinputs/samples
        
# Dataset
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()
import matplotlib.pyplot as plt

X,y = spiral_data(samples=100,classes=3)
print(X.shape)
plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
loss_activation = Activation_softmax_loss_categoricalcrossentropy()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output,y)

print(loss_activation.output[:5])
print(f'Loss:{loss}')

#calculate accuracy
predictions = np.argmax(loss_activation.output,axis=1)
if len(y.shape)==2:
    y = np.argmax(y,axis=1)
    
accuracy = np.mean(predictions==y)
print(f'accurcay: {accuracy}')

#backward pass
loss_activation.backward(loss_activation.output,y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)
print(dense1.biases)
print(dense2.dweights)
print(dense2.dbiases)

        
        