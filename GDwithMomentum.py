import numpy as np
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
#Dense layer

class Layer_Dense:
    def __init__(self,n_inputs,n_neuron):
        self.weights = 0.01* np.random.randn(n_inputs,n_neuron)
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
        self.dinputs[self.inputs <=0] = 0     
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


class Optimizer_GD:
    def __init__(self,learning_rate=1.,decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0 
        self.momentum = momentum
        
    def pre_update_param(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1./(1. + self.decay * self.iteration))
        
    def update_params(self,layer):
        
        if self.momentum:
            if not hasattr(layer,'weights_momentums'):
                layer.weights_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            weights_update = self.momentum * layer.weights_momentums - \
                self.current_learning_rate * layer.dweights
            bias_update = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
                
            layer.weights_momentums = weights_update
            layer.bias_momentums = bias_update
        else:
            
            weights_update = -self.current_learning_rate * layer.dweights
            #print('layer weights',layer.weights)
            bias_update = -self.current_learning_rate * layer.dbiases 
        
        layer.weights += weights_update
        layer.biases += bias_update
        
    def post_update_param(self):
        self.iteration +=1       
# Dataset
X,y = spiral_data(samples=100,classes=3)
print(X.shape)
plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()

dense1 = Layer_Dense(2,64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_softmax_loss_categoricalcrossentropy()
optimizer = Optimizer_GD(decay=1e-3,momentum=0.5)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)

    #print(loss_activation.output[:5])
    #print(f'Loss:{loss}')

    #calculate accuracy
    predictions = np.argmax(loss_activation.output,axis=1)
    if len(y.shape)==2:
        y = np.argmax(y,axis=1)
        
    accuracy = np.mean(predictions==y)
    
    if not epoch%100:
        print(f'epoch: {epoch},'+
              f'acc: {accuracy:.3f},'+
              f'loss: {loss:.3f},'+
              f'Learning Rate: {optimizer.current_learning_rate}')
    #print(f'accurcay: {accuracy}')

    #backward pass
    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.pre_update_param()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_param()
    
   
        