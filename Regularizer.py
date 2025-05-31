import numpy as np
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
#Dense layer

class Layer_Dense:
    def __init__(self,n_inputs,n_neuron,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.01* np.random.randn(n_inputs,n_neuron)
        self.biases = np.zeros((1,n_neuron))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
              
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
        
    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights<0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
            
        if self.weight_regularizer_l2 > 0:
            self.dweights +=2*self.weight_regularizer_l2 * self.weights
            
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases<0] = -1
            self.dbiases +=self.bias_regularizer_l1 * dL1
            
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2*self.bias_regularizer_l2 * self.biases
            
            
            
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
    def regularization_loss(self,layer):
        regularization_loss = 0
        
        if layer.weight_regularizer_l1 > 0:
            regularization_loss +=layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        
        if layer.weight_regularizer_l2 > 0:
            regularization_loss +=layer.weight_regularizer_l2 * np.sum(np.abs(layer.weights))    
        if layer.bias_regularizer_l1 > 0:
            regularization_loss +=layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss +=layer.bias_regularizer_l2 * np.sum(np.abs(layer.biases))        
        return regularization_loss
    
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


class Optimizer_Adam:
    def __init__(self,learning_rate=1.,decay=0.,epsilon=1e-7,beta1=0.9,beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0 
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        
    def pre_update_param(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1./(1. + self.decay * self.iteration))
        
    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            
        layer.weight_momentum = self.beta1 * layer.weight_momentum + (1 - self.beta1)*layer.dweights
        layer.bias_momentum = self.beta1 * layer.bias_momentum + (1 - self.beta1)*layer.dbiases  
        
        layer.weight_cache = self.beta2 * layer.weight_cache + (1-self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1-self.beta2) * layer.dbiases**2
        
        weight_momentum_corrected = layer.weight_momentum / (1 - self.beta1 ** (self.iteration+1))
        bias_momentum_corrected = layer.bias_momentum / (1 - self.beta1 ** (self.iteration+1))
        
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iteration + 1)) 
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iteration + 1))     
        
        layer.weights += -self.current_learning_rate * \
            weight_momentum_corrected/(np.sqrt(weight_cache_corrected)+self.epsilon)
        #print('layer weights',layer.weights)
        layer.biases += -self.current_learning_rate * \
            bias_momentum_corrected/(np.sqrt(bias_cache_corrected)+self.epsilon)
        
    def post_update_param(self):
        self.iteration +=1       
# Dataset
X,y = spiral_data(samples=100,classes=3)
print(X.shape)
plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
plt.show()

dense1 = Layer_Dense(2,64,weight_regularizer_l2=5e-4,bias_regularizer_l2=5e-4)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64,3)
loss_activation = Activation_softmax_loss_categoricalcrossentropy()
optimizer = Optimizer_Adam(learning_rate=0.02,decay=5e-7)

for epoch in range(10001):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    data_loss = loss_activation.forward(dense2.output,y)
    regularization_loss = (loss_activation.loss.regularization_loss(dense1) +
                           loss_activation.loss.regularization_loss(dense2)
    )
    loss = data_loss+regularization_loss
    
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
              f'data_loss: {data_loss:.3f}'+
              f'reg loss: {regularization_loss:.3f}'+
              f'Learning Rate: {optimizer.current_learning_rate:.3f}')
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
    
# testing model
X_test,y_test = spiral_data(samples=100,classes=3)
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output,y_test)
predictions = np.argmax(loss_activation.output,axis=1)
if len(y_test.shape)==2:
    y_test = np.argmax(y_test,axis=1)
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')
   
        