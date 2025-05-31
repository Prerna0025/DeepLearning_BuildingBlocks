import numpy as np

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
    
class Activation_softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1,keepdims=True)
        self.output = probabilities
        
#====New in this lecture=====
class Activation_softmax_loss_categoricalCrossentropy:
    def __init__(self):
        self.activation = Activation_softmax
        self.loss = Loss_CategoricalCrossEntropy
        
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)
    
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true,axis=1)
            
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -=1
        self.dinputs = self.dinputs/samples
        
softmax_output = np.array([[0.7,0.1,0.2],
                           [0.1,0.5,0.4],
                           [0.02,0.9,0.08]])
class_target = np.array([0,1,1])
softmax_loss = Activation_softmax_loss_categoricalCrossentropy()
softmax_loss.backward(softmax_output,class_target)
dvalues1 = softmax_loss.dinputs
print('Gradients: combined loss and activation')
print(dvalues1)
        