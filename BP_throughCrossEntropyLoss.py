import numpy as np

class Loss_CategoricalCrossentropy(Loss):
    
    def backward(self,dvalues,y_true):
        # dvalues is the predicted output
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        self.dinputs = - y_true/ dvalues
        #normalize gradient
        self.dinputs = self.dinputs/samples