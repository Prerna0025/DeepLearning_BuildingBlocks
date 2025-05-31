import numpy as np

# Gradient of loss w.r.t weights

dvalues = np.array([[1.,1.,1.],
                    [2.,2.,2.],
                    [3.,3.,3.]])

inputs = np.array([[1,2,3,2.5],
                   [2.,5.,-1.,2],
                   [-1.5,2.7,3.3,-0.8]])

dweights = np.dot(inputs.T,dvalues)
print(f'dweights: {dweights}')

# Gradients of loss w.r.t biases
biases = np.array([[2,3,0.5]])

dbiases = np.sum(dvalues,axis=0,keepdims=True)
print(f'dbiases: {dbiases}')

# Gradient of loss w.r.t inputs
weights = np.array([[0.2,0.8,-0.5,1],
                    [0.5,-0.91,0.26,-0.5],
                    [-0.26,-0.27,0.17,0.87]])

dinputs = np.dot(dvalues,weights)
print(f'dinputs: {dinputs}')
