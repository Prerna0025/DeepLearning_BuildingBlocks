import numpy as np

inputs = np.array([1,2,3,4])

weights = np.array([[0.1,0.2,0.3,0.4],
                    [0.5,0.6,0.7,0.8],
                    [0.9,1,1.1,1.2]
                    ])

biases = np.array([0.1,0.2,0.3])

learning_rate = 0.001
def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0,1,0)

for iteration in range(100):
    # forward pass
    z = np.dot(weights,inputs)+biases
    a = relu(z)
    y = np.sum(a)
    loss = y**2
    
    #backward pass
    dL_dy = 2*y
    dy_da = np.ones_like(a)
    
    dL_da = dL_dy * dy_da
    da_dz = relu_derivative(z)
    dL_dz = dL_da * da_dz
    
    dL_dw = np.outer(dL_dz,inputs)
    dL_db = dL_dz
    
    weights -= learning_rate * dL_dw
    biases -= learning_rate * dL_db
    
    if iteration % 20 == 0:
        print(f"Iteration {iteration}, Loss: {loss}")

# Final weights and biases
print("Final weights:\n", weights)
print("Final biases:\n", biases)
     