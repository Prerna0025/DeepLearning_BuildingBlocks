import numpy as np
a = [
    [1,2,3,4],
    [5,6,7,8],
    [3,6,4,5]
]

print(np.sum(a))

print(np.sum(a,axis=0))
print(np.sum(a,axis=0).shape)

print(np.sum(a,axis=1))
print(np.sum(a,axis=1).shape)

print(np.sum(a,axis=0,keepdims=True))
print(np.sum(a,axis=0,keepdims=True).shape)

print(np.sum(a,axis=1,keepdims=True))
print(np.sum(a,axis=1,keepdims=True).shape)

print(np.max(a,axis=0))
print(np.max(a,axis=1))

print(np.max(a,axis=0,keepdims=True))
print(np.max(a,axis=1,keepdims=True))
