import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import random


image = io.imread('PIC2.jpg', as_gray=True)
pixels = image.flatten()

def parameter_initialization(pixels,k):
    mean = np.random.choice(pixels,k) 
    print(mean)    
    variance = np.full(k, np.var(pixels))
    print(variance)
    weights = np.full(k,1.0/k)
    print(weights)    
    return mean,variance, weights

def gaussian_dist(x, mean,variance):
    a1 = np.exp(-((x-mean)**2)/(2*variance))
    gaus = (1.0/np.sqrt(2*np.pi*variance))*a1
    return gaus
    
def expectation(pixels, mean, variance, weights, k):
    N = len(pixels)
    E = np.zeros((N,k))
    for i in range(k):
        E[:,i] = weights[i] * gaussian_dist(pixels,mean[i],variance[i])        
    E = E/(E.sum(axis=1,keepdims=True))    
    return E

def maximization(pixels, E, k):
    N = len(pixels)
    weights = E.sum(axis=0) / N
    mean = np.sum(E * pixels[:, np.newaxis], axis=0) / E.sum(axis=0)
    variance = np.sum(E * (pixels[:, np.newaxis] - mean) ** 2, axis=0) / E.sum(axis=0)
    return mean, variance, weights

def Algorithm(pixels, K, iter=100, epsilon=1e-4):
    means, variances, weights = parameter_initialization(pixels, K)
    for i in range(iter):
        E = expectation(pixels, means, variances, weights, K)
        new_means, new_variances, new_weights = maximization(pixels, E, K)
        if np.allclose(means, new_means, atol=epsilon):
            break
        means, variances, weights = new_means, new_variances, new_weights

    return means, variances, weights, E

def segment_image(pixels, E, image_shape):
    labels = np.argmax(E, axis=1)
    segmented = labels.reshape(image_shape)
    return segmented

def view_segments(segmented_image, K):
    plt.figure(figsize=(12, 4))
    for i in range(K):
        mask = (segmented_image == i).astype(float)
        plt.subplot(1, K, i + 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Segment {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


K = 3
means, variances, weights, E = Algorithm(pixels, K)
segmented_image = segment_image(pixels, E, image.shape)
view_segments(segmented_image, K)
