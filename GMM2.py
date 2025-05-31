import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.preprocessing import StandardScaler

# Load RGB image and reshape
image = io.imread('PIC2.jpg')  # Shape: (H, W, 3)
pixels = image.reshape(-1, 3).astype(np.float64)

def parameter_initialization(pixels, k):
    indices = np.random.choice(len(pixels), k, replace=False)
    means = pixels[indices]  # (k, 3)
    covariances = [np.cov(pixels, rowvar=False) for _ in range(k)]  # one covariance matrix per cluster
    weights = np.full(k, 1.0 / k)
    return means, covariances, weights

def multivariate_gaussian(x, mean, cov):
    D = x.shape[1]
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    norm_const = 1.0 / (np.power((2 * np.pi), D / 2) * np.sqrt(det))
    diff = x - mean
    exp_term = np.einsum('ij,jk,ik->i', diff, inv, diff)
    return norm_const * np.exp(-0.5 * exp_term)

def expectation(pixels, means, covariances, weights, k):
    N = len(pixels)
    E = np.zeros((N, k))
    for i in range(k):
        E[:, i] = weights[i] * multivariate_gaussian(pixels, means[i], covariances[i])
    E = E / E.sum(axis=1, keepdims=True)
    return E

def maximization(pixels, E, k):
    N, D = pixels.shape
    weights = E.sum(axis=0) / N
    means = np.dot(E.T, pixels) / E.sum(axis=0)[:, np.newaxis]
    covariances = []

    for i in range(k):
        diff = pixels - means[i]
        weighted_diff = diff.T * E[:, i]
        cov = np.dot(weighted_diff, diff) / E[:, i].sum()
        cov += np.eye(D) * 1e-6  # for numerical stability
        covariances.append(cov)

    return means, covariances, weights

def Algorithm(pixels, K, iter=100, epsilon=1e-4):
    means, covariances, weights = parameter_initialization(pixels, K)

    for i in range(iter):
        E = expectation(pixels, means, covariances, weights, K)
        new_means, new_covariances, new_weights = maximization(pixels, E, K)

        if np.allclose(means, new_means, atol=epsilon):
            break

        means, covariances, weights = new_means, new_covariances, new_weights

    return means, covariances, weights, E

def segment_image(E, image_shape):
    labels = np.argmax(E, axis=1)
    segmented = labels.reshape(image_shape[:2])
    return segmented

def display_segments(segmented_image, K):
    plt.figure(figsize=(12, 4))
    for i in range(K):
        mask = (segmented_image == i).astype(float)
        plt.subplot(1, K, i + 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Segment {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Run
K = 3
means, covariances, weights, E = Algorithm(pixels, K)
segmented_image = segment_image(E, image.shape)
display_segments(segmented_image, K)

# Optional: Visualize colored segmentation
def color_segments(segmented_image, K):
    from matplotlib import cm
    cmap = cm.get_cmap('tab10', K)
    colored = cmap(segmented_image / (K - 1))
    plt.imshow(colored)
    plt.title(f'{K} Color Segments')
    plt.axis('off')
    plt.show()

color_segments(segmented_image, K)
