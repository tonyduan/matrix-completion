import numpy as np
from scipy.stats import bernoulli


def gen_mask(m, n, prob_masked=0.5):
    """
    Generate a binary mask for m users and n movies.
    Note that 1 denotes observed, and 0 denotes unobserved.
    """
    return 1 - bernoulli.rvs(p=prob_masked, size=(m, n))


def gen_factorization_without_noise(m, n, k):
    """
    Generate non-noisy data for m users and n movies with k latent factors.
    Draws factors U, V from Gaussian noise and returns U Vᵀ.
    """
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    R = np.dot(U, V.T)
    return U, V, R


def gen_factorization_with_noise(m, n, k, sigma):
    """
    Generate noisy data for m users and n movies with k latent factors.
    Gaussian noise with variance sigma^2 is added to U V^T.
    Effect is a matrix with a few large singular values and many close to zero.
    """
    U = np.random.randn(m, k)
    V = np.random.randn(n, k)
    R = np.random.randn(m, n) * sigma + np.dot(U, V.T)
    return U, V, R
