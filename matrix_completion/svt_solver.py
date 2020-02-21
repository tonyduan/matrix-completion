from __future__ import division
import numpy as np
import logging

from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds

def _my_svd(M, k, algorithm):
    if algorithm == 'randomized':
        (U, S, V) = randomized_svd(
            M, n_components=min(k, M.shape[1]-1), n_oversamples=20)
    elif algorithm == 'arpack':
        (U, S, V) = svds(M, k=min(k, min(M.shape)-1))
        S = S[::-1]
        U, V = svd_flip(U[:, ::-1], V[::-1])
    else:
        raise ValueError("unknown algorithm")
    return (U, S, V)

def svt_solve(
        A, 
        mask, 
        tau=None, 
        delta=None, 
        epsilon=1e-2,
        rel_improvement=-0.01,
        max_iterations=1000,
        algorithm='arpack'):
    """
    Solve using iterative singular value thresholding.

    [ Cai, Candes, and Shen 2010 ]

    Parameters:
    -----------
    A : m x n array
        matrix to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    tau : float
        singular value thresholding amount;, default to 5 * (m + n) / 2

    delta : float
        step size per iteration; default to 1.2 times the undersampling ratio

    epsilon : float
        convergence condition on the relative reconstruction error

    max_iterations: int
        hard limit on maximum number of iterations

    algorithm: str, 'arpack' or 'randomized' (default='arpack')
        SVD solver to use. Either 'arpack' for the ARPACK wrapper in 
        SciPy (scipy.sparse.linalg.svds), or 'randomized' for the 
        randomized algorithm due to Halko (2009).

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    logger = logging.getLogger(__name__)
    if algorithm not in ['randomized', 'arpack']:
        raise ValueError("unknown algorithm %r" % algorithm)
    Y = np.zeros_like(A)

    if not tau:
        tau = 5 * np.sum(A.shape) / 2
    if not delta:
        delta = 1.2 * np.prod(A.shape) / np.sum(mask)

    r_previous = 0

    for k in range(max_iterations):
        if k == 0:
            X = np.zeros_like(A)
        else:
            sk = r_previous + 1
            (U, S, V) = _my_svd(Y, sk, algorithm)
            while np.min(S) >= tau:
                sk = sk + 5
                (U, S, V) = _my_svd(Y, sk, algorithm)
            shrink_S = np.maximum(S - tau, 0)
            r_previous = np.count_nonzero(shrink_S)
            diag_shrink_S = np.diag(shrink_S)
            X = np.linalg.multi_dot([U, diag_shrink_S, V])
        Y += delta * mask * (A - X)

        recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        if k % 1 == 0:
            logger.info("Iteration: %i; Rel error: %.4f" % (k + 1, recon_error))
        if recon_error < epsilon:
            break

    return X
