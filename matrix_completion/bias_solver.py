import numpy as np
import logging


def biased_mf_solve(A, mask, k, mu, epsilon=1e-3, max_iterations=100):
    """
    Solve biased probabilistic matrix factorization via alternating least
    squares.

    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.

    [ Paterek 2007 ]
    [ Koren, Bell, and Volinksy 2009 ]

    Parameters:
    -----------
    A : m x n array
        matrix to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    k : integer
        how many factors to use

    mu : float
        hyper-parameter penalizing norm of factored U, V and biases beta, gamma

    epsilon : float
        convergence condition on the difference between iterative results

    max_iterations: int
        hard limit on maximum number of iterations

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    logger = logging.getLogger(__name__)
    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    beta = np.random.randn(m)
    gamma = np.random.randn(n)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T) + \
                     np.outer(beta, np.ones(n)) + \
                     np.outer(np.ones(m), gamma)

    for _ in range(max_iterations):

        # iteration for U
        A_tilde = A - np.outer(np.ones(m), gamma)
        V_tilde = np.c_[np.ones(n), V]
        for i in range(m):
            U_tilde = np.linalg.solve(np.linalg.multi_dot([V_tilde.T, C_u[i],
                                                           V_tilde]) +
                                      mu * np.eye(k + 1),
                                      np.linalg.multi_dot([V_tilde.T, C_u[i],
                                                           A_tilde[i,:]]))
            beta[i] = U_tilde[0]
            U[i] = U_tilde[1:]

        # iteration for V
        A_tilde = A - np.outer(beta, np.ones(n))
        U_tilde = np.c_[np.ones(m), U]
        for j in range(n):
            V_tilde = np.linalg.solve(np.linalg.multi_dot([U_tilde.T, C_v[j],
                                                           U_tilde]) +
                                                           mu * np.eye(k + 1),
                                      np.linalg.multi_dot([U_tilde.T, C_v[j],
                                                           A_tilde[:,j]]))
            gamma[j] = V_tilde[0]
            V[j] = V_tilde[1:]

        X = np.dot(U, V.T) + \
            np.outer(beta, np.ones(n)) + \
            np.outer(np.ones(m), gamma)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        if _ % 1 == 0:
            logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if mean_diff < epsilon:
            break
        prev_X = X

    return X

