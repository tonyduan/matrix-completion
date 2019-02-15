import numpy as np
import logging


def pmf_solve(A, mask, k, mu, epsilon=1e-3, max_iterations=100):
    """
    Solve probabilistic matrix factorization using alternating least squares.

    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.

    [ Salakhutdinov and Mnih 2008 ]
    [ Hu, Koren, and Volinksy 2009 ]

    Parameters:
    -----------
    A : m x n array
        matrix to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    k : integer
        how many factors to use

    mu : float
        hyper-parameter penalizing norm of factored U, V

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

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)

    for _ in range(max_iterations):

        for i in range(m):
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([V.T, C_u[i], A[i,:]]))

        for j in range(n):
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([U.T, C_v[j], A[:,j]]))

        X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        if _ % 1 == 0:
            logger.info("Iteration: %i; Mean diff: %.4f" % (_ + 1, mean_diff))
        if mean_diff < epsilon:
            break
        prev_X = X

    return X
