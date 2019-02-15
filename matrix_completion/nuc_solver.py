from cvxpy import *


def nuclear_norm_solve(A, mask, mu=1.0):
    """
    Solve using a nuclear norm approach, using CVXPY.
    [ Candes and Recht, 2009 ]

    Parameters:
    -----------
    A : m x n array
        matrix we want to complete

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    mu : float
        hyperparameter controlling tradeoff between nuclear norm and square loss

    Returns:
    --------
    X: m x n array
        completed matrix
    """
    X = Variable(shape=A.shape)
    objective = Minimize(mu * norm(X, "nuc") +
                         sum_squares(multiply(mask, X - A)))
    problem = Problem(objective, [])
    problem.solve(solver=SCS)
    return X.value
