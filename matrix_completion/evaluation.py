import numpy as np


def calc_unobserved_rmse(U, V, A_hat, mask):
    """
    Calculate RMSE on all unobserved entries in mask, for true matrix UVᵀ.

    Parameters
    ----------
    U : m x k array
        true factor of matrix

    V : n x k array
        true factor of matrix

    A_hat : m x n array
        estimated matrix

    mask : m x n array
        matrix with entries zero (if missing) or one (if present)

    Returns:
    --------
    rmse : float
        root mean squared error over all unobserved entries
    """
    pred = np.multiply(A_hat, (1 - mask))
    truth = np.multiply(np.dot(U, V.T), (1 - mask))
    cnt = np.sum(1 - mask)
    return (np.linalg.norm(pred - truth, "fro") ** 2 / cnt) ** 0.5


def calc_validation_rmse(validation_data, A_hat):
    """
    Calculate validation RMSE on all validation entries.

    Parameters
    ----------
    validation_data : list
        list of tuples (i, j, r) where (i, j) are indices of matrix with entry r

    A_hat : m x n array
        estimated matrix
    """
    total_error = 0.0
    for (u, i, r) in validation_data:
        total_error += (r - A_hat[int(u),int(i)]) ** 2
    return np.sqrt(total_error / len(validation_data))
