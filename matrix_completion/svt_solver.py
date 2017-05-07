from __future__ import division
import numpy as np
import logging


def svt_solve(A, mask, tau=None, delta=None, epsilon=1e-2, max_iterations=1000):
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

  Returns:
  --------
  X: m x n array
    completed matrix
  """
  logger = logging.getLogger(__name__)
  Y = np.zeros_like(A)

  if not tau:
    tau = 5 * np.sum(A.shape) / 2
  if not delta:
    delta = 1.2 * np.prod(A.shape) / np.sum(mask)

  for _ in range(max_iterations):

    U, S, V = np.linalg.svd(Y, full_matrices=False)

    S = np.maximum(S - tau, 0)

    X = np.linalg.multi_dot([U, np.diag(S), V])
    Y += delta * mask * (A - X)

    rel_recon_error = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
    if _ % 1 == 0:
      logger.info("Iteration: %i; Rel error: %.4f" % (_ + 1, rel_recon_error))
    if rel_recon_error < epsilon:
      break

  return X
