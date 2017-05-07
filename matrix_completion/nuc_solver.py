from cvxpy import *


def nuclear_norm_solve(A, mask, mu):
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
    hyper-parameter controlling trade-off between nuclear norm and square loss

  Returns:
  --------
  X: m x n array
    completed matrix
  """
  X = Variable(*A.shape)
  objective = Minimize(mu * norm(X, "nuc") +
                       sum_squares(mul_elemwise(mask, X-A)))
  problem = Problem(objective, [])
  problem.solve(solver=SCS)
  return X.value
