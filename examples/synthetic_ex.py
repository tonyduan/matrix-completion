from matrix_completion import *


def run_experiment(m, n, k):

  U, V, R = gen_factorization_with_noise(m, n, k)


if __name__ == "__main__":
  run_experiment(50, 40, 10)
