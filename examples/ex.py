import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from matrix_completion import *


def plot_image(A):
    plt.imshow(A.T)
    plt.show()


if __name__ == "__main__":

    argparse = ArgumentParser()
    argparse.add_argument("--m", default=200, type=int)
    argparse.add_argument("--n", default=160, type=int)
    argparse.add_argument("--k", default=10, type=int)
    argparse.add_argument("--noise", default=0.1, type=float)
    argparse.add_argument("--mask-prob", default=0.75, type=float)

    args = argparse.parse_args()

    U, V, R = gen_factorization_with_noise(args.m, args.n, args.k, args.noise)
    mask = gen_mask(args.m, args.n, args.mask_prob)

    plot_image(R)
    plot_image(mask)

    R_hat = svt_solve(R, mask)
    print("== SVT")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat)

    R_hat = pmf_solve(R, mask, args.k, 1e-2)
    print("== PMF")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat)

    R_hat = biased_mf_solve(R, mask, args.k, 1e-2)
    print("== BMF")
    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
    plot_image(R_hat)
