
    Matrix completion library for Python

Last update: May 2017.

Methods:

    1. Nuclear norm minimization (Candes and Recht 2009)
    2. Singular value thresholding (Cai, Candes, and Shen 2010)
    3. Alternating least squares (Hu, Koren, and Volinsky 2008)
    4. Biased alternating least squares (Paterek 2007)

Usage:

    import numpy as np
    from matrix_completion import nuclear_norm_solve, calc_unobserved_rmse

    U = np.random.randn(20, 5)
    V = np.random.randn(15, 5)
    R = np.random.randn(20, 15) + np.dot(U, V.T)

    mask = np.round(np.random.rand(20, 15))
    R_hat = nuclear_norm_solve(A, mask, mu=1.0)

    print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))

License:

    This library is available under the Eclipse Public License.
