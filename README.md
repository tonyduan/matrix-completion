### Matrix completion in Python

Last update: February 2019.

---

Python code for a few approaches at low-dimensional matrix completion. 

These methods operate in-memory and do not scale beyond size 1000 x 1000 or so. 

#### Methods

1. Nuclear norm minimization (very slow) [1]
2. Singular value thresholding [2]
3. Alternating least squares [3,4]
4. Biased alternating least squares [5]

#### Usage

```python
import numpy as np
from matrix_completion import svt_solve, calc_unobserved_rmse

U = np.random.randn(20, 5)
V = np.random.randn(15, 5)
R = np.random.randn(20, 15) + np.dot(U, V.T)

mask = np.round(np.random.rand(20, 15))
R_hat = svt_solve(R, mask)

print("RMSE:", calc_unobserved_rmse(U, V, R_hat, mask))
```

Note that here, the mask is a matrix with entries either 1 (indicating observed) or 0 (indicating missing).

See the `examples/` directory for more details.

#### References

[1] Emmanuel Candès and Benjamin Recht. 2012. Exact matrix completion via convex optimization. Commun. ACM 55, 6 (June 2012), 111-119. DOI: https://doi.org/10.1145/2184319.2184343

[2] Jian-Feng Cai, Emmanuel J. Candès, and Zuowei Shen. 2010. A Singular Value Thresholding Algorithm for Matrix Completion. SIAM J. on Optimization 20, 4 (March 2010), 1956-1982. DOI=http://dx.doi.org/10.1137/080738970

[3] Yifan Hu, Yehuda Koren, and Chris Volinsky. 2008. Collaborative Filtering for Implicit Feedback Datasets. In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM '08). IEEE Computer Society, Washington, DC, USA, 263-272. DOI=http://dx.doi.org/10.1109/ICDM.2008.22

[4] Ruslan Salakhutdinov and Andriy Mnih. 2007. Probabilistic Matrix Factorization. In Proceedings of the 20th International Conference on Neural Information Processing Systems (NIPS'07), J. C. Platt, D. Koller, Y. Singer, and S. T. Roweis (Eds.). Curran Associates Inc., USA, 1257-1264.

[5] Paterek, Arkadiusz. “Improving regularized singular value decomposition for collaborative filtering.” (2007).

#### License

This code is available under the Eclipse Public License.
