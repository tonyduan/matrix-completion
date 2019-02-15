### Matrix completion in Python

Last update: February 2019.

---

Python code for a few approaches at low-dimensional matrix completion. 

These methods operate in-memory and do not scale beyond size 1000 x 1000 or so. 

#### Methods

1. Nuclear norm minimization (Candes and Recht 2009)
2. Singular value thresholding (Cai, Candes, and Shen 2010)
3. Alternating least squares (Hu, Koren, and Volinsky 2008)
4. Biased alternating least squares (Paterek 2007)

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

See the `examples/` directory for more details.

#### License

This code is available under the Eclipse Public License.
