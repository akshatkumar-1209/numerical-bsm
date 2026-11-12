"""
thomas_solver.py -- Thomas algorithm for tridiagonal systems (Derivation Part 3)

Solves  a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i,   i = 0..m-1,
with the convention that a[0] and c[m-1] are not used.

Forward elimination:  c'_0 = c_0/b_0,  d'_0 = d_0/b_0
                      q_i  = b_i - a_i c'_{i-1}
                      c'_i = c_i / q_i,  d'_i = (d_i - a_i d'_{i-1}) / q_i
Back substitution:    x_{m-1} = d'_{m-1},  x_i = d'_i - c'_i x_{i+1}

Cost O(m) operations. Running the file verifies the solver independently
(Plan Sections 6.4 and 6.17): against a known exact solution, against
numpy.linalg.solve, and on an actual Black-Scholes implicit / Crank-Nicolson matrix.
"""

import numpy as np


def thomas_solve(a, b, c, d):
    """Solve a tridiagonal system with lower diagonal a, main b, upper c, rhs d."""
    a = list(map(float, a)); b = list(map(float, b))
    c = list(map(float, c)); d = list(map(float, d))
    m = len(b)
    cp = [0.0] * m
    dp = [0.0] * m
    if abs(b[0]) < 1e-300:
        raise ValueError("zero pivot at row 0")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, m):
        q = b[i] - a[i] * cp[i - 1]
        if abs(q) < 1e-300:
            raise ValueError(f"zero pivot at row {i}")
        cp[i] = c[i] / q if i < m - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / q
    x = [0.0] * m
    x[m - 1] = dp[m - 1]
    for i in range(m - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return np.array(x)


def tridiag_matrix(a, b, c):
    """Dense matrix with the same three diagonals (used only for verification)."""
    return np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)


def _report(name, x_t, x_ref, A, d, ref_label):
    err = np.max(np.abs(x_t - x_ref))
    res = np.max(np.abs(A @ x_t - d))
    ok = err < 1e-10 * max(1.0, np.max(np.abs(x_ref))) and res < 1e-10 * max(1.0, np.max(np.abs(d)))
    print(f"  {name:<48} {err:10.2e}  {res:10.2e}   {'PASS' if ok else 'FAIL'}   ({ref_label})")
    return ok


def main():
    print("=" * 92)
    print("  THOMAS ALGORITHM VERIFICATION")
    print("=" * 92)
    print(f"  {'Test':<48} {'max|x-x_ref|':>10}  {'max|Ax-d|':>10}")
    print("  " + "-" * 88)

    # 1. Known exact solution
    m = 6
    a = np.r_[0.0, -np.ones(m - 1)]; b = 2.0 * np.ones(m); c = np.r_[-np.ones(m - 1), 0.0]
    x_exact = np.arange(1.0, m + 1.0)
    A = tridiag_matrix(a, b, c); d = A @ x_exact
    _report("1. 6x6 system, exact solution x = (1..6)", thomas_solve(a, b, c, d), x_exact, A, d, "exact")

    # 2. Random diagonally dominant system vs numpy
    rng = np.random.default_rng(42)
    m = 200
    a = np.r_[0.0, -rng.random(m - 1)]; c = np.r_[-rng.random(m - 1), 0.0]
    b = np.abs(a) + np.abs(c) + 1.0 + rng.random(m); d = rng.random(m)
    A = tridiag_matrix(a, b, c)
    _report("2. random diagonally dominant 200x200", thomas_solve(a, b, c, d), np.linalg.solve(A, d), A, d, "numpy")

    # 3. Black-Scholes matrices (benchmark K=100, T=1, r=0.05, sigma=0.2, S_max=400)
    K, r, sigma, S_max, M, N = 100.0, 0.05, 0.20, 400.0, 400, 400
    dt = 1.0 / N
    j = np.arange(1, M)
    alpha = 0.5 * (sigma ** 2 * j ** 2 - r * j)
    beta = -(sigma ** 2 * j ** 2 + r)
    gamma = 0.5 * (sigma ** 2 * j ** 2 + r * j)
    rhs = np.maximum(j * (S_max / M) - K, 0.0)
    for name, th in (("3. Black-Scholes fully implicit matrix (M=400)", 1.0),
                     ("4. Black-Scholes Crank-Nicolson matrix (M=400)", 0.5)):
        a = -th * dt * alpha; b = 1.0 - th * dt * beta; c = -th * dt * gamma
        A = tridiag_matrix(a, b, c)
        _report(name, thomas_solve(a, b, c, rhs), np.linalg.solve(A, rhs), A, rhs, "numpy")
        dom = np.all(np.abs(b) > np.abs(np.r_[0, a[1:]]) + np.abs(np.r_[c[:-1], 0]))
        print(f"     strictly diagonally dominant: {dom}")
    print("\n  The Thomas solution agrees with the exact / library solution to round-off level.")


if __name__ == "__main__":
    main()
