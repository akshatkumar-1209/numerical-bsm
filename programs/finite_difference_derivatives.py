"""
finite_difference_derivatives.py -- Taylor-series finite differences (Derivation Part 1)

    forward   D+ f(x)    = [f(x+h) - f(x)] / h               error  O(h)
    backward  D- f(x)    = [f(x) - f(x-h)] / h               error  O(h)
    central   D0 f(x)    = [f(x+h) - f(x-h)] / (2h)          error  O(h^2)
    second    D+D- f(x)  = [f(x+h) - 2f(x) + f(x-h)] / h^2   error  O(h^2)

The central formulas are exactly the stencils used for V_S and V_SS in the
Black-Scholes operator (Parts 2-5) and for Delta and Gamma (Part 7).

Test function f(x) = sin(x) exp(-x/5) at x0 = 1 (smooth, with known derivatives),
followed by a check on the Black-Scholes call price itself at S = K.
"""

import argparse
import os
from math import log2

import numpy as np

from black_scholes import bs_delta, bs_gamma, bs_price
from parameters import BENCH, FIG_DIR, figure_path


def f(x):
    return np.sin(x) * np.exp(-x / 5.0)


def f_prime_exact(x):
    return np.exp(-x / 5.0) * (np.cos(x) - np.sin(x) / 5.0)


def f_double_prime_exact(x):
    return np.exp(-x / 5.0) * (-np.sin(x) - 0.4 * np.cos(x) + np.sin(x) / 25.0)


def forward_diff(g, x, h):
    return (g(x + h) - g(x)) / h


def backward_diff(g, x, h):
    return (g(x) - g(x - h)) / h


def central_diff_first(g, x, h):
    return (g(x + h) - g(x - h)) / (2.0 * h)


def central_diff_second(g, x, h):
    return (g(x + h) - 2.0 * g(x) + g(x - h)) / h ** 2


def main():
    p = argparse.ArgumentParser(description="Finite-difference truncation-error study")
    p.add_argument("--x0", type=float, default=1.0)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()
    x0 = a.x0
    d1, d2 = f_prime_exact(x0), f_double_prime_exact(x0)
    hs = [0.2 / 2 ** k for k in range(7)]

    print("=" * 84)
    print(f"  FINITE-DIFFERENCE APPROXIMATIONS:  f(x) = sin(x) exp(-x/5),  x0 = {x0:g}")
    print(f"  exact f'(x0) = {d1:.12f}    exact f''(x0) = {d2:.12f}")
    print("=" * 84)
    print(f"  {'h':>9} {'forward':>11} {'p':>5} {'backward':>11} {'p':>5} {'central':>11} {'p':>5}"
          f" {'2nd deriv.':>11} {'p':>5}")
    rows, prev = [], None
    for h in hs:
        e = (abs(forward_diff(f, x0, h) - d1), abs(backward_diff(f, x0, h) - d1),
             abs(central_diff_first(f, x0, h) - d1), abs(central_diff_second(f, x0, h) - d2))
        ps = ["" if prev is None else f"{log2(prev[i] / e[i]):5.2f}" for i in range(4)]
        print(f"  {h:>9.5f} {e[0]:>11.3e} {ps[0]:>5} {e[1]:>11.3e} {ps[1]:>5} {e[2]:>11.3e} {ps[2]:>5}"
              f" {e[3]:>11.3e} {ps[3]:>5}")
        rows.append((h, *e)); prev = e
    print("  Observed orders p = log2(E_h / E_h/2):  one-sided -> 1,  central and second difference -> 2.")

    # Same stencils applied to the analytical Black-Scholes call price at S = K
    K, T, r, sig = BENCH["K"], BENCH["T"], BENCH["r"], BENCH["sigma"]
    C = lambda s: bs_price(s, K, T, r, sig, "call")    # noqa: E731
    dex, gex = bs_delta(K, K, T, r, sig, "call"), bs_gamma(K, K, T, r, sig)
    print(f"\n  Stencils applied to the Black-Scholes call C(S) at S = K = {K:g}"
          f"   (Delta = {dex:.6f}, Gamma = {gex:.6f})")
    print(f"  {'dS':>7} {'D0 C - Delta':>13} {'p':>5} {'D+D- C - Gamma':>15} {'p':>5}")
    prev = None
    for dS in (8.0, 4.0, 2.0, 1.0, 0.5):
        e = (abs((C(K + dS) - C(K - dS)) / (2 * dS) - dex), abs((C(K + dS) - 2 * C(K) + C(K - dS)) / dS ** 2 - gex))
        ps = ["" if prev is None else f"{log2(prev[i] / e[i]):5.2f}" for i in range(2)]
        print(f"  {dS:>7.2f} {e[0]:>13.3e} {ps[0]:>5} {e[1]:>15.3e} {ps[1]:>5}")
        prev = e
    print("  These are the stencil errors alone (exact prices as input); Part 7 adds the PDE error.")

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    R = np.array(rows)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for k, (lab, m) in enumerate((("forward  $D_+$", "s-"), ("backward $D_-$", "v-"),
                                  ("central $D_0$", "o-"), ("second $D_+D_-$", "^-")), start=1):
        ax.loglog(R[:, 0], R[:, k], m, ms=4, label=lab)
    ax.loglog(R[:, 0], R[0, 1] * R[:, 0] / R[0, 0] * 1.5, "k-.", lw=0.8, label="slope 1")
    ax.loglog(R[:, 0], R[0, 3] * (R[:, 0] / R[0, 0]) ** 2 * 0.5, "k:", label="slope 2")
    ax.set(xlabel="h", ylabel="|error|", title=f"Truncation error, f = sin(x) exp(-x/5), x0 = {x0:g}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout()
    out = figure_path("fig01_fd_truncation_error.png", a.x0 != 1.0); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  Figure saved: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
