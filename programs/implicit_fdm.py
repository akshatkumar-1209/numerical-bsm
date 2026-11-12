"""
implicit_fdm.py -- Fully Implicit (Backward Euler, theta = 1) solver (Derivation Part 4)

Discrete operator at interior node j (S_j = j dS):
    L_h V_j = alpha_j V_{j-1} + beta_j V_j + gamma_j V_{j+1}
    alpha_j = (sigma^2 j^2 - r j)/2,  beta_j = -(sigma^2 j^2 + r),  gamma_j = (sigma^2 j^2 + r j)/2

Fully implicit step  (V^{n+1} - V^n)/dtau = L_h V^{n+1}  gives the tridiagonal system
    a_j V_{j-1}^{n+1} + b_j V_j^{n+1} + c_j V_{j+1}^{n+1} = V_j^n
    a_j = -dtau alpha_j,  b_j = 1 - dtau beta_j,  c_j = -dtau gamma_j
with the known boundary values moved to the right-hand side:
    d_1     = V_1^n     - a_1     V_0^{n+1}
    d_{M-1} = V_{M-1}^n - c_{M-1} V_M^{n+1}

Usage
-----
    python implicit_fdm.py                         # benchmark, S_max = 4K, M = N = 400
    python implicit_fdm.py --M 200 --N 1000 --option put --sigma 0.3
"""

import argparse
import os

import numpy as np

from black_scholes import bs_price, bs_price_vec
from generate_grid import boundary_values, generate_s_grid, payoff, strike_index
from parameters import FIG_DIR, add_common_args, figure_path
from thomas_solver import thomas_solve, tridiag_matrix


def operator_coefficients(M, r, sigma):
    """alpha_j, beta_j, gamma_j for interior nodes j = 1..M-1 (S_j = j dS)."""
    j = np.arange(1, M, dtype=float)
    alpha = 0.5 * (sigma ** 2 * j ** 2 - r * j)
    beta = -(sigma ** 2 * j ** 2 + r)
    gamma = 0.5 * (sigma ** 2 * j ** 2 + r * j)
    return alpha, beta, gamma


def implicit_step(V_int, alpha, beta, gamma, dt, V0_new, VM_new):
    """One backward-Euler step of length dt on the interior values V_int."""
    a, b, c = -dt * alpha, 1.0 - dt * beta, -dt * gamma
    d = V_int.copy()
    d[0] -= a[0] * V0_new           # lower boundary contribution
    d[-1] -= c[-1] * VM_new         # upper boundary contribution
    return thomas_solve(a, b, c, d)


def implicit_solver(K, T, r, sigma, S_max, M, N, option="call"):
    """Return the grid S and the fully implicit solution V(S, tau = T)."""
    S, _ = generate_s_grid(S_max, M)
    dt = T / N
    alpha, beta, gamma = operator_coefficients(M, r, sigma)
    V = payoff(S, K, option)
    for n in range(N):
        V0, VM = boundary_values((n + 1) * dt, K, r, S_max, option)
        V[1:M] = implicit_step(V[1:M], alpha, beta, gamma, dt, V0, VM)
        V[0], V[M] = V0, VM
    return S, V


def error_norms(S, V, K, T, r, sigma, option="call"):
    """Interior max-norm and discrete L2 norm  (dS sum e_j^2)^{1/2}  against Black-Scholes."""
    e = V[1:-1] - bs_price_vec(S[1:-1], K, T, r, sigma, option)
    dS = S[1] - S[0]
    return np.max(np.abs(e)), np.sqrt(dS * np.sum(e ** 2))


def main():
    p = argparse.ArgumentParser(description="Fully implicit Black-Scholes FDM solver")
    add_common_args(p, M=400, N=400)
    a = p.parse_args()
    K, T, r, sig, Smax, M, N = a.K, a.T, a.r, a.sigma, a.Smax, a.M, a.N

    print("=" * 72)
    print("  FULLY IMPLICIT (BACKWARD EULER) FINITE-DIFFERENCE SOLVER")
    print(f"  K = {K:g}, T = {T:g}, r = {r:g}, sigma = {sig:g}")
    print(f"  S_max = {Smax:g}, M = {M}, N = {N}, dS = {Smax / M:g}, dtau = {T / N:g}")
    print("=" * 72)

    sols = {}
    for opt in ("call", "put"):
        S, V = implicit_solver(K, T, r, sig, Smax, M, N, opt)
        sols[opt] = V
        einf, e2 = error_norms(S, V, K, T, r, sig, opt)
        jK = strike_index(a.S0, Smax, M)
        v = V[jK] if jK is not None else np.interp(a.S0, S, V)
        ex = bs_price(a.S0, K, T, r, sig, opt)
        print(f"\n  {opt.upper()}:  V_FD(S={a.S0:g}) = {v:.6f}   V_BS = {ex:.6f}   |error| = {abs(v - ex):.3e}"
              + ("" if jK is not None else "  (linear interpolation: S not a node)"))
        print(f"         ||e||_inf = {einf:.3e}    ||e||_2 = {e2:.3e}")

    print("\n  Price verification table")
    print(f"  {'S':>7} {'C_BS':>11} {'C_Imp':>11} {'|err|':>10}   {'P_BS':>11} {'P_Imp':>11} {'|err|':>10}")
    for s in (60, 80, 90, 100, 110, 120, 140):
        if s > Smax:
            continue
        cb, pb = bs_price(s, K, T, r, sig, "call"), bs_price(s, K, T, r, sig, "put")
        cf, pf = np.interp(s, S, sols["call"]), np.interp(s, S, sols["put"])
        print(f"  {s:>7.1f} {cb:>11.6f} {cf:>11.6f} {abs(cf - cb):>10.2e}   {pb:>11.6f} {pf:>11.6f} {abs(pf - pb):>10.2e}")

    # Qualitative checks (Plan Section 6.16)
    C, P = sols["call"], sols["put"]
    tau_disc = K * np.exp(-r * T)
    print("\n  Qualitative checks")
    print(f"    call positivity  min V_j            = {C.min(): .3e}")
    print(f"    call monotone    min (V_j+1 - V_j)  = {np.diff(C).min(): .3e}")
    lb = np.maximum(S - tau_disc, 0.0)
    print(f"    call bounds      max(lb - V, V - S) = {max(np.max(lb - C), np.max(C - S)): .3e}  (<= 0 up to the O(dtau) discounting error)")
    print(f"    put-call parity  max|C - P - (S - K e^-rT)| = {np.max(np.abs(C - P - (S - tau_disc))):.2e}"
          "  (backward Euler discounts K by (1 + r dtau)^-N, not e^-rT)")

    # One time step solved by Thomas and by numpy.linalg.solve (Plan Section 6.17)
    dt = T / N
    al, be, ga = operator_coefficients(M, r, sig)
    V0 = payoff(S, K, "call")
    lo, hi = boundary_values(dt, K, r, Smax, "call")
    x_t = implicit_step(V0[1:M], al, be, ga, dt, lo, hi)
    A = tridiag_matrix(-dt * al, 1 - dt * be, -dt * ga)
    d = V0[1:M].copy(); d[0] -= -dt * al[0] * lo; d[-1] -= -dt * ga[-1] * hi
    print(f"    first step: max|V_Thomas - V_numpy| = {np.max(np.abs(x_t - np.linalg.solve(A, d))):.2e}")

    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(FIG_DIR, exist_ok=True)
        win = (S >= 0.4 * K) & (S <= 1.8 * K)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for opt, col in (("call", "C0"), ("put", "C3")):
            ex = bs_price_vec(S[win], K, T, r, sig, opt)
            ax[0].plot(S[win], ex, color=col, lw=2.5, alpha=0.35, label=f"{opt} (Black-Scholes)")
            ax[0].plot(S[win], sols[opt][win], "--", color=col, lw=1.2, label=f"{opt} (fully implicit)")
            ax[1].plot(S[win], sols[opt][win] - ex, color=col, label=opt)
        ax[0].plot(S[win], payoff(S[win], K, "call"), ":", color="gray", lw=1, label="payoffs")
        ax[0].plot(S[win], payoff(S[win], K, "put"), ":", color="gray", lw=1)
        ax[0].set(xlabel="S", ylabel="V(S, t=0)", title="Fully implicit vs analytical price")
        ax[1].axvline(K, color="gray", lw=0.8, ls=":")
        ax[1].set(xlabel="S", ylabel=r"$V_{FD}-V_{BS}$", title="Pointwise error")
        for x in ax:
            x.legend(fontsize=8); x.grid(alpha=0.3)
        fig.suptitle(f"K={K:g}, T={T:g}, r={r:g}, sigma={sig:g}; S_max={Smax:g}, M={M}, N={N}", fontsize=10)
        fig.tight_layout()
        out = figure_path("fig03_implicit_price_and_error.png", a.custom)
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"\n  Figure saved: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
