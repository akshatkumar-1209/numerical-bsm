"""
crank_nicolson.py -- Crank-Nicolson (theta = 1/2) solver and the improved
                     Crank-Nicolson with implicit (Rannacher) start-up
                     (Derivation Parts 5 and 8)

Crank-Nicolson step  (V^{n+1} - V^n)/dtau = (L_h V^{n+1} + L_h V^n)/2 :
    A V^{n+1} = B V^n + g^{n,n+1}
    A: a^A_j = -dtau/2 alpha_j,  b^A_j = 1 - dtau/2 beta_j,  c^A_j = -dtau/2 gamma_j
    B: a^B_j = +dtau/2 alpha_j,  b^B_j = 1 + dtau/2 beta_j,  c^B_j = +dtau/2 gamma_j
    boundary terms:
        d_1     += a^B_1 V_0^n - a^A_1 V_0^{n+1}
        d_{M-1} += c^B_{M-1} V_M^n - c^A_{M-1} V_M^{n+1}

Improved CN (Rannacher start-up): each of the first `startup` CN intervals is
replaced by two backward-Euler half-steps of length dtau/2; ordinary CN steps
follow. startup = 0 is pure Crank-Nicolson; startup = 1 is Rannacher's
original choice (two implicit half-steps).

Usage
-----
    python crank_nicolson.py                        # benchmark, S_max = 4K, M = N = 400
    python crank_nicolson.py --M 800 --N 25 --startup 1
"""

import argparse
import os

import numpy as np

from black_scholes import bs_price, bs_price_vec
from generate_grid import boundary_values, generate_s_grid, payoff, strike_index
from implicit_fdm import error_norms, implicit_step, operator_coefficients
from parameters import FIG_DIR, add_common_args, figure_path
from thomas_solver import thomas_solve, tridiag_matrix


def cn_step(V_int, alpha, beta, gamma, dt, bc_old, bc_new):
    """One Crank-Nicolson step of length dt.  bc_* = (V_0, V_M) at the old / new level."""
    h = 0.5 * dt
    aA, bA, cA = -h * alpha, 1.0 - h * beta, -h * gamma
    aB, bB, cB = h * alpha, 1.0 + h * beta, h * gamma
    d = bB * V_int
    d[1:] += aB[1:] * V_int[:-1]
    d[:-1] += cB[:-1] * V_int[1:]
    d[0] += aB[0] * bc_old[0] - aA[0] * bc_new[0]
    d[-1] += cB[-1] * bc_old[1] - cA[-1] * bc_new[1]
    return thomas_solve(aA, bA, cA, d)


def crank_nicolson_solver(K, T, r, sigma, S_max, M, N, option="call", startup=0):
    """Return grid S and CN solution V(S, tau = T); `startup` CN intervals use 2 BE half-steps."""
    S, _ = generate_s_grid(S_max, M)
    dt = T / N
    alpha, beta, gamma = operator_coefficients(M, r, sigma)
    V = payoff(S, K, option)
    tau = 0.0
    for n in range(N):
        if n < startup:                                   # two implicit half-steps
            for _ in range(2):
                tau += 0.5 * dt
                lo, hi = boundary_values(tau, K, r, S_max, option)
                V[1:M] = implicit_step(V[1:M], alpha, beta, gamma, 0.5 * dt, lo, hi)
                V[0], V[M] = lo, hi
        else:                                             # Crank-Nicolson step
            old = boundary_values(tau, K, r, S_max, option)
            tau = (n + 1) * dt
            new = boundary_values(tau, K, r, S_max, option)
            V[1:M] = cn_step(V[1:M], alpha, beta, gamma, dt, old, new)
            V[0], V[M] = new
    return S, V


def improved_crank_nicolson_solver(K, T, r, sigma, S_max, M, N, option="call", startup=1):
    """Crank-Nicolson with implicit (Rannacher) start-up."""
    return crank_nicolson_solver(K, T, r, sigma, S_max, M, N, option, startup=startup)


def main():
    p = argparse.ArgumentParser(description="Crank-Nicolson Black-Scholes FDM solver")
    add_common_args(p, M=400, N=400)
    p.add_argument("--startup", type=int, default=0,
                   help="number of CN intervals replaced by two BE half-steps (default 0 = pure CN)")
    a = p.parse_args()
    K, T, r, sig, Smax, M, N = a.K, a.T, a.r, a.sigma, a.Smax, a.M, a.N
    label = "Crank-Nicolson" + (f" + {a.startup} Rannacher start-up interval(s)" if a.startup else "")

    print("=" * 72)
    print(f"  {label.upper()} SOLVER")
    print(f"  K = {K:g}, T = {T:g}, r = {r:g}, sigma = {sig:g}")
    print(f"  S_max = {Smax:g}, M = {M}, N = {N}, dS = {Smax / M:g}, dtau = {T / N:g}")
    print("=" * 72)

    sols = {}
    for opt in ("call", "put"):
        S, V = crank_nicolson_solver(K, T, r, sig, Smax, M, N, opt, a.startup)
        sols[opt] = V
        einf, e2 = error_norms(S, V, K, T, r, sig, opt)
        jK = strike_index(a.S0, Smax, M)
        v = V[jK] if jK is not None else np.interp(a.S0, S, V)
        ex = bs_price(a.S0, K, T, r, sig, opt)
        print(f"\n  {opt.upper()}:  V_FD(S={a.S0:g}) = {v:.6f}   V_BS = {ex:.6f}   |error| = {abs(v - ex):.3e}"
              + ("" if jK is not None else "  (linear interpolation: S not a node)"))
        print(f"         ||e||_inf = {einf:.3e}    ||e||_2 = {e2:.3e}")

    print("\n  Price verification table")
    print(f"  {'S':>7} {'C_BS':>11} {'C_CN':>11} {'|err|':>10}   {'P_BS':>11} {'P_CN':>11} {'|err|':>10}")
    for s in (60, 80, 90, 100, 110, 120, 140):
        if s > Smax:
            continue
        cb, pb = bs_price(s, K, T, r, sig, "call"), bs_price(s, K, T, r, sig, "put")
        cf, pf = np.interp(s, S, sols["call"]), np.interp(s, S, sols["put"])
        print(f"  {s:>7.1f} {cb:>11.6f} {cf:>11.6f} {abs(cf - cb):>10.2e}   {pb:>11.6f} {pf:>11.6f} {abs(pf - pb):>10.2e}")

    C, P = sols["call"], sols["put"]
    disc = K * np.exp(-r * T)
    print("\n  Qualitative checks")
    print(f"    call positivity  min V_j            = {C.min(): .3e}")
    print(f"    call monotone    min (V_j+1 - V_j)  = {np.diff(C).min(): .3e}")
    print(f"    put-call parity  max|C - P - (S - K e^-rT)| = {np.max(np.abs(C - P - (S - disc))):.2e}")

    dt = T / N
    al, be, ga = operator_coefficients(M, r, sig)
    V0 = payoff(S, K, "call")
    old, new = boundary_values(0.0, K, r, Smax, "call"), boundary_values(dt, K, r, Smax, "call")
    x_t = cn_step(V0[1:M], al, be, ga, dt, old, new)
    h = 0.5 * dt
    A = tridiag_matrix(-h * al, 1 - h * be, -h * ga)
    B = tridiag_matrix(h * al, 1 + h * be, h * ga)
    d = B @ V0[1:M]
    d[0] += h * al[0] * old[0] + h * al[0] * new[0]
    d[-1] += h * ga[-1] * old[1] + h * ga[-1] * new[1]
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
            ax[0].plot(S[win], sols[opt][win], "--", color=col, lw=1.2, label=f"{opt} (Crank-Nicolson)")
            ax[1].plot(S[win], sols[opt][win] - ex, color=col, label=opt)
        ax[0].plot(S[win], payoff(S[win], K, "call"), ":", color="gray", lw=1, label="payoffs")
        ax[0].plot(S[win], payoff(S[win], K, "put"), ":", color="gray", lw=1)
        ax[0].set(xlabel="S", ylabel="V(S, t=0)", title=f"{label} vs analytical price")
        ax[1].axvline(K, color="gray", lw=0.8, ls=":")
        ax[1].set(xlabel="S", ylabel=r"$V_{FD}-V_{BS}$", title="Pointwise error")
        for x in ax:
            x.legend(fontsize=8); x.grid(alpha=0.3)
        fig.suptitle(f"K={K:g}, T={T:g}, r={r:g}, sigma={sig:g}; S_max={Smax:g}, M={M}, N={N}", fontsize=10)
        fig.tight_layout()
        out = figure_path("fig04_crank_nicolson_price_and_error.png", a.custom)
        fig.savefig(out, dpi=130); plt.close(fig)
        print(f"\n  Figure saved: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
