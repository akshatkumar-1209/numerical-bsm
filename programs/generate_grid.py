"""
generate_grid.py -- Grid, payoff and boundary data (Derivation Parts 1-2)

    S-grid   : S_j   = j * dS,    dS   = S_max / M,  j = 0..M
    tau-grid : tau_n = n * dtau,  dtau = T / N,      n = 0..N   (tau = T - t)

    Payoff (initial condition in tau):
        call  V(S,0) = max(S - K, 0)        put  V(S,0) = max(K - S, 0)
    Dirichlet boundary values:
        call  V(0,tau) = 0,                 V(S_max,tau) = S_max - K e^{-r tau}
        put   V(0,tau) = K e^{-r tau},      V(S_max,tau) = 0

These functions are imported by every finite-difference solver in the project.
Running the file performs the grid / payoff / boundary checks of Plan Section 6.16.
"""

import argparse
from math import exp

import numpy as np

from parameters import FIG_DIR, S_MAX, figure_path


def generate_s_grid(S_max, M):
    """Uniform asset grid with M intervals: returns (S, dS), S has M+1 nodes."""
    dS = S_max / M
    S = np.linspace(0.0, S_max, M + 1)
    return S, dS


def generate_tau_grid(T, N):
    """Uniform time-to-maturity grid with N steps: returns (tau, dtau)."""
    dtau = T / N
    tau = np.linspace(0.0, T, N + 1)
    return tau, dtau


def payoff(S, K, option="call"):
    """Terminal payoff, used as the initial vector V^0 of the tau-marching."""
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError("option must be 'call' or 'put'")


def boundary_values(tau, K, r, S_max, option="call"):
    """Dirichlet data (V(0,tau), V(S_max,tau)) at time-to-maturity tau."""
    disc = K * exp(-r * tau)
    if option == "call":
        return 0.0, S_max - disc
    if option == "put":
        return disc, 0.0
    raise ValueError("option must be 'call' or 'put'")


def strike_index(K, S_max, M):
    """Grid index j_K with S_j = K, or None if the strike is not a grid node."""
    x = K * M / S_max
    j = int(round(x))
    return j if abs(x - j) < 1e-9 else None


# Backwards-compatible aliases used in earlier versions of the project
european_call_payoff = lambda S, K: payoff(S, K, "call")   # noqa: E731
european_put_payoff = lambda S, K: payoff(S, K, "put")     # noqa: E731


def main():
    p = argparse.ArgumentParser(description="Grid, payoff and boundary checks")
    p.add_argument("--K", type=float, default=100.0)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--r", type=float, default=0.05)
    p.add_argument("--Smax", type=float, default=S_MAX)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args()

    S, dS = generate_s_grid(a.Smax, a.M)
    tau, dtau = generate_tau_grid(a.T, a.N)
    print("=" * 66)
    print("  GRID, PAYOFF AND BOUNDARY-CONDITION CHECKS")
    print(f"  K = {a.K:g}, T = {a.T:g}, r = {a.r:g}, S_max = {a.Smax:g}, M = {a.M}, N = {a.N}")
    print("=" * 66)
    print(f"  S-grid  : {a.M + 1} nodes, dS = {dS:g}")
    print("           " + np.array2string(S, precision=2))
    print(f"  tau-grid: {a.N + 1} levels, dtau = {dtau:g}")
    print("           " + np.array2string(tau, precision=4))

    checks = [
        ("S[0] == 0", S[0] == 0.0),
        ("S[M] == S_max", abs(S[-1] - a.Smax) < 1e-12),
        ("uniform dS", np.allclose(np.diff(S), dS)),
        ("tau[0] == 0 (payoff level)", tau[0] == 0.0),
        ("tau[N] == T (valuation level)", abs(tau[-1] - a.T) < 1e-12),
        ("uniform dtau", np.allclose(np.diff(tau), dtau)),
        ("call payoff >= 0", np.all(payoff(S, a.K, "call") >= 0)),
        ("put payoff >= 0", np.all(payoff(S, a.K, "put") >= 0)),
        ("call payoff - put payoff == S - K", np.allclose(payoff(S, a.K, "call") - payoff(S, a.K, "put"), S - a.K)),
    ]
    # compatibility of payoff and boundary data at tau = 0
    for opt in ("call", "put"):
        lo, hi = boundary_values(0.0, a.K, a.r, a.Smax, opt)
        V0 = payoff(S, a.K, opt)
        checks.append((f"{opt}: boundary data at tau=0 equal payoff", abs(lo - V0[0]) < 1e-12 and abs(hi - V0[-1]) < 1e-12))
    # boundary values satisfy put-call parity at every time level
    ok = True
    for t in tau:
        c0, cM = boundary_values(t, a.K, a.r, a.Smax, "call")
        p0, pM = boundary_values(t, a.K, a.r, a.Smax, "put")
        ok &= abs((c0 - p0) - (0.0 - a.K * exp(-a.r * t))) < 1e-12
        ok &= abs((cM - pM) - (a.Smax - a.K * exp(-a.r * t))) < 1e-12
    checks.append(("boundary data satisfy put-call parity", ok))
    jK = strike_index(a.K, a.Smax, a.M)
    checks.append((f"strike on grid (j_K = {jK})", jK is not None))

    print("\n  Check                                          Result")
    print("  " + "-" * 60)
    for name, res in checks:
        print(f"  {name:<46} {'PASS' if res else 'FAIL'}")

    print("\n  Boundary values V(0,tau), V(S_max,tau):")
    print(f"  {'tau':>8} {'call V0':>10} {'call VM':>12} {'put V0':>12} {'put VM':>8}")
    for t in tau:
        c0, cM = boundary_values(t, a.K, a.r, a.Smax, "call")
        p0, pM = boundary_values(t, a.K, a.r, a.Smax, "put")
        print(f"  {t:>8.4f} {c0:>10.4f} {cM:>12.4f} {p0:>12.4f} {pM:>8.4f}")

    if a.no_plot:
        return
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    Sf = np.linspace(0.0, 2.0 * a.K, 401)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for x, opt, col in ((ax[0], "call", "C0"), (ax[1], "put", "C3")):
        x.plot(Sf, payoff(Sf, a.K, opt), color=col, lw=2, label=f"{opt} payoff  V(S,0)")
        x.plot(S[S <= 2 * a.K], payoff(S[S <= 2 * a.K], a.K, opt), "o", color="k", ms=4, label=f"grid nodes (dS={dS:g})")
        x.axvline(a.K, color="gray", ls=":", lw=0.8)
        x.annotate("kink at S = K", (a.K, 0), xytext=(a.K * 1.08, 0.25 * a.K), fontsize=8,
                   arrowprops=dict(arrowstyle="->", lw=0.7))
        x.set(xlabel="S", ylabel="payoff", title=f"European {opt}: max({'S - K' if opt == 'call' else 'K - S'}, 0)")
        x.grid(alpha=0.3); x.legend(fontsize=8)
    fig.tight_layout()
    out = figure_path("fig02_payoffs.png", (a.K, a.Smax, a.M) != (100.0, S_MAX, 10)); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  Figure saved: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
