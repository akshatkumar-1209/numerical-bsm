"""
payoff_kink_cn_oscillation.py -- Payoff kink, Crank-Nicolson oscillations and
                                  implicit (Rannacher) start-up (Derivation Part 8)

Benchmark K=100, T=1, r=0.05, sigma=0.20, S_max = 5K.  Methods compared on identical grids:
    1. Fully implicit (backward Euler)
    2. Pure Crank-Nicolson
    3. Improved Crank-Nicolson: the first CN interval replaced by two backward-Euler
       half-steps of length dtau/2 (Rannacher start-up), then ordinary CN steps.

Mode-damping parameter at the strike (Derivation Part 8):
    lambda_K = sigma^2 K^2 dtau / dS^2
    stiffest mode near K:  G_CN = (1 - lambda_K)/(1 + lambda_K)  -> -1  as lambda_K -> infinity
                           G_BE(dtau/2)^2 = 1/(1 + lambda_K)^2     ->  0

  [1] Oscillating regime  M = 800, N = 25  (lambda_K = 41): V, Delta, Gamma near K
  [2] Effect of the time step   M = 800, N = 10, 25, 50, 100, 200
  [3] Effect of spatial refinement   N = 25, M = 200, 400, 800, 1600
  [4] Start-up variant: 2 half-steps (Rannacher) vs 4 half-steps
  [5] Cost: number of tridiagonal solves and run time

Usage
-----
    python payoff_kink_cn_oscillation.py
    python payoff_kink_cn_oscillation.py --M 1600 --N 50
"""

import argparse
import os
import time

import numpy as np

from black_scholes import bs_delta, bs_delta_vec, bs_gamma, bs_gamma_vec, bs_price, bs_price_vec
from crank_nicolson import crank_nicolson_solver
from implicit_fdm import implicit_solver
from numerical_greeks import compute_greeks, oscillation_count
from parameters import FIG_DIR, add_common_args, figure_path

METHODS = (
    ("Fully implicit", lambda *a: implicit_solver(*a)),
    ("Pure CN", lambda *a: crank_nicolson_solver(*a, startup=0)),
    ("Improved CN", lambda *a: crank_nicolson_solver(*a, startup=1)),
)


def kink_metrics(S, V, K, T, r, sig, opt, delta_win):
    """Errors of V, Delta, Gamma on [K - d, K + d], Gamma(K), min Gamma and spurious Gamma extrema."""
    Si, d, g = compute_greeks(S, V)
    w = (Si >= K - delta_win) & (Si <= K + delta_win)
    eV = np.max(np.abs(V[1:-1][w] - bs_price_vec(Si[w], K, T, r, sig, opt)))
    eD = np.max(np.abs(d[w] - bs_delta_vec(Si[w], K, T, r, sig, opt)))
    eG = np.max(np.abs(g[w] - bs_gamma_vec(Si[w], K, T, r, sig)))
    gK = np.interp(K, Si, g)
    return eV, eD, eG, gK, g[w].min(), oscillation_count(Si, g, K, delta_win, r, sig, T)


def lam(K, sig, T, N, dS):
    return sig ** 2 * K ** 2 * (T / N) / dS ** 2


def main():
    p = argparse.ArgumentParser(description="Payoff kink and Crank-Nicolson oscillation study")
    add_common_args(p, M=800, N=25)
    p.add_argument("--option", default="call", choices=["call", "put"])
    p.add_argument("--delta", type=float, default=10.0, help="half-width of the strike window (default 10)")
    a = p.parse_args()
    K, T, r, sig, Smax, M, N, opt, dw = a.K, a.T, a.r, a.sigma, a.Smax, a.M, a.N, a.option, a.delta
    dS = Smax / M
    lK = lam(K, sig, T, N, dS)
    gbs = bs_gamma(K, K, T, r, sig)

    print("=" * 92)
    print("  PAYOFF KINK, CRANK-NICOLSON OSCILLATIONS AND IMPLICIT START-UP")
    print(f"  K = {K:g}, T = {T:g}, r = {r:g}, sigma = {sig:g}; S_max = {Smax:g}, option = {opt}")
    print(f"  strike window [K - {dw:g}, K + {dw:g}];  exact Gamma(K) = {gbs:.6f}")
    print("=" * 92)

    # [1] ---------------------------------------------------------------------------------------
    print(f"\n  [1] OSCILLATING REGIME  M = {M}, N = {N}  (dS = {dS:g}, dtau = {T / N:g}, lambda_K = {lK:.1f})")
    print(f"      stiff-mode factors:  G_CN = {(1 - lK) / (1 + lK):+.3f} per step,"
          f"  two BE half-steps = {1 / (1 + lK) ** 2:.2e}")
    sol = {}
    for name, f in METHODS:
        sol[name] = f(K, T, r, sig, Smax, M, N, opt)
    print("\n  Values at nodes around the strike")
    hdr = f"  {'S':>7} | {'V_BS':>9} " + " ".join(f"{n[:9]:>9}" for n, _ in METHODS)
    hdr += f" | {'D_BS':>7} " + " ".join(f"{n[:9]:>9}" for n, _ in METHODS)
    hdr += f" | {'G_BS':>7} " + " ".join(f"{n[:9]:>9}" for n, _ in METHODS)
    print(hdr)
    greeks = {n: compute_greeks(*sol[n]) for n, _ in METHODS}
    S, _ = sol["Pure CN"]
    Si = greeks["Pure CN"][0]
    step = max(1, int(round(1.25 / dS)))
    for jj in np.where((Si >= K - 5) & (Si <= K + 5))[0][::step]:
        s = Si[jj]
        line = f"  {s:>7.2f} | {bs_price(s, K, T, r, sig, opt):>9.5f} "
        line += " ".join(f"{sol[n][1][jj + 1]:>9.5f}" for n, _ in METHODS)
        line += f" | {bs_delta(s, K, T, r, sig, opt):>7.4f} " + " ".join(f"{greeks[n][1][jj]:>9.4f}" for n, _ in METHODS)
        line += f" | {bs_gamma(s, K, T, r, sig):>7.4f} " + " ".join(f"{greeks[n][2][jj]:>9.4f}" for n, _ in METHODS)
        print(line)

    print(f"\n  Error summary on [K - {dw:g}, K + {dw:g}]")
    print(f"  {'Method':<15} {'|V(K) err|':>11} {'E_V':>10} {'E_Delta':>10} {'E_Gamma':>10} {'Gamma(K)':>10}"
          f" {'min Gamma':>10} {'spur.ext':>8}")
    vbs = bs_price(K, K, T, r, sig, opt)
    for name, _ in METHODS:
        S, V = sol[name]
        eV, eD, eG, gK, gmin, osc = kink_metrics(S, V, K, T, r, sig, opt, dw)
        print(f"  {name:<15} {abs(np.interp(K, S, V) - vbs):>11.3e} {eV:>10.3e} {eD:>10.3e} {eG:>10.3e}"
              f" {gK:>10.5f} {gmin:>10.2e} {osc:>8d}")

    # [2] ---------------------------------------------------------------------------------------
    print(f"\n  [2] EFFECT OF THE TIME STEP  (M = {M} fixed, dS = {dS:g})")
    print(f"  {'N':>5} {'lambda_K':>9} {'G_CN':>7} | " + " | ".join(f"{n:^28}" for n, _ in METHODS))
    print(f"  {'':>5} {'':>9} {'':>7} | " + " | ".join(f"{'E_Gamma':>9} {'Gamma(K)':>9} {'ext':>4} {'':>3}" for _ in METHODS))
    tstudy = {n: [] for n, _ in METHODS}
    Nlist = (10, 25, 50, 100, 200)
    for Nn in Nlist:
        l_ = lam(K, sig, T, Nn, dS)
        cells = []
        for name, f in METHODS:
            S, V = f(K, T, r, sig, Smax, M, Nn, opt)
            _, _, eG, gK, _, osc = kink_metrics(S, V, K, T, r, sig, opt, dw)
            tstudy[name].append(eG)
            cells.append(f"{eG:>9.2e} {gK:>9.5f} {osc:>4d} {'':>3}")
        print(f"  {Nn:>5} {l_:>9.1f} {(1 - l_) / (1 + l_):>+7.3f} | " + " | ".join(cells))

    # [3] ---------------------------------------------------------------------------------------
    print(f"\n  [3] EFFECT OF SPATIAL REFINEMENT  (N = {N} fixed, dtau = {T / N:g})")
    print(f"  {'M':>5} {'dS':>7} {'lambda_K':>9} | " + " | ".join(f"{n:^28}" for n, _ in METHODS))
    print(f"  {'':>5} {'':>7} {'':>9} | " + " | ".join(f"{'E_Gamma':>9} {'Gamma(K)':>9} {'ext':>4} {'':>3}" for _ in METHODS))
    sstudy = {}
    Mlist = (200, 400, 800, 1600)
    for Mm in Mlist:
        dSm = Smax / Mm
        cells = []
        for name, f in METHODS:
            S, V = f(K, T, r, sig, Smax, Mm, N, opt)
            _, _, eG, gK, _, osc = kink_metrics(S, V, K, T, r, sig, opt, dw)
            sstudy[(name, Mm)] = (S, V)
            cells.append(f"{eG:>9.2e} {gK:>9.5f} {osc:>4d} {'':>3}")
        print(f"  {Mm:>5} {dSm:>7.4f} {lam(K, sig, T, N, dSm):>9.1f} | " + " | ".join(cells))

    # [4] ---------------------------------------------------------------------------------------
    print(f"\n  [4] START-UP VARIANT  (M = {M}): number of CN intervals replaced by two BE half-steps")
    print(f"  {'N':>5} {'start-up':>22} {'|V(K) err|':>11} {'E_Delta':>10} {'E_Gamma':>10} {'spur.ext':>8}")
    for Nn in (10, 25):
        for su, lab in ((1, "1 interval (2 half)"), (2, "2 intervals (4 half)")):
            S, V = crank_nicolson_solver(K, T, r, sig, Smax, M, Nn, opt, startup=su)
            eV, eD, eG, gK, _, osc = kink_metrics(S, V, K, T, r, sig, opt, dw)
            print(f"  {Nn:>5} {lab:>22} {abs(np.interp(K, S, V) - vbs):>11.3e} {eD:>10.3e} {eG:>10.3e} {osc:>8d}")

    # [5] ---------------------------------------------------------------------------------------
    print(f"\n  [5] COMPUTATIONAL COST  (M = {M}, N = {N}; median of 5 runs)")
    print(f"  {'Method':<15} {'tridiagonal solves':>19} {'time [ms]':>10}")
    for name, f in METHODS:
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); f(K, T, r, sig, Smax, M, N, opt); ts.append(time.perf_counter() - t0)
        nsol = N + 1 if name == "Improved CN" else N
        print(f"  {name:<15} {nsol:>19d} {1e3 * np.median(ts):>10.1f}")

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    sty = {"Fully implicit": ("C0", "s", "-"), "Pure CN": ("C3", "o", "-"), "Improved CN": ("C2", "^", "--")}

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    z = (Si >= K - 12) & (Si <= K + 12)
    Sz = Si[z]
    ax[0].plot(Sz, bs_price_vec(Sz, K, T, r, sig, opt), "k", lw=3, alpha=0.25, label="analytical")
    ax[1].plot(Sz, bs_delta_vec(Sz, K, T, r, sig, opt), "k", lw=3, alpha=0.25, label="analytical")
    ax[2].plot(Sz, bs_gamma_vec(Sz, K, T, r, sig), "k", lw=3, alpha=0.25, label="analytical")
    for name, _ in METHODS:
        c, m_, ls = sty[name]
        S, V = sol[name]
        _, d, g = greeks[name]
        ax[0].plot(Sz, V[1:-1][z], ls, color=c, marker=m_, ms=2.5, lw=1, label=name)
        ax[1].plot(Sz, d[z], ls, color=c, marker=m_, ms=2.5, lw=1, label=name)
        ax[2].plot(Sz, g[z], ls, color=c, marker=m_, ms=2.5, lw=1, label=name)
    for x, t_ in zip(ax, ("Price", "Delta", "Gamma")):
        x.axvline(K, color="gray", ls=":", lw=0.8); x.set(title=t_, xlabel="S"); x.grid(alpha=0.3); x.legend(fontsize=7)
    fig.suptitle(f"Near-strike behaviour, K={K:g}, T={T:g}, r={r:g}, sigma={sig:g}; S_max={Smax:g}, M={M}, N={N}"
                 f" (lambda_K={lK:.0f})", fontsize=10)
    fig.tight_layout()
    out = figure_path("fig09_kink_price_delta_gamma.png", a.custom); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  Figure saved: {os.path.relpath(out)}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for name, x in zip(("Pure CN", "Improved CN"), ax[:2]):
        for Mm in Mlist:
            S, V = sstudy[(name, Mm)]
            S2, _, g = compute_greeks(S, V)
            w = (S2 >= K - 12) & (S2 <= K + 12)
            x.plot(S2[w], g[w], lw=1, marker=".", ms=2, label=f"M={Mm} (lambda_K={lam(K, sig, T, N, Smax / Mm):.0f})")
        w = (Si >= K - 12) & (Si <= K + 12)
        x.plot(Si[w], bs_gamma_vec(Si[w], K, T, r, sig), "k", lw=3, alpha=0.25, label="analytical")
        x.axvline(K, color="gray", ls=":", lw=0.8)
        x.set(title=f"{name}: Gamma, N={N}, spatial refinement", xlabel="S"); x.grid(alpha=0.3); x.legend(fontsize=7)
    for name, _ in METHODS:
        c, m_, ls = sty[name]
        ax[2].loglog([T / n for n in Nlist], tstudy[name], ls, color=c, marker=m_, label=name)
    ax[2].set(title=f"Gamma error on [K-{dw:g},K+{dw:g}], M={M}", xlabel=r"$\Delta\tau$", ylabel=r"max $|\Gamma_{FD}-\Gamma_{BS}|$")
    ax[2].grid(alpha=0.3, which="both"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    out = figure_path("fig10_kink_refinement.png", a.custom); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  Figure saved: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
