"""
numerical_greeks.py -- Delta and Gamma from the finite-difference price (Derivation Part 7)

Central divided differences on the computed price vector V_j at tau = T:
    Delta_j = (V_{j+1} - V_{j-1}) / (2 dS)
    Gamma_j = (V_{j+1} - 2 V_j + V_{j-1}) / dS^2          j = 1..M-1

Experiments (benchmark K=100, T=1, r=0.05, sigma=0.20, S_max = 5K):
  1. Analytical vs numerical Delta/Gamma (call and put) for fully implicit,
     pure Crank-Nicolson and improved Crank-Nicolson on the same grid.
  2. Price-Delta-Gamma error hierarchy  E_V, E_Delta, E_Gamma.
  3. Spatial mesh refinement for the Greeks, M = 50, 100, 200, 400 (fine N).
  4. Strike alignment: K on a grid node vs K midway between two nodes.

Usage
-----
    python numerical_greeks.py
    python numerical_greeks.py --M 200 --N 200
"""

import argparse
import os
from math import log2

import numpy as np

from black_scholes import bs_delta, bs_delta_vec, bs_gamma, bs_gamma_vec, bs_price, bs_price_vec
from crank_nicolson import crank_nicolson_solver
from implicit_fdm import implicit_solver
from parameters import FIG_DIR, add_common_args, figure_path

METHODS = (
    ("Fully implicit", lambda *a: implicit_solver(*a)),
    ("Crank-Nicolson", lambda *a: crank_nicolson_solver(*a, startup=0)),
    ("Improved CN", lambda *a: crank_nicolson_solver(*a, startup=1)),
)


def compute_greeks(S, V):
    """Central-difference Delta and Gamma at the interior nodes S_1..S_{M-1}."""
    dS = S[1] - S[0]
    delta = (V[2:] - V[:-2]) / (2.0 * dS)
    gamma = (V[2:] - 2.0 * V[1:-1] + V[:-2]) / dS ** 2
    return S[1:-1], delta, gamma


def _extrema(y):
    dy = np.sign(np.diff(y))
    dy = dy[dy != 0]
    return int(np.sum(dy[1:] != dy[:-1]))


def oscillation_count(S_int, gamma, K, half_width, r=None, sigma=None, T=None):
    """Spurious local extrema of the numerical Gamma in [K - w, K + w].

    Counts the local extrema of the numerical Gamma on the window and subtracts the
    number of extrema of the analytical Gamma on the same nodes (the exact Gamma has
    one smooth peak, located at S = K exp(-(r + 3 sigma^2/2) T)). Zero means that the
    numerical Gamma has exactly the shape of the analytical one; any positive value is
    a numerical oscillation."""
    from parameters import BENCH
    r = BENCH["r"] if r is None else r
    sigma = BENCH["sigma"] if sigma is None else sigma
    T = BENCH["T"] if T is None else T
    w = (S_int >= K - half_width) & (S_int <= K + half_width)
    exact = bs_gamma_vec(S_int[w], K, T, r, sigma)
    return max(0, _extrema(gamma[w]) - _extrema(exact))


def greek_errors(S, V, K, T, r, sigma, option, window=None):
    """Max-norm errors of price, Delta and Gamma over interior nodes (optionally a window)."""
    Si, d, g = compute_greeks(S, V)
    m = np.ones_like(Si, bool) if window is None else (Si >= window[0]) & (Si <= window[1])
    eV = np.max(np.abs(V[1:-1][m] - bs_price_vec(Si[m], K, T, r, sigma, option)))
    eD = np.max(np.abs(d[m] - bs_delta_vec(Si[m], K, T, r, sigma, option)))
    eG = np.max(np.abs(g[m] - bs_gamma_vec(Si[m], K, T, r, sigma)))
    return eV, eD, eG


def main():
    p = argparse.ArgumentParser(description="Numerical Delta and Gamma")
    add_common_args(p, M=400, N=400)
    a = p.parse_args()
    K, T, r, sig, Smax, M, N = a.K, a.T, a.r, a.sigma, a.Smax, a.M, a.N
    dS = Smax / M
    win = (K - 0.5 * K, K + 0.5 * K)                      # region of interest 0.5K <= S <= 1.5K

    print("=" * 84)
    print("  NUMERICAL GREEKS: CENTRAL DIVIDED DIFFERENCES ON THE FD PRICE")
    print(f"  K = {K:g}, T = {T:g}, r = {r:g}, sigma = {sig:g};  S_max = {Smax:g}, M = {M}, N = {N}, dS = {dS:g}")
    print("=" * 84)

    # ---- 1. Greeks at the evaluation point and selected nodes -------------------------------
    sol = {}
    for name, f in METHODS:
        for opt in ("call", "put"):
            sol[(name, opt)] = f(K, T, r, sig, Smax, M, N, opt)
    print("\n  [1] Greeks at S = S0 (benchmark: call Delta 0.636831, put Delta -0.363169, Gamma 0.018762)")
    print(f"  {'Method':<16} {'option':<5} {'Delta_FD':>11} {'Delta_BS':>11} {'|err|':>9}"
          f" {'Gamma_FD':>11} {'Gamma_BS':>11} {'|err|':>9}")
    for name, _ in METHODS:
        for opt in ("call", "put"):
            S, V = sol[(name, opt)]
            Si, d, g = compute_greeks(S, V)
            dfd, gfd = np.interp(a.S0, Si, d), np.interp(a.S0, Si, g)
            dbs, gbs = bs_delta(a.S0, K, T, r, sig, opt), bs_gamma(a.S0, K, T, r, sig)
            print(f"  {name:<16} {opt:<5} {dfd:>11.6f} {dbs:>11.6f} {abs(dfd - dbs):>9.2e}"
                  f" {gfd:>11.6f} {gbs:>11.6f} {abs(gfd - gbs):>9.2e}")

    S, V = sol[("Crank-Nicolson", "call")]
    Si, d, g = compute_greeks(S, V)
    print("\n  Call Greeks across the strike region (Crank-Nicolson)")
    print(f"  {'S':>7} {'Delta_BS':>10} {'Delta_FD':>10} {'err':>10}  {'Gamma_BS':>10} {'Gamma_FD':>10} {'err':>10}")
    for s in (80, 90, 95, 100, 105, 110, 120):
        db, gb = bs_delta(s, K, T, r, sig, "call"), bs_gamma(s, K, T, r, sig)
        df, gf = np.interp(s, Si, d), np.interp(s, Si, g)
        print(f"  {s:>7.1f} {db:>10.6f} {df:>10.6f} {df - db:>10.2e}  {gb:>10.6f} {gf:>10.6f} {gf - gb:>10.2e}")

    # ---- 2. Error hierarchy ------------------------------------------------------------------
    print(f"\n  [2] Price-Delta-Gamma error hierarchy (max-norm over {win[0]:g} <= S <= {win[1]:g})")
    print(f"  {'Method':<16} {'option':<5} {'E_V':>10} {'E_Delta':>10} {'E_Gamma':>10}  {'spurious extrema':>16}")
    for name, _ in METHODS:
        for opt in ("call", "put"):
            S, V = sol[(name, opt)]
            eV, eD, eG = greek_errors(S, V, K, T, r, sig, opt, win)
            Si, _, g = compute_greeks(S, V)
            print(f"  {name:<16} {opt:<5} {eV:>10.2e} {eD:>10.2e} {eG:>10.2e}  {oscillation_count(Si, g, K, 0.2 * K, r, sig, T):>16d}")

    # ---- 3. Spatial mesh refinement for the Greeks --------------------------------------------
    N_fine = 4000
    print(f"\n  [3] Spatial mesh refinement for the Greeks (call, N = {N_fine} fixed, max-norm over [{win[0]:g}, {win[1]:g}])")
    refine = {}
    for name, f in METHODS:
        print(f"  --- {name}")
        print(f"  {'M':>5} {'dS':>7} {'E_V':>10} {'p_V':>6} {'E_Delta':>10} {'p_D':>6} {'E_Gamma':>10} {'p_G':>6}")
        prev = None
        rows = []
        for Mm in (50, 100, 200, 400):
            S, V = f(K, T, r, sig, Smax, Mm, N_fine, "call")
            e = greek_errors(S, V, K, T, r, sig, "call", win)
            rows.append((Smax / Mm, *e))
            ps = ["" if prev is None else f"{log2(prev[i] / e[i]):6.2f}" for i in range(3)]
            print(f"  {Mm:>5} {Smax / Mm:>7.3f} {e[0]:>10.3e} {ps[0]:>6} {e[1]:>10.3e} {ps[1]:>6} {e[2]:>10.3e} {ps[2]:>6}")
            prev = e
        refine[name] = np.array(rows)

    # ---- 4. Strike alignment ------------------------------------------------------------------
    print("\n  [4] Strike alignment (Crank-Nicolson, call, N = 400): K on a node vs K midway between nodes")
    print("      aligned: dS = K/j_K;  off-node: dS = K/(j_K - 1/2), M kept, S_max = M dS (~5K)")
    print(f"  {'grid':<10} {'j_K':>6} {'dS':>7} {'S_max':>8} {'V(K) err':>10} {'Delta(K) err':>13}"
          f" {'Gamma(K) err':>13} {'E_V window':>11}")
    for jK in (10, 20, 40, 80):
        Mm = 5 * jK
        for label, dSa in (("aligned", K / jK), ("off-node", K / (jK - 0.5))):
            Sm = Mm * dSa
            S, V = crank_nicolson_solver(K, T, r, sig, Sm, Mm, 400, "call")
            Si, d, g = compute_greeks(S, V)
            ev = abs(np.interp(K, S, V) - bs_price(K, K, T, r, sig, "call"))
            ed = abs(np.interp(K, Si, d) - bs_delta(K, K, T, r, sig, "call"))
            eg = abs(np.interp(K, Si, g) - bs_gamma(K, K, T, r, sig))
            ew = greek_errors(S, V, K, T, r, sig, "call", win)[0]
            jk_txt = f"{jK}" if label == "aligned" else f"{jK - 0.5:g}"
            print(f"  {label:<10} {jk_txt:>6} {dSa:>7.4f} {Sm:>8.2f} {ev:>10.2e} {ed:>13.2e} {eg:>13.2e} {ew:>11.2e}")
    print("  (on the off-node grid the values at S = K are linearly interpolated between the two neighbouring nodes)")

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    tag = f"K={K:g}, T={T:g}, r={r:g}, sigma={sig:g}; S_max={Smax:g}, M={M}, N={N}"

    # Figure 5: analytical vs numerical Delta and Gamma (full range + errors)
    S, V = sol[("Crank-Nicolson", "call")]
    Si, d, g = compute_greeks(S, V)
    Sp, Vp = sol[("Crank-Nicolson", "put")]
    _, dp, _ = compute_greeks(Sp, Vp)
    m = Si <= 2.5 * K
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(Si[m], bs_delta_vec(Si[m], K, T, r, sig, "call"), lw=3, alpha=0.35, color="C0", label="call, analytical")
    ax[0, 0].plot(Si[m], d[m], "--", color="C0", lw=1.2, label="call, CN")
    ax[0, 0].plot(Si[m], bs_delta_vec(Si[m], K, T, r, sig, "put"), lw=3, alpha=0.35, color="C3", label="put, analytical")
    ax[0, 0].plot(Si[m], dp[m], "--", color="C3", lw=1.2, label="put, CN")
    ax[0, 0].set(title="Delta", xlabel="S")
    ax[0, 1].plot(Si[m], d[m] - bs_delta_vec(Si[m], K, T, r, sig, "call"), color="C0")
    ax[0, 1].set(title="Delta error (call = put)", xlabel="S", ylabel=r"$\Delta_{FD}-\Delta_{BS}$")
    ax[1, 0].plot(Si[m], bs_gamma_vec(Si[m], K, T, r, sig), lw=3, alpha=0.35, color="C2", label="analytical")
    ax[1, 0].plot(Si[m], g[m], "--", color="k", lw=1.0, label="CN")
    ax[1, 0].set(title="Gamma (calls and puts)", xlabel="S")
    z = (Si >= K - 15) & (Si <= K + 15)
    ax[1, 1].plot(Si[z], bs_gamma_vec(Si[z], K, T, r, sig), lw=3, alpha=0.35, color="C2", label="analytical")
    for name, mk in (("Fully implicit", "s"), ("Crank-Nicolson", "o"), ("Improved CN", "^")):
        S2, V2 = sol[(name, "call")]
        Si2, _, g2 = compute_greeks(S2, V2)
        ax[1, 1].plot(Si2[z], g2[z], mk, ms=3, label=name)
    ax[1, 1].set(title="Gamma near the strike (magnified)", xlabel="S")
    for x in ax.flat:
        x.axvline(K, color="gray", ls=":", lw=0.8); x.grid(alpha=0.3)
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=7)
    fig.suptitle(tag, fontsize=10); fig.tight_layout()
    out = figure_path("fig07_numerical_greeks.png", a.custom); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  Figure saved: {os.path.relpath(out)}")

    # Figure 6: Greek error vs dS (log-log)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for k, ttl in enumerate(("price", "Delta", "Gamma")):
        for name, mk in (("Fully implicit", "s-"), ("Crank-Nicolson", "o-"), ("Improved CN", "^--")):
            R = refine[name]
            ax[k].loglog(R[:, 0], R[:, k + 1], mk, ms=4, label=name)
        h = refine["Crank-Nicolson"][:, 0]
        ax[k].loglog(h, refine["Crank-Nicolson"][0, k + 1] * (h / h[0]) ** 2, "k:", label="slope 2")
        ax[k].set(title=f"{ttl} error, max over [{win[0]:g},{win[1]:g}]", xlabel="dS")
        ax[k].grid(alpha=0.3, which="both"); ax[k].legend(fontsize=7)
    fig.suptitle(f"Spatial refinement of Greeks, M = 50..400, N = {N_fine}; S_max={Smax:g}", fontsize=10)
    fig.tight_layout()
    out = figure_path("fig08_greek_convergence.png", a.custom); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  Figure saved: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
