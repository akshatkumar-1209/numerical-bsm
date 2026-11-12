"""
convergence_study.py -- Verification, convergence and domain truncation (Derivation Part 6)

Benchmark K=100, T=1, r=0.05, sigma=0.20, evaluated at S = K; S_max = 5K.

  [1] Price verification table   S, V_BS, V_Imp, V_CN               (M = N = 400)
  [2] Independent SPATIAL refinement   M = 50,100,200,400   with N = 20000 fixed
        error against the analytical Black-Scholes solution
  [3] Independent TEMPORAL refinement  N = 50,100,200,400   with M = 400 fixed
        (a) error against a same-grid reference V(M, N_ref = 12800)  -> isolates time error
        (b) error against the analytical solution                    -> shows the spatial floor
  [4] Paired refinement  (M,N) = (50,50),(100,100),(200,200),(400,400)
  [5] Domain truncation  S_max = 2K, 3K, 4K, 5K  at fixed dS = 1.25, N = 400
  [6] Final verification table  (method, ||e||_inf, ||e||_2, orders, oscillation)

Error norms (interior nodes j = 1..M-1):
    ||e||_inf = max_j |e_j|,     ||e||_2 = ( dS sum_j e_j^2 )^{1/2}
Observed order:  p = log(E_h / E_{h/2}) / log 2

Usage
-----
    python convergence_study.py            (about 1 minute)
"""

import argparse
import os
import time
from math import log2

import numpy as np

from black_scholes import bs_price, bs_price_vec
from crank_nicolson import crank_nicolson_solver
from generate_grid import strike_index
from implicit_fdm import error_norms, implicit_solver
from numerical_greeks import compute_greeks, oscillation_count
from parameters import BENCH, FIG_DIR, S_MAX_FACTOR, figure_path

SOLVERS = (("Fully implicit", implicit_solver, 1),
           ("Crank-Nicolson", lambda *a: crank_nicolson_solver(*a, startup=0), 2))


def order(e_prev, e):
    return log2(e_prev / e) if e_prev is not None and e > 0 else float("nan")


def fmt_p(p):
    return "   -  " if np.isnan(p) else f"{p:6.2f}"


def main():
    pa = argparse.ArgumentParser(description="Convergence and domain-truncation study")
    pa.add_argument("--no-plot", action="store_true")
    pa.add_argument("--option", default="call", choices=["call", "put"])
    a = pa.parse_args()
    K, T, r, sig, S0 = BENCH["K"], BENCH["T"], BENCH["r"], BENCH["sigma"], BENCH["S0"]
    Smax, opt = S_MAX_FACTOR * K, a.option
    exact0 = bs_price(S0, K, T, r, sig, opt)

    print("=" * 86)
    print("  VERIFICATION AND CONVERGENCE STUDY")
    print(f"  K = {K:g}, T = {T:g}, r = {r:g}, sigma = {sig:g}, S_max = {Smax:g}, option = {opt}")
    print(f"  Analytical value at S = {S0:g}:  {exact0:.6f}")
    print("=" * 86)

    # [1] ---------------------------------------------------------------------------------------
    M1 = N1 = 400
    S, Vi = implicit_solver(K, T, r, sig, Smax, M1, N1, opt)
    _, Vc = crank_nicolson_solver(K, T, r, sig, Smax, M1, N1, opt)
    print(f"\n  [1] PRICE VERIFICATION  (M = {M1}, N = {N1}, dS = {Smax / M1:g}, dtau = {T / N1:g})")
    print(f"  {'S':>7} {'V_BS':>11} {'V_Imp':>11} {'err_Imp':>10} {'V_CN':>11} {'err_CN':>10}")
    for s in (70, 80, 90, 95, 100, 105, 110, 120, 130):
        j = strike_index(s, Smax, M1)
        vb = bs_price(s, K, T, r, sig, opt)
        print(f"  {s:>7.1f} {vb:>11.6f} {Vi[j]:>11.6f} {Vi[j] - vb:>10.2e} {Vc[j]:>11.6f} {Vc[j] - vb:>10.2e}")

    # [2] ---------------------------------------------------------------------------------------
    N_sp, Ms = 20000, (50, 100, 200, 400)
    print(f"\n  [2] INDEPENDENT SPATIAL REFINEMENT  (N = {N_sp} fixed; error vs analytical solution)")
    spatial = {}
    for name, f, _ in SOLVERS:
        print(f"  --- {name}   (expected p = 2)")
        print(f"  {'M':>5} {'dS':>7} {'|e(K)|':>10} {'||e||_inf':>11} {'p_inf':>6} {'||e||_2':>11} {'p_2':>6}")
        pi = p2 = None
        rows = []
        for M in Ms:
            S, V = f(K, T, r, sig, Smax, M, N_sp, opt)
            einf, e2 = error_norms(S, V, K, T, r, sig, opt)
            eK = abs(V[strike_index(S0, Smax, M)] - exact0)
            print(f"  {M:>5} {Smax / M:>7.3f} {eK:>10.3e} {einf:>11.3e} {fmt_p(order(pi, einf))} {e2:>11.3e} {fmt_p(order(p2, e2))}")
            rows.append((Smax / M, einf, e2)); pi, p2 = einf, e2
        spatial[name] = np.array(rows)

    # [3] ---------------------------------------------------------------------------------------
    M_t, N_ref, Ns = 400, 12800, (50, 100, 200, 400)
    S, Vref = crank_nicolson_solver(K, T, r, sig, Smax, M_t, N_ref, opt, startup=1)
    jK = strike_index(S0, Smax, M_t)
    print(f"\n  [3] INDEPENDENT TEMPORAL REFINEMENT  (M = {M_t} fixed, dS = {Smax / M_t:g})")
    print(f"      reference: same spatial grid, N_ref = {N_ref} (Crank-Nicolson with implicit start-up)")
    temporal = {}
    for name, f, p_th in SOLVERS:
        print(f"  --- {name}   (expected p = {p_th})")
        print(f"  {'N':>5} {'dtau':>8} {'||e_t||_inf':>12} {'p_inf':>6} {'||e_t||_2':>11} {'p_2':>6} {'|e_t(K)|':>10}"
              f" {'p_K':>6}   {'||e||_inf vs BS':>15}")
        pi = p2 = pk = None
        rows = []
        for N in Ns:
            _, V = f(K, T, r, sig, Smax, M_t, N, opt)
            et = (V - Vref)[1:-1]
            einf, e2, eK = np.max(np.abs(et)), np.sqrt(Smax / M_t * np.sum(et ** 2)), abs(V[jK] - Vref[jK])
            ebs, _ = error_norms(S, V, K, T, r, sig, opt)
            print(f"  {N:>5} {T / N:>8.5f} {einf:>12.3e} {fmt_p(order(pi, einf))} {e2:>11.3e} {fmt_p(order(p2, e2))}"
                  f" {eK:>10.3e} {fmt_p(order(pk, eK))}   {ebs:>15.3e}")
            rows.append((T / N, einf, e2, ebs)); pi, p2, pk = einf, e2, eK
        temporal[name] = np.array(rows)
    e_sp_floor, _ = error_norms(S, Vref, K, T, r, sig, opt)
    print(f"  Spatial error of the M = {M_t} grid itself (reference vs analytical): ||e||_inf = {e_sp_floor:.3e}")

    # [4] ---------------------------------------------------------------------------------------
    print("\n  [4] PAIRED REFINEMENT  (M, N) = (50,50) ... (400,400); error vs analytical solution")
    paired = {}
    for name, f, _ in SOLVERS:
        print(f"  --- {name}")
        print(f"  {'(M, N)':>11} {'|e(K)|':>10} {'||e||_inf':>11} {'p_inf':>6} {'||e||_2':>11} {'p_2':>6} {'time [s]':>9}")
        pi = p2 = None
        rows = []
        for M in Ms:
            t0 = time.perf_counter()
            S, V = f(K, T, r, sig, Smax, M, M, opt)
            dt_run = time.perf_counter() - t0
            einf, e2 = error_norms(S, V, K, T, r, sig, opt)
            eK = abs(V[strike_index(S0, Smax, M)] - exact0)
            print(f"  {f'({M},{M})':>11} {eK:>10.3e} {einf:>11.3e} {fmt_p(order(pi, einf))} {e2:>11.3e}"
                  f" {fmt_p(order(p2, e2))} {dt_run:>9.3f}")
            rows.append((Smax / M, einf, e2, eK, S, V)); pi, p2 = einf, e2
        paired[name] = rows

    # [5] ---------------------------------------------------------------------------------------
    dS5, N5 = 1.25, 400
    print(f"\n  [5] DOMAIN TRUNCATION  (Crank-Nicolson, dS = {dS5} fixed, N = {N5})")
    print(f"  {'S_max':>6} {'M':>5} {'C(K)':>13} {'C(K) - C_BS':>12} {'P(K)':>13} {'P(K) - P_BS':>12}"
          f" {'|dC| vs 5K':>11} {'|dP| vs 5K':>11}")
    dom = {}
    for fct in (2, 3, 4, 5):
        Sm = fct * K
        M = int(round(Sm / dS5))
        vals = []
        for o in ("call", "put"):
            S, V = crank_nicolson_solver(K, T, r, sig, Sm, M, N5, o)
            vals.append(V[strike_index(K, Sm, M)])
        dom[fct] = (M, *vals)
    cb, pb = bs_price(K, K, T, r, sig, "call"), bs_price(K, K, T, r, sig, "put")
    for fct, (M, c, p) in dom.items():
        print(f"  {f'{fct}K':>6} {M:>5} {c:>13.9f} {c - cb:>12.3e} {p:>13.9f} {p - pb:>12.3e}"
              f" {abs(c - dom[5][1]):>11.1e} {abs(p - dom[5][2]):>11.1e}")
    print("  Changing S_max from 2K to 5K alters V(K) by far less than the discretisation error:")
    print("  domain truncation is negligible for this benchmark. S_max = 5K is used because it is")
    print("  conservative and is the only tested domain >= 4K that puts K on a grid node for every")
    print("  M in {50, 100, 200, 400}  (K/dS = M/5).")

    # [6] ---------------------------------------------------------------------------------------
    print("\n  [6] FINAL VERIFICATION TABLE  (M = N = 400, call, S_max = 5K)")
    print(f"  {'Method':<16} {'||e||_inf':>10} {'||e||_2':>10} {'p_space':>8} {'p_time':>7} {'p_paired':>9} {'spur. extrema':>14}")
    for name, f, _ in SOLVERS:
        sp, tp, pr = spatial[name], temporal[name], paired[name]
        S, V = pr[-1][4], pr[-1][5]
        Si, _, g = compute_greeks(S, V)
        print(f"  {name:<16} {pr[-1][1]:>10.3e} {pr[-1][2]:>10.3e} {log2(sp[-2, 1] / sp[-1, 1]):>8.2f}"
              f" {log2(tp[-2, 1] / tp[-1, 1]):>7.2f} {log2(pr[-2][1] / pr[-1][1]):>9.2f}"
              f" {oscillation_count(Si, g, K, 0.2 * K, r, sig, T):>14d}")
    print("  (orders from the finest pair; 'spur. extrema' = spurious local extrema of Gamma in [0.8K, 1.2K])")

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    col = {"Fully implicit": "C0", "Crank-Nicolson": "C1"}
    mk = {"Fully implicit": "s", "Crank-Nicolson": "o"}

    def ticks(x, vals):
        x.xaxis.set_major_locator(FixedLocator(vals)); x.xaxis.set_minor_locator(NullLocator())
        x.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for name in col:
        R = spatial[name]
        ax[0].loglog(R[:, 0], R[:, 1], "-", color=col[name], marker=mk[name], label=f"{name}  " + r"$\|e\|_\infty$")
        ax[0].loglog(R[:, 0], R[:, 2], "--", color=col[name], marker=mk[name], mfc="none", label=f"{name}  " + r"$\|e\|_2$")
    h = spatial["Crank-Nicolson"][:, 0]
    ax[0].loglog(h, spatial["Crank-Nicolson"][0, 1] * (h / h[0]) ** 2 * 0.5, "k:", label="slope 2")
    ax[0].set(xlabel=r"$\Delta S$", ylabel="error", title=f"Spatial refinement (N = {N_sp}; FI and CN coincide)")
    ticks(ax[0], h)
    for name in col:
        R = temporal[name]
        ax[1].loglog(R[:, 0], R[:, 1], "-", color=col[name], marker=mk[name], label=f"{name}: vs same-grid reference")
        ax[1].loglog(R[:, 0], R[:, 3], ":", color=col[name], marker=mk[name], mfc="none", label=f"{name}: vs analytical")
    k = temporal["Fully implicit"][:, 0]
    ax[1].loglog(k, temporal["Fully implicit"][0, 1] * (k / k[0]) * 0.5, "k-.", lw=0.8, label="slope 1")
    ax[1].loglog(k, temporal["Crank-Nicolson"][0, 1] * (k / k[0]) ** 2 * 0.5, "k:", label="slope 2")
    ax[1].axhline(e_sp_floor, color="gray", lw=0.8)
    ax[1].annotate(f"spatial error of the M={M_t} grid", (k[-1], e_sp_floor), xytext=(2, -11),
                   textcoords="offset points", fontsize=7, color="gray")
    ax[1].set(xlabel=r"$\Delta\tau$", title=f"Temporal refinement (M = {M_t}), " + r"$\|e\|_\infty$")
    ticks(ax[1], k)
    for name in col:
        R = np.array([row[:4] for row in paired[name]])
        ax[2].loglog(R[:, 0], R[:, 1], "-", color=col[name], marker=mk[name], label=name)
    hp = np.array([row[0] for row in paired["Crank-Nicolson"]])
    ax[2].loglog(hp, paired["Crank-Nicolson"][0][1] * (hp / hp[0]) ** 2 * 0.5, "k:", label="slope 2")
    ax[2].loglog(hp, paired["Fully implicit"][-1][1] * (hp / hp[-1]) * 1.6, "k-.", lw=0.8, label="slope 1")
    ax[2].set(xlabel=r"$\Delta S$  (with $\Delta\tau = T/M$)", title=r"Paired refinement $(M,N)=(M,M)$, $\|e\|_\infty$")
    ticks(ax[2], hp)
    for x in ax:
        x.grid(alpha=0.3, which="both"); x.legend(fontsize=7)
    fig.suptitle(f"K={K:g}, T={T:g}, r={r:g}, sigma={sig:g}, S_max={Smax:g}, {opt}", fontsize=10)
    fig.tight_layout()
    out = figure_path("fig05_convergence.png", opt != "call"); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n  Figure saved: {os.path.relpath(out)}")

    # Figure 4: pointwise error profiles of the paired refinement (shows where the error lives)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k_, name in enumerate(col):
        for row in paired[name]:
            S, V = row[4], row[5]
            w = (S >= 0.4 * K) & (S <= 2.0 * K)
            ax[k_].plot(S[w], V[w] - bs_price_vec(S[w], K, T, r, sig, opt), label=f"M=N={int(round(Smax / row[0]))}")
        ax[k_].axvline(K, color="gray", ls=":", lw=0.8)
        ax[k_].set(title=f"{name}: pointwise error", xlabel="S", ylabel=r"$V_{FD}-V_{BS}$")
        ax[k_].grid(alpha=0.3); ax[k_].legend(fontsize=7)
    fig.tight_layout()
    out = figure_path("fig06_pointwise_error_refinement.png", opt != "call"); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  Figure saved: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
