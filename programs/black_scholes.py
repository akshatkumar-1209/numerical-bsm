"""
black_scholes.py -- Analytical Black-Scholes benchmark (Derivation Part 2, Section 2.6)

Closed-form European call/put prices and the two Greeks studied in the project:

    C(S,tau) = S N(d1) - K e^{-r tau} N(d2)
    P(S,tau) = K e^{-r tau} N(-d2) - S N(-d1)
    d1 = [ln(S/K) + (r + sigma^2/2) tau] / (sigma sqrt(tau)),   d2 = d1 - sigma sqrt(tau)

    Delta_call = N(d1),   Delta_put = N(d1) - 1,   Gamma = n(d1) / (S sigma sqrt(tau))

This routine is deliberately independent of the finite-difference code: it shares
nothing with the solvers except the model parameters, so an error in one cannot be
silently reproduced in the other.

Usage
-----
    python black_scholes.py                       # professor's benchmark
    python black_scholes.py --S 110 --sigma 0.25  # any other parameter set
"""

import argparse
from math import erf, exp, log, pi, sqrt


def norm_cdf(x):
    """Standard normal CDF  N(x)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def norm_pdf(x):
    """Standard normal density  n(x)."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _d1_d2(S, K, tau, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqrt(tau))
    return d1, d1 - sigma * sqrt(tau)


def bs_price(S, K, tau, r, sigma, option="call"):
    """Black-Scholes price at asset level S and time-to-maturity tau."""
    if tau <= 0.0:                                     # payoff at maturity
        return max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
    if S <= 0.0:                                       # degenerate boundary S = 0
        return 0.0 if option == "call" else K * exp(-r * tau)
    d1, d2 = _d1_d2(S, K, tau, r, sigma)
    if option == "call":
        return S * norm_cdf(d1) - K * exp(-r * tau) * norm_cdf(d2)
    return K * exp(-r * tau) * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S, K, tau, r, sigma, option="call"):
    """Black-Scholes Delta = dV/dS."""
    if S <= 0.0:
        return 0.0 if option == "call" else -1.0
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return norm_cdf(d1) if option == "call" else norm_cdf(d1) - 1.0


def bs_gamma(S, K, tau, r, sigma):
    """Black-Scholes Gamma = d^2V/dS^2 (identical for calls and puts)."""
    if S <= 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, tau, r, sigma)
    return norm_pdf(d1) / (S * sigma * sqrt(tau))


# Vectorised helpers used by the FDM programs for error measurement
def bs_price_vec(S_arr, K, tau, r, sigma, option="call"):
    import numpy as np
    return np.array([bs_price(s, K, tau, r, sigma, option) for s in S_arr])


def bs_delta_vec(S_arr, K, tau, r, sigma, option="call"):
    import numpy as np
    return np.array([bs_delta(s, K, tau, r, sigma, option) for s in S_arr])


def bs_gamma_vec(S_arr, K, tau, r, sigma):
    import numpy as np
    return np.array([bs_gamma(s, K, tau, r, sigma) for s in S_arr])


def main():
    p = argparse.ArgumentParser(description="Analytical Black-Scholes benchmark")
    p.add_argument("--S", type=float, default=100.0, help="asset price (default 100)")
    p.add_argument("--K", type=float, default=100.0, help="strike (default 100)")
    p.add_argument("--T", type=float, default=1.0, help="maturity in years (default 1)")
    p.add_argument("--r", type=float, default=0.05, help="risk-free rate (default 0.05)")
    p.add_argument("--sigma", type=float, default=0.20, help="volatility (default 0.20)")
    a = p.parse_args()

    S, K, T, r, sig = a.S, a.K, a.T, a.r, a.sigma
    d1, d2 = _d1_d2(S, K, T, r, sig)
    C, P = bs_price(S, K, T, r, sig, "call"), bs_price(S, K, T, r, sig, "put")

    print("=" * 64)
    print("  ANALYTICAL BLACK-SCHOLES BENCHMARK")
    print(f"  S = {S:g}, K = {K:g}, T = {T:g}, r = {r:g}, sigma = {sig:g}")
    print("=" * 64)
    print(f"  d1                 = {d1: .6f}")
    print(f"  d2                 = {d2: .6f}")
    print(f"  European call      = {C: .6f}")
    print(f"  European put       = {P: .6f}")
    print(f"  Call Delta         = {bs_delta(S, K, T, r, sig, 'call'): .6f}")
    print(f"  Put  Delta         = {bs_delta(S, K, T, r, sig, 'put'): .6f}")
    print(f"  Gamma              = {bs_gamma(S, K, T, r, sig): .6f}")
    parity = C - P - (S - K * exp(-r * T))
    print(f"  Put-call parity    C - P - (S - K e^(-rT)) = {parity:.2e}")
    print("=" * 64)


if __name__ == "__main__":
    main()
