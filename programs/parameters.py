"""
parameters.py -- Single source of the professor's benchmark and of the numerical domain.

Benchmark (fixed by the project brief):
    K = 100, T = 1 year, r = 0.05, sigma = 0.20, evaluation point S = K = 100

Numerical domain (project assumption, justified by the domain-truncation study in
convergence_study.py / Derivation Part 6):
    S_max = 5K = 500

Every program reads its defaults from here and accepts command-line overrides, e.g.
    python crank_nicolson.py --sigma 0.3 --M 800 --N 200
"""

import os

BENCH = dict(K=100.0, T=1.0, r=0.05, sigma=0.20, S0=100.0)
S_MAX_FACTOR = 5.0                      # S_max = 5K
S_MAX = S_MAX_FACTOR * BENCH["K"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")


def add_common_args(parser, M=400, N=400):
    """Model parameters, domain and grid as command-line options with benchmark defaults."""
    parser.add_argument("--K", type=float, default=BENCH["K"], help="strike (default 100)")
    parser.add_argument("--T", type=float, default=BENCH["T"], help="maturity in years (default 1)")
    parser.add_argument("--r", type=float, default=BENCH["r"], help="risk-free rate (default 0.05)")
    parser.add_argument("--sigma", type=float, default=BENCH["sigma"], help="volatility (default 0.20)")
    parser.add_argument("--S0", type=float, default=BENCH["S0"], help="evaluation point (default 100)")
    parser.add_argument("--Smax", type=float, default=None, help="domain limit (default 5K)")
    parser.add_argument("--M", type=int, default=M, help=f"space intervals (default {M})")
    parser.add_argument("--N", type=int, default=N, help=f"time steps (default {N})")
    parser.add_argument("--no-plot", action="store_true", help="skip figure generation")
    _orig = parser.parse_args

    def parse_args(*a, **k):
        ns = _orig(*a, **k)
        smax_given = ns.Smax is not None
        if ns.Smax is None:
            ns.Smax = S_MAX_FACTOR * ns.K
        # any non-default setting -> figures go to figures/custom_*.png, so that the
        # committed benchmark figures are never overwritten by an exploratory run
        ns.custom = any(v != parser.get_default(key) for key, v in vars(ns).items()
                        if key not in ("no_plot", "Smax", "custom")) or (
                        smax_given and ns.Smax != S_MAX_FACTOR * BENCH["K"])
        return ns
    parser.parse_args = parse_args
    return parser


def figure_path(name, custom=False):
    """Path of an output figure; exploratory (non-benchmark) runs get a 'custom_' prefix."""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, ("custom_" if custom else "") + name)
