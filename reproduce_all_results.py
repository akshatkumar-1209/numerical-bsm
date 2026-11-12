"""
reproduce_all_results.py -- run every program of the project in order and store its output.

    python reproduce_all_results.py            # all text results -> results/, figures -> figures/

Each program is executed as a separate process from the programs/ folder with the
benchmark defaults of programs/parameters.py (K=100, T=1, r=0.05, sigma=0.20, S_max=5K).
Total run time is about one minute.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(ROOT, "programs")
RES = os.path.join(ROOT, "results")

RUNS = [
    ("01_black_scholes_analytic.txt", "black_scholes.py", "Analytical Black-Scholes benchmark"),
    ("02_finite_differences.txt", "finite_difference_derivatives.py", "Part 1  finite-difference stencils"),
    ("03_grid_payoff_boundaries.txt", "generate_grid.py", "Part 2  grid, payoff, boundary data"),
    ("04_thomas_solver.txt", "thomas_solver.py", "Part 3  Thomas algorithm"),
    ("05_fully_implicit.txt", "implicit_fdm.py", "Part 4  fully implicit solver"),
    ("06_crank_nicolson.txt", "crank_nicolson.py", "Part 5  Crank-Nicolson solver"),
    ("07_convergence_study.txt", "convergence_study.py", "Part 6  convergence and domain truncation"),
    ("08_numerical_greeks.txt", "numerical_greeks.py", "Part 7  numerical Delta and Gamma"),
    ("09_payoff_kink_oscillation.txt", "payoff_kink_cn_oscillation.py", "Part 8  payoff kink and CN start-up"),
]


def main():
    os.makedirs(RES, exist_ok=True)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    failed = []
    for out, script, label in RUNS:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, script], cwd=PROG, env=env, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        with open(os.path.join(RES, out), "w", encoding="utf-8") as fh:
            fh.write(proc.stdout)
            if proc.returncode != 0:
                fh.write("\n--- stderr ---\n" + proc.stderr)
        status = "ok" if proc.returncode == 0 else "FAILED"
        if proc.returncode != 0:
            failed.append(script)
        print(f"  {status:<6} {label:<44} {dt:6.1f} s  -> results/{out}")
    print("\n  All programs completed." if not failed else f"\n  Failed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
