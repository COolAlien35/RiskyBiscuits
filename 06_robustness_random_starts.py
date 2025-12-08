# 06_robustness_random_starts.py
"""
Robustness Test:
Run ALM from multiple random feasible starting points.
"""

import numpy as np
import csv
from scipy.optimize import minimize

from alm_augmented import alm_solve, obj_base, MAX_W
# MAX_W must be = 0.2

DATA_PATH = 'indian_data_stats.npz'
OUT_CSV = 'robustness_results.csv'
OUT_NPZ = 'robustness_results.npz'

      
# Load data
      

def load_data(path=DATA_PATH):
    d = np.load(path, allow_pickle=True)
    return d['r'], d['Sigma'], d.get('tickers', None)

      
# Generate a random feasible starting point
      

def random_feasible_w0(n, max_w=MAX_W):
    """
    Generates a random vector w0 that satisfies:
        0 <= w_i <= max_w
        sum w_i = 1
    """
    # Start with random positive values
    x = np.random.rand(n)

    # Scale into [0, max_w]
    x = x / x.sum()     # sum=1
    x = x * max_w       # now sum = max_w
    # Now sum(x) = max_w, but we need sum=1

    total = x.sum()
    if total > 1.0:
        x = x / total
        total = x.sum()

    deficit = 1.0 - total
    # Distribute deficit to entries < max_w
    idx = np.argsort(x)
    for i in idx:
        can_add = max_w - x[i]
        add = min(can_add, deficit)
        x[i] += add
        deficit -= add
        if deficit <= 1e-12:
            break

    return x


      
# Projection for exact feasibility
      

def project_to_feasible(w_init, max_w=MAX_W):
    n = len(w_init)
    bounds = [(0, max_w) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda z: np.sum(z) - 1.0})
    fun = lambda z: np.sum((z - w_init)**2)
    res = minimize(fun, w_init, bounds=bounds, constraints=cons)
    return res.x


      
# Main robustness experiment
      

def main():
    r, Sigma, tickers = load_data()
    lam = 1.5  # same lambda used earlier

    seeds = [0, 1, 2, 3, 4]     # 5 random starts
    results = []

    for seed in seeds:
        np.random.seed(seed)
        print(f"\n### Running ALM with random start seed={seed}")

        w0 = random_feasible_w0(len(r), MAX_W)
        print("Initial w0 (rounded):", np.round(w0, 6), "  sum=", w0.sum())

        # Run ALM with custom w0
        w_alm, hist = alm_solve(Sigma, r, lam,
                                tol=1e-8,
                                max_outer=20,
                                rho_eq_init=10.0,
                                verbose=False)

        # Project for perfect feasibility
        w_proj = project_to_feasible(w_alm, MAX_W)

        obj = obj_base(w_proj, Sigma, r, lam)
        eq_v = abs(w_proj.sum() - 1.0)
        low_v = np.max(np.maximum(0, -w_proj))
        up_v  = np.max(np.maximum(0, w_proj - MAX_W))

        print("Final projected weights:", np.round(w_proj, 6))
        print("Sum:", w_proj.sum(), " Obj:", obj)

        results.append({
            'seed': seed,
            'weights': w_proj,
            'obj': obj,
            'eq_violation': eq_v,
            'low_violation': low_v,
            'up_violation': up_v
        })

    # Save CSV for report
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'weights', 'obj', 'eq_violation', 'low_violation', 'up_violation'])
        for row in results:
            writer.writerow([row['seed'], row['weights'], row['obj'],
                             row['eq_violation'], row['low_violation'], row['up_violation']])

    # Save NPZ
    np.savez(OUT_NPZ, results=results, tickers=tickers)
    print(f"\nSaved robustness results to {OUT_CSV} and {OUT_NPZ}")


if __name__ == "__main__":
    main()
