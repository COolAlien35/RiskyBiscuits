# 07_lambda_sensitivity.py
"""
Lambda Sensitivity Sweep
Runs ALM for multiple lambda values and evaluates:
 - Final weights (projected)
 - Portfolio return
 - Portfolio variance (risk)
 - Objective value
Saves all results to CSV and NPZ.
"""

import numpy as np
import csv
from scipy.optimize import minimize

from alm_augmented import alm_solve, obj_base, MAX_W

DATA_PATH = 'indian_data_stats.npz'
CSV_OUT = 'lambda_sweep_results.csv'
NPZ_OUT = 'lambda_sweep_results.npz'

# Lambda grid (Option B)
LAMBDA_GRID = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


    # Load Data          

def load_data(path=DATA_PATH):
    d = np.load(path, allow_pickle=True)
    return d['r'], d['Sigma'], d.get('tickers', None)


    #   Projection (for exact feasibility)               -

def project_to_feasible(w_init, max_w=MAX_W):
    n = len(w_init)
    bounds = [(0, max_w) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda z: np.sum(z) - 1.0})
    fun = lambda z: np.sum((z - w_init)**2)
    res = minimize(fun, w_init, bounds=bounds, constraints=cons)
    return res.x


    #   Main  

def main():
    r, Sigma, tickers = load_data()

    results = []

    print("\n=== Running Lambda Sensitivity Sweep ===\n")

    for lam in LAMBDA_GRID:
        print(f"\n### Lambda = {lam}")

        # Run ALM
        w_alm, hist = alm_solve(Sigma, r, lam,
                                tol=1e-8,
                                max_outer=20,
                                rho_eq_init=10.0,
                                verbose=False)

        # Project to clear feasibility
        w_proj = project_to_feasible(w_alm, MAX_W)

        # Risk = wᵀ Σ w
        risk = float(w_proj.T @ Sigma @ w_proj)

        # Return = wᵀ r
        ret = float(w_proj.T @ r)

        # Objective = 0.5 wᵀΣw - λ (wᵀr)
        obj = obj_base(w_proj, Sigma, r, lam)

        print("Weights:", np.round(w_proj, 6))
        print("Sum:", w_proj.sum(), "Return:", ret, "Risk:", risk, "Objective:", obj)

        results.append({
            'lambda': lam,
            'weights': w_proj,
            'risk': risk,
            'return': ret,
            'objective': obj
        })

    # Save CSV
    with open(CSV_OUT, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lambda', 'weights', 'risk', 'return', 'objective'])
        for row in results:
            writer.writerow([row['lambda'], row['weights'], row['risk'], row['return'], row['objective']])

    # Save NPZ
    np.savez(NPZ_OUT, results=results, tickers=tickers, MAX_W=MAX_W)

    print(f"\nSaved results to {CSV_OUT} and {NPZ_OUT}")


if __name__ == "__main__":
    main()
