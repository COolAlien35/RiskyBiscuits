# 04_augmented_lagrangian.py
"""
Moderate Augmented Lagrangian Method (ALM)
for constrained portfolio optimization:

    minimize f(w) = 0.5 w^T Sigma w - lambda * w^T r

subject to:
    sum(w) = 1
    0 <= w_i <= MAX_W

ALM handles the equality constraint strongly and inequalities softly.
"""

import numpy as np
import time
from scipy.optimize import minimize

DATA_PATH = 'indian_data_stats.npz'
OUT_PATH = 'alm_results.npz'
MAX_W = 0.2   # upper-bound on weights

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_data(path=DATA_PATH):
    data = np.load(path, allow_pickle=True)
    return data['r'], data['Sigma'], data.get('tickers', None)

# ------------------------------------------------------------
# Base objective
# ------------------------------------------------------------

def obj_base(w, Sigma, r, lam):
    return 0.5 * w.T @ Sigma @ w - lam * w @ r

# ------------------------------------------------------------
# Build ALM objective + gradient
# ------------------------------------------------------------

def make_alm_functions(Sigma, r, lam, lam_eq, rho_eq):
    """
    Returns (fun, grad) for the ALM augmented objective:

        L(w) = f(w) + λ_eq (sum(w)-1) + (ρ_eq/2)(sum(w)-1)^2
               + penalty for inequality violations
    """

    def fun(w):
        # base
        base = obj_base(w, Sigma, r, lam)

        # equality part
        s = w.sum() - 1.0
        eq_term = lam_eq * s + 0.5 * rho_eq * (s ** 2)

        # inequality penalties
        low  = np.minimum(0.0, w)
        up   = np.maximum(0.0, w - MAX_W)
        ineq = 1e3 * ( (low**2).sum() + (up**2).sum() )

        return base + eq_term + ineq

    def grad(w):
        g = Sigma @ w - lam * r

        # equality grad
        s = w.sum() - 1.0
        g += (lam_eq + rho_eq * s) * np.ones_like(w)

        # inequality grads
        low_mask = w < 0
        up_mask  = w > MAX_W

        g[low_mask] += 2e3 * w[low_mask]
        g[up_mask]  += 2e3 * (w[up_mask] - MAX_W)

        return g

    return fun, grad

# ------------------------------------------------------------
# ALM main solver
# ------------------------------------------------------------

def alm_solve(Sigma, r, lam,
              tol=1e-8,
              max_outer=20,
              rho_eq_init=10.0,
              verbose=True):

    n = len(r)

    # start with a feasible-ish guess: uniform and clipped to upper bound
    w = np.ones(n) / n
    w = np.clip(w, 0, MAX_W)

    # distribute remaining weight (simple heuristic)
    deficit = 1 - w.sum()
    if deficit > 0:
        idx = np.argsort(w)
        for i in idx:
            can_add = MAX_W - w[i]
            add = min(can_add, deficit)
            w[i] += add
            deficit -= add
            if deficit <= 1e-12:
                break

    lam_eq = 0.0                 # equality multiplier
    rho_eq = rho_eq_init         # penalty parameter

    history = []

    for outer in range(1, max_outer+1):

        fun, grad = make_alm_functions(Sigma, r, lam, lam_eq, rho_eq)

        # inner solve
        res = minimize(fun, w, method='L-BFGS-B', jac=grad,
                       options={'maxiter': 2000, 'gtol':1e-8})

        w_new = res.x

        # constraint violations
        eq_v = abs(w_new.sum() - 1.0)
        low_v = np.max(np.maximum(0, -w_new))
        up_v  = np.max(np.maximum(0, w_new - MAX_W))
        tot_v = max(eq_v, low_v, up_v)

        # update multiplier (ALM rule)
        lam_eq = lam_eq + rho_eq * (w_new.sum() - 1.0)

        # optionally increase penalty
        if eq_v > tol:
            rho_eq *= 2.0

        # log progress
        history.append({
            'outer': outer,
            'obj': float(obj_base(w_new, Sigma, r, lam)),
            'eq_v': float(eq_v),
            'low_v': float(low_v),
            'up_v': float(up_v),
            'rho_eq': float(rho_eq),
            'lam_eq': float(lam_eq)
        })

        if verbose:
            print(f"[Outer {outer}] obj={history[-1]['obj']:.6f}, eq_v={eq_v:.2e}, up_v={up_v:.2e}, rho_eq={rho_eq}")

        # convergence
        if tot_v < tol:
            w = w_new
            break

        w = w_new.copy()

    return w, history

# ------------------------------------------------------------
# Run ALM
# ------------------------------------------------------------

def main():
    lam = 1.5
    r, Sigma, tickers = load_data()

    print("Loaded data for ALM. Using MAX_W =", MAX_W)
    if tickers is not None:
        print("Tickers:", list(tickers))

    w, hist = alm_solve(Sigma, r, lam, tol=1e-8, max_outer=20, verbose=True)

    print("\nALM finished.")
    print("Final weights:", np.round(w, 6))
    print("Sum of weights:", w.sum())
    print("Any negatives?", (w < 0).any(), " Any > MAX_W?", (w > MAX_W).any())
    print("Final obj:", obj_base(w, Sigma, r, lam))

    np.savez(OUT_PATH,
             lam=lam,
             w=w,
             history=hist,
             tickers=tickers,
             MAX_W=MAX_W)
    print(f"Results saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
