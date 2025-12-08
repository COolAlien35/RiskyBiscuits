# 03_penalty_method.py
"""
Advanced Penalty Method (REVISED upper bound = MAX_W = 0.2)
    minimize f(w) = 0.5 w^T Sigma w - lambda * w^T r
subject to:
    sum(w) = 1
    0 <= w_i <= MAX_W  (box constraints)

Key changes vs previous:
 - single constant MAX_W controls upper bound (now 0.2)
 - initial guess attempts to create a feasible starting point under new MAX_W
 - penalty and gradient adjusted to use MAX_W
"""

import numpy as np
import time
import csv
from functools import partial
from scipy.optimize import minimize

DATA_PATH = 'indian_data_stats.npz'
OUT_RESULTS = 'penalty_results.npz'
HISTORY_CSV = 'penalty_history.csv'

#   - User-tunable constant: maximum allowed weight per asset   -
MAX_W = 0.2

def load_data(path=DATA_PATH):
    data = np.load(path, allow_pickle=True)
    return data['r'], data['Sigma'], data.get('tickers', None)

def obj_base(w, Sigma, r, lam):
    return 0.5 * float(w.T.dot(Sigma).dot(w)) - lam * float(w.T.dot(r))

def make_penalty_objective(Sigma, r, lam,
                           rho_eq=1.0, rho_low=1.0, rho_up=1.0,
                           alpha_eq=1.0, alpha_low=1.0, alpha_up=1.0):
    """
    Returns (fun, grad) pair for objective + scaled penalties.
    Uses MAX_W for upper-bound.
    """
    def fun(w):
        base = obj_base(w, Sigma, r, lam)
        s = w.sum() - 1.0
        peq = alpha_eq * rho_eq * (s ** 2)
        neg = np.minimum(0.0, w)
        plow = alpha_low * rho_low * float((neg ** 2).sum())
        exceed = np.maximum(0.0, w - MAX_W)
        pup = alpha_up * rho_up * float((exceed ** 2).sum())
        return base + peq + plow + pup

    def grad(w):
        g = Sigma.dot(w) - lam * r
        s = w.sum() - 1.0
        g = g + (2.0 * alpha_eq * rho_eq * s) * np.ones_like(w)
        mask_low = (w < 0.0)
        g[mask_low] = g[mask_low] + (2.0 * alpha_low * rho_low * w[mask_low])
        mask_up = (w > MAX_W)
        g[mask_up] = g[mask_up] + (2.0 * alpha_up * rho_up * (w[mask_up] - MAX_W))
        return g

    return fun, grad

def constraint_violation(w):
    eq = abs(w.sum() - 1.0)
    low = float(np.maximum(0.0, -w).max())    # max negative magnitude
    up = float(np.maximum(0.0, w - MAX_W).max()) # max exceed amount relative to MAX_W
    return eq, low, up

def penalty_method(Sigma, r, lam,
                   w0=None,
                   rho_eq0=1.0, rho_low0=1.0, rho_up0=1.0,
                   alpha_eq=1.0, alpha_low=1.0, alpha_up=1.0,
                   tol_violation=1e-8,
                   max_outer=30,
                   inner_opts=None,
                   rho_increase_factor=10.0,
                   rho_max=1e12,
                   verbose=True):
    n = Sigma.shape[0]
    if w0 is None:
        # Create a feasible-ish starting guess under new MAX_W:
        # start with uniform, clip to [0, MAX_W], then distribute remaining mass
        w0 = np.ones(n) / n
        w0 = np.clip(w0, 0.0, MAX_W)
        if abs(w0.sum() - 1.0) > 1e-12:
            # rescale and distribute remaining mass greedily without violating MAX_W
            w = w0.copy()
            # If sum < 1, distribute to assets with spare capacity
            deficit = 1.0 - w.sum()
            if deficit > 0:
                # distribute starting from smallest weights
                idx = np.argsort(w)
                for i in idx:
                    can_add = MAX_W - w[i]
                    add = min(can_add, deficit)
                    w[i] += add
                    deficit -= add
                    if deficit <= 1e-12:
                        break
            else:
                # if sum > 1 (unlikely after clip), scale down proportionally
                w = w / w.sum()
            w0 = w.copy()
    else:
        w0 = w0.copy()

    rho_eq = float(rho_eq0)
    rho_low = float(rho_low0)
    rho_up = float(rho_up0)

    history = []
    w = w0.copy()

    if inner_opts is None:
        inner_opts = {'maxiter': 2000, 'gtol': 1e-8}

    prev_total_violation = max(constraint_violation(w)[0:3])

    for outer in range(1, max_outer + 1):
        t0_outer = time.time()
        fun, grad = make_penalty_objective(Sigma, r, lam,
                                           rho_eq=rho_eq, rho_low=rho_low, rho_up=rho_up,
                                           alpha_eq=alpha_eq, alpha_low=alpha_low, alpha_up=alpha_up)
        res = minimize(fun, w, method='L-BFGS-B', jac=grad, options=inner_opts)
        w_new = res.x.copy()
        inner_time = time.time() - t0_outer

        eq_v, low_v, up_v = constraint_violation(w_new)
        total_violation = max(eq_v, low_v, up_v)
        base_obj = obj_base(w_new, Sigma, r, lam)

        history.append({
            'outer': outer,
            'rho_eq': rho_eq,
            'rho_low': rho_low,
            'rho_up': rho_up,
            'inner_fun': float(res.fun),
            'inner_nit': int(res.nit) if hasattr(res, 'nit') else -1,
            'inner_success': bool(res.success),
            'inner_message': str(res.message),
            'time_inner_s': inner_time,
            'base_obj': float(base_obj),
            'eq_v': float(eq_v),
            'low_v': float(low_v),
            'up_v': float(up_v),
            'total_v': float(total_violation)
        })

        if verbose:
            print(f"[Outer {outer}] inner nit={res.nit}, success={res.success}, total_violation={total_violation:.3e}, base_obj={base_obj:.6f}, rho_eq={rho_eq:.1e}")

        if total_violation <= tol_violation:
            w = w_new
            if verbose:
                print("Termination: violations within tolerance.")
            break

        # Check reduction fraction
        reduction = prev_total_violation - total_violation
        reduction_frac = reduction / (prev_total_violation + 1e-18)

        if reduction_frac < 0.25:
            rho_eq = min(rho_eq * rho_increase_factor, rho_max)
            rho_low = min(rho_low * rho_increase_factor, rho_max)
            rho_up = min(rho_up * rho_increase_factor, rho_max)
            if verbose:
                print(f"  -> insufficient violation reduction ({reduction_frac:.2f}), increasing rhos -> {rho_eq:.1e}")
        else:
            rho_eq = min(rho_eq * 2.0, rho_max)
            rho_low = min(rho_low * 2.0, rho_max)
            rho_up = min(rho_up * 2.0, rho_max)

        prev_total_violation = total_violation
        w = w_new.copy()

    final_eq, final_low, final_up = constraint_violation(w)
    final_base_obj = obj_base(w, Sigma, r, lam)
    result = {
        'w': w,
        'base_obj': final_base_obj,
        'eq_v': final_eq,
        'low_v': final_low,
        'up_v': final_up,
        'history': history
    }
    return result

def save_history_csv(history, csv_path=HISTORY_CSV):
    keys = ['outer','rho_eq','rho_low','rho_up','inner_fun','inner_nit','inner_success','inner_message','time_inner_s','base_obj','eq_v','low_v','up_v','total_v']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            row2 = {k: row.get(k, '') for k in keys}
            writer.writerow(row2)

def main():
    lam = 1.5
    rho_eq0 = 1.0
    rho_low0 = 1.0
    rho_up0 = 1.0
    tol_violation = 1e-8
    max_outer = 30
    inner_opts = {'maxiter': 2000, 'gtol': 1e-8}

    r, Sigma, tickers = load_data()
    print("Loaded data. n =", len(r))
    if tickers is not None:
        print("Tickers:", list(tickers))
    print(f"Using MAX_W = {MAX_W}")

    res = penalty_method(Sigma, r, lam,
                         rho_eq0=rho_eq0, rho_low0=rho_low0, rho_up0=rho_up0,
                         alpha_eq=1.0, alpha_low=1.0, alpha_up=1.0,
                         tol_violation=tol_violation,
                         max_outer=max_outer,
                         inner_opts=inner_opts,
                         rho_increase_factor=10.0,
                         rho_max=1e12,
                         verbose=True)

    w = res['w']
    print("\nPenalty method finished.")
    print("Final base objective:", res['base_obj'])
    print("Final violations: eq={:.3e}, low={:.3e}, up={:.3e}".format(res['eq_v'], res['low_v'], res['up_v']))
    print("Weights (rounded):", np.round(w, 6))
    print("Sum of weights:", w.sum())
    print("Any negatives? ", (w < 0).any(), " Any > MAX_W? ", (w > MAX_W).any())

    np.savez(OUT_RESULTS,
             lam=lam,
             w=w,
             base_obj=res['base_obj'],
             eq_v=res['eq_v'],
             low_v=res['low_v'],
             up_v=res['up_v'],
             history=res['history'])
    save_history_csv(res['history'], HISTORY_CSV)
    print(f"Saved penalty results to {OUT_RESULTS} and history to {HISTORY_CSV}")

if __name__ == '__main__':
    main()
