# 02_unconstrained_solvers.py
"""
Unconstrained optimizers for:
    minimize f(w) = 0.5 w^T Sigma w - lambda * w^T r
Implements:
 - Gradient Descent (with step sized by spectral radius of Sigma)
 - Newton closed-form solution: w = lambda * Sigma^{-1} r
Loads data from 'indian_data_stats.npz' by default.
"""

import numpy as np
import time

# Path to data file (change to absolute path if needed)
DATA_PATH = 'indian_data_stats.npz'   # or '/mnt/data/indian_data_stats.npz'

def load_data(path=DATA_PATH):
    data = np.load(path, allow_pickle=True)
    r = data['r']
    Sigma = data['Sigma']
    tickers = data.get('tickers', None)
    return r, Sigma, tickers

def objective(w, Sigma, r, lam):
    return 0.5 * float(w.T.dot(Sigma).dot(w)) - lam * float(w.T.dot(r))

def grad(w, Sigma, r, lam):
    return Sigma.dot(w) - lam * r

def gradient_descent(Sigma, r, lam,
                     w0=None,
                     tol=1e-8,
                     max_iter=20000,
                     verbose=False):
    n = len(r)
    if w0 is None:
        w = np.ones(n) / n
    else:
        w = w0.copy()

    # safe step-size: 1 / (Lipschitz constant) where L = max eigenvalue of Sigma
    eigs = np.linalg.eigvalsh(Sigma)
    L = max(eigs.real) if eigs.size > 0 else 1.0
    alpha = 1.0 / (1.1 * L)   # slightly smaller than 1/L for safety

    if verbose:
        print(f"[GD] using alpha = {alpha:.3e} (1/(1.1*L)), L = {L:.6f}")

    obj_hist = []
    for k in range(1, max_iter+1):
        g = grad(w, Sigma, r, lam)
        w_new = w - alpha * g
        obj = objective(w_new, Sigma, r, lam)
        obj_hist.append(obj)
        if np.linalg.norm(w_new - w) < tol:
            return w_new, obj_hist, k
        w = w_new
    return w, obj_hist, max_iter

def newton_closed_form(Sigma, r, lam):
    # w* = lambda * Sigma^{-1} r
    # use pseudo-inverse for numerical stability
    pinv = np.linalg.pinv(Sigma)
    w = lam * pinv.dot(r)
    return w

def main():
    lam = 1.5   # default risk aversion (change if you want)
    r, Sigma, tickers = load_data()
    print("Loaded data. n =", len(r))
    if tickers is not None:
        print("Tickers:", list(tickers))

    # Newton closed form
    t0 = time.time()
    w_newton = newton_closed_form(Sigma, r, lam)
    t1 = time.time()
    obj_newton = objective(w_newton, Sigma, r, lam)
    print("\n[Newton closed-form]")
    print("time: {:.4f}s".format(t1 - t0))
    print("objective:", obj_newton)
    print("weights (first 7):", np.round(w_newton, 6))
    print("sum of weights:", w_newton.sum())

    # Gradient descent
    t0 = time.time()
    w_gd, hist, iters = gradient_descent(Sigma, r, lam, tol=1e-10, max_iter=20000, verbose=True)
    t1 = time.time()
    obj_gd = objective(w_gd, Sigma, r, lam)
    print("\n[Gradient Descent]")
    print("time: {:.4f}s, iterations:", iters)
    print("objective:", obj_gd)
    print("weights (first 7):", np.round(w_gd, 6))
    print("sum of weights:", w_gd.sum())

    # Compare solutions
    diff = np.linalg.norm(w_newton - w_gd)
    print("\nComparison:")
    print("L2 difference between Newton and GD:", diff)
    print("Relative objective diff (GD - Newton):", obj_gd - obj_newton)

    # Save results for later steps
    np.savez('unconstrained_results.npz',
             lam=lam,
             w_newton=w_newton,
             obj_newton=obj_newton,
             w_gd=w_gd,
             obj_gd=obj_gd,
             gd_hist=np.array(hist))

    print("\nSaved results to 'unconstrained_results.npz'")

if __name__ == '__main__':
    main()
