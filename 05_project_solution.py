# 05_project_solution.py
import numpy as np
from scipy.optimize import minimize

# load ALM result
data = np.load('alm_results.npz', allow_pickle=True)
w_alm = data['w']
MAX_W = float(data.get('MAX_W', 0.2))
tickers = data.get('tickers', None)

def project_to_feasible(w_init, max_w=0.2):
    n = len(w_init)
    bounds = [(0.0, max_w) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    fun = lambda x: np.sum((x - w_init)**2)   # squared distance
    res = minimize(fun, w_init, bounds=bounds, constraints=cons, options={'maxiter':2000})
    if not res.success:
        print("Projection solver warning:", res.message)
    return res.x

if __name__ == '__main__':
    w_proj = project_to_feasible(w_alm, MAX_W)
    print("Projected weights:", np.round(w_proj, 8))
    print("Sum:", w_proj.sum())
    print("Any negatives?", (w_proj < 0).any(), "Any>MAX_W?", (w_proj > MAX_W).any())
    # Save
    np.savez('alm_projected.npz', w_original=w_alm, w_projected=w_proj, tickers=tickers, MAX_W=MAX_W)
    print("Saved alm_projected.npz")
