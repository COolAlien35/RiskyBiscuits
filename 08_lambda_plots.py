# 08_lambda_plots.py
"""
Advanced, stylish visualization of lambda sensitivity results.
Generates publication-quality plots:
 - weights vs lambda
 - return vs lambda
 - risk vs lambda
 - objective vs lambda
 - risk vs return curve
"""

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = 'lambda_sweep_results.npz'

# ---------------------------------------------------------
# Load sweep results
# ---------------------------------------------------------

def load_sweep(path=DATA_PATH):
    d = np.load(path, allow_pickle=True)
    results = d['results']
    tickers = d.get('tickers', None)
    max_w = float(d.get('MAX_W', 0.2))
    return results, tickers, max_w


# ---------------------------------------------------------
# Plot settings (stylistic)
# ---------------------------------------------------------

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['font.size'] = 12


# ---------------------------------------------------------
# Main Plotting Function
# ---------------------------------------------------------

def main():
    results, tickers, max_w = load_sweep()

    lambdas = [row['lambda'] for row in results]
    weights = np.array([row['weights'] for row in results])
    risks = np.array([row['risk'] for row in results])
    rets  = np.array([row['return'] for row in results])
    objs  = np.array([row['objective'] for row in results])

    # -----------------------------------------------------
    # 1. Weights vs Lambda
    # -----------------------------------------------------
    plt.figure()
    for i in range(weights.shape[1]):
        label = tickers[i] if tickers is not None else f"Asset {i+1}"
        plt.plot(lambdas, weights[:, i], marker='o', label=label)
    plt.axhline(max_w, color='red', linestyle='--', label=f'Max weight = {max_w}')
    plt.title("Weights vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_weights_vs_lambda.png", dpi=300)
    print("Saved plot_weights_vs_lambda.png")

    # -----------------------------------------------------
    # 2. Return vs Lambda
    # -----------------------------------------------------
    plt.figure()
    plt.plot(lambdas, rets, marker='o', color='green')
    plt.title("Portfolio Return vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig("plot_return_vs_lambda.png", dpi=300)
    print("Saved plot_return_vs_lambda.png")

    # -----------------------------------------------------
    # 3. Risk (Variance) vs Lambda
    # -----------------------------------------------------
    plt.figure()
    plt.plot(lambdas, risks, marker='o', color='orange')
    plt.title("Portfolio Risk vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Risk (Variance)")
    plt.tight_layout()
    plt.savefig("plot_risk_vs_lambda.png", dpi=300)
    print("Saved plot_risk_vs_lambda.png")

    # -----------------------------------------------------
    # 4. Objective vs Lambda
    # -----------------------------------------------------
    plt.figure()
    plt.plot(lambdas, objs, marker='o', color='purple')
    plt.title("Objective Value vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Objective")
    plt.tight_layout()
    plt.savefig("plot_objective_vs_lambda.png", dpi=300)
    print("Saved plot_objective_vs_lambda.png")

    # -----------------------------------------------------
    # 5. Efficient Frontier-like: Risk vs Return
    # -----------------------------------------------------
    plt.figure()
    plt.plot(risks, rets, marker='o', color='blue')
    for i, lam in enumerate(lambdas):
        plt.text(risks[i], rets[i], f"λ={lam}", fontsize=10)
    plt.title("Risk vs Return (Across Lambda)")
    plt.xlabel("Risk (Variance)")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.savefig("plot_risk_return_curve.png", dpi=300)
    print("Saved plot_risk_return_curve.png")


if __name__ == "__main__":
    main()
