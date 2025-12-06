# RiskyBiscuits

# Constrained Portfolio Optimization  
### Penalty Method • Augmented Lagrangian Method • Gradient Descent • Newton  
**Author:** Pulkit Pandey (BT2024060), Dayal Gupta (BT2024167), Harsh Kumar (BT2024008) 
**Course:** AIT 203 — Optimization (Section A)  
**Instructor:** —  

---

## 📌 Project Overview

This project implements and analyzes **constrained mean–variance portfolio optimization** on seven Indian large-cap equities (2018–2025).  
The optimization problem is:

\[
\min_w \; \frac{1}{2} w^\top \Sigma w - \lambda r^\top w
\]

Subject to:
- Full investment:  
  \[
  \sum_i w_i = 1
  \]
- No short selling:  
  \[
  w_i \ge 0
  \]
- Per-asset exposure cap:  
  \[
  w_i \le 0.2
  \]

We implement and compare:

### **Unconstrained Solvers**
- Gradient Descent (GD)  
- Newton’s Closed-Form Solution  

### **Constrained Solvers**
- Penalty Method  
- Augmented Lagrangian Method (ALM)  
- Final Feasible Projection QP  

### **Additional Experiments**
- Robustness to initialization  
- Lambda (λ) sensitivity sweep  
- Efficient frontier visualization  

All results, tables, and figures appear in the final LaTeX report:
```

Portfolio_Optimization_Report_Final.tex

```

---

## 📁 Directory Structure

```

project/
│
├── data/
│   └── indian_data_stats.npz        # mean returns, covariance, tickers, dates
│
├── figures/
│   ├── plot_weights_vs_lambda.png
│   ├── plot_return_vs_lambda.png
│   ├── plot_risk_vs_lambda.png
│   ├── plot_objective_vs_lambda.png
│   └── plot_risk_return_curve.png
│
├── results/
│   ├── logs/                        # console logs from solvers
│   ├── unconstrained_results.npz
│   ├── penalty_results.npz
│   ├── alm_results.npz
│   └── lambda_sweep_results.npz
│
├── scripts/
│   ├── run_all.sh
│   ├── run_unconstrained.py
│   ├── run_penalty.py
│   ├── run_alm.py
│   ├── run_projection.py
│   ├── robustness_random_starts.py
│   └── lambda_sensitivity.py
│
├── Portfolio_Optimization_Report_Final.tex
└── README.md

````

---

## ⚙️ Installation & Setup

### **1. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
````

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### Core packages used:

* **numpy** — linear algebra
* **scipy.optimize** — L-BFGS-B, SLSQP
* **pandas** — data processing
* **matplotlib** — plotting
* **yfinance** — data download (initial stage)
* **cvxpy** (optional) — projection QP alternative

---

## ▶️ Running Experiments

### **Run everything (recommended):**

```bash
bash scripts/run_all.sh
```

This will:

* Run all solvers
* Generate all plots
* Save all `.npz` result files
* Save logs to `results/logs/`
* Prepare figures for the report

---

### **Run solvers individually**

#### **Unconstrained (GD + Newton)**

```bash
python scripts/run_unconstrained.py
```

#### **Penalty Method**

```bash
python scripts/run_penalty.py
```

#### **Augmented Lagrangian Method**

```bash
python scripts/run_alm.py
```

#### **Projection to Feasible Set**

```bash
python scripts/run_projection.py
```

#### **Robustness checks (random initializations)**

```bash
python scripts/robustness_random_starts.py
```

#### **Lambda sensitivity sweep**

```bash
python scripts/lambda_sensitivity.py
```

---

## 📊 Key Results Summary

### **Final Feasible Portfolio (λ = 1.5):**

[
w^* = [0,;0.2,;0.2,;0.2,;0.2,;0.2,;0]
]

### Solver Comparison:

* **Newton:** Fast closed-form; infeasible (allows shorting and >100% leverage)
* **Gradient Descent:** Matches Newton but slow; also infeasible
* **Penalty Method:** Works but requires very large penalties (ρ > 10⁷)
* **Augmented Lagrangian Method:**
  ✔ Best convergence
  ✔ Best numerical stability
  ✔ Feasible solution
  ✔ Robust to initialization

### Weight Saturation Insight:

The optimizer saturates the 20% cap for the **five highest-return assets**.
This is economically consistent: caps limit concentration, and λ shifts aggressiveness toward returns.

---

## 📘 Reproducibility Guarantees

This project includes:

* Fixed seeds for all stochastic experiments
* Saved `.npz` result files
* Saved logs in `results/logs/`
* A fully documented and compilable LaTeX report

To regenerate the report:

```bash
pdflatex Portfolio_Optimization_Report_Final.tex
```

---

## 📄 Final Report

The full academic + business-style report is in:

```
Portfolio_Optimization_Report_Final.tex
Portfolio_Optimization_Report_Final.pdf   (after compilation)
```

It contains:

* Full mathematical formulation
* Solver derivations
* Iteration summaries
* Sensitivity analysis
* Robustness tests
* Economic interpretation
* All charts and tables

---
Just say **“generate badge header”**, **“make run_all.sh”**, or **“make executive summary”**.
```
