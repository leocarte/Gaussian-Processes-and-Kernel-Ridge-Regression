## Gaussian Processes and Kernel Ridge Regression

### Credits
- Done as part of Statistical Machine Learning Course (MATH-412) - EPFL 2025 Autumn Offering

### Team
- Leonardo Cartesegna
- Chandrasekhara Devarakonda
- Giulia Scagliarini

### Dataset
- Stock Portfolio Performance dataset from the UCI Machine Learning Repository. Can be accessed [here](https://archive.ics.uci.edu/dataset/390/stock+portfolio+performance).
- A copy is included in the repo [here](stock%20portfolio%20performance%20data%20set.xlsx).
- We use the Excel sheet **"all period"**: 6 portfolio-concept weights as inputs and the **normalized Annual Return** as target.

### Goal of the Project
- Make the GP–KRR connection explicit: the **GP posterior mean / MAP** coincides with the **KRR predictor** under a matching noise–regularization identification, while GP additionally provides predictive uncertainty.
- Compare hyperparameter selection strategies and highlight the practical impact of **calibration** (not just point-error).

### Hyperparameter Methods Compared
- GP-ML: Optimize GP hyperparameters by (negative) log marginal likelihood.
- GP-CV (MSE): K-fold cross-validation grid search, selecting by validation MSE.
- GP-CV (NLPD): K-fold cross-validation grid search, selecting by mean NLPD (uses predictive variances).
- KRR-CV (MSE): K-fold cross-validation grid search, selecting by validation MSE.

### Observations
- Point-prediction MSE is often similar across methods, but **predictive calibration differs materially**.
- Selecting hyperparameters by MSE alone can drive **overconfident noise/uncertainty estimates**, which degrades NLPD.

- Detailed report can be found [here](Report.pdf).

### Code
- The implementation is contained in a single notebook [here](Notebook.ipynb).
- Running all cells will:
  - load the dataset,
  - fit GP (ML + CV variants) and KRR (CV),
  - print a summary table,
  - generate diagnostic plots (parity + 1D feature slices).

### Documentation of the code/functions

#### Util functions
<details>
<summary><b>rbf_kernel</b> — Compute an RBF (squared-exponential) Gram matrix</summary>

Computes the kernel matrix between two input sets:

k(x, x') = variance · exp( -0.5 · ||x - x'||² / lengthscale² )

Inputs:
- X1: (n1, d) array
- X2: (n2, d) array
- lengthscale: ℓ > 0
- variance: σ_f² > 0

Returns:
- K: (n1, n2) array, with K[i, j] = k(X1[i], X2[j])

</details>

--------------------------------

#### GP relevant functions
<details>
<summary><b>gp_neg_log_marginal_likelihood</b> — Negative log marginal likelihood for GP regression</summary>

Computes the GP negative log marginal likelihood under an RBF kernel and Gaussian observation noise.

- Parameterization uses log-space: theta_log = [log(ℓ), log(σ_f), log(σ_n)]
- Uses Cholesky factorization of K + σ_n² I for numerical stability.

Returns:
- scalar NLL value to be minimized.

</details>

<details>
<summary><b>gp_fit_marginal_likelihood</b> — Fit GP hyperparameters via marginal likelihood</summary>

Optimizes (ℓ, σ_f, σ_n) by minimizing the negative log marginal likelihood (L-BFGS-B).

Returns a dict:
- ell, sigma_f, sigma_n: optimized hyperparameters (in the scaled feature/target space used in training)
- opt_result: full scipy optimizer result

</details>

<details>
<summary><b>gp_predict</b> — GP posterior mean and (optional) latent variance at test points</summary>

Computes GP posterior predictions for the **latent function** f(x) at X_test.

Returns:
- mean: (n_test,) posterior mean of f(X_test)
- var:  (n_test,) posterior variance of f(X_test) (if return_var=True)

Note:
- Predictive variance for an observed y adds noise: Var[y*] = Var[f*] + σ_n².

</details>

<details>
<summary><b>gaussian_nlpd</b> — Per-point negative log predictive density under a Gaussian</summary>

Computes NLPD for targets under N(y_mean, y_var), with a small variance floor for stability.

Returns:
- (n,) array of per-point NLPD contributions.

</details>

<details>
<summary><b>gp_cross_validation</b> — K-fold GP hyperparameter grid search (MSE or NLPD selection)</summary>

Runs K-fold CV over grids of (ℓ, σ_f, σ_n) for an RBF GP.

For each grid point, it computes:
- CV MSE of the predictive mean
- CV mean NLPD using predictive variance: Var[y_val] = Var[f_val] + σ_n²

Selection:
- objective="mse": selects hyperparameters minimizing CV MSE
- objective="nlpd": selects hyperparameters minimizing CV NLPD

Returns:
- best_params: dict with ell, sigma_f, sigma_n
- best_mse: CV MSE at the selected point (scaled y)
- best_nlpd: CV NLPD at the selected point (scaled y)
- cv_results: DataFrame with all grid evaluations (sorted by the selection metric)

</details>

-------------------------

#### KRR relevant functions
<details>
<summary><b>krr_fit</b> — Fit Kernel Ridge Regression coefficients (dual form)</summary>

Fits KRR with an RBF kernel using the closed-form dual solution:

α = (K + λ_eff I)^{-1} y

where λ_eff is the diagonal shift applied to the Gram matrix.

Returns a dict containing:
- X_train, coef (α), ell, sigma_f, lambd_eff

</details>

<details>
<summary><b>krr_predict</b> — Predict with a fitted KRR model</summary>

Computes predictions at X_test via:

f̂(X_test) = K(X_train, X_test)^T α

Returns:
- (n_test,) array of predictions.

</details>

<details>
<summary><b>krr_cross_validation</b> — K-fold KRR hyperparameter grid search (select by MSE)</summary>

Runs K-fold CV grid search over:
- λ_eff (ridge/diagonal shift),
- ℓ and σ_f (RBF kernel hyperparameters),

selecting the combination with the lowest mean validation MSE (scaled y).

Returns:
- best_params: dict with lambd_eff, ell, sigma_f
- best_mse: corresponding CV MSE (scaled y)

</details>

------------------------------

#### Data loading utils
<details>
<summary><b>load_portfolio_dataset_all_period</b> — Load features/target from the Excel file</summary>

Loads the **"all period"** sheet and builds:
- X: (n, 6) array of portfolio concept weights (columns starting with "the weight")
- y: (n,) array of normalized Annual Return

Returns:
- X, y as numpy arrays.

</details>
