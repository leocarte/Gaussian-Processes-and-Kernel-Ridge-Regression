"""
Gaussian Processes vs Kernel Ridge Regression
---------------------------------------------
Analysis code for the portfolio performance dataset.

Notation follows Rasmussen & Williams and the lecture notes:

- Training inputs:  X \in R^{n x D}
- Training targets: y \in R^n
- Kernel matrix:    K_{ij} = k(x_i, x_j)
- GP noise variance: sigma_n^2
- KRR regularization: lambda

We use:
- Inputs: weights of the stock-picking concepts (6 features).
- Target: normalized annual return ("all period" sheet).
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize


# -------------------------------------------------------------------
# RBF kernel
# -------------------------------------------------------------------

def rbf_kernel(X1, X2, lengthscale, variance):
    """
    Squared exponential / RBF kernel.

    k(x, x') = variance * exp( - 0.5 * ||x - x'||^2 / lengthscale^2 )

    Parameters
    ----------
    X1 : array, shape (n1, D)
    X2 : array, shape (n2, D)
    lengthscale : float > 0
    variance : float > 0   (this is sigma_f^2 in R&W)

    Returns
    -------
    K : array, shape (n1, n2)
        Kernel matrix.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    # Squared Euclidean distance matrix
    sqdist = (
        np.sum(X1**2, axis=1)[:, None]
        + np.sum(X2**2, axis=1)[None, :]
        - 2.0 * X1.dot(X2.T)
    )
    return variance * np.exp(-0.5 * sqdist / (lengthscale ** 2))


# -------------------------------------------------------------------
# Gaussian Process regression (zero mean, RBF kernel, Gaussian noise)
# -------------------------------------------------------------------

def gp_neg_log_marginal_likelihood(theta_log, X, y):
    """
    Negative log marginal likelihood for GP regression with RBF kernel.

    theta_log = [log(ell), log(sigma_f), log(sigma_n)]
    """
    ell, sigma_f, sigma_n = np.exp(theta_log)

    K = rbf_kernel(X, X, ell, sigma_f ** 2)
    Ky = K + sigma_n ** 2 * np.eye(X.shape[0])

    try:
        L = cholesky(Ky, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        # Numerical issue -> huge penalty
        return 1e10

    alpha = solve_triangular(
        L.T,
        solve_triangular(L, y, lower=True, check_finite=False),
        lower=False,
        check_finite=False,
    )

    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    n = X.shape[0]
    nll = 0.5 * y.dot(alpha) + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
    return nll


def gp_fit_marginal_likelihood(X, y, initial_log_theta=None, bounds=None):
    """
    Fit GP hyperparameters by minimizing the negative log marginal likelihood.

    Returns a dict with ell, sigma_f, sigma_n and the full optimizer result.
    """
    if initial_log_theta is None:
        y_std = np.std(y)
        if y_std == 0:
            y_std = 1.0
        initial = np.log([1.0, y_std, 0.1 * y_std])  # heuristic
    else:
        initial = np.asarray(initial_log_theta)

    if bounds is None:
        # log-params in [log(1e-3), log(1e3)]
        bounds = [(-7, 7), (-7, 7), (-7, 7)]

    res = minimize(
        gp_neg_log_marginal_likelihood,
        initial,
        args=(X, y),
        method="L-BFGS-B",
        bounds=bounds,
    )

    ell, sigma_f, sigma_n = np.exp(res.x)
    return {"ell": ell, "sigma_f": sigma_f, "sigma_n": sigma_n, "opt_result": res}


def gp_predict(X_train, y_train, X_test, ell, sigma_f, sigma_n, return_var=True):
    """
    GP posterior predictive at X_test.

    Returns
    -------
    mean : (n_test,)
    var  : (n_test,)  (if return_var=True)
    """
    K = rbf_kernel(X_train, X_train, ell, sigma_f ** 2)
    Ky = K + sigma_n ** 2 * np.eye(X_train.shape[0])

    L = cholesky(Ky, lower=True, check_finite=False)
    alpha = solve_triangular(
        L.T,
        solve_triangular(L, y_train, lower=True, check_finite=False),
        lower=False,
        check_finite=False,
    )

    K_star = rbf_kernel(X_train, X_test, ell, sigma_f ** 2)
    mean = K_star.T.dot(alpha)

    if not return_var:
        return mean

    v = solve_triangular(L, K_star, lower=True, check_finite=False)
    K_starstar = rbf_kernel(X_test, X_test, ell, sigma_f ** 2)
    cov = K_starstar - v.T.dot(v)
    var = np.clip(np.diag(cov), 0.0, np.inf)  # numeric safety

    return mean, var


def gp_cross_validation(X, y, ell_grid, sigma_f_grid, sigma_n_grid,
                        n_splits=5, random_state=0, verbose=False):
    """
    Grid-search K-fold cross-validation for GP hyperparameters.

    For each (ell, sigma_f, sigma_n), we treat them as fixed and
    evaluate CV MSE using the GP predictive mean.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_params = None
    best_mse = np.inf

    for ell in ell_grid:
        for sigma_f in sigma_f_grid:
            for sigma_n in sigma_n_grid:
                mse_folds = []
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]
                    y_pred_val = gp_predict(
                        X_tr, y_tr, X_val,
                        ell, sigma_f, sigma_n,
                        return_var=False
                    )
                    mse_folds.append(mean_squared_error(y_val, y_pred_val))

                mse_cv = float(np.mean(mse_folds))
                if verbose:
                    print(f"GP-CV ell={ell:.3g}, sigma_f={sigma_f:.3g}, "
                          f"sigma_n={sigma_n:.3g}, CV MSE={mse_cv:.4g}")

                if mse_cv < best_mse:
                    best_mse = mse_cv
                    best_params = {"ell": ell, "sigma_f": sigma_f, "sigma_n": sigma_n}

    return best_params, best_mse


# -------------------------------------------------------------------
# Kernel Ridge Regression
# -------------------------------------------------------------------

def krr_fit(X, y, lambd, ell, sigma_f):
    """
    Kernel ridge regression with RBF kernel.

    Objective:
        (1/(2n)) * ||y - f||^2 + (lambda / 2) * ||f||^2_{H_k}

    Solution:
        alpha = (K + lambda * n * I)^{-1} y

    Returns a model dict.
    """
    n = X.shape[0]
    K = rbf_kernel(X, X, ell, sigma_f ** 2)
    A = K + lambd * n * np.eye(n)
    alpha = np.linalg.solve(A, y)
    return {
        "alpha": alpha,
        "X_train": X.copy(),
        "ell": ell,
        "sigma_f": sigma_f,
        "lambd": lambd,
    }


def krr_predict(model, X_test):
    """Predict with KRR model."""
    K_star = rbf_kernel(
        model["X_train"], X_test,
        model["ell"], model["sigma_f"] ** 2
    )
    y_pred = K_star.T.dot(model["alpha"])
    return y_pred


def krr_cross_validation(X, y, lambd_grid, ell_grid, sigma_f_grid,
                         n_splits=5, random_state=0, verbose=False):
    """
    Grid-search K-fold CV for KRR hyperparameters (lambda, ell, sigma_f).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_params = None
    best_mse = np.inf

    for lambd in lambd_grid:
        for ell in ell_grid:
            for sigma_f in sigma_f_grid:
                mse_folds = []
                for train_idx, val_idx in kf.split(X):
                    X_tr, X_val = X[train_idx], X[val_idx]
                    y_tr, y_val = y[train_idx], y[val_idx]

                    model = krr_fit(X_tr, y_tr, lambd, ell, sigma_f)
                    y_pred = krr_predict(model, X_val)
                    mse_folds.append(mean_squared_error(y_val, y_pred))

                mse_cv = float(np.mean(mse_folds))
                if verbose:
                    print(
                        f"KRR-CV lambda={lambd:.3g}, ell={ell:.3g}, "
                        f"sigma_f={sigma_f:.3g}, CV MSE={mse_cv:.4g}"
                    )

                if mse_cv < best_mse:
                    best_mse = mse_cv
                    best_params = {"lambd": lambd, "ell": ell, "sigma_f": sigma_f}

    return best_params, best_mse


# -------------------------------------------------------------------
# Data loading: portfolio dataset ("all period" sheet)
# -------------------------------------------------------------------

def load_portfolio_dataset_all_period(path):
    """
    Load 'all period' sheet and build:
      X = weights of stock-picking concepts (n x 6)
      y = normalized annual return (n,)

    Columns are taken directly from the Excel multi-index header.
    """
    xls = pd.ExcelFile(path)
    df = pd.read_excel(xls, sheet_name="all period", header=[0, 1])

    # Input columns: the weight of the stock-picking concept
    weight_cols = [
        col for col in df.columns
        if col[0].startswith("the weight")
    ]

    # Target column: normalized annual return
    target_col = (
        'the normalized  investment performance indicator',
        'Annual Return',
    )

    df_sub = df[weight_cols + [target_col]].dropna()
    X = df_sub[weight_cols].to_numpy(dtype=float)
    y = df_sub[target_col].to_numpy(dtype=float)

    return X, y


# -------------------------------------------------------------------
# Full experiment: GP-ML, GP-CV, KRR-CV
# -------------------------------------------------------------------

def run_experiment_with_gp_and_krr(path, test_size=0.3, random_state=0):
    # ----- Load data -----
    X, y = load_portfolio_dataset_all_period(path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Standardize X and y (common in GP practice)
    X_scaler = StandardScaler().fit(X_train)
    X_train_s = X_scaler.transform(X_train)
    X_test_s = X_scaler.transform(X_test)

    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std == 0:
        y_std = 1.0
    y_train_s = (y_train - y_mean) / y_std

    print("Training points:", X_train_s.shape[0],
          " Test points:", X_test_s.shape[0])

    # ===========================
    # 1) GP with marginal likelihood
    # ===========================
    print("\nFitting GP (RBF kernel) by marginal likelihood...")
    gp_ml_params = gp_fit_marginal_likelihood(X_train_s, y_train_s)
    print("GP-ML hyperparameters (scaled space):")
    print(gp_ml_params)

    gp_ml_mean_s, gp_ml_var_s = gp_predict(
        X_train_s, y_train_s, X_test_s,
        gp_ml_params["ell"], gp_ml_params["sigma_f"], gp_ml_params["sigma_n"],
        return_var=True,
    )
    gp_ml_mean = y_mean + y_std * gp_ml_mean_s
    gp_ml_std = np.sqrt(gp_ml_var_s) * y_std
    gp_ml_mse = mean_squared_error(y_test, gp_ml_mean)

    # Mean negative log predictive density (NLPD)
    nlpd_terms = []
    for yi, mu, sigma in zip(y_test, gp_ml_mean, gp_ml_std):
        var = max(sigma ** 2, 1e-9)
        nlpd = 0.5 * np.log(2.0 * np.pi * var) + 0.5 * (yi - mu) ** 2 / var
        nlpd_terms.append(nlpd)
    gp_ml_nlpd = float(np.mean(nlpd_terms))

    print(f"GP-ML test MSE:  {gp_ml_mse:.4f}")
    print(f"GP-ML mean NLPD: {gp_ml_nlpd:.4f}")

    # ===========================
    # 2) GP hyperparameters via cross-validation
    # ===========================
    print("\nTuning GP hyperparameters by cross-validation (grid search)...")
    ell_grid = np.linspace(2, 4, 50)
    sigma_f_grid = np.linspace(1.2, 2, 50)
    sigma_n_grid = np.logspace(-2, -1, 50)  

    gp_cv_best, gp_cv_mse_cv = gp_cross_validation(
        X_train_s, y_train_s,
        ell_grid, sigma_f_grid, sigma_n_grid,
        n_splits=5, random_state=random_state, verbose=False,
    )
    print("Best GP-CV hyperparameters (scaled space):")
    print(gp_cv_best)
    print(f"Best GP-CV CV MSE (scaled y): {gp_cv_mse_cv:.4f}")

    gp_cv_mean_s, gp_cv_var_s = gp_predict(
        X_train_s, y_train_s, X_test_s,
        gp_cv_best["ell"], gp_cv_best["sigma_f"], gp_cv_best["sigma_n"],
        return_var=True,
    )
    gp_cv_mean = y_mean + y_std * gp_cv_mean_s
    gp_cv_std = np.sqrt(gp_cv_var_s) * y_std
    gp_cv_mse = mean_squared_error(y_test, gp_cv_mean)

    nlpd_terms = []
    for yi, mu, sigma in zip(y_test, gp_cv_mean, gp_cv_std):
        var = max(sigma ** 2, 1e-9)
        nlpd = 0.5 * np.log(2.0 * np.pi * var) + 0.5 * (yi - mu) ** 2 / var
        nlpd_terms.append(nlpd)
    gp_cv_nlpd = float(np.mean(nlpd_terms))

    print(f"GP-CV test MSE:  {gp_cv_mse:.4f}")
    print(f"GP-CV mean NLPD: {gp_cv_nlpd:.4f}")

    # ===========================
    # 3) Kernel ridge regression via CV
    # ===========================
    print("\nTuning Kernel Ridge Regression by cross-validation...")
    lambd_grid = np.logspace(-4.5, -3.5, 50)     
    ell_grid_krr = np.linspace(2, 4, 50)
    sigma_f_grid_krr = np.linspace(1.2, 2, 50)

    krr_best, krr_best_mse_cv = krr_cross_validation(
        X_train_s, y_train_s,
        lambd_grid, ell_grid_krr, sigma_f_grid_krr,
        n_splits=5, random_state=random_state, verbose=False,
    )
    print("Best KRR hyperparameters (scaled space):")
    print(krr_best)
    print(f"Best KRR CV MSE (scaled y): {krr_best_mse_cv:.4f}")

    krr_model = krr_fit(
        X_train_s, y_train_s,
        krr_best["lambd"], krr_best["ell"], krr_best["sigma_f"],
    )
    krr_pred_s = krr_predict(krr_model, X_test_s)
    krr_pred = y_mean + y_std * krr_pred_s
    krr_mse = mean_squared_error(y_test, krr_pred)
    print(f"KRR test MSE: {krr_mse:.4f}")

    # ===========================
    # 4) Direct comparison GP mean vs KRR prediction
    # ===========================
    gp_ml_mean_s_only = gp_predict(
        X_train_s, y_train_s, X_test_s,
        gp_ml_params["ell"], gp_ml_params["sigma_f"], gp_ml_params["sigma_n"],
        return_var=False,
    )
    diff = gp_ml_mean_s_only - krr_pred_s
    print("\nDifference between GP-ML mean and KRR prediction (scaled y):")
    print(f"  Mean |diff|: {np.mean(np.abs(diff)):.4e}")
    print(f"  Max  |diff|: {np.max(np.abs(diff)):.4e}")

    # ===========================
    # 5) Summary table of results
    # ===========================
    summary = pd.DataFrame(
        {
            "Test MSE": [
                gp_ml_mse,
                gp_cv_mse,
                krr_mse,
            ],
            "Mean NLPD": [
                gp_ml_nlpd,
                gp_cv_nlpd,
                np.nan,      # not defined for KRR
            ],
            "CV MSE (scaled y)": [
                np.nan,          # GP-ML doesn't use CV
                gp_cv_mse_cv,
                krr_best_mse_cv,
            ],
            "ell": [
                gp_ml_params["ell"],
                gp_cv_best["ell"],
                krr_best["ell"],
            ],
            "sigma_f": [
                gp_ml_params["sigma_f"],
                gp_cv_best["sigma_f"],
                krr_best["sigma_f"],
            ],
            "sigma_n / lambda": [
                gp_ml_params["sigma_n"],
                gp_cv_best["sigma_n"],
                krr_best["lambd"],
            ],
        },
        index=["GP-ML", "GP-CV", "KRR-CV"],
    )

    print("\n========== Summary of results ==========")
    print(summary)
    print("========================================\n")

    # Return everything for further analysis / plotting
    results = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "y_train_s": y_train_s,
        "gp_ml_params": gp_ml_params,
        "gp_ml_test_mse": gp_ml_mse,
        "gp_ml_test_nlpd": gp_ml_nlpd,
        "gp_cv_best": gp_cv_best,
        "gp_cv_test_mse": gp_cv_mse,
        "gp_cv_test_nlpd": gp_cv_nlpd,
        "krr_best": krr_best,
        "krr_test_mse": krr_mse,
    }
    return results


# -------------------------------------------------------------------
# Script entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    # Change this if the file is elsewhere
    DATA_PATH = "stock portfolio performance data set.xlsx"
    results = run_experiment_with_gp_and_krr(DATA_PATH)