## Gaussian-Processes and Kernel Ridge Regression

### Credits
- Done as part of Statistical Machine Learning Course(MATH 412) - EPFL 2025 Autumn Offering

### Team
- Giulia Scagliarini
- Leonardo Cartesegna
- Chandrasekhara Devarakonda

### Dataset
- Stock Portfolio performance Dataset obtained from UCI Machine Learning Repository. Can be accessed [here](https://archive.ics.uci.edu/dataset/390/stock+portfolio+performance).
- Copy of the data is also included in the repo [here](stock%20portfolio%20performance%20data%20set.xlsx)

### Goal of the Project
- To understand the theoretical connection between Gaussian Process Regression(GPR) and Kernel Ridge Regression(KRR). Establish the conditions for their exact equivalence

- Understand the practical importance of GPR over KRR. 

- Experiment with different hyperparameter selection methods and interpret the quantitative results comparatively. 

### Hyperparameter Methods Compared
- GP-ML(Maximum Likelihood)
- GP-CV-MSE(Cross Validation based on MSE)
- GP-CV-NLPD(Cross Validation based on mean-predicted-density)
- KRR-CV-MSE(Cross Validation based on MSE)

### Observations
- All the methods obtain nearly similar MSE values, yet significantly differ in their mean-predicitive-density
- We observe that MSE based hyperparameter selection, leads to overconfident noise variance estimates, and thereby leading to poor mean-predictive density

- Detailed report of the project can be found [here](Report.pdf)

### Code
- The code is present in the single python notebook [here](Notebook.ipynb)
- One can Simply run all cells to generate all the necesary plots and the results. 

### Documentation of the code/functions

#### Util functions
<details>
<summary>rbf_kernel - Compute RBF kernel Gram matrix between two datasets</summary>

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

</details>

--------------------------------

#### GP relevant functions
<details>
<summary>gp_neg_log_marginal_likelihood - Compute the negative log likelihood</summary>


    Negative log marginal likelihood for GP regression with RBF kernel.

    theta_log = [log(ell), log(sigma_f), log(sigma_n)]

</details>


<details>
<summary>gp_fit_marginal_likelihood - Fit GP params to minimize log likelihood</summary>


    Fit GP hyperparameters by minimizing the negative log marginal likelihood.

    Returns a dict with ell, sigma_f, sigma_n and the full optimizer result.

</details>


<details>
<summary>gp_predict - Predict mean and variance of posterior at test-point</summary>


    GP posterior predictive at X_test.

    Returns
    -------
    mean : (n_test,)
    var  : (n_test,)  (if return_var=True)

</details>


<details>
<summary>gp_cross_validation - Cross validation for GP to select hyperparameters. Multiple methods supported. </summary>


    Grid-search K-fold CV for GP regression hyperparameters (RBF kernel).

    We evaluate *both*:
      - MSE of the predictive mean
      - Mean NLPD of the predictive density p(y_val | x_val, D_train, theta)

    For GP regression with Gaussian observation noise, the predictive density of
    the noisy target y_* is:
        y_* | D, x_* ~ N(mu_f(x_*),  Var[f_*(x_*) | D] + sigma_n^2)

    Per GPML (Rasmussen & Williams), Sec. 5.4.2, using the (negative) log
    predictive density is the natural CV criterion for probabilistic regression
    models, because it accounts for both mean *and* calibrated predictive
    variances. MSE alone ignores variances.

    objective: "nlpd" (default) or "mse" determines how we pick the best model.

</details>

-------------------------
#### KRR relevant functions

<details>
<summary>krr_fit - Fit and obtain the KRR coefficients</summary>


    Kernel Ridge Regression (KRR) with an RBF kernel.

    We use the closed-form predictor:
        coef = (K + lambd_eff * I)^{-1} y

    Here `lambd_eff` is the *effective* ridge parameter (diagonal shift) in the
    kernel matrix. If you write the KRR objective as

        (1/(2n)) * ||y - f||^2 + (lambda/2) * ||f||_{H_k}^2,

    then the corresponding diagonal shift is:
        lambd_eff = n * lambda.

    This parameterization is convenient for comparing directly with GP regression,
    where the predictive mean uses (K + sigma_n^2 I)^{-1}.

</details>


<details>
<summary>krr_predict - Calculate y_pred based on fitted coef for test</summary>


    Predict with a fitted KRR model.

</details>


<details>
<summary>krr_cross_validation - Crossvalidation based hyperparam selection</summary>


    Predict with a fitted KRR model.

</details>

------------------------------
#### Data loading utils

<details>
<summary>load_portfolio_dataset_all_period - Load the dataset and sanitize data</summary>


    Load 'all period' sheet and build:
      X = weights of stock-picking concepts (n x 6)
      y = normalized annual return (n,)

    Columns are taken directly from the Excel multi-index header.

</details>


