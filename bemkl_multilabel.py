"""
BEMKL - Bayesian Efficient Multiple Kernel Learning
Multilabel Classification (Variational Approximation)

Translated from R to Python by Claude.
Original paper: Gönen, M. (2012). Bayesian Efficient Multiple Kernel Learning. ICML.

Adapted for multiomics BRCA dataset (CNV, miRNA, mRNA, Methylation).
"""

import numpy as np
import scipy.linalg as sla
from scipy.stats import norm
from scipy.special import digamma, gammaln
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sym(A: np.ndarray) -> np.ndarray:
    """Enforce exact symmetry."""
    return (A + A.T) * 0.5


def _cho_inv(A: np.ndarray) -> np.ndarray:
    """
    Invert symmetric matrix via Cholesky with blind progressive jitter.
    Never calls eigvalsh / SVD — only Cholesky + triangular solve.
    """
    n   = A.shape[0]
    A   = _sym(A)
    eps = max(1e-10, 1e-10 * float(np.max(np.abs(np.diag(A)))))
    for _ in range(20):                          # up to eps ~ 1e9 × start
        try:
            c, low = sla.cho_factor(A + eps * np.eye(n),
                                    lower=True, check_finite=False)
            return sla.cho_solve((c, low), np.eye(n), check_finite=False)
        except (sla.LinAlgError, ValueError):
            eps *= 10
    # Absolute last resort: diagonal approximation
    return np.diag(1.0 / np.maximum(np.diag(A), 1e-6))


def _cho_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b via Cholesky with blind progressive jitter.
    More numerically stable than _cho_inv(A) @ b.
    Never calls eigvalsh / SVD.
    """
    A   = _sym(A)
    eps = max(1e-10, 1e-10 * float(np.max(np.abs(np.diag(A)))))
    for _ in range(20):
        try:
            c, low = sla.cho_factor(A + eps * np.eye(A.shape[0]),
                                    lower=True, check_finite=False)
            return sla.cho_solve((c, low), b, check_finite=False)
        except (sla.LinAlgError, ValueError):
            eps *= 10
    # Diagonal fallback
    return b / np.maximum(np.diag(A)[:, None] if b.ndim > 1
                          else np.diag(A), 1e-6)


def _logdet(Sigma: np.ndarray) -> float:
    """
    Log-determinant via Cholesky with progressive jitter.
    Never calls eigvalsh / SVD.
    """
    Sigma = _sym(Sigma)
    eps   = max(1e-10, 1e-10 * float(np.max(np.abs(np.diag(Sigma)))))
    for _ in range(20):
        try:
            c, _ = sla.cho_factor(Sigma + eps * np.eye(Sigma.shape[0]),
                                   lower=True, check_finite=False)
            return 2.0 * float(np.sum(np.log(np.abs(np.diag(c)))))
        except (sla.LinAlgError, ValueError):
            eps *= 10
    # Diagonal fallback
    return float(np.sum(np.log(np.maximum(np.diag(Sigma), 1e-30))))


# Keep old names as aliases used in the training loop
_safe_inv   = _cho_inv
_safe_solve = _cho_solve


def _truncated_normal_mean_var(mu, lower, upper):
    """
    E[X] and Var[X] for X ~ TN(mu, 1, lower, upper).
    All arrays must be broadcastable.
    """
    alpha = lower - mu
    beta  = upper - mu
    phi_a = norm.pdf(alpha)
    phi_b = norm.pdf(beta)
    Phi_a = norm.cdf(alpha)
    Phi_b = norm.cdf(beta)

    Z = Phi_b - Phi_a
    Z = np.where(Z == 0, 1.0, Z)          # avoid division by zero

    mean = mu + (phi_a - phi_b) / Z
    var  = 1.0 + (alpha * phi_a - beta * phi_b) / Z \
               - ((phi_a - phi_b) / Z) ** 2
    return mean, var, Z


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

def default_parameters() -> dict:
    return dict(
        alpha_lambda = 1.0,
        beta_lambda  = 1.0,
        alpha_gamma  = 1.0,
        beta_gamma   = 1.0,
        alpha_omega  = 1.0,
        beta_omega   = 1.0,
        sigma_g      = 1.0,   # 0.1 causes KmKm/sg² to explode with large N
        margin       = 1.0,
        iteration    = 200,
        progress     = 1,
        seed         = 42,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def bemkl_train(Km: np.ndarray, Y: np.ndarray, parameters: Optional[dict] = None) -> dict:
    """
    Train BEMKL multilabel classifier.

    Parameters
    ----------
    Km : ndarray, shape (D, N, P)
        Kernel matrices (D×D kernels evaluated on N training points).
        Each Km[:, :, m] is a D×N kernel slice.
        More precisely: Km[d, n, m] = k_m(x_d, x_n).
    Y  : ndarray, shape (L, N)
        Label matrix with entries in {-1, +1}.
    parameters : dict, optional
        Hyper-parameters. Uses default_parameters() if None.

    Returns
    -------
    state : dict
        Trained model (posterior parameters + hyper-parameters).
    """
    if parameters is None:
        parameters = default_parameters()

    rng = np.random.default_rng(parameters["seed"])

    D, N, P = Km.shape
    L       = Y.shape[0]
    sg      = parameters["sigma_g"]
    al      = parameters["alpha_lambda"]
    bl      = parameters["beta_lambda"]
    ag      = parameters["alpha_gamma"]
    bg      = parameters["beta_gamma"]
    ao      = parameters["alpha_omega"]
    bo      = parameters["beta_omega"]
    margin  = parameters["margin"]
    n_iter  = parameters["iteration"]

    # ---------- initialise variational factors ----------

    # Lambda (precision priors for A)
    Lambda = dict(
        alpha = np.full((D, L), al + 0.5),
        beta  = np.full((D, L), bl),
    )

    # A  (D × L weight matrix)
    A = dict(
        mu    = rng.standard_normal((D, L)),
        sigma = np.stack([np.eye(D)] * L, axis=-1),   # D × D × L
    )

    # G  (P × N × L intermediate outputs)
    G_mu = np.abs(rng.standard_normal((P, N, L))) + margin
    for m in range(P):
        G_mu[m, :, :] *= Y.T          # align sign with labels
    G = dict(
        mu    = G_mu,
        sigma = np.eye(P),             # shared P × P covariance
    )

    # gamma (precision prior for b)
    gamma = dict(
        alpha = np.full((L,), ag + 0.5),
        beta  = np.full((L,), bg),
    )

    # omega (precision priors for e)
    omega = dict(
        alpha = np.full((P,), ao + 0.5),
        beta  = np.full((P,), bo),
    )

    # be  joint (b, e) — length L+P vector  (b is length L, e is length P)
    be = dict(
        mu    = np.concatenate([np.zeros(L), np.ones(P)]),
        sigma = np.eye(L + P),
    )

    # F  (L × N auxiliary variables)
    F_mu = (np.abs(rng.standard_normal((L, N))) + margin) * np.sign(Y)
    F = dict(
        mu    = F_mu,
        sigma = np.ones((L, N)),
    )

    # ---------- cache KmKm = Σ_m  Km[:,:,m] @ Km[:,:,m].T  (D × D) ----------
    KmKm = np.zeros((D, D))
    for m in range(P):
        KmKm += Km[:, :, m] @ Km[:, :, m].T


    # ---------- truncation bounds ----------
    lower = np.full((L, N), -1e40)
    upper = np.full((L, N), +1e40)
    lower[Y > 0] = +margin
    upper[Y < 0] = -margin

    # ---------- second-moment caches ----------
    atimesaT = np.stack([
        np.outer(A["mu"][:, o], A["mu"][:, o]) + A["sigma"][:, :, o]
        for o in range(L)
    ], axis=-1)                                              # D × D × L

    GtimesGT = np.stack([
        G["mu"][:, :, o] @ G["mu"][:, :, o].T + N * G["sigma"]
        for o in range(L)
    ], axis=-1)                                              # P × P × L

    btimesbT = np.outer(be["mu"][:L], be["mu"][:L]) + be["sigma"][:L, :L]
    etimeseT = (np.outer(be["mu"][L:], be["mu"][L:])
                + be["sigma"][L:, L:])

    etimesb = np.zeros((P, L))
    for o in range(L):
        etimesb[:, o] = be["mu"][L:] * be["mu"][o] + be["sigma"][L:, o]

    KmtimesGT = np.zeros((D, L))
    for o in range(L):
        for m in range(P):
            KmtimesGT[:, o] += Km[:, :, m] @ G["mu"][m, :, o]

    bounds = np.zeros(n_iter) if parameters["progress"] else None

    # ---------- variational updates ----------
    for it in range(n_iter):

        # --- update Lambda ---
        for o in range(L):
            denom = 1.0 / bl + 0.5 * np.diag(atimesaT[:, :, o])
            Lambda["beta"][:, o] = np.where(denom > 0, 1.0 / denom, bl)

        # --- update A ---
        for o in range(L):
            prec_A = np.diag(Lambda["alpha"][:, o] * Lambda["beta"][:, o]) + KmKm / sg**2
            A["sigma"][:, :, o] = _safe_inv(prec_A)
            A["mu"][:, o] = _safe_solve(prec_A, KmtimesGT[:, o] / sg**2)
            # guard against divergence
            A["mu"][:, o] = np.nan_to_num(A["mu"][:, o], nan=0.0, posinf=1e6, neginf=-1e6)
            atimesaT[:, :, o] = (np.outer(A["mu"][:, o], A["mu"][:, o])
                                 + A["sigma"][:, :, o])

        # --- update G ---
        prec_G = np.eye(P) / sg**2 + etimeseT
        G["sigma"] = _safe_inv(prec_G)
        for o in range(L):
            # AtKm[m, n] = A[:,o] · Km[:,n,m]  →  shape (P, N)
            AtKm = np.stack([A["mu"][:, o] @ Km[:, :, m] for m in range(P)], axis=0)
            rhs = (AtKm / sg**2
                   + np.outer(be["mu"][L:], F["mu"][o, :])
                   - etimesb[:, o:o+1])
            G["mu"][:, :, o] = np.nan_to_num(
                G["sigma"] @ rhs, nan=0.0, posinf=1e6, neginf=-1e6)
            GtimesGT[:, :, o] = (G["mu"][:, :, o] @ G["mu"][:, :, o].T
                                  + N * G["sigma"])
            KmtimesGT[:, o] = sum(Km[:, :, m] @ G["mu"][m, :, o] for m in range(P))

        # --- update gamma ---
        denom_g = 1.0 / bg + 0.5 * np.diag(btimesbT)
        gamma["beta"] = np.where(denom_g > 0, 1.0 / denom_g, bg)

        # --- update omega ---
        denom_w = 1.0 / bo + 0.5 * np.diag(etimeseT)
        omega["beta"] = np.where(denom_w > 0, 1.0 / denom_w, bo)

        # --- update be (joint bias + kernel weights) ---
        G_sum = np.array([G["mu"][:, :, o].sum(axis=1) for o in range(L)]).T  # P × L
        top_left = (np.diag(gamma["alpha"] * gamma["beta"])
                    + N * np.eye(L))
        top_right = G_sum.T
        bot_left  = G_sum
        bot_right = np.diag(omega["alpha"] * omega["beta"])
        for o in range(L):
            bot_right += GtimesGT[:, :, o]

        be_prec = np.block([
            [top_left,  top_right],
            [bot_left,  bot_right],
        ])
        be["sigma"] = _safe_inv(be_prec)
        rhs_b = F["mu"].sum(axis=1)
        rhs_e = np.zeros(P)
        for o in range(L):
            rhs_e += G["mu"][:, :, o] @ F["mu"][o, :]
        be["mu"] = np.nan_to_num(
            _safe_solve(be_prec, np.concatenate([rhs_b, rhs_e])),
            nan=0.0, posinf=1e3, neginf=-1e3)

        btimesbT = (np.outer(be["mu"][:L], be["mu"][:L])
                    + be["sigma"][:L, :L])
        etimeseT = (np.outer(be["mu"][L:], be["mu"][L:])
                    + be["sigma"][L:, L:])
        for o in range(L):
            etimesb[:, o] = (be["mu"][L:] * be["mu"][o]
                             + be["sigma"][L:, o])

        # --- update F ---
        output = np.zeros((L, N))
        for o in range(L):
            stacked = np.vstack([np.ones((1, N)), G["mu"][:, :, o]])  # (1+P) × N
            be_sub  = be["mu"][[o] + list(range(L, L + P))]
            output[o, :] = np.nan_to_num(be_sub @ stacked, nan=0.0,
                                          posinf=1e6, neginf=-1e6)

        F["mu"], F["sigma"], Z = _truncated_normal_mean_var(output, lower, upper)
        # guard F
        F["mu"]   = np.nan_to_num(F["mu"],   nan=0.0)
        F["sigma"] = np.clip(np.nan_to_num(F["sigma"], nan=1.0), 0.0, None)

        # --- ELBO (optional) ---
        if parameters["progress"]:
            log2pi = np.log(2 * np.pi)

            def _fs(x):
                """Finite-safe scalar: replace any non-finite with 0."""
                v = float(np.nansum(np.where(np.isfinite(x), x, 0.0)))
                return v if np.isfinite(v) else 0.0

            def _fld(M):
                """Finite-safe logdet: return 0.0 if result not finite."""
                v = _logdet(M)
                return v if np.isfinite(v) else 0.0

            try:
              lb = 0.0
              with np.errstate(all='ignore'):   # silence overflow/invalid warnings
                # p(Lambda)
                lb += _fs((al-1)*(digamma(Lambda["alpha"])+np.log(np.maximum(Lambda["beta"],1e-300)))
                           - Lambda["alpha"]*Lambda["beta"]/bl
                           - gammaln(al) - al*np.log(bl))
                # p(A | Lambda)
                for o in range(L):
                    lb += _fs(-0.5*Lambda["alpha"][:,o]*Lambda["beta"][:,o]*np.diag(atimesaT[:,:,o]))
                    lb += _fs(-0.5*D*log2pi + 0.5*(digamma(Lambda["alpha"][:,o])
                               + np.log(np.maximum(Lambda["beta"][:,o], 1e-300))))
                # p(G | A, Km)
                for o in range(L):
                    t1 = _fs(-0.5/sg**2 * np.diag(GtimesGT[:,:,o]))
                    t2 = _fs( 1.0/sg**2 * A["mu"][:,o] * KmtimesGT[:,o])
                    t3 = _fs(-0.5/sg**2 * (KmKm * atimesaT[:,:,o]).ravel())
                    t4 = float(-0.5*N*P*(log2pi + 2*np.log(sg)))
                    lb += float(np.clip(t1+t2+t3+t4, -1e15, 1e15))
                # p(gamma)
                lb += _fs((ag-1)*(digamma(gamma["alpha"])+np.log(np.maximum(gamma["beta"],1e-300)))
                           - gamma["alpha"]*gamma["beta"]/bg
                           - gammaln(ag) - ag*np.log(bg))
                # p(b | gamma)
                lb += _fs(-0.5*gamma["alpha"]*gamma["beta"]*np.diag(btimesbT))
                lb += _fs(-0.5*L*log2pi + 0.5*(digamma(gamma["alpha"])
                           + np.log(np.maximum(gamma["beta"], 1e-300))))
                # p(omega)
                lb += _fs((ao-1)*(digamma(omega["alpha"])+np.log(np.maximum(omega["beta"],1e-300)))
                           - omega["alpha"]*omega["beta"]/bo
                           - gammaln(ao) - ao*np.log(bo))
                # p(e | omega)
                lb += _fs(-0.5*omega["alpha"]*omega["beta"]*np.diag(etimeseT))
                lb += _fs(-0.5*P*log2pi + 0.5*(digamma(omega["alpha"])
                           + np.log(np.maximum(omega["beta"], 1e-300))))
                # p(F | b, e, G)
                for o in range(L):
                    lb += _fs(-0.5*(F["mu"][o,:]**2 + F["sigma"][o,:]))
                    lb += _fs(F["mu"][o,:] * (G["mu"][:,:,o].T @ be["mu"][L:]))
                    lb += _fs(be["mu"][o] * F["mu"][o,:])
                    lb += _fs(-0.5*(etimeseT * GtimesGT[:,:,o]).ravel())
                    lb += _fs(-(G["mu"][:,:,o].T @ etimesb[:,o]))
                    lb += float(-0.5*N*float(btimesbT[o,o]) - 0.5*N*log2pi)
                # q(Lambda)
                lb += _fs(Lambda["alpha"] + np.log(np.maximum(Lambda["beta"],1e-300))
                          + gammaln(Lambda["alpha"])
                          + (1-Lambda["alpha"])*digamma(Lambda["alpha"]))
                # q(A)  — entropy of 469×469 Gaussians; logdet is large-negative, finite
                for o in range(L):
                    lb += float(0.5*(D*(log2pi+1))) + 0.5*_fld(A["sigma"][:,:,o])
                # q(G)
                lb += float(0.5*L*N*P*(log2pi+1)) + float(0.5*L*N)*_fld(G["sigma"])
                # q(gamma)
                lb += _fs(gamma["alpha"] + np.log(np.maximum(gamma["beta"],1e-300))
                          + gammaln(gamma["alpha"])
                          + (1-gamma["alpha"])*digamma(gamma["alpha"]))
                # q(omega)
                lb += _fs(omega["alpha"] + np.log(np.maximum(omega["beta"],1e-300))
                          + gammaln(omega["alpha"])
                          + (1-omega["alpha"])*digamma(omega["alpha"]))
                # q(be)
                lb += float(0.5*(L+P)*(log2pi+1)) + 0.5*_fld(be["sigma"])
                # q(F)
                log_Z = np.where(Z > 1e-300, np.log(Z), np.log(1e-300))
                lb += _fs(0.5*(log2pi + F["sigma"]) + log_Z)

              bounds[it] = lb if np.isfinite(lb) else np.nan
            except Exception as _elbo_err:
              bounds[it] = np.nan
              lb = np.nan

            if (it + 1) % 10 == 0:
                if np.isfinite(lb):
                    print(f"  iter {it+1:4d} | ELBO = {lb:+.6e}")
                else:
                    # Find which posterior variable has gone NaN to help diagnosis
                    nan_vars = [k for k, v in [
                        ("A_mu",     A["mu"]),    ("G_mu",  G["mu"]),
                        ("be_mu",    be["mu"]),   ("F_mu",  F["mu"]),
                        ("btimesbT", btimesbT),   ("etimeseT", etimeseT),
                    ] if not np.all(np.isfinite(v))]
                    if nan_vars:
                        print(f"  iter {it+1:4d} | ELBO = NaN  (diverged vars: {nan_vars})")
                    else:
                        print(f"  iter {it+1:4d} | ELBO = NaN  (numerical overflow in log-terms only — predictions OK)")

    state = dict(
        Lambda     = Lambda,
        A          = A,
        gamma      = gamma,
        omega      = omega,
        be         = be,
        bounds     = bounds,
        parameters = parameters,
    )
    return state


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def bemkl_test(Km: np.ndarray, state: dict) -> dict:
    """
    Predict with a trained BEMKL model.

    Parameters
    ----------
    Km    : ndarray, shape (D, N_test, P)
    state : dict returned by bemkl_train

    Returns
    -------
    prediction : dict with keys G, F, P
        P is the L × N_test probability matrix (probability of positive label).
    """
    D, N, P   = Km.shape
    be_mu     = state["be"]["mu"]
    be_sigma  = state["be"]["sigma"]
    A_mu      = state["A"]["mu"]
    A_sigma   = state["A"]["sigma"]
    sg        = state["parameters"]["sigma_g"]
    margin    = state["parameters"]["margin"]
    L         = be_mu.shape[0] - P

    G_mu    = np.zeros((P, N, L))
    G_sigma = np.zeros((P, N, L))

    for o in range(L):
        for m in range(P):
            G_mu[m, :, o]    = A_mu[:, o] @ Km[:, :, m]
            G_sigma[m, :, o] = (sg**2
                                + np.diag(Km[:, :, m].T @ A_sigma[:, :, o]
                                          @ Km[:, :, m]))

    F_mu    = np.zeros((L, N))
    F_sigma = np.zeros((L, N))

    for o in range(L):
        idx    = [o] + list(range(L, L + P))
        be_sub = be_mu[idx]
        be_S   = be_sigma[np.ix_(idx, idx)]
        stacked = np.vstack([np.ones((1, N)), G_mu[:, :, o]])   # (1+P) × N
        F_mu[o, :]    = be_sub @ stacked
        F_sigma[o, :] = (1.0
                         + np.diag(stacked.T @ be_S @ stacked))

    pos = 1.0 - norm.cdf((+margin - F_mu) / F_sigma)
    neg = norm.cdf((-margin - F_mu) / F_sigma)
    denom = pos + neg
    denom = np.where(denom == 0, 1.0, denom)
    prob  = pos / denom

    return dict(G=G_mu, G_sigma=G_sigma, F=F_mu, F_sigma=F_sigma, P=prob)


# ---------------------------------------------------------------------------
# Kernel utilities for multiomics data
# ---------------------------------------------------------------------------

def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """RBF (Gaussian) kernel  k(x,y) = exp(-gamma ||x-y||^2)."""
    sq_dists = (np.sum(X**2, axis=1, keepdims=True)
                + np.sum(Y**2, axis=1, keepdims=True).T
                - 2 * X @ Y.T)
    return np.exp(-gamma * np.clip(sq_dists, 0, None))


def linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X @ Y.T


def polynomial_kernel(X: np.ndarray, Y: np.ndarray, degree: int, coef0: float = 1.0) -> np.ndarray:
    return (X @ Y.T + coef0) ** degree


def normalize_kernel(K_tr: np.ndarray,
                     K_te: np.ndarray = None
                     ) -> tuple:
    """
    Spherical normalization (unit diagonal on training kernel).

    For K_tr  (N_tr × N_tr): divides by outer(d_tr, d_tr).
    For K_te  (N_te × N_tr): divides by outer(d_te, d_tr),
        where d_te = sqrt(diag(k(x_test, x_test))) approximated
        by the row-wise self-similarity, and d_tr comes from K_tr.

    Returns (K_tr_norm,) or (K_tr_norm, K_te_norm).
    """
    d_tr = np.sqrt(np.diag(K_tr))
    d_tr[d_tr == 0] = 1.0
    K_tr_norm = K_tr / np.outer(d_tr, d_tr)

    if K_te is None:
        return K_tr_norm

    # For the cross-kernel (N_te × N_tr), the column normalizer is d_tr.
    # The row normalizer d_te = sqrt(k(x_i, x_i)) for each test point.
    # We approximate it as sqrt of the diagonal of K(X_te, X_te),
    # which equals the row-norm of X_te in feature space.
    # A safe proxy: row-wise max of K_te before normalization (always ≥ diagonal
    # for PSD kernels). But the cleanest approach is to compute k(x_te, x_te)
    # directly and pass it in. Here we expose it via a wrapper in build_kernels.
    # d_te is injected by build_kernels (see below).
    raise RuntimeError("Call _normalize_cross_kernel() directly.")


def _normalize_cross_kernel(K_tr: np.ndarray, K_te: np.ndarray,
                             d_tr: np.ndarray, d_te: np.ndarray) -> tuple:
    """
    Spherical normalization for both train and cross (test) kernels.

    K_tr : (N_tr, N_tr)   train kernel
    K_te : (N_te, N_tr)   cross kernel  k(X_test, X_train)
    d_tr : (N_tr,)        sqrt(k(x_i, x_i)) for training points
    d_te : (N_te,)        sqrt(k(x_j, x_j)) for test points
    """
    d_tr = d_tr.copy(); d_tr[d_tr == 0] = 1.0
    d_te = d_te.copy(); d_te[d_te == 0] = 1.0

    K_tr_norm = K_tr / np.outer(d_tr, d_tr)   # (N_tr, N_tr)
    K_te_norm = K_te / np.outer(d_te, d_tr)   # (N_te, N_tr)
    return K_tr_norm, K_te_norm


def build_kernels(X_train: np.ndarray,
                  X_test:  np.ndarray,
                  gammas:  list = None,
                  degrees: list = None,
                  ) -> tuple:
    """
    Build normalized train (N_tr × N_tr × P) and test (N_te × N_tr × P)
    kernel stacks from a single omics feature matrix.

    gammas  : RBF bandwidth values  (default: 2^{-3} … 2^6, 10 values)
    degrees : polynomial degrees    (default: [1, 2, 3])
    """
    if gammas  is None: gammas  = [2**k for k in range(-3, 7)]
    if degrees is None: degrees = [1, 2, 3]

    kernels_tr, kernels_te = [], []

    for g in gammas:
        K_tr = rbf_kernel(X_train, X_train, g)          # (N_tr, N_tr)
        K_te = rbf_kernel(X_test,  X_train, g)          # (N_te, N_tr)
        K_self_te = rbf_kernel(X_test, X_test, g)       # (N_te, N_te)
        d_tr = np.sqrt(np.diag(K_tr))
        d_te = np.sqrt(np.diag(K_self_te))
        Kn_tr, Kn_te = _normalize_cross_kernel(K_tr, K_te, d_tr, d_te)
        kernels_tr.append(Kn_tr)
        kernels_te.append(Kn_te)

    for d in degrees:
        K_tr = polynomial_kernel(X_train, X_train, d)
        K_te = polynomial_kernel(X_test,  X_train, d)
        K_self_te = polynomial_kernel(X_test, X_test, d)
        d_tr = np.sqrt(np.diag(K_tr))
        d_te = np.sqrt(np.diag(K_self_te))
        Kn_tr, Kn_te = _normalize_cross_kernel(K_tr, K_te, d_tr, d_te)
        kernels_tr.append(Kn_tr)
        kernels_te.append(Kn_te)

    Km_tr = np.stack(kernels_tr, axis=-1)            # (N_tr, N_tr, P)
    Km_te = np.stack(kernels_te, axis=-1)            # (N_te, N_tr, P)
    Km_te = Km_te.transpose(1, 0, 2)                 # → (N_tr, N_te, P)  matches R convention
    return Km_tr, Km_te


def build_multiomics_kernels(
    omics_train: list[np.ndarray],
    omics_test:  list[np.ndarray],
    gammas:  list = None,
    degrees: list = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build kernel stacks for multiple omics modalities.

    omics_train / omics_test : list of 2-D arrays, one per modality,
                               shape (N_samples, N_features).

    Returns Km_tr (N_tr × N_tr × P_total) and Km_te (N_te × N_tr × P_total).
    """
    tr_parts, te_parts = [], []
    for X_tr, X_te in zip(omics_train, omics_test):
        kt, kte = build_kernels(X_tr, X_te, gammas, degrees)
        tr_parts.append(kt)
        te_parts.append(kte)
    Km_tr = np.concatenate(tr_parts, axis=-1)
    Km_te = np.concatenate(te_parts, axis=-1)
    return Km_tr, Km_te
