"""Orthogonal Matching Pursuit with protected (forced) support.

Dependency-free numpy implementation. Selection is performed on
standardized columns; the final fit is on the original scale.
Constant (zero-variance) columns are NOT standardized so the intercept
remains selectable.
"""
import numpy as np


def omp(Phi, y, max_terms, init_support=(), tol=1e-10):
    """Returns (support_indices, coef_full)."""
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, p = Phi.shape
    support = [int(s) for s in init_support]

    mu = Phi.mean(axis=0)
    sigma = Phi.std(axis=0)
    const_col = sigma < 1e-12
    sigma_safe = np.where(const_col, 1.0, sigma)
    Phi_s = (Phi - mu) / sigma_safe
    # constant columns stay as-is (not zeroed out) so they stay selectable
    Phi_s[:, const_col] = Phi[:, const_col]

    def refit(supp):
        if not supp:
            return np.zeros(0), y.copy()
        coef, *_ = np.linalg.lstsq(Phi[:, supp], y, rcond=None)
        return coef, y - Phi[:, supp] @ coef

    coef, r = refit(support)
    while len(support) < max_terms:
        scores = np.abs(Phi_s.T @ r)
        scores[support] = -np.inf
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]) or scores[j] < tol:
            break
        support.append(j)
        coef, r = refit(support)
        if float(np.linalg.norm(r)) < tol:
            break

    coef_full = np.zeros(p)
    if support:
        c, *_ = np.linalg.lstsq(Phi[:, support], y, rcond=None)
        coef_full[np.array(support)] = c
    return sorted(set(support)), coef_full


def support_terms(support_indices, feature_names):
    return [feature_names[i] for i in support_indices]


def bic_value(Phi, y, support):
    """BIC of a least-squares fit on the given support (lower is better)."""
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    supp = sorted(int(s) for s in support)
    if not supp:
        rss = float(y @ y)
    else:
        c, *_ = np.linalg.lstsq(Phi[:, supp], y, rcond=None)
        rss = float(((y - Phi[:, supp] @ c) ** 2).sum())
    rss = max(rss, 1e-12)
    return n * np.log(rss / n) + len(supp) * np.log(n)


def omp_bic(Phi, y, max_terms=12, init_support=()):
    """OMP forward pass followed by BIC backward pruning.

    Fixes greedy early mistakes under correlated dictionaries.
    Protected (init_support) terms are never pruned.
    Returns (support_indices, coef_full).
    """
    Phi = np.asarray(Phi, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = Phi.shape[0]
    prot = set(int(s) for s in init_support)
    # avoid degenerate fits: never fit more terms than n//3
    max_terms = int(min(max_terms, max(len(prot) + 1, n // 3)))

    support, _ = omp(Phi, y, max_terms=max_terms, init_support=sorted(prot))

    def bic(supp):
        return bic_value(Phi, y, supp)

    best = list(support)
    best_bic = bic(best)
    while len(best) > len(prot):
        cands = [s for s in best if s not in prot]
        scored = sorted((bic([s for s in best if s != c]), c) for c in cands)
        if scored and scored[0][0] < best_bic - 1e-9:
            best_bic, drop = scored[0]
            best = [s for s in best if s != drop]
        else:
            break

    coef_full = np.zeros(Phi.shape[1])
    if best:
        sb = sorted(best)
        c, *_ = np.linalg.lstsq(Phi[:, sb], y, rcond=None)
        coef_full[np.array(sb)] = c
    return sorted(best), coef_full
