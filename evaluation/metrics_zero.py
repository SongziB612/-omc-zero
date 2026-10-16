"""Pure-function metrics for the zero experiment."""
import numpy as np


def learning_cost(first_below_round_queries, query_cost, budget, penalty_factor):
    """Cost to first reach theta_star.

    first_below_round_queries: queries_used at the first round with mse<=theta,
        or None if never reached -> capped at budget*cost*penalty_factor
        (preregistered; see zero.yaml).
    """
    if first_below_round_queries is None:
        return float(budget * query_cost * penalty_factor)
    return float(first_below_round_queries * query_cost)


def median_ci(values, n_boot=10000, seed=0, ci=0.95):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    med = float(np.median(v))
    boots = np.empty(n_boot)
    n = len(v)
    for i in range(n_boot):
        boots[i] = np.median(rng.choice(v, size=n, replace=True))
    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100
    return med, float(np.percentile(boots, lo_q)), float(np.percentile(boots, hi_q))


def paired_permutation_test(diffs, n_perm=20000, seed=0):
    """One-sided test H1: mean(diffs) > 0, where
    diffs[i] = L_control_i - L_treatment_i (paired by seed)."""
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    obs = float(np.mean(d))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm_means = (signs * d).mean(axis=1)
    # one-sided "greater": treatment reduces L
    p = float(np.mean(perm_means >= obs))
    return p


def disagreement(pred_a, pred_b):
    """RMSE between two prediction vectors on a probe set (D for 2 agents)."""
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))
