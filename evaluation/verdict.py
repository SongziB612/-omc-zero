"""Preregistered verdict rule for the zero experiment.

Reads per-seed results and emits GO / CAUTION / STOP, executed verbatim
from the frozen config (see configs/zero.yaml -> decision.*).

  GO:      p < alpha (treatment reduces L) AND no diversity collapse
  CAUTION: p < alpha but disagreement collapsed -> sharing helps but
           homogenizes; add diversity-preserving mechanisms before scaling
  STOP:    p >= alpha -> H1 fails in this setting; do NOT scale
"""
from .metrics_zero import paired_permutation_test, median_ci


def verdict_from_results(per_seed, decision_cfg):
    """per_seed: list of dicts with keys
       L_control, L_treatment, disagreement_control, disagreement_treatment,
       plus diagnostics. Returns a verdict dict."""
    alpha = decision_cfg["alpha"]
    floor_ratio = decision_cfg["disagreement_floor_ratio"]

    diffs = [s["L_control"] - s["L_treatment"] for s in per_seed]
    p = paired_permutation_test(diffs)
    med, lo, hi = median_ci(diffs)

    dis_c = [s["disagreement_control"] for s in per_seed]
    dis_t = [s["disagreement_treatment"] for s in per_seed]
    med_dis_c = float(sum(dis_c) / len(dis_c)) if dis_c else float("nan")
    med_dis_t = float(sum(dis_t) / len(dis_t)) if dis_t else float("nan")
    ratio = (med_dis_t / med_dis_c) if med_dis_c > 0 else float("nan")

    if p < alpha and ratio >= floor_ratio:
        verdict = "GO"
    elif p < alpha:
        verdict = "CAUTION"
    else:
        verdict = "STOP"

    # diagnostics for the STOP path: is it "theory wrong" or "experiment broken"?
    rep_rates = [s.get("replication_rate") for s in per_seed
                 if s.get("replication_rate") == s.get("replication_rate")]
    ctrl_hit_ceiling = sum(1 for s in per_seed if s.get("control_hit_ceiling"))
    treat_hit_floor = sum(1 for s in per_seed if s.get("treatment_at_floor"))

    return {
        "verdict": verdict,
        "p_value": p,
        "median_L_diff_control_minus_treatment": med,
        "ci95": [lo, hi],
        "n_seeds": len(per_seed),
        "mean_disagreement_control": med_dis_c,
        "mean_disagreement_treatment": med_dis_t,
        "disagreement_ratio_treatment_over_control": ratio,
        "diagnostics": {
            "mean_replication_rate_treatment":
                float(sum(rep_rates) / len(rep_rates)) if rep_rates else float("nan"),
            "control_seeds_hitting_budget_ceiling": ctrl_hit_ceiling,
            "treatment_seeds_at_query_floor": treat_hit_floor,
        },
    }
