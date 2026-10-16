"""Analysis tests: verdict rule on synthetic data."""
import numpy as np

from evaluation.metrics_zero import (learning_cost, median_ci,
                                     paired_permutation_test)
from evaluation.verdict import verdict_from_results


def _per_seed(n, l_ctrl, l_treat, dis_c=1.0, dis_t=0.9):
    return [{"L_control": l_ctrl, "L_treatment": l_treat,
             "disagreement_control": dis_c, "disagreement_treatment": dis_t,
             "replication_rate": 0.8, "control_hit_ceiling": False,
             "treatment_at_floor": False} for _ in range(n)]


def _dec():
    return {"alpha": 0.05, "disagreement_floor_ratio": 0.5}


def test_verdict_go_on_known_effect():
    rng = np.random.default_rng(0)
    per_seed = []
    for _ in range(20):
        lc = float(rng.uniform(40, 60))
        per_seed.append({"L_control": lc, "L_treatment": lc * 0.6,
                         "disagreement_control": 1.0,
                         "disagreement_treatment": 0.9,
                         "replication_rate": 0.8,
                         "control_hit_ceiling": False,
                         "treatment_at_floor": False})
    v = verdict_from_results(per_seed, _dec())
    assert v["verdict"] == "GO", v
    assert v["p_value"] < 0.05


def test_verdict_stop_on_null():
    rng = np.random.default_rng(1)
    per_seed = []
    for _ in range(20):
        lc = float(rng.uniform(40, 60))
        lt = float(rng.uniform(40, 60))
        per_seed.append({"L_control": lc, "L_treatment": lt,
                         "disagreement_control": 1.0,
                         "disagreement_treatment": 0.9,
                         "replication_rate": 0.5,
                         "control_hit_ceiling": False,
                         "treatment_at_floor": False})
    v = verdict_from_results(per_seed, _dec())
    assert v["verdict"] == "STOP", v
    assert v["p_value"] >= 0.05


def test_verdict_caution_on_collapse():
    rng = np.random.default_rng(2)
    per_seed = []
    for _ in range(20):
        lc = float(rng.uniform(40, 60))
        per_seed.append({"L_control": lc, "L_treatment": lc * 0.6,
                         "disagreement_control": 1.0,
                         "disagreement_treatment": 0.1,  # collapsed
                         "replication_rate": 0.9,
                         "control_hit_ceiling": False,
                         "treatment_at_floor": False})
    v = verdict_from_results(per_seed, _dec())
    assert v["verdict"] == "CAUTION", v


def test_type_one_error_near_nominal():
    # null sims: rejection rate should be ~alpha (loose tolerance)
    rng = np.random.default_rng(3)
    rej = 0
    trials = 200
    for _ in range(trials):
        diffs = rng.normal(0, 10, size=20)
        if paired_permutation_test(diffs, n_perm=2000, seed=_) < 0.05:
            rej += 1
    rate = rej / trials
    assert 0.01 <= rate <= 0.12, f"type-I rate {rate} far from 0.05"


def test_learning_cost_penalty_for_unreached():
    c = learning_cost(None, query_cost=1.0, budget=80, penalty_factor=2.0)
    assert c == 160.0
    c2 = learning_cost(42, query_cost=1.0, budget=80, penalty_factor=2.0)
    assert c2 == 42.0


def test_median_ci_covers():
    med, lo, hi = median_ci([1, 2, 3, 4, 100], n_boot=2000, seed=0)
    assert med == 3.0
    assert lo <= 3.0 <= hi
