import numpy as np
import pytest

from environment.lawlab import LawLabEnv, EnvConfig, BudgetExhausted
from environment.features import FEATURE_NAMES, design_matrix


def test_same_seed_same_target():
    e1 = LawLabEnv(EnvConfig(seed_target=7))
    e1.reset()
    e2 = LawLabEnv(EnvConfig(seed_target=7))
    e2.reset()
    assert e1.target_terms == e2.target_terms
    assert np.array_equal(e1.test_X, e2.test_X)


def test_different_seed_usually_different_target():
    terms = set()
    for s in range(10):
        e = LawLabEnv(EnvConfig(seed_target=s))
        e.reset()
        terms.add(tuple(t for t, _ in e.target_terms))
    assert len(terms) > 1


def test_queries_reproducible_including_noise():
    e1 = LawLabEnv(EnvConfig(seed_target=7))
    e1.reset()
    e2 = LawLabEnv(EnvConfig(seed_target=7))
    e2.reset()
    xs = [np.array([0.5, -0.3]), np.array([-1.0, 1.0])]
    for x in xs:
        assert e1.query(x) == e2.query(x)


def test_budget_exhausted():
    e = LawLabEnv(EnvConfig(seed_target=1, query_budget=2))
    e.reset()
    e.query(np.array([0.0, 0.0]))
    e.query(np.array([0.0, 0.0]))
    with pytest.raises(BudgetExhausted):
        e.query(np.array([0.0, 0.0]))
    assert e.queries_used == 2
    assert e.budget_remaining == 0


def test_query_outside_domain():
    e = LawLabEnv(EnvConfig(seed_target=1))
    e.reset()
    with pytest.raises(ValueError):
        e.query(np.array([5.0, 0.0]))


def test_evaluate_perfect_predictor_near_zero():
    e = LawLabEnv(EnvConfig(seed_target=3, noise_std=0.0))
    e.reset()
    terms = e.target_terms
    names = [t for t, _ in terms]
    coefs = np.array([c for _, c in terms])
    mse = e.evaluate_batch(
        lambda X: design_matrix(np.asarray(X), names) @ coefs)
    assert mse < 1e-12


def test_query_log_ids_unique():
    e = LawLabEnv(EnvConfig(seed_target=1))
    e.reset()
    for _ in range(5):
        e.query(np.array([0.1, 0.2]))
    ids = [q["id"] for q in e.query_log]
    assert len(set(ids)) == 5
