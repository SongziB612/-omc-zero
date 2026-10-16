import numpy as np
import pytest

from environment.lawlab import LawLabEnv, EnvConfig
from environment.features import FEATURE_NAMES, NAME_TO_IDX, eval_expression
from agents.agent import Agent, AgentConfig
from culture.kb import KnowledgeRecord, Provenance


def make_agent(seed_target=11, seed_agent=1, task_cap=150, kb=None):
    env = LawLabEnv(EnvConfig(seed_target=seed_target,
                              query_budget=task_cap))
    env.reset()
    cfg = AgentConfig(agent_id="test-a", feature_library=FEATURE_NAMES,
                      seed_agent=seed_agent, task_query_cap=task_cap)
    return Agent(cfg, env, kb), env


def test_solitary_agent_learns():
    a, env = make_agent(task_cap=150)
    mses = []
    for _ in range(60):
        a.step()
        if len(a.mem_X) % 10 == 0:
            mses.append(env.evaluate_batch(a.predict_batch))
    final = env.evaluate_batch(a.predict_batch)
    assert final < 0.05, f"solitary agent failed to learn: mse={final}"
    # roughly monotone improvement (allow some wiggle)
    assert mses[-1] <= mses[0]


def test_propose_records_are_executable():
    a, env = make_agent(task_cap=150)
    for _ in range(40):
        a.step()
    recs = a.propose()
    assert len(recs) > 0
    for r in recs:
        preds = np.asarray(r.expression(env.test_X))
        assert preds.shape == (441,)
        assert np.all(np.isfinite(preds))
        assert r.evidence["support"], "record must carry support terms"
        assert r.provenance.proposer_id == "test-a"


def test_propose_dedups_repeated_calls():
    a, env = make_agent(task_cap=150)
    for _ in range(40):
        a.step()
    first = a.propose()
    second = a.propose()
    assert len(first) > 0
    assert len(second) == 0  # same belief -> no new records


def _true_expression_record(env):
    """Oracle expression record (the true target): must replicate."""
    terms = env.target_terms
    support = [t for t, _ in terms]
    coefs = [c for _, c in terms]
    return KnowledgeRecord(
        id="t0", kind="expression",
        proposition="oracle expression",
        expression=lambda X, _s=support, _c=coefs: eval_expression(
            np.asarray(X, dtype=float).reshape(-1, 2), _s, _c),
        evidence={"support": support, "query_ids": [], "n_queries": 0},
        conditions="", provenance=Provenance("other", 0, [], 0),
        confidence=1.0, status="candidate")


def _garbage_expression_record():
    support = ["sin(2*x1)", "x1^3"]
    coefs = [5.0, -5.0]
    return KnowledgeRecord(
        id="t1", kind="expression", proposition="garbage expression",
        expression=lambda X, _s=support, _c=coefs: eval_expression(
            np.asarray(X, dtype=float).reshape(-1, 2), _s, _c),
        evidence={"support": support, "query_ids": [], "n_queries": 0},
        conditions="", provenance=Provenance("other", 0, [], 0),
        confidence=1.0, status="candidate")


def test_replicate_true_expression_succeeds():
    a, env = make_agent(task_cap=150)
    for _ in range(20):
        a.step()
    rec = _true_expression_record(env)
    res = a.replicate(rec)
    assert res.ran
    assert res.success, f"true expression failed: mse={res.mse_on_fresh}"
    assert res.n_fresh_queries == 0


def test_replicate_garbage_expression_fails():
    a, env = make_agent(task_cap=150)
    for _ in range(20):
        a.step()
    rec = _garbage_expression_record()
    res = a.replicate(rec)
    assert res.ran
    assert not res.success


def test_replicate_term_validates_true_term():
    # term-level replication via BIC on own data
    a, env = make_agent(task_cap=150)
    for _ in range(20):
        a.step()
    term, coef = env.target_terms[0]
    rec = KnowledgeRecord(
        id="tt", kind="term", proposition=f"term {term}",
        expression=lambda X, _t=term, _c=coef: eval_expression(
            np.asarray(X, dtype=float).reshape(-1, 2), [_t], [_c]),
        evidence={"support": [term], "coef_mean": coef,
                  "query_ids": [], "n_queries": 0},
        conditions="", provenance=Provenance("other", 0, [], 0),
        confidence=1.0, status="candidate")
    res = a.replicate(rec)
    assert res.ran and res.success


def test_replicate_term_rejects_false_term():
    a, env = make_agent(task_cap=150)
    for _ in range(20):
        a.step()
    true_terms = {t for t, _ in env.target_terms}
    absent = next(n for n in FEATURE_NAMES if n not in true_terms)
    rec = KnowledgeRecord(
        id="ft", kind="term", proposition=f"term {absent}",
        expression=lambda X, _t=absent: eval_expression(
            np.asarray(X, dtype=float).reshape(-1, 2), [_t], [1.0]),
        evidence={"support": [absent], "coef_mean": 1.0,
                  "query_ids": [], "n_queries": 0},
        conditions="", provenance=Provenance("other", 0, [], 0),
        confidence=1.0, status="candidate")
    res = a.replicate(rec)
    assert res.ran and not res.success


def test_adopt_protects_terms():
    a, env = make_agent(task_cap=150)
    rec = _true_expression_record(env)
    rec.id = "v0"
    a.adopt(rec)
    for term in rec.evidence["support"]:
        assert term in a.protected_terms
    for _ in range(20):
        a.step()
    # protected terms must stay in every committee member's support
    for term in rec.evidence["support"]:
        j = NAME_TO_IDX[term]
        assert all(j in m["support_idx"] for m in a.committee)


def test_replication_is_free_and_uses_own_data():
    # validation must not spend query budget (computational, free)
    a, env = make_agent(task_cap=150)
    for _ in range(20):
        a.step()
    before = env.queries_used
    rec = _true_expression_record(env)
    res = a.replicate(rec)
    assert res.ran and res.success
    assert env.queries_used == before
    assert res.n_fresh_queries == 0


def test_replication_needs_min_own_data():
    a, env = make_agent(task_cap=150)
    for _ in range(5):  # fewer than min_own_data_for_validation=15
        a.step()
    rec = _true_expression_record(env)
    res = a.replicate(rec)
    assert not res.ran  # retry later, not a failure
