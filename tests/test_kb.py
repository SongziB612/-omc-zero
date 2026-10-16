import numpy as np
import pytest

from culture.kb import KnowledgeBase, KnowledgeRecord, Provenance, ReplicationResult
from environment.features import eval_expression


def _rec(proposer="a0", kind="term", coef=1.5, term="sin(2*x1)"):
    return KnowledgeRecord(
        id="", kind=kind, proposition=f"term {term}: coef={coef}",
        expression=lambda X, _t=term, _c=coef: eval_expression(
            np.asarray(X, dtype=float).reshape(-1, 2), [_t], [_c]),
        evidence={"support": [term], "coef_mean": coef,
                  "query_ids": ["q1"], "n_queries": 5, "residual_mse": 0.02},
        conditions="trig", provenance=Provenance(proposer, 0, [], 0),
        confidence=0.9, status="")


def _rep(rid, by, success=True, ran=True):
    return ReplicationResult(record_id=rid, replicator_id=by, success=success,
                             mse_on_fresh=0.01 if success else 1.0,
                             n_fresh_queries=15, fresh_query_ids=["qx"],
                             ran=ran)


def test_false_record_never_promoted_without_replication():
    kb = KnowledgeBase()
    rid = kb.publish(_rec())
    assert kb.records[rid].status == "candidate"
    assert kb.pull("a1", k=5) == []  # candidates invisible via pull


def test_one_successful_replication_validates():
    kb = KnowledgeBase()
    rid = kb.publish(_rec(proposer="a0"))
    kb.report_replication(rid, "a1", _rep(rid, "a1", success=True))
    assert kb.records[rid].status == "validated"
    pulled = kb.pull("a1", k=5)
    assert [r.id for r in pulled] == [rid]


def test_two_failures_refute():
    kb = KnowledgeBase()
    rid = kb.publish(_rec(proposer="a0"))
    kb.report_replication(rid, "a1", _rep(rid, "a1", success=False))
    assert kb.records[rid].status == "candidate"  # one failure is not enough
    kb.report_replication(rid, "a2", _rep(rid, "a2", success=False))
    assert kb.records[rid].status == "refuted"
    assert kb.pull("a1", k=5) == []


def test_self_replication_asserts():
    kb = KnowledgeBase()
    rid = kb.publish(_rec(proposer="a0"))
    with pytest.raises(AssertionError):
        kb.report_replication(rid, "a0", _rep(rid, "a0", success=True))


def test_non_run_replication_is_not_evidence():
    kb = KnowledgeBase()
    rid = kb.publish(_rec(proposer="a0"))
    kb.report_replication(rid, "a1", _rep(rid, "a1", success=False, ran=False))
    kb.report_replication(rid, "a2", _rep(rid, "a2", success=False, ran=False))
    assert kb.records[rid].status == "candidate"  # non-runs don't refute


def test_predictive_equivalence_dedup_merges():
    kb = KnowledgeBase()
    r1 = _rec(proposer="a0", coef=1.5)
    r2 = _rec(proposer="a1", coef=1.5)  # same term, same coef -> equivalent
    r2.evidence["query_ids"] = ["q2", "q3"]  # distinct evidence to merge
    r2.evidence["n_queries"] = 2
    id1 = kb.publish(r1)
    id2 = kb.publish(r2)
    assert id1 == id2
    assert len(kb.records) == 1
    # evidence merged (union of query ids)
    assert kb.records[id1].evidence["n_queries"] == 3


def test_different_coef_not_deduped():
    kb = KnowledgeBase()
    id1 = kb.publish(_rec(proposer="a0", coef=1.5))
    id2 = kb.publish(_rec(proposer="a1", coef=0.5))
    assert id1 != id2
    assert len(kb.records) == 2


def test_pull_excludes_own_and_sorts_by_confidence():
    kb = KnowledgeBase()
    r1 = _rec(proposer="a0", coef=1.5); r1.confidence = 0.5
    r2 = _rec(proposer="a0", coef=0.5); r2.confidence = 0.95  # different coef: distinct record
    id1, id2 = kb.publish(r1), kb.publish(r2)
    kb.report_replication(id1, "a1", _rep(id1, "a1", success=True))
    kb.report_replication(id2, "a1", _rep(id2, "a1", success=True))
    pulled = kb.pull("a1", k=5)
    assert [r.id for r in pulled] == [id2, id1]  # confidence order
    assert kb.pull("a0", k=5) == []  # proposer pulls nothing of its own


def test_provenance_complete():
    kb = KnowledgeBase()
    rid = kb.publish(_rec(proposer="a7"))
    p = kb.records[rid].provenance
    assert p.proposer_id == "a7"
    assert isinstance(p.generation, int)
    assert isinstance(p.parent_ids, list)
    assert isinstance(p.seed, int)


def test_candidates_from_others_excludes_attempted_and_own():
    kb = KnowledgeBase()
    id0 = kb.publish(_rec(proposer="a0", coef=1.5))
    id1 = kb.publish(_rec(proposer="a1", coef=0.5))  # distinct record
    assert id0 != id1
    cands = kb.candidates_from_others("a2")
    assert {r.id for r in cands} == {id0, id1}
    kb.report_replication(id0, "a2", _rep(id0, "a2", success=False))
    cands = kb.candidates_from_others("a2")
    assert {r.id for r in cands} == {id1}
