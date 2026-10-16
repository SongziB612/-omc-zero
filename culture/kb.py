"""Civilization knowledge base (minimal, for the zero experiment).

Promotion rule (preregistered, do not tune):
  candidate --(>=1 successful replication by a DIFFERENT agent)--> validated
  candidate --(>=2 failed replications)--> refuted
An agent can never replicate its own record (asserted).

Dedup: a published record that is predictively equivalent to an existing
non-refuted record merges evidence instead of creating a new entry.
This is what keeps the KB from degenerating into an append-only RAG dump:
records are executable and equivalence is predictive, not textual.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Provenance:
    proposer_id: str
    generation: int
    parent_ids: list
    seed: int


@dataclass
class ReplicationResult:
    record_id: str
    replicator_id: str
    success: bool
    mse_on_fresh: float
    n_fresh_queries: int
    fresh_query_ids: list
    ran: bool = True  # False if replication could not run (budget exhausted);
                      # a non-run is NOT evidence and must not count as failure


@dataclass
class KnowledgeRecord:
    id: str
    kind: str                 # "term" | "expression"
    proposition: str          # human-readable
    expression: object        # callable: (N,2) -> (N,), executable predictor
    evidence: dict            # {n_queries, residual_mse, query_ids, support, ...}
    conditions: str
    provenance: Provenance
    confidence: float
    status: str               # "candidate" | "validated" | "refuted" | "superseded"
    replications: list = field(default_factory=list)
    counterexamples: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)


def _probe_grid():
    g = np.linspace(-2.0, 2.0, 11)
    G1, G2 = np.meshgrid(g, g)
    return np.column_stack([G1.ravel(), G2.ravel()])


PROBE = _probe_grid()
PRED_EQ_TOL = 1e-8


class KnowledgeBase:
    def __init__(self):
        self.records = {}       # id -> KnowledgeRecord
        self._counter = 0
        self._attempts = set()  # (record_id, agent_id) with a completed run
        self.comm_ops = {"publishes": 0, "pulls": 0, "replications": 0}

    # ------------------------------------------------------------------ publish
    def _predictive_equal(self, a: KnowledgeRecord, b: KnowledgeRecord):
        if a.expression is None or b.expression is None:
            return False
        try:
            pa = np.asarray(a.expression(PROBE), dtype=float)
            pb = np.asarray(b.expression(PROBE), dtype=float)
        except Exception:
            return False
        return bool(np.max(np.abs(pa - pb)) < PRED_EQ_TOL)

    def publish(self, record: KnowledgeRecord) -> str:
        self.comm_ops["publishes"] += 1
        for existing in self.records.values():
            if existing.status == "refuted":
                continue
            if existing.kind == record.kind and self._predictive_equal(existing, record):
                # merge evidence, keep the original id
                ev = existing.evidence
                ev["query_ids"] = list(dict.fromkeys(
                    ev.get("query_ids", []) + record.evidence.get("query_ids", [])))
                ev["n_queries"] = len(ev["query_ids"])
                ev["residual_mse"] = min(ev.get("residual_mse", float("inf")),
                                         record.evidence.get("residual_mse", float("inf")))
                return existing.id
        record.id = f"k{self._counter}"
        self._counter += 1
        record.status = "candidate"
        self.records[record.id] = record
        return record.id

    # ------------------------------------------------------------------ replicate
    def candidates_from_others(self, agent_id: str):
        return [r for r in self.records.values()
                if r.status == "candidate"
                and r.provenance.proposer_id != agent_id
                and (r.id, agent_id) not in self._attempts]

    def report_replication(self, record_id: str, agent_id: str,
                           res: ReplicationResult):
        self.comm_ops["replications"] += 1
        rec = self.records[record_id]
        assert agent_id != rec.provenance.proposer_id, \
            "an agent must never replicate its own record"
        rec.replications.append(res)
        if not res.ran:
            return  # not enough data yet: retry later, don't mark attempted
        self._attempts.add((record_id, agent_id))
        if res.success and rec.status == "candidate":
            rec.status = "validated"
        fails = sum(1 for r in rec.replications if r.ran and not r.success)
        if fails >= 2 and rec.status == "candidate":
            rec.status = "refuted"

    # ------------------------------------------------------------------ pull
    def pull(self, agent_id: str, k: int):
        self.comm_ops["pulls"] += 1
        valid = [r for r in self.records.values()
                 if r.status == "validated"
                 and r.provenance.proposer_id != agent_id]
        valid.sort(key=lambda r: r.confidence, reverse=True)
        return valid[:k]

    def get_validated(self):
        return [r for r in self.records.values() if r.status == "validated"]

    def status_counts(self):
        counts = {"candidate": 0, "validated": 0, "refuted": 0, "superseded": 0}
        for r in self.records.values():
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    @property
    def replication_rate(self):
        runs = [r for rec in self.records.values() for r in rec.replications if r.ran]
        if not runs:
            return float("nan")
        return sum(1 for r in runs if r.success) / len(runs)
