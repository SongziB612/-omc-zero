"""Programmatic LawLab agent.

Belief = committee of sparse (OMP) fits over a fixed feature library,
built on bootstrap resamples of episodic memory with varying sparsity.
Exploration = query-by-committee (query where members disagree most).

M -> K (propose): emit candidate knowledge records for consensus terms
and for the best member's full expression.
K -> M (adopt): validated terms become protected features (forced support).
"""
from dataclasses import dataclass, field
import numpy as np

from environment.features import design_matrix, eval_expression, NAME_TO_IDX
from environment.lawlab import BudgetExhausted
from culture.kb import KnowledgeRecord, Provenance, ReplicationResult
from .sparse import omp_bic, bic_value, support_terms

# First queries before the committee takes over: corners + center (fixed,
# reproducible, identical across paired groups).
INIT_POINTS = [(-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5), (0.0, 0.0)]


@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    feature_library: list
    solver: str = "omp"
    committee_size: int = 5
    candidate_pool_n: int = 200
    consensus_threshold: float = 0.8
    min_memory_for_propose: int = 10
    seed_agent: int = 0
    theta_rep: float = 0.05
    min_fresh_queries: int = 5   # legacy; free validation uses own data
    min_own_data_for_validation: int = 15
    # Propose gate: only publish the best expression when its train MSE is
    # below this (don't spam the KB with junk early beliefs).
    propose_mse_threshold: float = 0.05
    # Task query budget (matched exactly across groups; validation is free).
    task_query_cap: int = 45


@dataclass
class BeliefState:
    committee: list
    memory_n: int
    protected_terms: set
    train_mse_best: float


class Agent:
    def __init__(self, cfg: AgentConfig, env, kb=None):
        self.cfg = cfg
        self.env = env
        self.kb = kb
        self.rng = np.random.default_rng(cfg.seed_agent)
        self.mem_X = []        # list of (2,) arrays
        self.mem_y = []        # list of floats
        self.mem_qids = []     # env query ids, parallel to mem_X
        self.committee = []    # list of dicts {support_idx, coef, train_mse}
        self.protected_terms = set()
        self.adopted_ids = set()
        self.published_sigs = set()
        self._best_proposed_mse = float("inf")
        self._init_idx = 0
        self._n_task = 0   # task queries spent (step)
        lo = np.array([b[0] for b in env.config.domain])
        hi = np.array([b[1] for b in env.config.domain])
        self._lo, self._hi = lo, hi

    # ------------------------------------------------------------------ core loop
    def step(self):
        """One step: choose x, query, observe, refit. Returns a log dict."""
        if self._n_task >= self.cfg.task_query_cap:
            raise BudgetExhausted("task query cap reached")
        if len(self.mem_X) < len(INIT_POINTS):
            x = np.array(INIT_POINTS[self._init_idx], dtype=float)
            self._init_idx += 1
            disagreement = float("nan")
        else:
            pool = self.rng.uniform(self._lo, self._hi,
                                    size=(self.cfg.candidate_pool_n, 2))
            preds = self._committee_predict_batch(pool)  # (pool, members)
            disagreement = float(preds.var(axis=1).max())
            x = pool[int(np.argmax(preds.var(axis=1)))]
        y = self.env.query(x)
        self._n_task += 1
        qid = self.env.query_log[-1]["id"]
        self.mem_X.append(np.asarray(x, dtype=float))
        self.mem_y.append(float(y))
        self.mem_qids.append(qid)
        self._refit()
        return {"x": x.tolist(), "y": float(y), "qid": qid,
                "disagreement": disagreement,
                "queries_used": self.env.queries_used}

    def _design(self, X):
        return design_matrix(X, self.cfg.feature_library)

    def _refit(self):
        n = len(self.mem_X)
        if n == 0:
            self.committee = []
            return
        X = np.stack(self.mem_X)
        y = np.array(self.mem_y)
        Phi = self._design(X)
        prot_idx = sorted(NAME_TO_IDX[t] for t in self.protected_terms
                          if t in NAME_TO_IDX)
        members = []
        for i in range(self.cfg.committee_size):
            bidx = self.rng.choice(n, size=n, replace=True)
            # varying forward depth across members (diversity) + BIC prune
            supp, coef = omp_bic(Phi[bidx], y[bidx], max_terms=8 + 2 * i,
                                 init_support=prot_idx)
            train_mse = float(np.mean((Phi @ coef - y) ** 2))
            members.append({"support_idx": supp, "coef": coef,
                            "train_mse": train_mse})
        # best member first (used by propose())
        members.sort(key=lambda m: m["train_mse"])
        self.committee = members

    # ------------------------------------------------------------------ prediction
    def _member_predict_batch(self, member, X):
        Phi = self._design(np.asarray(X, dtype=float))
        return Phi @ member["coef"]

    def _committee_predict_batch(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([self._member_predict_batch(m, X)
                                for m in self.committee])

    def predict(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self.predict_batch(x)[0])

    def predict_batch(self, X):
        X = np.asarray(X, dtype=float)
        if not self.committee:
            return np.zeros(X.shape[0])
        return self._committee_predict_batch(X).mean(axis=1)

    @property
    def belief(self):
        best = self.committee[0]["train_mse"] if self.committee else float("inf")
        return BeliefState(committee=self.committee, memory_n=len(self.mem_X),
                           protected_terms=set(self.protected_terms),
                           train_mse_best=best)

    # ------------------------------------------------------------------ M -> K
    def propose(self):
        """Emit candidate knowledge records. Dedups against own history.

        Zero experiment: expression records only. (Term-level records stay
        in the schema for V0.1, where cross-target transfer makes them
        valuable; term replication is not statistically reliable at an
        affordable fresh-query budget — see calibration notes 2026-10-16.)
        A propose gate suppresses junk early beliefs.
        """
        out = []
        if len(self.mem_X) < self.cfg.min_memory_for_propose or not self.committee:
            return out
        # Term proposals: unanimous committee agreement => strong early signal.
        # These carry PARTIAL knowledge (a few confident terms) long before
        # the full expression is ready -- the complementary-sharing mechanism.
        k = len(self.committee)
        votes = {}
        coef_acc = {}
        for m in self.committee:
            for j in m["support_idx"]:
                name = self.cfg.feature_library[j]
                votes[name] = votes.get(name, 0) + 1
                coef_acc.setdefault(name, []).append(m["coef"][j])
        for name, v in votes.items():
            if v == k:  # unanimous
                coefs = np.array(coef_acc[name])
                rec = self._make_term_record(
                    name, float(coefs.mean()), float(coefs.std()))
                if rec is not None:
                    out.append(rec)
        # Expression proposal: only on substantial breakthrough.
        best = self.committee[0]
        if best["train_mse"] > self.cfg.propose_mse_threshold:
            return out
        if best["train_mse"] >= 0.5 * self._best_proposed_mse:
            return out  # no substantial breakthrough since last proposal
        self._best_proposed_mse = best["train_mse"]
        rec = self._make_expression_record(best)
        if rec is not None:
            out.append(rec)
        return out

    def _sig(self, kind, support, coefs):
        # structural signature: same support = same claim (coef jitter
        # at 2 decimals is not a new belief)
        return (kind, tuple(sorted(support)))

    def _register(self, kind, support, coefs):
        sig = self._sig(kind, support, coefs)
        if sig in self.published_sigs:
            return None
        self.published_sigs.add(sig)
        return sig

    def _base_record(self, kind, proposition, expression, support, residual_mse,
                     confidence):
        return KnowledgeRecord(
            id="", kind=kind, proposition=proposition, expression=expression,
            evidence={"n_queries": len(self.mem_X),
                      "residual_mse": float(residual_mse),
                      "query_ids": list(self.mem_qids),
                      "support": list(support),
                      "domain": "x in [-2,2]^2"},
            conditions="trig family, x in [-2,2]^2",
            provenance=Provenance(proposer_id=self.cfg.agent_id, generation=0,
                                  parent_ids=[], seed=self.cfg.seed_agent),
            confidence=float(confidence), status="candidate",
            replications=[], counterexamples=[], open_questions=[])

    def _make_term_record(self, name, coef_mean, coef_std):
        if self._register("term", [name], [coef_mean]) is None:
            return None
        expr = lambda X, _n=name, _c=coef_mean: eval_expression(
            np.asarray(X, dtype=float).reshape(-1, 2), [_n], [_c])
        prop = f"term {name}: coef={coef_mean:.3f}±{coef_std:.3f}"
        rec = self._base_record("term", prop, expr, [name],
                                residual_mse=self.committee[0]["train_mse"],
                                confidence=1.0)  # unanimous => full confidence
        rec.evidence["coef_mean"] = float(coef_mean)
        return rec

    def _make_expression_record(self, member):
        support = support_terms(member["support_idx"], self.cfg.feature_library)
        coefs = [float(member["coef"][j]) for j in member["support_idx"]]
        if self._register("expression", support, coefs) is None:
            return None
        expr = lambda X, _s=support, _c=coefs: eval_expression(
            np.asarray(X, dtype=float).reshape(-1, 2), _s, _c)
        terms = " + ".join(f"{c:.3f}*{n}" for n, c in zip(support, coefs))
        return self._base_record("expression", f"expr: {terms}", expr, support,
                                 residual_mse=member["train_mse"],
                                 confidence=1.0 / (1.0 + member["train_mse"]))

    # ------------------------------------------------------------------ replication
    def replicate(self, record):
        """Computational independent validation using OWN task data.

        Free (no query cost). Tests whether the record's claim holds on data
        the proposer never saw (different agent, different query points).
        Requires >= min_own_data_for_validation points; otherwise returns
        ran=False so the candidate can be retried later.
        """
        if len(self.mem_X) < self.cfg.min_own_data_for_validation:
            return ReplicationResult(
                record_id=record.id, replicator_id=self.cfg.agent_id,
                success=False, mse_on_fresh=float("inf"), n_fresh_queries=0,
                fresh_query_ids=[], ran=False)
        X = np.stack(self.mem_X)
        y = np.array(self.mem_y)
        if record.kind == "term":
            success, mse = self._validate_term(record, X, y)
        else:
            preds = np.asarray(record.expression(X), dtype=float)
            mse = float(np.mean((preds - y) ** 2))
            success = mse <= self.cfg.theta_rep
        return ReplicationResult(
            record_id=record.id, replicator_id=self.cfg.agent_id,
            success=bool(success), mse_on_fresh=mse,
            n_fresh_queries=0, fresh_query_ids=[], ran=True)

    def _validate_term(self, record, X, y):
        """Targeted BIC test: does the claimed term earn its place?

        Compares BIC of the best sparse fit WITHOUT the term vs WITH the
        term forced in, on the replicator's own (independent) data.
        Success = positive BIC evidence (ΔBIC > 2) AND coefficient magnitude
        agreement (lenient: existence is the main claim; adopt() refits coef).
        """
        from environment.features import design_matrix
        name = record.evidence["support"][0]
        prop_coef = float(record.evidence.get("coef_mean", 0.0))
        j = NAME_TO_IDX[name]
        Phi = self._design(X)
        lib_wo = [f for f in self.cfg.feature_library if f != name]
        Phi_wo = design_matrix(X, lib_wo)
        supp_wo, _ = omp_bic(Phi_wo, y, max_terms=12)
        supp_w, coef_w = omp_bic(Phi, y, max_terms=12, init_support=[j])
        dBIC = bic_value(Phi_wo, y, supp_wo) - bic_value(Phi, y, supp_w)
        c_est = float(coef_w[j])
        tol = 0.5 * abs(prop_coef) + 0.1
        mag_ok = abs(c_est - prop_coef) <= tol
        resid = y - Phi @ coef_w
        mse = float(np.mean(resid ** 2))
        return (dBIC > 2.0) and mag_ok, mse

    # ------------------------------------------------------------------ K -> M
    def adopt(self, record):
        """Adopt a validated record: protect its support terms."""
        if record.id in self.adopted_ids:
            return
        self.adopted_ids.add(record.id)
        self.protected_terms.update(record.evidence.get("support", []))
        self._refit()
