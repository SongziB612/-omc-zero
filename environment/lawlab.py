"""LawLab environment: hidden symbolic law discovery.

An agent chooses query points x in [-2,2]^2, observes y = f(x) + noise,
and must drive held-out MSE below theta_star. Every query costs budget;
exceeding the budget raises BudgetExhausted.

The target function is drawn ONLY from `seed_target`, so two experiment
groups run with the same seed see the identical target (paired design).
"""
from dataclasses import dataclass
import numpy as np

from .features import FEATURE_NAMES, TRIG_NAMES, design_matrix


@dataclass(frozen=True)
class EnvConfig:
    family: str = "trig"
    domain: tuple = ((-2.0, 2.0), (-2.0, 2.0))
    noise_std: float = 0.01
    query_cost: float = 1.0
    query_budget: int = 80
    test_grid_n: int = 21
    seed_target: int = 0


class BudgetExhausted(Exception):
    pass


class LawLabEnv:
    def __init__(self, config: EnvConfig):
        self.config = config
        self._target_terms = None      # list[(feature_name, coef)]
        self._test_X = None
        self._test_y = None
        self._queries_used = 0
        self._query_id = 0
        self._query_log = []           # list of dicts {id, x, y}
        self._noise_rng = None

    # ------------------------------------------------------------------ setup
    def reset(self):
        rng = np.random.default_rng(self.config.seed_target)
        n_terms = int(rng.integers(3, 6))  # 3..5 terms
        # ensure at least one trig term so the family keeps its character
        trig_pick = [rng.choice(TRIG_NAMES)]
        rest_pool = [n for n in FEATURE_NAMES if n not in trig_pick]
        others = list(rng.choice(rest_pool, size=n_terms - 1, replace=False))
        terms = trig_pick + others
        coefs = []
        for _ in terms:
            c = 0.0
            while abs(c) < 0.3:  # avoid near-zero terms that are pure noise
                c = float(rng.uniform(-2.0, 2.0))
            coefs.append(round(c, 2))
        self._target_terms = list(zip(terms, coefs))

        n = self.config.test_grid_n
        lo = [b[0] for b in self.config.domain]
        hi = [b[1] for b in self.config.domain]
        g1 = np.linspace(lo[0], hi[0], n)
        g2 = np.linspace(lo[1], hi[1], n)
        G1, G2 = np.meshgrid(g1, g2)
        self._test_X = np.column_stack([G1.ravel(), G2.ravel()])
        self._test_y = self._target_fn(self._test_X)

        self._queries_used = 0
        self._query_id = 0
        self._query_log = []
        # Seeded observation-noise stream: reproducible per seed_target, so
        # paired groups and kill/resume runs are bit-identical (spec §4).
        self._noise_rng = np.random.default_rng(
            (self.config.seed_target + 1) * 7919 + 13)

    # ------------------------------------------------------------------ target
    def _target_fn(self, X):
        X = np.asarray(X, dtype=float)
        names = [t for t, _ in self._target_terms]
        coefs = np.array([c for _, c in self._target_terms])
        return design_matrix(X, names) @ coefs

    @property
    def target_terms(self):
        """For oracle checks / analysis only. Agents must never touch this."""
        return list(self._target_terms)

    # ------------------------------------------------------------------ interaction
    def query(self, x):
        x = np.asarray(x, dtype=float).reshape(2)
        lo = np.array([b[0] for b in self.config.domain])
        hi = np.array([b[1] for b in self.config.domain])
        if np.any(x < lo) or np.any(x > hi):
            raise ValueError(f"query point {x} outside domain")
        if self._queries_used >= self.config.query_budget:
            raise BudgetExhausted(
                f"query budget {self.config.query_budget} exhausted")
        rng = self._noise_rng  # seeded at reset: reproducible observations
        y = float(self._target_fn(x.reshape(1, -1))[0]
                  + rng.normal(0.0, self.config.noise_std))
        self._queries_used += 1
        qid = f"q{self._query_id}"
        self._query_id += 1
        self._query_log.append({"id": qid, "x": x.tolist(), "y": y})
        return y

    def evaluate(self, predict_fn):
        """MSE of predict_fn (takes (2,) -> float) on the held-out grid."""
        preds = np.array([predict_fn(x) for x in self._test_X])
        return float(np.mean((preds - self._test_y) ** 2))

    def evaluate_batch(self, predict_batch_fn):
        """Vectorized version: predict_batch_fn takes (N,2) -> (N,)."""
        preds = np.asarray(predict_batch_fn(self._test_X), dtype=float)
        return float(np.mean((preds - self._test_y) ** 2))

    # ------------------------------------------------------------------ bookkeeping
    @property
    def queries_used(self):
        return self._queries_used

    @property
    def budget_remaining(self):
        return self.config.query_budget - self._queries_used

    @property
    def query_log(self):
        return list(self._query_log)

    @property
    def test_X(self):
        return self._test_X.copy()

    @property
    def test_y(self):
        return self._test_y.copy()
