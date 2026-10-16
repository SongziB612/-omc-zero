import numpy as np

from agents.sparse import omp, omp_bic
from environment.features import FEATURE_NAMES, design_matrix, NAME_TO_IDX


def _synthetic(n=60, terms=("sin(2*x1)", "x1*x2"), coefs=(1.5, -0.8), seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = design_matrix(X, list(terms)) @ np.array(coefs)
    y += rng.normal(0, 0.01, size=n)
    Phi = design_matrix(X, FEATURE_NAMES)
    return Phi, y, [NAME_TO_IDX[t] for t in terms]


def test_omp_recovers_support():
    Phi, y, true_idx = _synthetic()
    supp, coef = omp(Phi, y, max_terms=6)
    assert set(true_idx).issubset(set(supp)), f"missed terms: {supp}"
    for j in true_idx:
        assert abs(coef[j]) > 0.1


def test_omp_protected_terms_forced():
    Phi, y, true_idx = _synthetic()
    prot = [NAME_TO_IDX["cos(3*x2)"]]  # a distractor, forced in
    supp, coef = omp(Phi, y, max_terms=6, init_support=prot)
    assert prot[0] in supp


def test_omp_respects_max_terms():
    Phi, y, _ = _synthetic()
    supp, _ = omp(Phi, y, max_terms=3)
    assert len(supp) <= 3


def test_omp_bic_recovers_exact_support_dense():
    # the calibration oracle scenario: dense noiseless grid, correlated dict
    import numpy as _np
    from environment.lawlab import LawLabEnv, EnvConfig
    env = LawLabEnv(EnvConfig(seed_target=0))
    env.reset()
    Phi = design_matrix(env.test_X, FEATURE_NAMES)
    supp, coef = omp_bic(Phi, env.test_y, max_terms=16)
    mse = float(_np.mean((Phi @ coef - env.test_y) ** 2))
    true = {t for t, _ in env.target_terms}
    found = {FEATURE_NAMES[i] for i in supp}
    assert true == found, f"true={sorted(true)} found={sorted(found)}"
    assert mse < 1e-10


def test_omp_bic_never_prunes_protected():
    Phi, y, _ = _synthetic()
    prot = [NAME_TO_IDX["cos(3*x2)"]]
    supp, _ = omp_bic(Phi, y, max_terms=12, init_support=prot)
    assert prot[0] in supp


def test_omp_intercept_selectable():
    # target with a constant term: OMP must be able to select "1"
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(80, 2))
    y = 2.0 + design_matrix(X, ["x1"]) @ np.array([1.0])
    Phi = design_matrix(X, FEATURE_NAMES)
    supp, coef = omp(Phi, y, max_terms=4)
    assert NAME_TO_IDX["1"] in supp
    assert abs(coef[NAME_TO_IDX["1"]] - 2.0) < 0.1
