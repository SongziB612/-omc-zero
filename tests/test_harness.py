"""Harness tests: pairing, budget invariants, determinism, logging."""
import json
import os

import numpy as np
import yaml

from experiments.run_zero import run_group, run_seed


def _cfg(**kw):
    base = {
        "theta_star": 0.01, "theta_rep": 0.05, "query_cost": 1.0,
        "task_query_cap": 25,
        "max_rounds": 25, "pull_k": 5,
        "failed_run_penalty_factor": 2.0,
        "committee_size": 5, "candidate_pool_n": 200,
    }
    base.update(kw)
    return base


def test_paired_targets_identical_across_groups():
    t = run_group(3, "treatment", _cfg())
    c = run_group(3, "control", _cfg())
    assert t["target_terms"] == c["target_terms"]


def test_budget_caps_equal_across_groups(tmp_path):
    # THE sacred invariant: per-agent task query caps identical across groups.
    # Validation is computational and free (uses own data).
    cfg = _cfg(task_query_cap=25)
    t = run_group(4, "treatment", cfg)
    c = run_group(4, "control", cfg)
    for res in (t, c):
        assert all(q <= 25 for q in
                   [rl["queries_used"][0] for rl in res["round_logs"]])


def test_run_is_deterministic(tmp_path):
    cfg = _cfg()
    r1 = run_group(5, "treatment", cfg)
    r2 = run_group(5, "treatment", cfg)
    assert r1["L_agents"] == r2["L_agents"]
    assert r1["disagreement"] == r2["disagreement"]
    assert r1["kb_status"] == r2["kb_status"]


def test_jsonl_logs_written_and_schema_versioned(tmp_path):
    cfg = _cfg()
    out = run_seed(6, cfg, str(tmp_path))
    for group in ("treatment", "control"):
        p = tmp_path / "seed_006" / f"{group}.jsonl"
        assert p.exists()
        lines = p.read_text().strip().split("\n")
        assert len(lines) == out[group]["rounds"]
        first = json.loads(lines[0])
        assert first["seed"] == 6 and first["group"] == group
        assert "mse" in first and "queries_used" in first


def test_treatment_kb_actually_used(tmp_path):
    # on at least one seed, the treatment KB should validate something:
    # otherwise the treatment == control + overhead and the test is vacuous.
    cfg = _cfg()
    validated_total = 0
    for seed in (7, 8, 9):
        t = run_group(seed, "treatment", cfg)
        validated_total += (t["kb_status"] or {}).get("validated", 0)
    assert validated_total > 0, "KB never validated anything: check propose/replicate"
