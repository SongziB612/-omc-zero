"""Zero-experiment runner: paired 2-agent fuse test.

Usage:
  python -m experiments.run_zero --config configs/zero.yaml --outdir runs/<ts>
  python -m experiments.run_zero --config configs/zero.yaml --outdir runs/<ts> --seeds 0-2   # smoke
  python -m experiments.run_zero --config configs/zero.yaml --calibrate                       # calibration only

For each seed: run treatment (2 agents + shared KB) and control (2 isolated
agents) against the SAME target function (paired design). Budgets are
per-agent caps, identical across groups; replication queries spend the
replicator's own budget (validation has a price).
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.lawlab import LawLabEnv, EnvConfig, BudgetExhausted
from environment.features import FEATURE_NAMES
from agents.agent import Agent, AgentConfig
from culture.kb import KnowledgeBase
from evaluation.metrics_zero import learning_cost, disagreement
from evaluation.verdict import verdict_from_results


def _probe():
    g = np.linspace(-2.0, 2.0, 11)
    G1, G2 = np.meshgrid(g, g)
    return np.column_stack([G1.ravel(), G2.ravel()])


PROBE = _probe()


def parse_seeds(s):
    if s is None:
        return None
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return list(range(int(s)))


def run_group(seed, group, cfg):
    """Run one group (treatment/control) for one seed. Returns summary dict."""
    n_agents = 2
    kb = KnowledgeBase() if group == "treatment" else None
    # MATCHED task budget (the sacred invariant): identical per-agent task
    # query caps in both groups. Validation is computational and free, so
    # there is no val/task split anymore.
    task_cap = cfg["task_query_cap"]
    total_cap = task_cap
    assert total_cap == cfg["task_query_cap"], \
        "total budget must match across groups"
    agents = []
    envs = []
    for i in range(n_agents):
        env = LawLabEnv(EnvConfig(
            query_budget=total_cap,
            seed_target=seed))
        env.reset()
        envs.append(env)
        ac = AgentConfig(
            agent_id=f"{group}-a{i}",
            feature_library=FEATURE_NAMES,
            committee_size=cfg.get("committee_size", 5),
            candidate_pool_n=cfg.get("candidate_pool_n", 200),
            seed_agent=int(hashlib.sha256(
                f"{seed}/{group}/{i}".encode()).hexdigest(), 16) % (2 ** 31),
            theta_rep=cfg["theta_rep"],
            task_query_cap=task_cap)
        agents.append(Agent(ac, env, kb))

    # paired-design invariant: identical targets
    t0 = envs[0].target_terms
    assert all(e.target_terms == t0 for e in envs[1:]), "target mismatch"

    theta = cfg["theta_star"]
    first_below = [None, None]   # queries_used at first mse<=theta per agent
    exhausted = [False, False]
    reached = [False, False]     # per-agent: already at theta* (stop spending,
                                 # keep sharing)
    round_logs = []

    for rnd in range(cfg["max_rounds"]):
        rlog = {"round": rnd}
        # 1) act + propose -> publish (solved agents stop querying but keep sharing)
        for idx, a in enumerate(agents):
            if exhausted[idx] or reached[idx]:
                continue
            try:
                slog = a.step()
            except BudgetExhausted:
                exhausted[idx] = True
                continue
            rlog[f"agent{idx}_step"] = {k: v for k, v in slog.items()
                                       if k in ("disagreement", "queries_used")}
            if kb is not None:
                for rec in a.propose():
                    kb.publish(rec)
        # 1b) solved agents still publish their good news
        if kb is not None:
            for idx, a in enumerate(agents):
                if reached[idx] and not exhausted[idx]:
                    for rec in a.propose():
                        kb.publish(rec)
        # 2) evaluate (vectorized, no query cost) -> update reached
        mses = []
        for idx, a in enumerate(agents):
            m = a.env.evaluate_batch(a.predict_batch)
            mses.append(m)
            if first_below[idx] is None and m <= theta:
                first_below[idx] = a.env.queries_used
                reached[idx] = True
        # 3) pull validated -> adopt ; replicate best candidate (only if stuck)
        if kb is not None:
            for idx, a in enumerate(agents):
                if reached[idx] or exhausted[idx]:
                    continue
                for rec in kb.pull(a.cfg.agent_id, k=cfg["pull_k"]):
                    a.adopt(rec)
            for idx, a in enumerate(agents):
                if exhausted[idx] or reached[idx]:
                    continue
                # replicate only the highest-confidence unattempted candidate:
                # validation budget is scarce, don't waste it on stale beliefs
                cands = kb.candidates_from_others(a.cfg.agent_id)
                if not cands:
                    continue
                cand = max(cands, key=lambda r: r.confidence)
                try:
                    res = a.replicate(cand)
                except BudgetExhausted:
                    exhausted[idx] = True
                    continue
                kb.report_replication(cand.id, a.cfg.agent_id, res)
                if res.ran and res.success:
                    # the replicator just verified it: adopt immediately
                    a.adopt(cand)
        rlog["mse"] = mses
        rlog["queries_used"] = [a.env.queries_used for a in agents]
        if kb is not None:
            rlog["kb_status"] = kb.status_counts()
            rlog["replication_rate"] = kb.replication_rate
            rlog["comm_ops"] = dict(kb.comm_ops)
        round_logs.append(rlog)
        if all(m <= theta for m in mses):
            break

    # disagreement between the two agents on the fixed probe set
    dis = disagreement(agents[0].predict_batch(PROBE),
                       agents[1].predict_batch(PROBE))
    L = [learning_cost(first_below[i], cfg["query_cost"],
                       total_cap,
                       cfg["failed_run_penalty_factor"]) for i in range(2)]
    return {
        "seed": seed, "group": group,
        "L_agents": L,
        "L_mean": float(np.mean(L)),
        "disagreement": float(dis),
        "rounds": len(round_logs),
        "n_unreached": sum(1 for f in first_below if f is None),
        "kb_status": kb.status_counts() if kb else None,
        "replication_rate": kb.replication_rate if kb else None,
        "target_terms": t0,
        "round_logs": round_logs,
    }


def run_seed(seed, cfg, outdir):
    out = {}
    for group in ("treatment", "control"):
        res = run_group(seed, group, cfg)
        out[group] = res
        path = os.path.join(outdir, f"seed_{seed:03d}", f"{group}.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for rl in res["round_logs"]:
                f.write(json.dumps({"seed": seed, "group": group, **rl}) + "\n")
        # strip bulky logs from the returned summary (kept on disk)
        res = dict(res)
        del res["round_logs"]
        out[group] = res
    return out


def build_verdict_input(all_seeds, cfg):
    per_seed = []
    for seed, groups in all_seeds.items():
        t, c = groups["treatment"], groups["control"]
        rr = t["replication_rate"]
        per_seed.append({
            "seed": seed,
            "L_control": c["L_mean"],
            "L_treatment": t["L_mean"],
            "disagreement_control": c["disagreement"],
            "disagreement_treatment": t["disagreement"],
            "replication_rate": rr if rr == rr else float("nan"),
            "control_hit_ceiling": c["n_unreached"] > 0,
            "treatment_at_floor": t["L_mean"] <= 0.2 * cfg["task_query_cap"] * cfg["query_cost"],
        })
    return per_seed


def calibrate(cfg, n_calib=10):
    """Solitary-agent calibration: budget suggestion, oracle check,
    replication sanity (true-positive AND false-positive rate).
    Prints results; does not modify config."""
    print("== calibration: solitary agent queries-to-theta ==")
    qs = []
    for seed in range(n_calib):
        res = run_group(seed, "control", cfg)
        qs.append(res["L_agents"][0])
    qs = np.array(qs)
    for p in (50, 80, 90, 95):
        print(f"  p{p}: {np.percentile(qs, p):.1f} queries")
    p50 = float(np.percentile(qs, 50))
    p90 = float(np.percentile(qs, 90))
    print(f"  preregistered rule: task_query_cap = solitary p50 = {p50:.0f} "
          f"(p90={p90:.0f} for reference; validation is free)")
    print(f"  current config task_query_cap={cfg['task_query_cap']}")

    print("== oracle check: omp_bic on dense noiseless grid ==")
    from agents.sparse import omp_bic
    from environment.features import design_matrix
    env = LawLabEnv(EnvConfig(seed_target=0))
    env.reset()
    Phi = design_matrix(env.test_X, FEATURE_NAMES)
    supp, coef = omp_bic(Phi, env.test_y, max_terms=16)
    mse = float(np.mean((Phi @ coef - env.test_y) ** 2))
    true_support = {t for t, _ in env.target_terms}
    found = {FEATURE_NAMES[i] for i in supp}
    print(f"  oracle mse={mse:.2e} (theta*/10 = {cfg['theta_star']/10:.2e}), "
          f"support recall={len(true_support & found)}/{len(true_support)}")
    assert mse < cfg["theta_star"] / 10, "solver/library cannot fit the truth: fix first"

    print("== replication sanity: true vs garbage expressions ==")
    from environment.features import eval_expression
    from culture.kb import KnowledgeRecord, Provenance

    def _expr_record(support, coefs):
        return KnowledgeRecord(
            id="calib", kind="expression", proposition="calib expr",
            expression=lambda X, _s=support, _c=coefs: eval_expression(
                np.asarray(X, dtype=float).reshape(-1, 2), _s, _c),
            evidence={"support": support, "query_ids": [], "n_queries": 0},
            conditions="", provenance=Provenance("other", 0, [], 0),
            confidence=1.0, status="candidate")

    def _fresh_agent(seed):
        # agent with 20 own task points (validation needs >=15)
        env = LawLabEnv(EnvConfig(query_budget=500, seed_target=seed))
        env.reset()
        ac = AgentConfig(agent_id="calib", feature_library=FEATURE_NAMES,
                         seed_agent=999, theta_rep=cfg["theta_rep"],
                         task_query_cap=500)
        a = Agent(ac, env, None)
        for _ in range(20):
            a.step()
        return a, env

    tp = fp = 0
    n = 6
    for seed in range(100, 100 + n):
        a, env = _fresh_agent(seed)
        support = [t for t, _ in env.target_terms]
        coefs = [c for _, c in env.target_terms]
        r = a.replicate(_expr_record(support, coefs))
        tp += int(r.success and r.ran)
        r2 = a.replicate(_expr_record(["sin(2*x1)", "x1^3"], [5.0, -5.0]))
        fp += int(r2.success and r2.ran)
    print(f"  true-expression replication:    {tp}/{n} succeeded (want >= 0.8)")
    print(f"  garbage-expression replication: {fp}/{n} succeeded (want == 0)")
    if tp / n < 0.8:
        print("  WARNING: true-positive too low -> check theta_rep")
    if fp / n > 0:
        print("  WARNING: false-positive > 0 -> theta_rep too lenient")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seeds", default=None,
                    help="'0-19', '0,1,5' or '20' (=0..19)")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.calibrate:
        calibrate(cfg)
        return

    seeds = parse_seeds(args.seeds)
    if seeds is None:
        seeds = list(range(cfg["n_seeds"]))
    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(outdir, exist_ok=True)

    # freeze config into the run dir (preregistration artifact)
    with open(os.path.join(outdir, "zero.yaml.frozen"), "w") as f:
        yaml.safe_dump(cfg, f)

    all_seeds = {}
    t0 = time.time()
    for s in seeds:
        all_seeds[s] = run_seed(s, cfg, outdir)
        t_, c_ = all_seeds[s]["treatment"], all_seeds[s]["control"]
        print(f"seed {s:3d}: L_treat={t_['L_mean']:6.1f} "
              f"L_ctrl={c_['L_mean']:6.1f} "
              f"dis_t={t_['disagreement']:.3f} dis_c={c_['disagreement']:.3f} "
              f"kb={t_['kb_status']}", flush=True)

    per_seed = build_verdict_input(all_seeds, cfg)
    verdict = verdict_from_results(per_seed, cfg["decision"])
    verdict["elapsed_s"] = time.time() - t0
    verdict["per_seed"] = per_seed
    with open(os.path.join(outdir, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print("\n=== VERDICT ===")
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_seed"},
                     indent=2))
    print(f"run dir: {outdir}")


if __name__ == "__main__":
    main()
