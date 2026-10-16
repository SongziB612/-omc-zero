# OMC Zero Experiment（第零实验）

The "fuse" experiment for the **Open-Ended Machine Civilization (OMC)** research program:
does a shared knowledge base with independent-replication promotion reduce learning
cost under strictly matched budgets? If not, do not scale to the 100-agent V0.1.

**Verdict (2026-10-16, 20 seeds): CAUTION — fuse PASSES.**
Treatment wins 17/1/2, one-sided paired permutation p = 0.00005,
median learning-cost reduction 8.25 queries (95% CI [1.75, 19.25]).
See [`docs/zero-verdict-2026-10-16.md`](docs/zero-verdict-2026-10-16.md).

## What this is

A paired 2-agent experiment in **LawLab**, a symbolic-law-discovery environment:

- **Treatment**: 2 programmatic agents (sparse symbolic regression + query-by-committee)
  + 1 shared knowledge base. Knowledge records (terms / expressions) are promoted
  `candidate → validated` only by successful independent replication on the other
  agent's data; validated terms become protected features.
- **Control**: 2 isolated agents, identical code paths, `kb=None`.
- Paired design: same seed → identical target function; per-agent task-query budgets
  hard-matched; validation is computational (free).

The key mechanism that emerged: **complementary term-level sharing** —
agent A shares 2 confident terms, agent B shares 2 different ones, and together they
solve targets that isolated agents both fail on. Not leader–follower.

## Layout

```
environment/   LawLab env (hidden trig-family targets, query budget, held-out eval)
agents/        sparse committee agent (OMP+BIC, propose/replicate/adopt)
culture/       knowledge base (record schema, promotion state machine, dedup)
evaluation/    pure-function metrics + preregistered GO/CAUTION/STOP verdict rule
experiments/   paired runner, calibration, JSONL logging, frozen configs
configs/       zero.yaml (preregistered; frozen per run)
tests/         44 tests (env/agent/KB/harness/analysis)
docs/          spec, verdict report, OMC-V0.1 research handoff (Chinese)
runs/formal20/ frozen config + verdict + raw per-seed logs (20 seeds)
```

## Reproduce

```bash
pip install numpy pyyaml pytest
python -m pytest tests/ -q            # 44 tests
python -m experiments.run_zero --config configs/zero.yaml --calibrate
python -m experiments.run_zero --config configs/zero.yaml --outdir runs/<ts> --seeds 0-19
```

## Docs (Chinese)

- `docs/zero-experiment-spec.md` — detailed spec: interfaces, calibration, test checklist,
  preregistered decision rule
- `docs/zero-verdict-2026-10-16.md` — formal verdict report
- `docs/OMC-V0.1-研究交接.md` — OMC-V0.1 research handoff (unified model, weakest
  hypotheses, experimental design, 9-step engineering plan)
