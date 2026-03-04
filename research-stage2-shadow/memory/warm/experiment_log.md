# Experiment Log

| # | Version | Batch | Hypothesis | EV-VC@100 Δ | Key Δ | Promoted |
|---|---------|-------|-----------|-------------|-------|----------|
| 1 | v0001 | smoke-test-20260303-223300 | Value-weighted regressor training | WORKER FAILED — no metrics | No artifacts produced (phantom completion) | No |
| 2 | v0002 | ralph-v1-20260304-003317 | A: lr=0.03+700 trees; B: lambda=5+mcw=25 | WORKER FAILED — no metrics | Worker ignored direction, made unauthorized changes (unfroze classifier, modified evaluate.py), phantom completion | No |
| 3 | v0003 | ralph-v1-20260304-003317 | Screen: A=lr+trees, B=L2+leaves → B wins → full benchmark | +0.0034 (mean), +0.0013 (bottom-2) | Worker produced valid results on worktree branch but artifacts not on main. v0003 passes ALL committed gates (3 layers). Dirty v0/gates on main from iter-1 worker contamination discovered. | No (infra failure) |
