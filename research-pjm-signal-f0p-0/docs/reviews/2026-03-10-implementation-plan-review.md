# Implementation Plan Review

## Findings

1. **BLOCK**: Task 6 wires the formula feature under the wrong column name, so one of the intended 9 model features can be silently replaced with all zeros. `ml.config` defines the model on `v7_formula_score` (`implementation-plan.md:310-316`), and Task 12 tunes `v7_formula_score` (`implementation-plan.md:2715-2720`), but Task 6 tells the implementer to derive and validate `v62b_formula_score` instead (`implementation-plan.md:1352-1471`). Because `prepare_features()` also fills missing features with `0` instead of failing (`implementation-plan.md:1393-1408`), this mismatch would not crash; it would quietly train the wrong model.

2. **BLOCK**: Task 4 explicitly permits the known-bad naive DA join as a fallback, which contradicts the core project requirement. The plan architecture says the PJM-critical adaptation is the branch-level join because the naive join captures only about 46% of binding value (`implementation-plan.md:7`, `implementation-plan.md:35`). But `fetch_and_cache_month()` falls back to aggregating raw `monitored_facility` directly into `branch_name` when `constraint_info` is missing (`implementation-plan.md:1094-1113`). That can silently poison cached labels and any downstream evaluation or deployment cache built from them. This path should fail closed, or at minimum be an explicit manual-recovery path that blocks promotion.

3. **BLOCK**: The shared realized-DA cache is populated with an implicit `f0` mapping, even when it is later consumed by `f1` slices. `fetch_and_cache_month()` defaults `period_type` to `"f0"` (`implementation-plan.md:1065-1070`), and the callers in both the cache script and deployment preflight omit `period_type` when fetching missing months (`implementation-plan.md:1885-1887`, `implementation-plan.md:2943-2949`). Since `constraint_info` is stored under `period_type={P}`, the plan is assuming the branch mapping is period-invariant without proving it. If `f1` has branch mappings not present in `f0`, `f1` targets and binding-frequency sets will be silently incomplete.

4. **HIGH**: Task 3 cannot pass in the stated task order because its test imports a Task 4 symbol before Task 4 exists. The Task 3 test calls `from ml.realized_da import _fetch_raw_da` (`implementation-plan.md:669-671`), but Task 3 is scheduled before `ml.realized_da.py` is created (`implementation-plan.md:624-883`, `implementation-plan.md:3083-3084`). As written, the branch-mapping task is not independently completable or reviewable.

5. **HIGH**: Task 8's `ml/pipeline.py` is not equivalent to the actual `v2` model the rest of the plan is building. The plan says to copy/adapt the MISO pipeline and treats it as identical once `data_loader.py` handles the PJM join (`implementation-plan.md:1658-1763`). But the actual `v2` contract in Task 11 depends on per-row binding-frequency enrichment and `collect_usable_months()` (`implementation-plan.md:2288-2293`, `implementation-plan.md:2462-2474`), neither of which exists in Task 8's pipeline. If anyone uses the promised pipeline/benchmark path, it will evaluate a materially different model than the research scripts.

6. **HIGH**: The blend-search and holdout artifact contract is internally inconsistent, so the "best blend" has no single canonical home. Task 12 says the search "Saves the best blend as v1 in the registry" (`implementation-plan.md:2715`), but the same task's review checks for blend weights in `registry/.../v2/config.json` (`implementation-plan.md:2754-2765`), and Task 13 says holdout re-runs "v2 with optimized blend" (`implementation-plan.md:2777-2785`). That leaves promotion semantics undefined and makes it easy to run holdout or deployment with a different blend than the one the search selected.

7. **HIGH**: The plan dropped the upstream point-in-time provenance check for f1 snapshot features, leaving a leakage assumption untested. This plan uses f1 delivery-month logic and forecast features from both V6.2B and Spice6 (`implementation-plan.md:194-199`, `implementation-plan.md:1219-1245`, `implementation-plan.md:1638-1649`), but it never adds any task to verify that the f1 snapshots were produced pre-auction. The upstream stage5 f1 plan added a dedicated provenance task for exactly this reason. For this project, that means a meaningful leakage risk remains outside the acceptance criteria.

8. **HIGH**: Deployment validation is missing the most important output-integrity checks. Task 14's review covers importability, tier range, naming, and passthrough existence (`implementation-plan.md:3021-3073`), but it does not verify that for ML slices only `rank_ori`, `rank`, and `tier` change, nor that all other constraint columns and all shift factors remain bit-identical to V6.2B. The PJM handoff makes that invariant explicit and downstream-facing. Without it, the deployment plan can pass review while still breaking consumers that depend on exact schema, row universe, column order, or SF fidelity.

9. **MED**: Task 1's import verification cannot succeed before Task 2 creates `ml.config.py`. The plan asks the implementer to import `ml.train` immediately after copying it (`implementation-plan.md:142-151`), but the upstream `train.py` imports `LTRConfig` from `ml.config`. This is a sequencing bug in the plan itself.

10. **MED**: The blend-search space is narrower than the upstream reference without any PJM-specific justification. Task 12 hard-codes `w_dmix = 0` and only searches the `w_da`/`w_dori` edge (`implementation-plan.md:2720`), even though the referenced upstream search script explores the full `(w_da, w_dmix, w_dori)` simplex. Unless PJM-specific evidence has already ruled `density_mix_rank_value` out, this restriction can miss the best slice-specific blend.

11. **MED**: Deployment naming is still underspecified at the point where the plan expects implementation. Task 14 says to switch to PJM signal names (`implementation-plan.md:3008-3012`), but it never states the concrete output signal names or adds the handoff's recommended verification that downstream PJM code actually consumes the chosen predecessor/successor names. For a deployment task, "PJM signal names" is not specific enough.

12. **MED**: The plan's declared file/module scope is inconsistent with its execution steps. The file structure promises an adapted `ml/benchmark.py` (`implementation-plan.md:44-67`, `implementation-plan.md:108`), and Task 8 says `ml/pipeline.py` is "Used by the benchmark harness" (`implementation-plan.md:1658-1667`), but no task ever creates `ml/benchmark.py`. That leaves the documented architecture incomplete and makes the `pipeline.py` task harder to justify or review.

## Scope Checked

Reviewed against:

- `/home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0/CLAUDE.md`
- `/home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0/human-input/data-gap-audit.md`
- `/home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0/human-input/handoff-codex.md`
- `/home/xyz/workspace/research-qianli-v2/research-pjm-signal-f0p-0/human-input/handoff-claude.md`
- `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/train.py`
- `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/pipeline.py`
- `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/data_loader.py`
- `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py`
- `/home/xyz/workspace/research-qianli-v2/research-stage5-tier/docs/v70-deployment-handoff.md`
- `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/v70/inference.py`
- `/home/xyz/workspace/research-qianli-v2/research-miso-signal7/scripts/generate_v70_signal.py`
