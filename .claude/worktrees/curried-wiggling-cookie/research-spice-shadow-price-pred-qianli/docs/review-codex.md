# Review of `document/design-planning.md` (Harsh Critique)

## Critical Findings

1. **The core reproducibility claim is not credible.**
   - Reference: `document/design-planning.md:151`, `document/design-planning.md:460`
   - The doc says reproducibility is "guaranteed" without persisted model artifacts. That is an overclaim. With XGBoost, distributed loading, and likely nontrivial training environment variance, config + summary manifests do not guarantee bitwise or behavioral reproducibility.
   - Missing minimum controls: fixed random seeds policy, deterministic training flags, library/build fingerprint, feature pipeline code hash, training dataset snapshot hash, and model artifact fingerprint.
   - Consequence: you cannot defend a forensic question like "reproduce exact predictions from version X".

2. **Checksum strategy is too weak for integrity and audit.**
   - Reference: `document/design-planning.md:188`
   - Only `config.json` is checksummed in `manifest.json`. That protects almost nothing. `metrics.json`, thresholds, feature importances, train manifests, and even code version can be tampered independently while still passing the active checksum check.
   - Consequence: false sense of integrity; easy silent drift.

3. **The plan contradicts itself on train manifest layout.**
   - Reference: `document/design-planning.md:124`, `document/design-planning.md:233`, `document/design-planning.md:329`, `document/design-planning.md:471`
   - Section 3.1 describes a single `train_manifest.json` in the version root.
   - Section 5.1 defines per-month/class manifests under `manifests/train_manifest_{YYYYMM}_{class_type}.json`.
   - Section 11 verification goes back to expecting root-level `train_manifest.json`.
   - Consequence: implementers will produce incompatible outputs; audit scripts will disagree with producers.

4. **"Five hard gates" are not hard because one gate is undefined.**
   - Reference: `document/design-planning.md:192`, `document/design-planning.md:200`
   - `C-RMSE` floor is "ceiling TBD". That is not a gate; it is a placeholder.
   - Consequence: promotion policy can be bypassed by interpretation.

## High-Severity Findings

1. **Gate aggregation can hide catastrophic subgroup regressions.**
   - Reference: `document/design-planning.md:207`
   - Averaging onpeak/offpeak allows one class to tank while the other compensates.
   - At minimum, enforce per-class, per-period floors and "no regression" checks.

2. **No statistical decision framework for promotion.**
   - Reference: `document/design-planning.md:204`
   - "No regression" is undefined: tolerance? confidence interval? paired test across periods? minimum effect size?
   - Consequence: noisy period variance can trigger random promote/reject decisions.

3. **Audit scripts are specified as health checks but validate too little.**
   - Reference: `document/design-planning.md:356`, `document/design-planning.md:382`, `document/design-planning.md:408`
   - Audits focus on file presence/basic schema and a config-default comparison.
   - Missing high-value checks: code commit hash consistency, dataset lineage hashes, feature list ordering/hash, threshold schema version, metric recomputation spot-checks.

4. **Hard-coded machine paths undermine portability and repeatability.**
   - Reference: `document/design-planning.md:239`, `document/design-planning.md:477`
   - `/opt/temp/...` and `/home/xyz/workspace/pmodel` are environment-specific.
   - Consequence: doc is not runnable outside one workstation.

5. **Proposed dead-code deletions are weakly justified and risky.**
   - Reference: `document/design-planning.md:433`, `document/design-planning.md:435`
   - "never instantiated" / "never used in notebooks" is not enough evidence for safe deletion.
   - Require call-graph/static reference scan + test coverage impact before deleting shared modules.

## Medium-Severity Findings

1. **Version naming format is inconsistent in examples.**
   - Reference: `document/design-planning.md:160`, `document/design-planning.md:163`
   - Format requires `{SEQ}`, but `v000-legacy-20260220` omits it.
   - Either make `SEQ` optional or fix examples.

2. **`feature_importance.json` is presented as behavior verification, but this is weak.**
   - Reference: `document/design-planning.md:155`, `document/design-planning.md:306`
   - Feature importances are unstable across retrains and are not a substitute for model identity.

3. **Doc mixes architecture, policy, and execution details without clear ownership boundaries.**
   - Reference: whole document
   - This reads like a blended RFC + runbook + to-do list. It should be split so policy and implementation cannot drift.

## What Must Be Fixed Before This Is Implementable

1. Define a single artifact contract (exact file tree + schema versions) and remove all naming contradictions.
2. Replace config-only checksum with a version-level manifest hash over all required artifacts plus code revision and environment fingerprint.
3. Reword reproducibility claims to "best effort" unless deterministic guarantees are actually implemented.
4. Finalize all gate thresholds (no TBDs), define tie/noise handling, and enforce per-class/per-period constraints.
5. Require lineage integrity: data snapshot IDs/hashes for every train and eval input.
6. Move path configuration to environment/config, remove workstation-specific paths from canonical workflow.

## Bottom Line

The document has good intent but currently overpromises and under-specifies. In its present form, it will produce a registry that looks rigorous while failing key auditability and reproducibility requirements.
