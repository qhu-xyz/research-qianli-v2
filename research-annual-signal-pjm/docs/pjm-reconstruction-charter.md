# PJM Annual Signal — Reconstruction Charter

**Date**: 2026-03-25
**Status**: Source policy and modeling objective
**Goal**: define what we are and are not trying to reconstruct from PJM V4.6, using only reproducible inputs

---

## 1. Objective

The goal is **not** to reproduce PJM V4.6 bit-for-bit.

The goal is to build a **fully reproducible annual signal** that:

- uses only inputs we can load and rebuild from current shared data
- is benchmarked directly against the released `V4.6` annual signal
- captures the same core economic signal as V4.6
- can be benchmarked directly against V4.6 on the same annual publish cells
- beats last year's release, `V4.6`, on honest branch-level capture metrics

Short version:

- **do not clone V4.6 internals**
- **do build a reproducible density + DA history signal**
- **treat V4.6 as the release baseline to beat**
- **judge success by benchmarked branch-level capture, not formula similarity**

---

## 2. Source Policy

### 2.1 Allowed baseline inputs

The baseline reconstruction is allowed to use:

- `PJM_SPICE_DENSITY_DISTRIBUTION.parquet`
- `PJM_SPICE_CONSTRAINT_INFO.parquet`
- `PJM_SPICE_CONSTRAINT_LIMIT.parquet`
- `PJM_SPICE_SF.parquet` if needed for publishing or branch propagation
- `PJM_DA_SHADOW_PRICE.parquet`

These are the reproducible shared inputs already audited in Phase 1.

### 2.2 Disallowed baseline dependencies

The baseline reconstruction must **not** depend on:

- `flow_memo.parquet`
- `flow_onpeak.parquet`
- `flow_offpeak.parquet`
- `pw_flows/`
- simulator-only artifacts from notebook submissions
- upstream PowerWorld/PTDF simulation outputs that are not available as reproducible pipeline inputs in this repo

Reason:

- these are downstream simulator outputs, not a stable, rebuildable raw-input contract for this project
- relying on them would recreate the benchmark with hidden external dependencies rather than produce a reproducible PJM signal pipeline

### 2.3 Interpretation of density

For this reconstruction, treat `PJM_SPICE_DENSITY_DISTRIBUTION.parquet` as:

- a **reproducible pre-aggregated representation** of simulated flow behavior
- suitable as model input
- **not** an invertible representation of the exact V4.6 deviation calculation

That means:

- we may derive new density-based scores from it
- we should not promise exact recovery of V4.6 deviation values from it

---

## 3. What We Believe About V4.6

Current working conclusion:

- V4.6 deviation features are downstream of simulated flow percentages
- the currently available density parquet contains related information, but not enough information to invert the original V4.6 calculation exactly
- DA history can be reconstructed independently and used as a separate signal component

Therefore the right task is:

- **learn or design a density-derived proxy score**
- combine it with reproducible DA-history features
- benchmark the resulting annual signal against V4.6

---

## 4. Reconstruction Strategy

### 4.1 Baseline family

Start with a family of reproducible branch-level signals built from:

- density summary features
- density-tail features
- branch limit features
- DA-history features

Examples:

- right-tail max / sum blends over bins `60..100`
- branch-level max/min collapse across CIDs
- simple historical DA rank / branch SP features

### 4.2 What “simulate V4.6 flow” means here

For this repo, “simulate V4.6 flow” should be interpreted narrowly:

- reconstruct the **behavioral role** of the V4.6 flow component
- not re-run the hidden upstream PTDF/PowerWorld simulator

In practice:

- build a density-derived proxy for the flow/deviation component
- combine it with a DA component
- compare against V4.6 on the same GT

### 4.3 Exact replication is not a gate

The project does **not** require:

- exact `deviation_max`
- exact `deviation_sum`
- exact V4.6 rank parity

The project **does** require:

- honest benchmark comparison
- reproducibility
- measurable branch-level improvement or at least competitive performance

---

## 5. Success Criteria

### 5.1 Baseline success

A baseline is useful if it:

- is built only from allowed inputs
- runs reproducibly across annual cells
- achieves credible GT and universe coverage
- is a credible challenger to the released `V4.6` baseline
- matches or exceeds V4.6 on at least some annual-cell `Abs_SP@K` results

### 5.2 “Better” means

For this project, “better than V4.6” primarily means:

- higher `Abs_SP@200`
- higher `Abs_SP@400`

on the same annual cell `(planning_year, market_round, class_type)`.

Secondary support:

- equal or better dangerous-branch capture
- stronger performance on the most valuable branches
- better overlap-only reranking on shared branches

### 5.3 Not sufficient

These are not enough by themselves:

- formula resemblance to V4.6
- high correlation with V4.6 score columns
- matching V4.6 top-K overlap without improved capture

---

## 6. Immediate Modeling Phases

### Phase A: density-to-score reconstruction sweep

Build simple reproducible score candidates from density only:

- tail max
- tail sum
- weighted tail blends
- bin-window blends across `60..100`

Output:

- one score table per candidate
- one benchmark comparison against V4.6 for checkpoint slice

### Phase B: density + DA challenger

Add DA-history features:

- historical branch SP
- DA rank
- simple binding frequency

Output:

- one reproducible challenger that is better than pure density-only and is directly comparable to `V4.6`

### Phase C: branch-level model

Train a simple branch-level model from the reproducible features.

Output:

- one baseline model table
- one model-vs-V4.6 comparison at `K=200/400`

---

## 7. First Checkpoint

The first benchmark slice remains:

- `planning_year=2024-06`
- `market_round=2`
- `class_type=onpeak`

For that slice, the next worker should answer:

1. which density-only score best approximates useful flow stress?
2. does adding DA-history improve `Abs_SP@200/400`?
3. where does V4.6 still beat us on dangerous branches?
4. where do we beat V4.6 already?

---

## 8. Bottom Line

This repo is now oriented around a reproducible challenge:

- **build a better PJM annual signal from density + DA history**
- **do not depend on unreproducible simulator outputs**
- **benchmark honestly against V4.6**
