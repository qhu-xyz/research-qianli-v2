# PJM Annual Signal Handoff

**Date**: 2026-03-25  
**Scope**: port the MISO annual-signal production structure into the PJM repo, without blindly copying MISO assumptions

## Goal

Build a PJM annual-signal pipeline that follows the same repo shape and engineering discipline as:

- `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2`

But the implementation must be PJM-native:

- PJM data sources
- PJM bridge / mapping logic
- PJM GT construction
- PJM publish contract
- PJM benchmark comparison against:
  - `TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{auction_round}`

This should not start as a full model rewrite. Start by porting the pipeline shape and proving one baseline path end to end.

Before implementation, the worker should do two things:

1. understand the existing MISO code structure and why it is organized that way
2. do a preliminary PJM data analysis to understand the actual data it will be working with and whether there are abnormalities

That preliminary analysis is mandatory. Do not start coding blindly.

## Recommended sequence

### Phase 0: Read the code and freeze the PJM problem statement

Before writing code, read and understand:

- the MISO production repo:
  - `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2`
- the PJM notebook / exploratory material:
  - `/home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/human-input/pjm-spice-signal-annual.ipynb`

The worker should be able to explain, in plain language:

- how MISO defines the branch universe
- how MISO builds GT
- how MISO separates GT coverage from model-universe coverage
- how MISO handles round-sensitive cutoff logic
- how MISO evaluates against full-quarter DA SP rather than only the filtered universe

Only after that, freeze the PJM problem statement.

Answer these first:

- What is the PJM annual planning-year grid?
- What are the auction rounds?
- Are there class types only, or class type plus another partition?
- Is there any PJM equivalent of MISO `aq1-aq4`, or is the evaluation grain different?
- What is the publish surface for `V4.6`:
  - path layout
  - schema
  - rank direction

Do not start coding until the PJM base grain is explicit.

Target output:
- one short contract note defining the PJM evaluation/publish grain

### Phase 1: Preliminary data analysis and source audit

Before building modules, do a real data audit. The goal is not just “paths exist”; it is to understand what data PJM actually has, where it is incomplete, and where it may differ materially from MISO.

Prove the existence and shape of the PJM inputs:

- density files
- bridge / constraint info files
- limit files
- SF files
- realized DA files

For DA, confirm:

- whether monthly aggregates exist
- whether daily cache is needed
- exact date coverage
- whether 2025 / current holdout is incomplete

Pay special attention to these three items:

1. **Density file date range**
- what planning years and market months exist
- whether round partitions exist for all intended periods
- whether historical years are complete
- whether there are missing months or partial rounds
- do not assume there is only one canonical density root
- current PJM density layout appears split across two roots:
  - legacy `network_model=miso/...`
  - newer `spice_version=v6/...`
- for overlapping `2024-06`, the roots are not duplicates:
  - legacy root holds `R2-R4`
  - newer root holds `R1`
- for `2025-06`, the newer root appears to hold `R1-R4`
- the worker must build one explicit root/round resolution rule before implementation

2. **PJM DA loading**
- what the canonical DA loader should be
- whether daily and monthly DA both exist
- whether `load_data_with_replacement` / backward-fill logic is required
- whether replacement/backfill changes the effective usable date range
- whether DA is naturally branch-level, CID-level, or node-level before transformation

3. **DA CID shadow-price recovery rate**
- after whatever PJM mapping / recovery ladder is defined, what fraction of actual DA shadow price can be mapped to branch-level GT
- measure this by year, quarter, round, and ctype if applicable
- identify abnormal years early, especially current / incomplete holdout years

The worker should explicitly look for abnormalities, not just average behavior:

- missing density months
- incomplete DA date coverage
- suspiciously low GT recovery in certain years
- benchmark path holes
- structural changes in later years
- differences between historical years and current holdout

Use the MISO coverage work as the pattern:

- separate GT recovery from model-universe coverage
- explain losses by cause
- do not hide missing recoverability behind filtered-universe metrics

Target output:
- one source-of-truth doc for PJM data paths and partition shapes
- one preliminary anomaly report covering:
  - density date coverage
  - DA availability / loading behavior
  - DA CID recovery rate
  - likely risky years or quarters

### Phase 2: Baseline architecture port

Port the MISO structure, not the exact code behavior:

- `ml/core`
- `ml/markets/pjm`
- `ml/products/annual`

Recommended first PJM modules:

- `ml/markets/pjm/config.py`
- `ml/markets/pjm/bridge.py`
- `ml/markets/pjm/realized_da.py`
- `ml/markets/pjm/ground_truth.py`
- `ml/markets/pjm/data_loader.py`

Do this before model work. The point is to establish:

- data ownership
- round handling
- cache handling
- GT mapping contract

### Phase 3: GT and coverage proof

Before training any model, prove that PJM GT is well-defined.

Need two separate coverage metrics:

1. **GT mapping coverage**
- how much realized DA SP can be mapped onto branch-level truth

2. **Model-universe coverage**
- how much of that mapped branch-level SP survives the model universe filter

Do not collapse these into one number.

This was a major source of confusion on MISO.

Target output:
- per year / quarter / ctype coverage table
- top missed branches outside the model universe
- root-cause split:
  - unmapped ex ante
  - mapped but filtered out
  - any PJM-specific recovery failure mode discovered during preliminary analysis

### Phase 4: Baseline model first

Do not start with the hybrid model.

Start with one simple baseline path equivalent in spirit to MISO `v0c` / `7.1b`:

- class-specific
- round-aware if PJM supports round-specific inputs
- reproducible from raw features

Important:
- do not rely on saved parquets that cannot be rebuilt
- keep the baseline publishable and explainable

Target output:
- one baseline eval artifact
- one benchmark comparison against `V4.6`

### Phase 5: Evaluation framework

Use the MISO evaluation lessons:

- evaluate at the true PJM base grain
- compare against the full DA total, not only the filtered universe
- keep both metrics:
  - in-universe conditional metric
  - grand-universe capture metric

The honest end-to-end metric is the one using total quarter DA SP as denominator.

Target output:
- normalized registry artifact
- `spec.json`
- `metrics.json`
- optional `analysis.json`

### Phase 6: Hybrid candidate only after baseline is stable

Once the baseline path is correct:

- add the base model
- add the dormant specialist
- define whether final policy is branch-level or constraint-level

Do not mix this phase into the baseline port.

## What to reuse from MISO

Reuse the engineering pattern:

- contracts first
- explicit registry schema
- explicit output schema
- release manifest
- smoke test
- round-sensitive daily cutoff handling
- strict partition failure behavior

Relevant reference repo:

- `/home/xyz/workspace/research-qianli-v2/research-annual-signal-v2`

Relevant reference areas:

- `ml/core/`
- `ml/markets/miso/`
- `docs/contracts/`
- `releases/miso/annual/7.1b/`

Also inspect `/pbase` where needed, especially for PJM DA / MCP loading and replacement behavior.

## What not to assume from MISO

Do not assume PJM has the same:

- quarter structure
- ctype semantics
- mapping tables
- fallback ladder
- benchmark schema
- path layout
- DA availability pattern
- publish contract

PJM should reuse the architecture, not inherit hidden MISO business rules.

## Main implementation traps

1. **Cutoff-day logic**
- this is easy to get subtly wrong
- define whether helper returns:
  - last included day
  - or first excluded day
- make the loader convention explicit

2. **GT coverage vs model-universe coverage**
- keep them separate in all reporting

3. **Benchmark path assumptions**
- verify `V4.6` rank direction and schema from disk
- do not assume it matches MISO `V4.4` or `7.x`

4. **Silent partition fallback**
- missing round/month partitions should usually fail, not warn-and-continue

5. **Training before GT proof**
- do not spend time on model tuning until GT and coverage are understood

6. **Assuming PJM DA behaves like MISO DA**
- PJM may require different loader behavior
- PJM backward-fill / replacement logic may be materially important
- the worker must confirm this from code and data, not from analogy

## First concrete milestone

The first serious checkpoint should be:

- one PJM slice
- one baseline eval table
- one GT mapping report
- one benchmark overlap report

Suggested slice:

- one historical planning year with complete DA
- one auction round
- both ctypes if applicable

Do not scale to all years until that one slice is understood.

## Deliverables for the next worker

1. A PJM base-grain contract doc
2. A PJM data-source contract doc
3. A preliminary anomaly report
4. A GT coverage report
5. A baseline baseline-model design note
6. A concrete implementation order for:
- `config`
- `bridge`
- `realized_da`
- `ground_truth`
- `data_loader`

## Bottom line

The right first objective is not “port the whole MISO model.”

It is:

- understand the codebase first
- understand the PJM data and its abnormalities first
- prove the PJM data and GT contracts
- prove coverage honestly
- stand up one reproducible baseline
- only then move to hybrid modeling and publication
