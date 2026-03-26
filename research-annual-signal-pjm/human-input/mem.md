# Task

- port miso's overall structure in /home/xyz/workspace/research-qianli-v2/research-annual-signal-v2 to produce pjm signal in this repo.

## reference scripts and other files
- /home/xyz/workspace/research-qianli-v2/research-annual-signal-pjm/human-input/pjm-spice-signal-annual.ipynb
- TEST.Signal.PJM.SPICE_ANNUAL_V4.6.R{auction_round}

## steps
- understand data sources
    - density files
    - DA files
        - need to be cached per day
        - need to check data range
        - data from 2025py is not complete as we are still in mar 2025. 
- need slices at (year, round, ctype) level (there is no ptype in pjm right?)
- read miso's implementation
    this is my understanding of how miso's procedure works:
    - branch universe defining by filtering out low-density constraints 
    - map constraints into branches. 
    - use **mapping** to get that branch's GT
        - use annual mapping + f0 fallback ==> miso has another layer that pjm doesn't have
        - before anything, we need to make sure that pjm can achiveve roughly the same coverage as miso. check 2025 carefully as this can be an outlier
    - train model
        - one baseline model v0c/7.1b -> i say 7.1b here because 7.0b is not ctype aware. in pjm, if we can know exactly pjm annual v4/6's recipe, we can fine-tune this.but here be careful (do not use the saved parquets that we cannot reproduce)
        - then a hybrid model, one base model trained on all data, another dormant nb-12 model trained to find new signals. the final model is a blend.
    - publication
        - use 
    - evaluation:
        - comprehensive evaluation, check ~/workspace/research-qianli-v2/.claude/worktrees/annual-worktree to understand how. 
        - during evaluation, use something to evaluate a model's true capability at the grand universe (not the filtered universe). example: for whatever model, pick the top200/400 constraints/branches, see how much sp it captures. compare with that years overall shadow price to see capture rate. 

## other remarks
- check /pbase if needed and skills if needed
- the cutoff date logic is very nuanced. 

## data loading
---
  MCP Data with Backward Fill

  Use the load_data_with_replacement or load_monthly_data_with_replacement methods on any MCP loader (PJM, MISO, etc.).

  Single auction month

  from pbase.data.dataset.ftr.mcp.pjm import PjmMcp

  mcp = PjmMcp()
  df = mcp.load_data_with_replacement(
      auction_month="2026-01-01",
      market_round=1,
      period_type="f0",
  )

  Date range

  df = mcp.load_monthly_data_with_replacement(
      from_month="2025-06-01",
      to_month="2026-01-01",
      market_round=1,
      period_type="f0",
  )

  For MISO, use MisoMcp from pbase.data.dataset.ftr.mcp.miso.

  Under the hood, these methods:
  1. Load raw MCP data via load_data()
  2. Load the nodal replacement table via self._replacement_loader.load_data()
  3. Group by class_type and call prepare_backward_fill() with node_col="pnode_id", time_col="auction_date", value_col="mcp"

  ---
  DA Data with Backward Fill

  Use load_data_with_replacement on DaLmpDailyAgg or DaLmpMonthlyAgg (in pbase.data.dataset.da.base). These are abstract — use
   the RTO-specific subclass.

  DA Daily Agg — single month

  from pbase.data.dataset.da.pjm import PjmDaLmpDailyAgg  # or equivalent RTO subclass

  da_daily = PjmDaLmpDailyAgg()
  df = da_daily.load_data_with_replacement(loading_month="2026-01-01")

  DA Daily Agg — date range

  df = da_daily.load_data_with_replacement(
      from_date="2025-06-01",
      to_date="2026-01-01",
  )

  DA Monthly Agg — single month

  from pbase.data.dataset.da.pjm import PjmDaLmpMonthlyAgg  # or equivalent

  da_monthly = PjmDaLmpMonthlyAgg()
  df = da_monthly.load_data_with_replacement(loading_month="2026-01-01")

  DA Monthly Agg — date range

  df = da_monthly.load_data_with_replacement(
      from_date="2025-06-01",
      to_date="2026-01-01",
  )

  Key differences from MCP:
  - time_col is "datetime_beginning_utc" (not "auction_date")
  - value_col is "congestion_price_da_daily" for daily agg, "congestion_price_da_monthly" for monthly agg

  ---
  How Backward Fill Works

  The prepare_backward_fill() function (from pbase.data.dataset.replacement) uses the nodal replacement table to impute prices
   for nodes that were renamed/replaced. It builds a replacement graph, traces node lineage backwards, and fills missing
  prices using predecessor node data within the [st, et) window.
  *** end of data loading


  QA:
  other questions:
  - for spice, whether it is pjm/miso, DA constraint id is the same as spice id right?
  - then for DA constraint, the mapping from da to branch is based on monitored line?
  - "quarterly mapping": we need to find each quarter's mapping then **sum right?**
  15. i think you need to verify now


*** verifying lightgbm drift
• Partially.

  I accept:

  - stale caches were a real problem
  - the cache cleanup fixed most of the drift
  - CP1/CP2/CP4/CP5 now being exact is strong evidence the port is basically right

  I do not accept, yet, that the remaining drift is proven to be “pure LightGBM
  nondeterminism.”

  Why:

  - that’s still an inference
  - CP3 mean score diff +0.089 is not tiny
  - CP6 K=200 at -3.4% is much better than -10.3%, but still material enough that it should
    be explained, not just named

  What I would require before Phase C:

  1. hash / compare the exact training matrix and labels
  2. train the base model twice in the same process on the same matrix
  3. train it again in a fresh process
  4. compare predictions

  If:

  - same-process is exact
  - fresh-process drifts
    then I’m willing to call the residual drift LightGBM/process variance and move on.

  If not, there is still an upstream mismatch.

  So my answer is:

  - I accept stale-cache cleanup as the major fix
  - I do not yet accept “pure LightGBM nondeterminism” as proven
  - one more small determinism test should settle it before Phase C
  ***

❯ that is not wht i mean. i thought for miso
- we have many constraints under same branc, we map them all onto one branch (from 2 sides:
 spice constraint to branch for feature, da constraint to branch for label/target)


 ## cutoff date
  - there are 3 different ctypes, 4 rounds, each needs one signal, right?
 - check miso's v2 repo to understand how miso uses 
1. a daily cutoff to find each round's last date
2. and the data can only be loaded cutoff - 1 (check the exact logic and run test to understand how)
3. find the equivalent logic for pjm, cutoff date for each round should be specified somewhere in the code
. we need some partial April data without leaking
 - daily cache
    - we need to cache all previous daily data for faster feature build-up, similar to miso

## after this, let's prepare releasing this as the 1st pjm version, call it version 7. not 7b
- make sure each groupby has a signal (year, round, ctype)
- make sure each round gets different DA lookback data and there is no bug in the output. check
- sf matrix's index should be pnode_id, 
- check last pjm's release for sf matrix and constraint tier list and we need to match them
- need to include limit
- 
